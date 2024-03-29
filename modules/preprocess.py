''' preprocess.py

This file provides methods to preprocess one or more prokaryotic genomes by passing in an 
input text file containing a list of GenBank accession numbers (specific to genome). The
accession numbers are used by the Entrez API to retrieve the GenBank file of the prokaryote.

The preprocessing step consists of extracting coding sequences (CDS) and noncoding sequences 
(NCS) from each genome. CDS are found by parsing the GenBank file for CDS tags. NCS are found 
by searching for all ORF-sets (set of open reading frames that have the same stop codon) within 
an intergenic region (space between genes). CDS and NCS data is stored as a Biopython SeqRecord.
'''

import random
import warnings
import pandas as pd
import seaborn as sns
from math import log
from scipy import stats
from Bio import SeqIO, Entrez
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation

from modules.utils import extract_orf_sets
from modules.extract import extract_gc_content

##########################################################################################
########################################--PUBLIC--########################################
##########################################################################################

def preprocess_genome(filename, n_genomes=None, seq = 'all', e=0.007, tol=0.2, random_state=None, OFFSET=30, noise=False, LMIN=60, email=None):
    '''
    INPUT: filename (string), LMIN=60, OFFSET=60
    OUTPUT: cds, ncs
    
        filename: txt file containing list of accession numbers
        OFFSET: number of basepairs up-stream of start codon/down-stream of stop codon (used for the TIS feature)
        LMIN: minimum length of gene/non-gene sequences (default = 60 - ref: Hoff et al.)
        cds: list of SeqRecords containing information about gene sequences
        ncs: list of SeqRecords containing information about non-gene sequences
    '''
    
    Entrez.email = email
    
    # get input genome accession numbers
    prok_ids = get_input_ids(filename)
    
    if n_genomes > len(prok_ids): n_genomes = None
    
    _cds, _ncs = [], []
    for acc_num in prok_ids[0:n_genomes]:
        
        # GET GENOME/DNA USING ENTREZ
        try:
            seq_record = entrez(acc_num, db="sequences", rettype="gbwithparts", file_format="gb")
            print(f'Successfully fetched {acc_num}')
        except Exception:
            print(f'Error: Invalid accession number: {acc_num}')
            continue

        dna = coding_noncoding_strands(seq_record)                

        # GET POSITIVE ORFS (CDS)
        cds = get_cds_records(seq_record, dna, acc_num, LMIN, OFFSET) 
        mu, std = fit_norm(log_norm(sequence_lengths(cds, OFFSET)))
        _cds.extend(cds)
        print(f'\t{len(cds)} CDS Records\n')
        
        # GET NEGATIVE ORFS (NCS)
        if seq.lower() == 'ncs' or seq.lower() == 'all':
            itr = get_interregions(cds, seq_record)
            ncs = get_ncs_records(itr, dna, acc_num,\
                                  [mu,std], e, tol, LMIN, OFFSET, random_state)
            _ncs.extend(ncs)
            print(f'\t{len(ncs)} NCS Records\n')
    
    print(f'{len(_cds)} Total CDS Records')
    print(f'{len(_ncs)} Total NCS Records')
    
    return _cds, _ncs


def split_training_records(SeqRecords, size=None, random_state=None, shuffle=True):
    if random_state: random.seed(random_state)
    if shuffle: random.shuffle(SeqRecords)
    m = int(len(SeqRecords) * size)
    return SeqRecords[0:m], SeqRecords[m:]


def plot_data_distributions(cds, ncs, x, element='poly'):
    # put data into tidy lists
    labels = ['cds']*len(cds) + ['ncs']*len(ncs)

    if x == 'length':
        features = sequence_lengths(cds) + sequence_lengths(ncs)
    elif x == 'log_length':
        features = log_norm(sequence_lengths(cds)) + log_norm(sequence_lengths(ncs))
    elif x == 'gc':
        features = get_gc_content(cds) + get_gc_content(ncs)
    else:
        print('Input valid feature - ex. length, log_length, gc.')
        return -1

    # add data to pandas dataframe
    data = {'label': labels, x: features}
    df = pd.DataFrame(data)
    sns.histplot(df, x=x, hue="label", element=element)


##########################################################################################
######################################--PRIVATE--#########################################
##########################################################################################

def get_input_ids(filename):  
    '''
    INPUT: filename (string)
    OUTPUT: prok_ids (list)
    
        filename: txt file containing list of accession numbers
        prok_ids: list of accession numbers parsed from input txt file
    '''
    
    try:
        with open(filename) as file:
            prok_ids = file.readlines()
            prok_ids = [line.rstrip() for line in prok_ids]
        
    except IOError:
        print("Error: File does not appear to exist. Try 'train.txt'.")
        return 0
    
    return prok_ids

def entrez(acc_num, db="sequences", rettype="gbwithparts", file_format="gb"):
    # seq_records (features) from GenBank (or FASTA) file
    handle = Entrez.efetch(db=db, id=acc_num, rettype=rettype, retmode="text")
    seq_record = SeqIO.read(handle, file_format)
    handle.close()    
    return seq_record

def coding_noncoding_strands(seq_record):
    return [seq_record, seq_record.reverse_complement()]

##########################################################################################

def sequence_lengths(records, OFFSET=0):
    lengths = []
    for record in records:
        lengths.append(len(record.seq)-(2*OFFSET))
    return lengths

def log_norm(data):
    return [log(x) for x in data]

def fit_norm(data):
    mu, std = stats.norm.fit(pd.Series(data))
    return mu, std

##########################################################################################

def create_seqrecord(start, end, strand, typ, seq, name, seqID):
    f = [SeqFeature(FeatureLocation(start, end, strand), type=typ)]
    r = SeqRecord(seq, name=name, id=seqID, features=f)
    return r

def get_cds_records(seq_record, dna, acc_num, LMIN, OFFSET):
    cds = []
    for feature in seq_record.features:
        if feature.type == 'CDS':
            tag    = feature.qualifiers['locus_tag'][0]
            start  = feature.location.start.position
            end    = feature.location.end.position
            strand = feature.strand
             
            m30    = (dna[0][start-OFFSET:start] if strand == 1 else dna[1][::-1][end:end+OFFSET][::-1]).seq
            seq    = (dna[0][start:end] if strand == 1 else dna[1][::-1][start:end][::-1]).seq
            p30    = (dna[0][end:end+OFFSET] if strand == 1 else dna[1][::-1][start-OFFSET:start][::-1]).seq
            
            if len(seq) < LMIN: continue
            
            r = create_seqrecord(start, end, strand, 'CDS', m30+seq+p30, tag, acc_num)
            cds.append(r)
    
    return cds

# Copyright(C) 2009 Iddo Friedberg & Ian MC Fleming
# Released under Biopython license. http://www.biopython.org/DIST/LICENSE
# Do not remove this comment
def get_interregions(coding_records, seq_record):
    cds_list_plus = []
    cds_list_minus = []
    intergenic_records = []
    
    initials = coding_records[0].name[0:2]
    acc_num = seq_record.id

    # Loop over the genome file, get the CDS features on each of the strands
    for record in coding_records:
        feature = record.features[0]
        mystart = feature.location.start.position
        myend = feature.location.end.position
        if feature.strand == -1:
            cds_list_minus.append((mystart,myend,-1))
        elif feature.strand == 1:
            cds_list_plus.append((mystart,myend,1))
        else:
            sys.stderr.write("No strand indicated %d-%d. Assuming +\n" %(mystart, myend))
            cds_list_plus.append((mystart,myend,1))
    buffer = 0
    for i,pospair in enumerate(cds_list_plus[1:]):
        # Compare current start position to previous end position
        last_end = cds_list_plus[i][1]
        this_start = pospair[0]
        if this_start - last_end >= 1:
            intergene_seq = seq_record.seq[last_end:this_start]
            strand_string = +1
            name = initials + '_NC' + str(i).zfill(5)
            feature = [SeqFeature(FeatureLocation(last_end,this_start,strand_string), type='interregion')]
            intergenic_records.append(SeqRecord(intergene_seq, name=name, id=acc_num, features=feature))
        count = i
    for i,pospair in enumerate(cds_list_minus[1:]):
        last_end = cds_list_minus[i][1]
        this_start = pospair[0]
        if this_start - last_end >= 1:
            intergene_seq = seq_record.seq[last_end:this_start]
            strand_string = -1
            name = initials + '_NC' + str(i+buffer).zfill(5)
            feature = [SeqFeature(FeatureLocation(last_end,this_start,strand_string), type='interregion')]
            intergenic_records.append(SeqRecord(intergene_seq, name=name, id=acc_num, features=feature))
    return intergenic_records


def pnorm(x, loc, scale):
    return stats.norm.pdf(log(x), loc=loc, scale=scale)


def get_ncs_records(interregions, dna, acc_num, p, e, tol, LMIN, OFFSET, random_state=None):
    
    if random_state: random.seed(random_state)
    
    ncs = []
    mu, std = p
    for i,interregion in enumerate(interregions):
        # get record info
        tag     = interregion.name
        strand  = interregion.features[0].strand
        i_start = interregion.features[0].location.start.position 
        
        # find all orf-sets in longest orf
        orf_sets = extract_orf_sets(interregion.seq)
        if not orf_sets: continue
        
        for orf_set in orf_sets:
            # longest orf in set
            orf = orf_set[0]
            orf_len = orf[1] - orf[0]
            
            # check edge cases
            if orf_len < LMIN: continue
            if pnorm(orf_len, mu, std) < e: continue
            if random.uniform(0,1) < tol: continue
            
            start = i_start+orf[0]
            end   = i_start+orf[1]
            seq = dna[0][start-OFFSET:end+OFFSET].seq 
            
            r = create_seqrecord(start, end, strand, 'NCS', seq, tag, acc_num)
            ncs.append(r)
    return ncs

##########################################################################################

def get_gc_content(records):
    gc = []
    for record in records:
        gc.append(extract_gc_content(record.seq))
    return gc