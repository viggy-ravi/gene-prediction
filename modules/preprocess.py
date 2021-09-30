''' preprocess.py

This provides code to extract all valid open reading frames (ORFs) in a genome. 
ORFs consist of positive ORFs or genes, and negative ORFs or ORFs found within the interregions. 
'''

import random
import warnings
import pandas as pd
import seaborn as sns
from math import log
from scipy import stats
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import Entrez

from modules.extract import extract_all_orfs, extract_longest_orf, extract_gc_content

##########################################################################################
##########################################################################################
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

def genome(filename, n_genomes=None, seq = 'all', e=0.007, tol=0.2, random_state=None, OFFSET=30, noise=False, LMIN=60):
    '''
    INPUT: filename (string), LMIN=60, OFFSET=60
    OUTPUT: cds, ncs
    
        filename: txt file containing list of accession numbers
        OFFSET: number of basepairs up-stream of start codon/down-stream of stop codon (used for the TIS feature)
        LMIN: minimum length of gene/non-gene sequences (default = 60 - ref: Hoff et al.)
        cds: list of SeqRecords containing information about gene sequences
        ncs: list of SeqRecords containing information about non-gene sequences
    '''
    
    # get input genome accession numbers
    prok_ids = get_input_ids(filename)
    
    if n_genomes > len(prok_ids): n_genomes = None
    
    _cds, _ncs = [], []
    for acc_num in prok_ids[0:n_genomes]:
        
        # GET GENOME USING ENTREZ
        try:
            seq_record = entrez(acc_num, db="sequences", rettype="gbwithparts", file_format="gb")
            print(f'Successfully fetched {acc_num}')
            
        except Exception:
            print(f'Error: Invalid accession number: {acc_num}')
            continue

        # DEFINE DNA: CODING, NONCODING STRANDS
        dna = [seq_record, seq_record.reverse_complement()]                

        # GET POSITIVE ORFS (CDS)
        cds = get_cds_records(seq_record, dna, acc_num, LMIN, OFFSET) 
        
        if seq.lower() == 'ncs' or seq.lower() == 'all':
            # GET INTERGENIC SEQUENCES (INTERREGIONS)
            interregions = get_interregions(cds, seq_record)

            # GET NEGATIVE ORFS (NCS)
            mu, std = fit_norm(log_norm(sequence_lengths(cds)))
            ncs = get_ncs_records_v3(interregions, dna, acc_num, [mu,std], e, tol, LMIN, OFFSET, random_state)

        # SAVE DATA
        if seq.lower() == 'ncs':
            _ncs.extend(ncs)
            print(f'\t{len(ncs)} NCS Records\n')
        elif seq.lower() == 'cds':
            _cds.extend(cds)
            print(f'\t{len(cds)} CDS Records\n')
        else:
            _cds.extend(cds)
            _ncs.extend(ncs)
            print(f'\t{len(cds)} CDS Records, {len(ncs)} NCS Records')
    
    print(f'{len(_cds)} Total CDS Records')
    print(f'{len(_ncs)} Total NCS Records')
    
    return _cds, _ncs


def entrez(acc_num, db="sequences", rettype="gbwithparts", file_format="gb"):
    # seq_records (features) from GenBank (or FASTA) file
    handle = Entrez.efetch(db=db, id=acc_num, rettype=rettype, retmode="text")
    seq_record = SeqIO.read(handle, file_format)
    handle.close()
    
    return seq_record


##########################################################################################
##########################################################################################
##########################################################################################


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
            
            if len(seq) < LMIN: continue # check if orf length > l_min = 60
            
            f = [SeqFeature(FeatureLocation(start, end, strand), type='CDS')]
            r = SeqRecord(m30+seq+p30, name=tag, id=acc_num, features=f)            
            cds.append(r)
    
    return cds


##########################################################################################
##########################################################################################
##########################################################################################


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


##########################################################################################
##########################################################################################
##########################################################################################


def sequence_lengths(records, offset=60):
    lengths = []
    for record in records:
        lengths.append(len(record.seq)-offset)
    return lengths

def log_norm(data):
    return [log(x) for x in data]

def fit_norm(data):
    mu, std = stats.norm.fit(pd.Series(data))
    return mu, std

def pnorm(x, loc, scale):
    return stats.norm.pdf(log(x), loc=loc, scale=scale)

def get_ncs_records_v3(interregions, dna, acc_num, p, e, tol, LMIN, OFFSET, random_state=None):
    
    if random_state: random.seed(random_state)
    
    ncs = []
    mu, std = p
    for i,interregion in enumerate(interregions):
        # get record info
        tag     = interregion.name
        strand  = interregion.features[0].strand
        i_start = interregion.features[0].location.start.position 
        
        # find longest orf in interregion
        long_orf = extract_longest_orf(interregion.seq)
        if not long_orf: continue
        long_seq = interregion.seq[long_orf[0]:long_orf[1]]
        l_start  = i_start + long_orf[0]
        
        # find all orf-sets in longest orf
        orf_sets = extract_orf_sets(long_seq)
        if not orf_sets: continue
        
        for orf_set in orf_sets:
            # longest orf in set
            orf = orf_set[0]
            orf_len = orf[1] - orf[0]
            
            # check edge cases
            if orf_len < LMIN: continue
            if pnorm(orf_len, mu, std) < e: continue
            if random.uniform(0,1) < tol: continue
            
            start = l_start+orf[0]
            end   = l_start+orf[1]
            seq = dna[0][start-OFFSET:end+OFFSET].seq       
            f = [SeqFeature(FeatureLocation(start, end, strand), type='NCS')]
            r = SeqRecord(seq, name=tag, id=acc_num, features=f)
            ncs.append(r)
    return ncs


##########################################################################################
##########################################################################################
##########################################################################################


def get_gc_content(records):
    gc = []
    for record in records:
        gc.append(extract_gc_content(record.seq))
    return gc

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