''' preprocess.py

This provides code to extract all valid open reading frames (ORFs) in a genome. 
ORFs consist of positive ORFs or genes, and negative ORFs or ORFs found within the interregions. 
'''

import random
import warnings
from scipy import stats
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import Entrez

from modules.utils import codon_pos, orf_finder, longest_orf

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

def genome(filename, n_genomes=None, seq = 'all', OFFSET=30, noise=False, LMIN=60):
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
            seq_record = entrez(acc_num, 
                                db="sequences", 
                                rettype="gbwithparts", 
                                file_format="gb")
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
            ncs = get_ncs_records(interregions, dna, acc_num, LMIN, OFFSET)

        # SAVE DATA
        if seq.lower() == 'ncs':
            _ncs.extend(ncs)
            print(f'{len(ncs)} NCS Records\n')
        elif seq.lower() == 'cds':
            _cds.extend(cds)
            print(f'{len(cds)} CDS Records\n')
        else:
            _cds.extend(cds)
            _ncs.extend(ncs)
            print(f'{len(cds)} CDS Records, {len(ncs)} NCS Records\n')
    
    print(f'{len(_cds)} Total CDS Records')
    print(f'{len(_ncs)} Total NCS Records')
    
    return _cds, _ncs


def entrez(acc_num, db="sequences", rettype="gbwithparts", file_format="gb"):
    # seq_records (features) from GenBank (or FASTA) file
    handle = Entrez.efetch(db=db, id=acc_num, rettype=rettype, retmode="text")
    seq_record = SeqIO.read(handle, file_format)
    handle.close()
    
    return seq_record


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
            feature = [SeqFeature(FeatureLocation(last_end+1,this_start,strand_string), type='interregion')]
            intergenic_records.append(SeqRecord(intergene_seq, name=name, id=acc_num, features=feature))
        count = i
    for i,pospair in enumerate(cds_list_minus[1:]):
        last_end = cds_list_minus[i][1]
        this_start = pospair[0]
        if this_start - last_end >= 1:
            intergene_seq = seq_record.seq[last_end:this_start]
            strand_string = -1
            name = initials + '_NC' + str(i+buffer).zfill(5)
            feature = [SeqFeature(FeatureLocation(last_end+1,this_start,strand_string), type='interregion')]
            intergenic_records.append(SeqRecord(intergene_seq, name=name, id=acc_num, features=feature))
    return intergenic_records


def get_ncs_records(interregions, dna, acc_num, LMIN, OFFSET):
    ## interregion (i or intr)
    ## longest orf (long)
    
    ncs = []
    for interregion in interregions:
        feature = interregion.features[0]
        strand  = feature.strand
        
        # find longest orf in interregion
        i_start  = feature.location.start.position
        i_end    = feature.location.end.position
        intr_seq = dna[0][i_start:i_end] if strand == 1 else dna[1][::-1][i_start:i_end][::-1]
        long_orf = longest_orf(intr_seq.seq)
        
        if not long_orf: continue                       # check if orf exists 
        if long_orf[1] - long_orf[0] < LMIN: continue   # check if orf length > l_min = 60
        
        tag   = interregion.name
        start = i_start + long_orf[0]
        end   = i_start + long_orf[1]

        m_start = start - OFFSET
        m_end = end + OFFSET
        
        m30   = (dna[0][m_start:start] if strand == 1 else dna[1][::-1][end-1:m_end-1][::-1]).seq
        seq   = intr_seq[long_orf[0]:long_orf[1]].seq
        p30   = (dna[0][end:m_end] if strand == 1 else dna[1][::-1][m_start-1:start-1][::-1]).seq
        
        f = [SeqFeature(FeatureLocation(start, end, strand), type='NCS')]
        r = SeqRecord(m30+seq+p30, name=tag, id=acc_num, features=f)
        ncs.append(r)
    
    return ncs