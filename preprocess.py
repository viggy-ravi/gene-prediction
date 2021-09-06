''' preprocess.py

This provides code to extract all valid open reading frames (ORFs) in a genome. 
ORFs consist of positive ORFs or genes, and negative ORFs or ORFs found within the interregions. 
'''

import re
import itertools
import textwrap

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import Entrez

START_CODONS = ['ATG','CTG','GTG','TTG']
STOP_CODONS = ['TAG','TGA','TAA']


'''
INPUT: name of text file (string) with accession numbers
OUTPUT: list of accession numbers
'''
def get_input_ids(filename):    
    try:
        open('./inputs/' + filename, 'r')
    except IOError:
        print("Error: File does not appear to exist. Try 'input_test', 'input_sm', 'input_med', or 'input_lg'.")
        return 0
    
    with open('./inputs/' + filename) as file:
        prok_ids = file.readlines()
        prok_ids = [line.rstrip() for line in prok_ids]
        
    return prok_ids

def preprocess_genome(filename, LMIN=60, OFFSET=30):
    # get input genome accession numbers
    prok_ids = get_input_ids(filename)    
    
    _cds, _ncs = [], []
    for acc_num in prok_ids:
        
        # GET GENOME USING ENTREZ
        try:
            seq_record = fetch_genome(acc_num)                           
            print(f'Successfully fetched {acc_num}')
        except Exception:
            print(f'Error: Invalid accession number: {acc_num}')
            continue
        
        # DEFINE DNA: CODING, NONCODING STRANDS
        dna = [seq_record, seq_record.reverse_complement()]                
        
        # FIND POS/NEG ORFS 
        cds, ncs = get_orfs(seq_record, dna, acc_num, LMIN=LMIN, OFFSET=OFFSET) 
        
        # SAVE DATA
        _cds.extend(cds)
        _ncs.extend(ncs)
        
        print(f'{len(cds)} CDS Records, {len(ncs)} NCS Records\n')
    
    print(f'{len(_cds)} Total CDS Records')
    print(f'{len(_ncs)} Total NCS Records')
    
    return _cds, _ncs

def fetch_genome(acc_num):
    # seq_records (features) from GenBank file
    handle = Entrez.efetch(db="sequences", id=acc_num, rettype="gbwithparts", retmode="text")
    seq_record = SeqIO.read(handle, "gb")
    handle.close()
    
    return seq_record


def get_orfs(seq_record, dna, acc_num, LMIN, OFFSET):
    # Get positive ORFs (coding sequences (CDS) or genes)
    cds = get_cds_records(seq_record, dna, acc_num, LMIN, OFFSET)
            
    # Find interregions
    interregions = get_interregions(cds, seq_record)
    
    # Get negative ORFs
    ncs = get_ncs_records(interregions, dna, acc_num, LMIN, OFFSET)
    
    return cds, ncs


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
        
        
def codon_pos(seq, codon_list):
    pos = []
    for codon in codon_list:
        matches = re.finditer(codon, str(seq))
        matches_positions = [match.start() for match in matches]
        pos.extend(matches_positions)
    return sorted(pos)


'''
Input: Bio.Seq.Seq
Output: returns start and end position of longest orf in a sequence
'''
def longest_orf(seq):
    all_starts = codon_pos(seq, START_CODONS)
    all_stops  = codon_pos(seq, STOP_CODONS)

    orfs = []
    found = False;            
    for s in all_starts:
        for e in all_stops[::-1]:
            if (e >= s) and ((e-s)%3 == 0):
                found = True; orfs = [s,e+3];
                break
        if found: break
            
    return orfs


'''
Input: Bio.Seq.Seq
Output: returns all in-frame start codons within an ORF sequence
'''
def orf_finder(orf_seq):
    all_starts = codon_pos(orf_seq, START_CODONS)

    # find all ORF
    orfs = []
    e = len(orf_seq)
    for s in all_starts:
        if (e >= s) and (s%3 == 0):
            orfs.append([s,e])
    return orfs