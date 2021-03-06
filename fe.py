''' fe.py (feature extraction)

Provides code extract codon usage or the frequencies of codons found in a sequence ('codons' can be of length 3 (tri/monocodon) or 6 (hex/dicodon)),
    extract codon and tis features, 
    and extract tis features for linear discriminant tis training. 
'''

import numpy as np
import re
import itertools
import textwrap
from Bio.SeqUtils import GC
from tensorflow.keras.utils import to_categorical

from preprocess import orf_finder

# Produces all tri/hex codon permutations 4^n (ex. AAA, AAC,...,TTT for n=3) 
def permutations_with_replacement(n):
    for i in itertools.product(("A", "C", "G", "T"), repeat=n):
        yield i

# Function to create codon:idx dictionary. Converts codons to integers (ex. {AAA:0, AAC:1, ..., TTT:63} for n=3)
def populate_codon_idx_dict(nbases=3):
    codon_idx_dict = {}
    count = 0
    for i in permutations_with_replacement(nbases):
        key = "".join(i) #codon or dicodon sequence
        codon_idx_dict[key] = count
        count += 1
    return codon_idx_dict


TRICODON_IDX_DICT = populate_codon_idx_dict(nbases=3)
HEXCODON_IDX_DICT = populate_codon_idx_dict(nbases=6)


# Parse dna sequence into a list of codons separated by nbases
# Input: Bio.Seq.Seq
'''
Ex. Input  - AAATTTGGG
    Output - ['AAA','TTT','GGG'] if nbases=3
           = ['AAATTT','TTTGGG'] if nbases=6
'''
def codon_parser(seq, nbases):
    codon_seq = [seq[i:i+nbases] for i in range(0,len(seq),3) if (i+nbases)<len(seq)]
    return codon_seq

# Returns count of tri/hex codon frequencies in list form
# Input: Bio.Seq.Seq
def codon_frequency(seq, nbases):      
    codon_idx_dict = TRICODON_IDX_DICT if nbases==3 else HEXCODON_IDX_DICT
    frame = [0]*len(codon_idx_dict)
    
    parsed_codons = codon_parser(seq, nbases)
    for codon in parsed_codons:
        frame[codon_idx_dict[codon]] += 1
    frame[codon_idx_dict[parsed_codons[-1]]] += 1 # add last codon
    
    return frame

# reference for unit Euclidean norm
# https://stackoverflow.com/questions/18524112/norm-along-row-in-pandas
def extract_codon_usage(seq, n):
    frame = codon_frequency(seq, n)
    frame_norm = frame / np.sqrt(np.sum(np.square(frame))) # faster than np.linalg.norm
    data = np.array(frame_norm)
    return data


def extract_codon_tis_features(orfs, train_nn=False, OFFSET=30):

    _tri, _hex, _gc, _yc = [], [], [], []
    _tis_nn = []
    
    for record in orfs:
        seq = record[OFFSET:-OFFSET].seq

        # MONOCODON FREQUENCE
        tri_data = extract_codon_usage(seq, 3)
        _tri.append(tri_data)
        
        # DICODON FREQUENCE
        hex_data = extract_codon_usage(seq, 6)
        _hex.append(hex_data)
        
        # TIS
        if train_nn:
            seq60 = record[0:(2*OFFSET)].seq
            frame = [TRICODON_IDX_DICT[seq60[i:i+3]] for i in range(0,len(seq60)-2)]
            tis_data = np.array(to_categorical(frame, num_classes=64).astype(int).reshape(-1))
            _tis_nn.append(tis_data)

        # GC
        gc_data = np.float16(GC(seq) / 100)
        _gc.append(gc_data)
        
        # Y
        typ = record.features[0].type
        if typ == 'CDS': _yc.append(1)
        else: _yc.append(-1)
        
    return np.array(_tri), np.array(_hex), np.array(_tis_nn), np.array(_gc), np.array(_yc)


def LD_tis_training_features(cds, OFFSET=30):
    _tis, _yt = [], []
    for record in cds:
        seq = record.seq
        
        # find orf-set
        orf_set = orf_finder(seq[OFFSET:-OFFSET])
        
        for i,cand in enumerate(orf_set):
            cand_start = cand[0] + OFFSET
            
            # find sequence window and populate codon information (58,) at each position
            seq60 = record[(cand_start-OFFSET):(cand_start+OFFSET)].seq 
            frame = [TRICODON_IDX_DICT[seq60[i:i+3]] for i in range(0,len(seq60)-2)]

            # convert frame to (58,64) using to_categorical and flatten to 1D array
            tis_data = np.array(to_categorical(frame, num_classes=64).astype(int).reshape(-1))
            
            _tis.append(tis_data)
            if i == 0: _yt.append(1)
            else: _yt.append(-1)
                
    return np.array(_tis), np.array(_yt)