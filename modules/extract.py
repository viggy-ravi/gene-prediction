''' extract.py 

This file provides methods to extract features from all data. Features include:
monocodon frequency, dicodon frequency, translation initiation site (tis), length,
and gc content. 
'''

import re
import itertools
import numpy as np
from Bio.SeqUtils import GC
from tensorflow.keras.utils import to_categorical

from modules.utils import extract_orf_sets, extract_tis_candidates

##########################################################################################
#######################################--CONSTANTS--######################################
##########################################################################################

from modules.utils import get_start_codons, get_stop_codons, populate_codon_idx_dict

START_CODONS = get_start_codons()
STOP_CODONS = get_stop_codons()

TRICODON_IDX_DICT = populate_codon_idx_dict(nbases=3)
HEXCODON_IDX_DICT = populate_codon_idx_dict(nbases=6)

##########################################################################################
########################################--PUBLIC--########################################
##########################################################################################

'''NEED TO TEST'''
def extract_features(seq_records):
    _tri, _hex, _tis, _len, _gc = [], [], [], [], []
    _y = []
    
    for record in seq_records:
        orf_sets = extract_orf_sets(record.seq)
        if not orf_sets: continue
        
        for orf_set in orf_sets:
            orf = orf_set[0]
            if len(orf) < 60: continue
            
            # FEATURES (TRI, HEX, TIS, LEN, GC)
            _tri.append(extract_codon_usage(seq, 3))
            _hex.append(extract_codon_usage(seq, 6))
            _tis.append(extract_tis_frame(record, start_pos=orf[0]))
            _len.append(extract_len(seq))
            _gc.append(extract_gc_content(seq))
    
    feat = [_tri, _hex, _tis, _len, _gc]
    return feat

def extract_nn_training_features(seq_records, OFFSET=30):    
    _tri, _hex, _tis, _len, _gc = [], [], [], [], []
    _y = []
    
    for record in seq_records:
        seq = record.seq[OFFSET:-OFFSET]
                
        # FEATURES (TRI, HEX, TIS, LEN, GC)
        _tri.append(extract_codon_usage(seq, 3))
        _hex.append(extract_codon_usage(seq, 6))
        _tis.append(extract_tis_frame(record, start_pos=OFFSET))
        _len.append(extract_len(seq))
        _gc.append(extract_gc_content(seq))
        
        # Y
        _y.append(extract_codon_label(record))
    
    feat = [_tri, _hex, _tis, _len, _gc]
    return feat, np.array(_y)


def extract_ld_training_features(seq_records, OFFSET=30):
    _tri, _hex, _yc = [], [], []
    _tis, _yt = [], []
    
    for record in seq_records:        
        seq = record.seq[OFFSET:-OFFSET]
        
        # CODON FEATURES (TRI, HEX)
        _tri.append(extract_codon_usage(seq, 3))
        _hex.append(extract_codon_usage(seq, 6))
        _yc.append(extract_codon_label(record))
                
        # TIS FEATURES
        if record.features[0].type == 'CDS':
            tis, yt = extract_tis_training_frames(record, OFFSET)
            _tis.extend(tis)
            _yt.extend(yt)
        
    return np.array(_tri), np.array(_hex), np.array(_yc), np.array(_tis), np.array(_yt)

##########################################################################################
######################################--PRIVATE--#########################################
##########################################################################################

def extract_tis_training_frames(record, OFFSET):
    '''
    INPUT: seq_record, OFFSET
    OUTPUT: _tis, _yt
    
        seq_record: list of all discriminant training SeqRecords
        OFFSET: number of basepairs up-stream of start codon/down-stream of stop codon (used for the TIS feature)
        _tis: list of arrays (3712,) of flattened tis features (58,64)
        _yt: corresponding label for _tis feature
    '''
    _tis, _yt = [], []

    # find tis candidates
    orf_set = extract_tis_candidates(record[OFFSET:-OFFSET].seq)

    for i,cand in enumerate(orf_set):
        cand_start = cand[0] + OFFSET

        # find sequence window and populate codon information (58,) at each position
        seq60 = record[(cand_start-OFFSET):(cand_start+OFFSET)].seq 
        frame = [TRICODON_IDX_DICT[seq60[i:i+3]] for i in range(0,len(seq60)-2)]

        # convert frame to (58,64) using to_categorical and flatten to 1D array
        tis_data = np.array(to_categorical(frame, num_classes=64).astype(int).reshape(-1))

        _tis.append(tis_data)
        if i == 0: 
            _yt.append(1)
        else: 
            _yt.append(0)
                
    return _tis, _yt

##########################################################################################

def extract_codon_usage(seq, n):
    #ref: https://stackoverflow.com/questions/18524112/norm-along-row-in-pandas
    frame = codon_frequency(seq, n)
    # normalize: faster than np.linalg.norm
    frame_norm = frame / np.sqrt(np.sum(np.square(frame)))
    data = np.array(frame_norm)
    return data

def codon_frequency(seq, nbases): 
    # Returns count of tri/hex codon frequencies in list form
    # Input: Bio.Seq.Seq
    
    codon_idx_dict = TRICODON_IDX_DICT if nbases==3 else HEXCODON_IDX_DICT
    frame = [0]*len(codon_idx_dict)
    
    parsed_codons = codon_parser(seq, nbases)
    for codon in parsed_codons:
        frame[codon_idx_dict[codon]] += 1
    frame[codon_idx_dict[parsed_codons[-1]]] += 1 # add last codon
    
    return frame

def codon_parser(seq, nbases):
    '''
    INPUT:
    OUTPUT:
    
    Ex. Input  - AAATTTGGG
    Output - ['AAA','TTT','GGG'] if nbases=3
           = ['AAATTT','TTTGGG'] if nbases=6
    '''
    # Parse dna sequence into a list of codons separated by nbases
    # Input: Bio.Seq.Seq
    
    codon_seq = [seq[i:i+nbases] for i in range(0,len(seq),3) if (i+nbases)<len(seq)]
    return codon_seq

def extract_tis_frame(record, start_pos, OFFSET=30):
    seq = None
    
    # check if 30 bp before start codon
    if start_pos >= OFFSET:
        seq = record[start_pos-OFFSET : start_pos+OFFSET].seq
    else:
        seq = record[0 : start_pos+OFFSET].seq
    
    # create buffer frame (if incomplete tis frame)
    n = 0 if start_pos >= OFFSET else (OFFSET-start_pos)   
    buffer = [0] * n
    
    # find tis frame
    frame = buffer + [TRICODON_IDX_DICT[seq[i:i+3]] for i in range(0,len(seq)-2)]
    
    # convert to to_categorical and correct buffer (make all zeros)
    tis_frame = to_categorical(frame, num_classes=64)
    for i in range(0,n): 
        tis_frame[i][0] = 0
    
    return np.array(tis_frame).astype(int).reshape(-1)

def extract_len(seq):
    return len(seq)

def extract_gc_content(seq):
    return (GC(seq) / 100)

def extract_codon_label(record):
    y = 0
    try:
        label = record.features[0].type
        if label == 'CDS': 
            y = 1
        else: 
            y = 0
    except Exception:
        return -1
    
    return y