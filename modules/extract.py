''' extract.py 

Provides code extract codon usage or the frequencies of codons found in a sequence ('codons' can be of length 3 (tri/monocodon) or 6 (hex/dicodon)),
    extract tis features for linear discriminant tis training. 
'''

import re
import itertools
import numpy as np
from scipy.stats import norm
from Bio.SeqUtils import GC

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.utils import to_categorical

##########################################################################################
##########################################################################################
##########################################################################################


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


##########################################################################################
##########################################################################################
##########################################################################################


START_CODONS = ['ATG','CTG','GTG','TTG']
STOP_CODONS = ['TAG','TGA','TAA']
    
def extract_all_orfs(seq):
    '''
    INPUT: seq (Bio.Seq.Seq)
    OUTPUT: returns all in-frame start codons within an ORF sequence
    '''
    
    all_starts = codon_pos(seq, START_CODONS)

    # find all ORF
    orfs = []
    e = len(seq)
    for s in all_starts:
        if (e >= s) and (s%3 == 0):
            orfs.append([s,e])
    
    return orfs


def extract_longest_orf(seq):
    '''
    INPUT: seq (Bio.Seq.Seq)
    OUTPUT: returns start and end position of longest orf in a sequence
    '''
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


def codon_pos(seq, codon_list):
    '''
    INPUT: seq, codon_list
    OUTPUT:
    
        seq: Bio.Seq.Seq gene/non-gene sequence
        codon_list: list of start or stop codons
    '''
    pos = []
    for codon in codon_list:
        matches = re.finditer(codon, str(seq))
        matches_positions = [match.start() for match in matches]
        pos.extend(matches_positions)
    
    return sorted(pos)


##########################################################################################
##########################################################################################
##########################################################################################


def features(seq_records):
    '''
    INPUT:
    OUTPUT:
    '''
    
    _tri, _hex, _tis, _gc = [], [], [], []
    _y = []
    
    for record in seq_records:
        # TRY EXCEPT
        orf = extract_longest_orf(record.seq)
        if not orf: continue
        seq = record[orf[0]:orf[1]].seq
        if len(seq) < 60: continue
        
        # FEATURES (TRI, HEX, TIS, GC)
        _tri.append(extract_codon_usage(seq, 3))
        _hex.append(extract_codon_usage(seq, 6))
        _tis.append(extract_tis_frame(record, start_pos=orf[0]))
        _gc.append(extract_gc_content(seq))
        
        # Y
        _y.append(extract_codon_label(record))
    
    feat = [_tri, _hex, _tis, _gc]
    return feat, _y


def training_features(seq_records, OFFSET=30):
    '''
    INPUT:
    OUTPUT:
    '''
    
    _tri, _hex, _yc = [], [], []
    _tis, _yt = [], []
    
    for record in seq_records:        
        seq = record.seq[OFFSET:-OFFSET]
        # TODO - rely on longest orf function instead of OFFSET
#         orf = extract_longest_orf(record.seq)
#         seq = record[orf[0]:orf[1]].seq
        
        # CODON FEATURES (TRI, HEX)
        _tri.append(extract_codon_usage(seq, 3))
        _hex.append(extract_codon_usage(seq, 6))
        _yc.append(extract_codon_label(record))
                
        # TIS FEATURES
        if record.features[0].type == 'CDS':
            tis, yt = extract_tis_training_frames(record, OFFSET)
            _tis.extend(tis)
            _yt.extend(yt)
        
    _cod = [_tri, _hex]
    return _cod, _yc, _tis, _yt


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

    # find orf-set
    orf_set = extract_all_orfs(record[OFFSET:-OFFSET].seq)

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


def extract_codon_usage(seq, n):
    '''
    INPUT:
    OUTPUT:
    
    
    ref: https://stackoverflow.com/questions/18524112/norm-along-row-in-pandas
    '''
    
    frame = codon_frequency(seq, n)
    # normalize: faster than np.linalg.norm
    frame_norm = frame / np.sqrt(np.sum(np.square(frame)))
    data = np.array(frame_norm)
    return data


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


def extract_tis_frame(record, start_pos, OFFSET=30):
    '''
    INPUT:
    OUTPUT:
    '''
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


def extract_gc_content(seq):
    return (GC(seq) / 100)


def extract_len(seq):
    return len(seq)


def codon_frequency(seq, nbases): 
    '''
    INPUT:
    OUTPUT:
    '''
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


##########################################################################################
##########################################################################################
##########################################################################################

    
def pos_score(s, gaus_params):
    '''
    INPUT:
    OUTPUT:
    '''
    # returns probability that tis sequence is tis positive (from a gene)
    pi = gaus_params[0]
    mu = gaus_params[1]
    sd = gaus_params[2]
    
    likelihood = norm.pdf(s, mu[0], sd[0])
    marginal = pi[0]*norm.pdf(s, mu[0], sd[0]) + pi[1]*norm.pdf(s, mu[1], sd[1])
    
    return (pi[0] * likelihood) / marginal


def neg_score(s, gaus_params):
    '''
    INPUT:
    OUTPUT:
    '''
    # returns probability that tis sequence is tis negative (not from a gene)
    pi = gaus_params[0]
    mu = gaus_params[1]
    sd = gaus_params[2]
    
    likelihood = norm.pdf(s, mu[1], sd[1])
    marginal = pi[0]*norm.pdf(s, mu[0], sd[0]) + pi[1]*norm.pdf(s, mu[1], sd[1])
    
    return (pi[1] * likelihood) / marginal


def dimentional_reduction(features, w, p):
    '''
    INPUT: features (4,), w (3,), p (3,2)
    OUTPUT: reduced_features_nn_input
    
        features: (tri, hex, tis, gc)
        w (weights): (wM, wD, wT)
        p (params): ((pi_pos, pi_neg), (mu_pos, mu_neg), (sd_pos, sd_neg))
    '''
    
    # x1 = reduced tri (Monocodon) feature
    x1 = features[0] @ w[0]
    
    # x2 = reduced hex (Dicodon) feature
    x2 = features[1] @ w[1]
    
    # x3 = positive tis score (s, p)
    x3 = pos_score(features[2] @ w[2], p)
    
    # x4 = negative tis score (s, p)
    x4 = neg_score(features[2] @ w[2], p)
    
    # x7 = GC content (doesn't need to be reduced)
    x7 = features[3]
    
    reduced_features_nn_input = np.stack((x1,x2,x3,x4,x7)).T
    
    return reduced_features_nn_input