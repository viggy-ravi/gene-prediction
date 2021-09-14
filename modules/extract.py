''' extract.py 

Provides code extract codon usage or the frequencies of codons found in a sequence ('codons' can be of length 3 (tri/monocodon) or 6 (hex/dicodon)),
    extract tis features for linear discriminant tis training. 
'''

import numpy as np
from scipy.stats import norm
from Bio.SeqUtils import GC
from tensorflow.keras.utils import to_categorical

from modules.utils import orf_finder, longest_orf, populate_codon_idx_dict

TRICODON_IDX_DICT = populate_codon_idx_dict(nbases=3)
HEXCODON_IDX_DICT = populate_codon_idx_dict(nbases=6)


def features(seq_records):
    '''
    INPUT:
    OUTPUT:
    '''
    
    _tri, _hex, _tis, _gc = [], [], [], []
    _y = []
    
    for record in seq_records:
        # TRY EXCEPT
        orf = longest_orf(record.seq)
        seq = record[orf[0]:orf[1]].seq
        
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
        orf = longest_orf(record.seq)
        seq = record[orf[0]:orf[1]].seq
        
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
    orf_set = orf_finder(record[OFFSET:-OFFSET].seq)

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
            _yt.append(-1)
                
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
            y = -1
    except Exception:
        return 0
    
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