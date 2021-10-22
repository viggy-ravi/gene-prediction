''' predict.py

NEED TO TEST

'''

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from modules.preprocess import preprocess_genome
from modules.extract import extract_features
from modules.utils import extract_longest_orf, load_params, load_model

import numpy as np
from scipy.stats import norm

# suppress warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

##########################################################################################
########################################--PUBLIC--########################################
##########################################################################################

def from_genome(filename, params, model, outfile, n_genomes=None, seq='all', OFFSET=30, LMIN=60):
    # preprocess genome
    CDS, NCS = preprocess_genome(filename=filename,\
                                   n_genomes=n_genomes, seq=seq, \
                                   OFFSET=OFFSET, LMIN=LMIN)
    
    # seqrecord2fasta
    tmp_file = "./input/tmp.fna"
    SEQ = CDS + NCS
    seqrecord2fasta(tmp_file, SEQ)
    
    # from_fasta
    y_pred = from_fasta(tmp_file, params, model, outfile)
    
    return y_pred


def from_fasta(filename, params, model, outfile):   
    # extract sequences from fasta file
    records = fasta2seqrecords(filename)
    
    # extract features
    feat = extract_features(records)
    
    # dim red features
    data = load_params(params)
    w, p = data[0:3], data[3:]
    nn_input = dimentional_reduction(feat, w, p)
    
    # predict + convert to 1s or 0s
    predictions = predict(nn_input, model)
    y_pred = [1 if val >= 0.5 else 0 for val in predictions]
    
    # save output fasta file
    save_predictions(outfile, y_pred, records)
    
    return y_pred


def dimensional_reduction(features, w, p):
    '''
    INPUT: features (4,), w (3,), p (3,2)
    OUTPUT: reduced_features_nn_input
    
        features: (tri, hex, tis, len, gc)
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
    
    # x5 = length (doesn't need to be reduced)
    x5 = features[3]
    
    # x7 = GC content (doesn't need to be reduced)
    x7 = features[4]
    
    reduced_features_nn_input = np.stack((x1,x2,x3,x4,x7)).T
    
    return reduced_features_nn_input

##########################################################################################
######################################--PRIVATE--#########################################
##########################################################################################

def seqrecord2fasta(filename, seq_records):
    fasta_records = []
    
    for i,record in enumerate(seq_records):
        r = SeqRecord(record.seq, description=f"Seq No {i+1}", name="", id="")
        fasta_records.append(r)

    SeqIO.write(fasta_records, filename, "fasta")
    print(f"Saved genome fragments to {filename}.")

def fasta2seqrecords(filename):
    seq_records = []
    
    try:
        for frag in SeqIO.parse(filename, "fasta"):
            r = SeqRecord(frag.seq, description=frag.description)
            seq_records.append(r)
    except Exception:
        print(f"Error: no file named {filename}.")
        return 0
        
    return seq_records

def predict(data, model):
    model = load_model(model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    prediction = model.predict(data)
    return prediction
    
def save_predictions(outfile, predictions, seq_records):  
    results = []
    
    for (pred, record) in zip(predictions, seq_records):
        if pred > 0.5:
            orf = extract_longest_orf(record.seq)
            SEQ = record[orf[0]:orf[1]].seq
            r = SeqRecord(SEQ, id=record.id, name=record.name, description=record.description)
            results.append(r)
            
    SeqIO.write(results, outfile, "fasta")
    print(f"Saved genome fragments to {outfile}.")
    

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