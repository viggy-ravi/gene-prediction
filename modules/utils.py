''' utils.py 

Frequently used methods. 
'''

import numpy as np
import json
import csv
from itertools import zip_longest
from tensorflow.keras.models import model_from_json

import re
import itertools

START_CODONS = ['ATG','CTG','GTG','TTG']
STOP_CODONS = ['TAG','TGA','TAA']


def save_model2json(filename, model):    
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(filename + '.h5')
    print('Saved model to disk.')

    
def load_model(model):
    try:
        # load json and create model
        with open(model + '.json', 'r') as f:
            loaded_model_json = f.read()
            loaded_model = model_from_json(loaded_model_json)

            # load weights into new model
            loaded_model.load_weights(model + '.h5')
            print("Loaded model from disk.")
    
    except Exception:
        print(f"Error: model {model} not found.")
        return 0
    
    return loaded_model


def save_params2csv(filename, w, gaus_params):
    data = []
    for l in w: data.append(l.tolist())
    for l in gaus_params: data.append(list(l))
    
    with open(filename,"w+") as f:
        writer = csv.writer(f)
        for values in zip_longest(*data):
            writer.writerow(values)
            
    print("Saved parameters to disk.")


def load_param(filename):
    wM, wD, wT = [], [], []
    pi, mu, sd = [], [], []
    
    try:
        with open(filename, "r") as f:
            csv_reader = csv.reader(f, delimiter=',')
            for lines in csv_reader:
                if lines[0] != '': wM.append(lines[0])
                if lines[1] != '': wD.append(lines[1])
                if lines[2] != '': wT.append(lines[2])
                if lines[3] != '': pi.append(lines[3])
                if lines[4] != '': mu.append(lines[4])
                if lines[5] != '': sd.append(lines[5])

        print("Loaded parameters from disk.")
        
    except Exception:
        print(f"Error: parameters {filename} not found.")
        return 0
    
    w = [np.array(wM).astype(np.float64), np.array(wD).astype(np.float64), np.array(wT).astype(np.float64)]
    p = [np.array(pi).astype(np.float64), np.array(mu).astype(np.float64), np.array(sd).astype(np.float64)]
                
    return w, p


def orf_finder(orf_seq):
    '''
    INPUT: seq (Bio.Seq.Seq)
    OUTPUT: returns all in-frame start codons within an ORF sequence
    '''
    
    all_starts = codon_pos(orf_seq, START_CODONS)

    # find all ORF
    orfs = []
    e = len(orf_seq)
    for s in all_starts:
        if (e >= s) and (s%3 == 0):
            orfs.append([s,e])
    
    return orfs


def longest_orf(seq):
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