''' utils.py 

Frequently used methods. 
'''

import random
import numpy as np
import re
import itertools
import json
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import model_from_json

START_CODONS = ['ATG','CTG','GTG','TTG']
STOP_CODONS = ['TAG','TGA','TAA']


def split_training_records(SeqRecords, size=None, random_state=None, shuffle=True):
    if random_state: random.seed(random_state)
    if shuffle: random.shuffle(SeqRecords)
    m = int(len(SeqRecords) * size)
    return SeqRecords[0:m], SeqRecords[m:] 


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
    
    
def save_params2csv(filename, data):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)            
    print("Saved data to disk.")


def load_params(filename):
    loaded_data = []    
    
    try:
        with open(filename, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                loaded_data.append(np.array(row).astype(np.float64))
        print("Loaded parameters from disk.")
        
    except Exception:
        print(f"Error: parameters {filename} not found.")
        return 0
                
    return loaded_data


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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