''' genepred.py 

Frequently used methods. 
'''

import random
import numpy as np
import itertools
from itertools import zip_longest
import matplotlib.pyplot as plt

import csv
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import model_from_json


##########################################################################################
##########################################################################################
##########################################################################################

# move to preprocess.py

def split_training_records(SeqRecords, size=None, random_state=None, shuffle=True):
    if random_state: random.seed(random_state)
    if shuffle: random.shuffle(SeqRecords)
    m = int(len(SeqRecords) * size)
    return SeqRecords[0:m], SeqRecords[m:] 


##########################################################################################
##########################################################################################
##########################################################################################


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


##########################################################################################
##########################################################################################
##########################################################################################

# move to train.py

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