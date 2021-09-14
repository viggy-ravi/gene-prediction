# gene-prediction

## Introduction

This project is based on the 2008 paper ['Gene prediction in metagenomic fragments: A large scale machine learning approach'](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-217) by Hoff et at. The goal of the `gene_prediction_pipeline.ipynb` notebook is to train the gene prediction algorithm using a 2-stage approach - linear discriminant and neural network - and provide predictions for sequences in an easy to use way. The linear discriminant model reduces high dimensional features and the neural network predicts if a given sequence is a gene or not. My intentions for starting this project were to gain a deeper understanding of machine learning approaches behind the `Orphelia gene prediction algorithm` and sharpen my ML/programming skills.  
    
## Contents
1. Preprocess Genome 
    * extract coding sequences/noncoding sequences
    * shuffle/split data
2. Extract Features 
    * monocodon (tricodon) frequency, 
    * dicodon (hexcodon) frequency, 
    * tis, 
    * gc content
3. Train Linear Discriminant for Dimensional Reduction
4. Train Binary Neural Network for Gene Prediction
5. Predict Sequences (FASTA)
    * Prediction from Genome Sequence
    * Prediction from FASTA Input File

## TODO
* Create visual for tricodon/hexcodon weight representations
* Create visual for occurances of TIS codons up- and down-stream of start codon
* Create visual for gc content vs sequence length
* Ensure NCS lengths follow the same distribution as CDS lengths (normalize)
    * fit distribution (Erlang) to positive data (CDS)
    * sample negative data (NCS) from fitted distribution
* Implement batch training (to overcome large amount of training data) 

## Setup

### With Google Colab
First open a new Google Colab file. Mount your Google Drive and move to your working directory. 

    from google.colab import drive
    drive.mount('/content/gdrive')

    # Change working directory
    %cd gdrive/MyDrive/
    
Next, clone this repository and move into the directory.

    # Clone git repository (copy all files to Google Colab)
    !git clone https://github.com/viggy-ravi/gene-prediction.git
    
    # Go to gene-prediction folder
    %cd gene-prediction/

Open the `gene_prediction_pipeline.ipynb` file and install the necessary dependendies. You will then be able to replicate the results from this notebook.
