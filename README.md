# gene-prediction

## Introduction

This project is based on the 2008 paper ['Gene prediction in metagenomic fragments: A large scale machine learning approach'](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-217) by Hoff et at. The goal of the `Binary-Classification-of-Coding-and-Noncoding-Genes.ipynb` notebook is to train the gene prediction algorithm using a 2-stage approach - linear discriminant and neural network. The linear discriminant model reduces high dimensional features and the neural network predicts if a given sequence is a gene or not. My intentions for starting this project were to gain a deeper understanding of machine learning approaches behind the `Orphelia gene prediction algorithm` and sharpen my ML/programming skills.  
    
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

## Quick tour
* Add example of how the pipeline will work

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

Open the `Binary-Classification-of-Coding-and-Noncoding-Genes.ipynb` file and install the necessary dependendies. You will then be able to replicate the results from this notebook.
    
### Dependencies
* os
* sys
* re
* itertools
* textwrap
* random
* numpy
* scipy
* Bio
* tensorflow
* pandas
* matplotlib
* seaborn

## TODO
* Write more descriptive explanations in notebook
* Create visuals for tricodon/hexcodon weight representations
* Create visuals for occurances of TIS codons at positions
* Train LD and NN by inputing a text file of prokaryote genome IDs
* Add more statistical analysis + more in depth analysis of where the model fails
* Train NN on more data and save NN model
* Create pipeline for predicting if any input sequence is a gene or not (FASTA) 
