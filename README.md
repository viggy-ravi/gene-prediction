# gene-prediction

## Introduction

This project was based on the 2008 paper ['Gene prediction in metagenomic fragments: A large scale machine learning approach'](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-217) by Hoff et at. My intentions for starting this project were to gain a deeper understanding of machine learning approaches behind the `Orphelia gene prediction algorithm` and sharpen my programming skills. 
    
The project consists of several sections:
1. Preprocess Genome (extract coding sequences/noncoding sequences, shuffle/split data)
2. Extract Features (monocodon frequency, dicodon frequency, tis, gc)
3. Linear Discriminant - Dimentionality reduction on high dimensional features
4. Binary classification neural network for coding/noncoding gene prediction
 
### TODO
* Write more descriptive explanations in notebook
* Move Jupyter Notebook and Google Colab setups to README
* Create visuals for tricodon/hexcodon weight representations
* Create visuals for occurances of TIS codons at positions
* Train LD and NN by inputing a text file of prokaryote genome IDs
* Add more statistical analysis + more in depth analysis of where the model fails
* Train NN on more data and save NN model
* Create pipeline for predicting if any input sequence is a gene or not (FASTA) 

## Quick tour
* Add example of how the pipeline will work

## Setup
This project was written using Python 3.8. 

### Dependencies
* os
* sys
* re
* itertools
* textwrap
* numpy
* scipy
* Bio
* tensorflow

### With Google Colab
First mount your Google Drive and move to your working directory. 

    from google.colab import drive
    drive.mount('/content/gdrive')

    # Change working directory
    %cd gdrive/MyDrive/

### With pip
* TODO

### With conda
* TODO
