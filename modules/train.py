''' ld.py (linear discriminant)

Provides code to train linear discriminant weight vectors (discrinimant) and 
    find gaussian distributions for positive and negative tis features (tis_gaussian).
'''

import numpy as np
from scipy.sparse import csr_matrix

def discriminant(X, y, l=1):    
    '''
    INPUT: X, y, l
    OUTPUT: w
    
        X: (_,n) where n = 64, 4096, or 3712
        y: (n,) where n = 64, 4096, or 3712
        l: lambda - hyperparameter for discriminant training
    '''
    
    X = csr_matrix(X)
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T@X + l*I) @ X.T@y
    return np.array(w).reshape(-1)


def tis_gaussian(LD_tis, LD_yt, wT):
    '''
    INPUT: LD_tis, LD_yt, wT, OFFSET
    OUTPUT: ((pi_pos, pi_neg), (mu_pos, mu_neg), (sd_pos, sd_neg))
    
        Fits a Gaussian distribution around positive tis features (start codon = start of gene) and 
        negative tis features (start codon = within gene). Function returns the prior (pi), mean (mu),
        and standard deviation (sd) of the corresponding Gaussians. 
    '''
    tis_pos, tis_neg = tis_pos_neg(LD_tis, LD_yt, wT)
    
    pos_y = len(tis_pos)
    neg_y = len(tis_neg)
    tis_y = pos_y + neg_y
    
    # prior probabilities
    pi_pos = pos_y/tis_y
    pi_neg = neg_y/tis_y

    # gaussian mean and sd
    mu_pos = np.mean(tis_pos)
    sd_pos = np.std(tis_pos)

    mu_neg = np.mean(tis_neg)
    sd_neg = np.std(tis_neg)

    return ((pi_pos, pi_neg), (mu_pos, mu_neg), (sd_pos, sd_neg))

def tis_pos_neg(tis, y, wT):
    '''
    INPUT: tis, y, wT
    OUTPUT: LD_tis_pos, LD_tis_neg
    
        tis: list of tis features (58x64) from coding sequences only
        y: positive (1) or negative (-1) feature
        wT: discriminant weight vector for tis feature
        LD_tis_pos: list of positive tis features (start codon = start of gene)
        LD_tis_neg: list of negative tis features (start codon = within gene)
    '''
    
    LD_tis_pos = []
    LD_tis_neg = []
    
    LD_tis_red = tis @ wT
    for element in zip(LD_tis_red, y):
        if element[1] == 1: 
            LD_tis_pos.append(element[0])
        else: 
            LD_tis_neg.append(element[0])
            
    return LD_tis_pos, LD_tis_neg