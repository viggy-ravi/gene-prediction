''' ld.py (linear discriminant)

Provides code to train linear discriminant weight vectors (train_linear_discrinimant), 
    find gaussian distributions for positive and negative tis features (prior_gaussian), 
    and find posterior probabilities for positive and negative tis features (X3, X4).
    
See original paper (Hoff et. al) for more information about the gaussian distributions for tis features.
'''

import numpy as np
from scipy.stats import norm

'''
X (_,n), where n = 64, 4096, or 3712
'''
def train_linear_discriminant(X, y, l=1):    
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T@X + l*I) @ X.T@y
    return np.array(w).reshape(-1)


def prior_gaussian(tis_pos, tis_neg):
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

    return [pi_pos, pi_neg], [mu_pos, mu_neg], [sd_pos, sd_neg]

# returns probability that tis sequence is tis positive (from a gene) 
def X3(s, pi, mu, sd):
    likelihood = norm.pdf(s, mu[0], sd[0])
    marginal = pi[0]*norm.pdf(s, mu[0], sd[0]) + pi[1]*norm.pdf(s, mu[1], sd[1])
    
    return (pi[0] * likelihood) / marginal

# returns probability that tis sequence is tis negative (not from a gene)
def X4(s, pi, mu, sd):
    likelihood = norm.pdf(s, mu[1], sd[1])
    marginal = pi[0]*norm.pdf(s, mu[0], sd[0]) + pi[1]*norm.pdf(s, mu[1], sd[1])
    
    return (pi[1] * likelihood) / marginal