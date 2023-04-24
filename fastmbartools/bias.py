import numpy as np
import sys
import os

def linear_bias(x, p):
    """
    Define linear bias k*(x-x0). 
    
    Prameters
    ---------
    x : 2d array-like
        Samples of target collective variables (CVs).
    
    p : 1d array-like
        Bias parameters. The format should be [k_0, x0_0, k_1, x0_1, ...]
    
    Returns
    -------
    energy : 1d array-like
        Bias potential for each sample. 

    """
    p = np.reshape(p, (-1, 2))
    k = p[:, 0]
    x0 = p[:, 1]
    energy = np.sum(k*(x - x0), axis=1)
    return energy
    

def harmonic_bias(x, p):
    """
    Define harmonic bias 0.5*k*((x-x0)**2). 
    
    Prameters
    ---------
    x : 2d array-like
        Samples of target collective variables (CVs).
    
    p : 1d array-like
        Bias parameters. The format should be [k_0, x0_0, k_1, x0_1, ...]
    
    Returns
    -------
    energy : 1d array-like
        Bias potential for each sample. 

    """
    p = np.reshape(p, (-1, 2))
    k = p[:, 0]
    x0 = p[:, 1]
    energy = np.sum(0.5*k*((x - x0)**2), axis=1)
    return energy


