# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import quad # TODO: integration of continuous distribution



def vfe(p,q,x,debug=False):
    """
        p: numpy array representing p(w,x), two-dimensional (w rows, x columns)
        q: numpy array representing q(w), one-dimensional (w columns)
        x: integer representing the value of x observed. Corresponds to a column of p.
        debug: set to True to see verbose output
        
        Variational free energy is a function of three things:
            + a joint distribution p(w,x) treated as a generative model of the 
                statistical relationship between unobserved states w
                and observed states x
            + a distribution q(w) treated as an approximation of p(w),
                the marginal distribution of p(w,x)
            + an observed input value x
    
        Calculate the variational free energy between p and q.
        F = Energy          - Entropy
          = <log(1/p(x,w))>_q(w)    - <log(1/q(w))>_q(w)
        
        For discrete distributions, <.>_q(w) is the sum over values of q(w).
        For continuous distributions, <.>_q(w) is the integral over values of q(w).
        
        This function is the discrete version.
        
    """
    
    ## 0. Normalise.
    ##    This step ensures that the distributions sum to 1,
    ##     as is required for probability distributions.
    p = p / np.sum(p) # joint distribution of w and x
    q = q / np.sum(q) # single distribution of w
    
    ## 1. Value of x selects a column of p(w,x)
    p_col = p[:,x]
    
    ## 1. Calculate the "energy"
    energy = np.sum(q * np.log(1/p_col)) # element-wise multiplication
    if debug: print("Energy: "+str(energy))
    
    ## 2. Calculate the entropy of q
    entropy = np.sum(q * np.log(1/q)) # element-wise multiplication
    if debug: print("Entropy: "+str(entropy))
    
    ## 3. subtract the entropy from the energy
    F = energy - entropy
    
    return F

def vfe_cont(p,q):
    """
        Variational free energy for continuous distributions.
        
        p and q should both be of some useful type from scipy or numpy 
    """

    ## 0. Normalise??
    
    """
    ## test
    mean = 5
    std = 1

    #y = norm.pdf(x,5,1)
    def normal_distribution_function(x):
        value = norm.pdf(x,mean,std)
        return value
    
    #x1 = mean + std
    #x2 = mean + 2.0 * std

    res, err = quad(normal_distribution_function, x1, x2)
    
    return (res,err)
    """
    
    def erg(x):
        value = q.pdf(x)*np.log(1/p.pdf(x))
        return value
    
    def ent(x):
        value = q.pdf(x)*np.log(1/q.pdf(x))
        return value
    
    ## 1. Calculate Energy
    #energy = quad(q*np.log(1/p),-np.inf,np.inf) # quad takes function, start, stop
    energy,err = quad(erg,-np.inf,np.inf) # quad takes function, start, stop
    
    ## 2. Calculate Entropy
    ## Maybe there is just a simple scipy or numpy function for entropy of q?
    entropy,err = quad(ent,-np.inf,np.inf)
    
    ## 3. subtract entropy from energy
    F = energy - entropy
    
    return F

    