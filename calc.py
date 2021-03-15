# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import quad # TODO: integration of continuous distribution



def vfe_discrete(
        p,
        q,
        x,
        units='n',
        debug=False
        ):
    """
        Calculate variational free energy for discrete distributions.    
    
        p: numpy array representing p(w,x), two-dimensional (w along rows, x along columns)
        q: numpy vector representing q(w), one-dimensional (w as columns)
        x: integer representing the value of x observed. Corresponds to a column of p.
        units: [n]ats or [b]its
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
        For continuous distributions, <.>_q(w) would be the integral over values of q(w).
        
        This function is the discrete version.
        See vfe_cont for the continuous version.
        
    """
    
    ## 0. Check inputs
    if units!='n' and units!='b':
        print('Error: units must be nats or bits.')
        return False
    
    ## 1. Normalise.
    ##    This step ensures that the distributions sum to 1,
    ##     as is required for probability distributions.
    p = p / np.sum(p) # joint distribution of w and x
    q = q / np.sum(q) # single distribution of w
    
    ## 2. Value of x selects a column of p(w,x)
    p_col = p[:,x]
    
    ## 3. Calculate the "energy"
    if units=='n':      energy = np.sum(q * np.log(1/p_col)) # element-wise multiplication
    if units=='b':    energy = np.sum(q * np.log2(1/p_col)) # element-wise multiplication
    
    if debug: print("Energy: "+str(energy))
    
    ## 4. Calculate the entropy of q
    if units=='n':      entropy = np.sum(q * np.log(1/q)) # element-wise multiplication
    if units=='b':    entropy = np.sum(q * np.log2(1/q)) # element-wise multiplication
    
    if debug: print("Entropy: "+str(entropy))
    
    ## 5. subtract the entropy from the energy
    F = energy - entropy
    
    return F

def vfe_cont(
        p,
        q,
        x,
        units='n',
        debug=False
        ):
    """
        Calculate variational free energy for continuous distributions.
        
        p: [TODO TYPE] representing p(w,x), two-dimensional
        q: [TODO TYPE] representing q(w), one-dimensional
        x: integer representing the value of x observed. 
        units: [n]ats or [b]its
        debug: set to True to see verbose output
        
        Variational free energy is a function of three things:
            + a joint distribution p(w,x) treated as a generative model of the 
                statistical relationship between unobserved states w
                and observed states x
            + a distribution q(w) treated as an approximation of p(w),
                the marginal distribution of p(w,x)
            + an observed input value x
    
        Calculate the variational free energy between p and q.
        F = Energy - Entropy
          = <log(1/p(x,w))>_q(w) - <log(1/q(w))>_q(w)
        
        For continuous distributions, <.>_q(w) is the integral over values of q(w).
        For discrete distributions, <.>_q(w) would be the sum over values of q(w).
        
        This function is the continuous version.
        See vfe_discrete for the discrete version
        
    """
    
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
    
    def erg(z):
        ## TODO: p should be chosen with respect to observed value x
        value = q.pdf(z)*np.log(1/p.pdf(z))
        return value
    
    def ent(z):
        value = q.pdf(z)*np.log(1/q.pdf(z))
        return value
    
    ## 1. Calculate Energy
    energy,err = quad(erg,-np.inf,np.inf) # quad takes function, start, stop
    
    ## 2. Calculate Entropy
    ## Maybe there is just a simple scipy or numpy function for entropy of q?
    entropy,err = quad(ent,-np.inf,np.inf)
    
    ## 3. subtract entropy from energy
    F = energy - entropy
    
    return F

    