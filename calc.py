# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad #integrate over continuous distribution
from scipy.stats import entropy
import logging # error reporting


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
        F = Energy                  - Entropy
          = <log(1/p(x,w))>_q(w)    - <log(1/q(w))>_q(w)
        
        For discrete distributions, <.>_q(w) is the sum over values of q(w).
        For continuous distributions, <.>_q(w) would be the integral over values of q(w).
        
        This function is the DISCRETE version.
        See vfe_cont() for the continuous version.
        
    """
    
    ## 0. Check inputs
    if units!='n' and units!='b':
        logging.error('Error: units must be nats or bits.')
        return False
    
    ## 1. Normalise.
    ##    This step ensures that the distributions sum to 1,
    ##     as is required for probability distributions.
    p = p / np.sum(p) # joint distribution of w and x
    q = q / np.sum(q) # single distribution of w
    
    ## 2. Value of x selects a column of p(w,x)
    p_col = p[:,x]
    
    ## 3. Calculate the "energy" in nats
    energy = np.sum(q * np.log(1/p_col)) # element-wise multiplication
    
    ## 3b. If required, convert to bits
    if units=='b': energy /= np.log(2)
    
    if debug: logging.info("Energy: "+str(energy))
    
    ## 4. Calculate the entropy of q in nats
    entropy = np.sum(q * np.log(1/q)) # element-wise multiplication
    
    ## 4b. If required, convert to bits
    if units=='b': entropy /= np.log(2)
    
    if debug: logging.info("Entropy: "+str(entropy))
    
    ## 5. subtract the entropy from the energy
    F = energy - entropy
    
    return F

"""
    vfe_cont_2()
    Calculate variational free energy for continuous distributions
     using the identity:
         free energy = energy - entropy.
"""

def vfe_cont(
        p,
        p_cond,
        q,
        x,
        units='n',
        debug=False
        ):
    """
        Calculate variational free energy for continuous distributions.
        
        p: subclass of scipy.stats.rv_continuous
            representing p(w)
        p_cond: Represents p(x|w).
                 Generator function that returns a
                 subclass of scipy.stats.rv_continuous
                 given a value of w.
        q: subclass of scipy.stats.rv_continuous
            representing q(w)
        x: [TODO TYPE] representing the value of x observed. 
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
        F = Energy                  - Entropy
          = <log(1/p(x,w))>_q(w)    - <log(1/q(w))>_q(w)
        
        For discrete distributions, <.>_q(w) would be the sum over values of q(w).
        For continuous distributions, <.>_q(w) is the integral over values of q(w).
        
        Instead of p(x,w) we use p(w)*p(x|w)
        
        This function is the CONTINUOUS version.
        See vfe_discrete() for the discrete version.
        
    """
    
    ## 0. Check inputs
    if units!='n' and units!='b':
        logging.error('Error: units must be nats or bits.')
        return False
    
    ## 1. Calculate Energy
    def erg(w):
    ## Integrating: For each value of w,
        ##  get q(w) at that value of w,
        q_w = q.pdf(w)
        ##  get p(w) at that value of w,
        p_w = p.pdf(w)
        ##  get p(x|w),
        p_x_given_w = p_cond(w).pdf(x)
        
        ## Calculate energy at that point
        if p_w*p_x_given_w==0:return 0
        value = q_w*np.log(1/(p_w*p_x_given_w))
        
        ## Convert to bits if necessary.
        if units=='b': value /= np.log(2)
        return value
    
    energy,err = quad(erg,-np.inf,np.inf,epsabs=1e-3) # quad takes function, start, stop
    if debug: logging.info(f"Calculated energy {energy} with error {err}")
    
    ## 2. Calculate Entropy
    entropy = q.entropy()
    if units=='b': entropy/=np.log(2)
    
    if debug: logging.info("Entropy: "+str(entropy))
    
    ## 3. subtract entropy from energy
    F = energy - entropy
    
    return F

"""
    vfe_cont_2()
    Calculate variational free energy for continuous distributions
     using the relative entropy.
"""

def vfe_cont_2(
        p,
        p_cond,
        q,
        x,
        units='n',
        debug=False
        ):
    """
        Calculate variational free energy for continuous distributions.
        
        p: subclass of scipy.stats.rv_continuous
            representing p(w)
        p_cond: Represents p(x|w).
                 Generator function that returns a
                 subclass of scipy.stats.rv_continuous
                 given a value of w.
        q: subclass of scipy.stats.rv_continuous
            representing q(w)
        x: [TODO TYPE] representing the value of x observed. 
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
        F = Energy                  - Entropy
          = <log(1/p(x,w))>_q(w)    - <log(1/q(w))>_q(w)
          
          = D(Q||P) - <log(p(x|w))>_q(w)
        
        For discrete distributions, <.>_q(w) would be the sum over values of q(w).
        For continuous distributions, <.>_q(w) is the integral over values of q(w).
        
        Instead of p(x,w) we use p(w)*p(x|w)
        
        This function is the CONTINUOUS version.
        See vfe_discrete() for the discrete version.
        
    """
    
    ## 0. Check inputs
    if units!='n' and units!='b':
        logging.error('Error: units must be nats or bits.')
        return False
    
    ## 1. Calculate relative entropy from P to Q
    ## NB You have to feed them in the wrong way round,
    ##     so the distribution that is called P here
    ##     is called Q in kld_cont(), and vice versa.
    ##     That's because of the definition of free energy.
    ##    How to interpret this form of KLD is
    ##     another matter.
    kld = kld_cont(q,p,units,debug=debug)
    
    
    ## 2. Calculate second part of the free energy
    ##  <log(p(x|w))>_q(w)
    
    def component_of_integral(w):
    ## Integrating: For each value of w,
        ##  get q(w) at that value of w,
        q_w = q.pdf(w)
        ##  get p(x|w),
        p_x_given_w = p_cond(w).pdf(x)
        
        ## Calculate q(w)*log(p_x_given_w) at that point
        if p_x_given_w==0:return 0
        value = q_w*np.log(p_x_given_w)
        
        ## Convert to bits if necessary.
        if units=='b': value /= np.log(2)
        return value
    
    second_term,err = quad(component_of_integral,
                           -np.inf,
                           np.inf,
                           epsabs=1e-8) # quad takes function, start, stop
    if debug: logging.info(f"Calculated term {second_term} with error {err}")
    
    
    ## 3. subtract second term from kld
    F = kld - second_term
    
    return F

"""
    kld_cont()
    Relative entropy (i.e. Kullback-Leibler divergence)
     from Q to P for continuous distributions.
    Note that the order in which the distributions are supplied
     matters: in general, D(P||Q) != D(Q||P)
"""
def kld_discrete(p,
             q,
             units='n',
             debug=False
             ):
    """
        p: numpy array
        q: numpy array
        debug: Boolean. Prints information to console.
        
        The definition is:
            D(P||Q) = <log(p/q)>_p
        
        Compare: D(Q||P) = <log(q/p)>_q
    """
    
    ## 1. Check inputs
    if units!='n' and units!='b': logging.error("Units must be nats or bits")
    
    ## 2. Normalise.
    ##    This step ensures that the distributions sum to 1,
    ##     as is required for probability distributions.
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    ## 3. Get KLD in nats
    kld = entropy(p,q)
    
    ## 4. Convert to bits if required
    if units=='b': kld /= np.log(2)
    
    ## 5. Return
    return kld
    


"""
    kld_cont()
    Relative entropy (i.e. Kullback-Leibler divergence)
     from Q to P for continuous distributions.
    Note that the order in which the distributions are supplied
     matters: in general, D(P||Q) != D(Q||P)
"""
def kld_cont(p,
             q,
             units='n',
             tolerance=1e-10,
             debug=False
             ):
    """
        p: subclass of scipy.stats.rv_continuous
        q: subclass of scipy.stats.rv_continuous
        units: [n]ats or [b]its
        tolerance: float. Usually the integrate function
                will complain if q(x)=0 and p(x)!=0.
                But if p(x)<tolerance it won't complain.
        debug: Boolean. Prints information to console.
        
        The definition is:
            D(P||Q) = <log(p/q)>_p
        
        Compare: D(Q||P) = <log(q/p)>_q
    """
    
    ## 1. Check inputs
    if units!='n' and units!='b': logging.error("Units must be nats or bits")
    
    def component_of_integral(x):
        p_x = p.pdf(x) # the value of p at this point
        q_x = q.pdf(x) # the value of q at this point
        
        if p_x == 0: return 0
        if q_x == 0: 
            ## Check if p_x is negligible
            if p_x < tolerance: return 0 # no problem
            ## Otherwise, it's an error.
            logging.error(f"Values of P p({x})={p_x} must be in the support of Q")
        
        value = p_x*np.log(p_x/q_x) # value of integral at this point
        
        ## Convert to bits if necessary
        if units == 'b': value /= np.log(2)
        
        ## return
        return value 
    
    kld,err = quad(component_of_integral,-np.inf,np.inf,epsabs=1e-6)
    if debug: logging.info(f"KLD of {kld} obtained with error {err}")
    
    
    return kld
    