# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 23:04:04 2022

All about expected free energy.
"""

import numpy as np
from matplotlib import pyplot as plt

## Wrapper
def G(p,q,units='b'):return efe_discrete_two_variables(p,q,units)

def efe_discrete_two_variables(
        p,
        q,
        units='b'
        ):
    """
    Calculate the two-variable form of expected free energy, defined as:
        
        G(p,q) = SUM_z [ q(z) . SUM_x [ p(x|z) . log(q(z)/p(x,z)) ] ]

    Parameters
    ----------
    p : array-like.
        Joint distribution over x and z.
        The z are the rows, the x are the columns.
        Sum over axis=0 is p(x)
        Sum over axis=1 is p(z)
    q : array-like
        Marginal distribution over z.
    units : Char1, optional
        [b]its or [n]ats. The default is 'b'.

    Returns
    -------
    G : float.
        The expected free energy.

    """
    
    ## Initialise
    p = np.array(p)
    q = np.array(q)
    
    ## Get p_z marginal array
    p_z = p.sum(axis=1)
    
    ## Sum over z
    total = 0
    for i in range(len(q)):
        
        ## Get z value
        q_zi = q[i]
        
        ## Sum over x
        for j in range(len(p[0])):
            
            ## Get conditional probabilities p(x|z)
            p_x_given_z = (p.T/p_z).T
            
            ## The conditional probability of THIS value of x given THIS value of z.
            p_cond = p_x_given_z[i][j]
            
            ## The logarithm.
            logger = np.log(q_zi/p[i][j])
            
            if units == 'b':
                logger /= np.log(2)
            
            ## Add this component of the sum to the total.
            total += q_zi * p_cond * logger
            
    return total

def entropy(
        p_x,
        units = 'b'
        ):
    """
    Entropy over marginal distribution p_x.

    Parameters
    ----------
    p_x : array-like
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    p_x = np.array(p_x)
    
    assert p_x.sum() == 1
    
    total = 0
    for i in range(len(p_x)):
        
        logger = np.log(1/p_x[i])
        
        if units == 'b':
            logger /= np.log(2)
        
        total += p_x[i] * logger
    
    return total
            
def conditional_entropy(
        p,
        units='b'
        ):
    """
    Calculate H(X|Z) from joint distribution p(x,z). Definition:
        
        H(X|Z) = SUM_x,z[ p(x,z) log(p(z)/p(x,z)) ]

    Parameters
    ----------
    p : array-like.
        Joint distribution over x and z.
        The z are the rows, the x are the columns.
        Sum over axis=0 is p(x)
        Sum over axis=1 is p(z).
    units : Char1, optional
        [b]its or [n]ats. The default is 'b'.

    Returns
    -------
    H : float.
        The conditional entropy.

    """
    
    ## Initialise
    p = np.array(p)
    
    ## Get p_z
    p_z = p.sum(axis=1)
    
    
    total = 0
    
    ## Sum over Z
    for i in range(len(p)):
        
        ## Get the value of p(z) for this row
        p_zi = p_z[i]
        
        ## Sum over X
        for j in range(len(p[i])):
            
            ## Get joint
            p_joint = p[i][j]
        
            ## Get log
            logger = np.log(p_zi/p_joint)
            
            ## Check units
            if units == 'b':
                logger /= np.log(2)
    
            ## Add this component to the total
            total += p_joint * logger
    
    return total


"""
    Components of G
"""

def G_epistemic(
        p,
        q,
        units = 'b'
        ):
    """
    Definition:
        
        G_epistemic = SUM_z[ q(z) SUM_x[ p(x|z) . log(q(z)/p(z|x)) ] ]

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    units : Char1, optional
        [b]its or [n]ats. The default is 'b'.

    Returns
    -------
    None.

    """
    
    ## Initialise
    p = np.array(p)
    q = np.array(q)
    
    ## Get p_z and p_x marginal arrays
    p_z = p.sum(axis=1)
    p_x = p.sum(axis=0)
    
    ## Sum over z
    total = 0
    for i in range(len(q)):
        
        ## Get z value
        q_zi = q[i]
        
        ## Sum over x
        for j in range(len(p[0])):
            
            ## Get conditional probabilities p(x|z) and p(z|x)
            p_x_given_z = (p.T/p_z).T
            p_z_given_x = p/p_x
            
            ## The conditional probability of THIS value of x given THIS value of z.
            p_cond_x_z = p_x_given_z[i][j]
            
            ## This conditional probability of THIS value of z given THIS value of x.
            p_cond_z_x = p_z_given_x[i][j]
            
            ## The logarithm.
            logger = np.log(q_zi/p_cond_z_x)
            
            if units == 'b':
                logger /= np.log(2)
            
            ## Add this component of the sum to the total.
            total += q_zi * p_cond_x_z * logger
            
    return total

def G_pragmatic(
        p,
        q,
        units = 'b'
        ):
    """
    Definition:
        
        G_pragmatic = SUM_z[ q(z) SUM_x[ p(x|z) . log(1/p(x)) ] ]

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    units : Char1, optional
        [b]its or [n]ats. The default is 'b'.

    Returns
    -------
    None.

    """
    
    ## Initialise
    p = np.array(p)
    q = np.array(q)
    
    ## Get p_z and p_x marginal arrays
    p_z = p.sum(axis=1)
    p_x = p.sum(axis=0)
    
    ## Sum over z
    total = 0
    for i in range(len(q)):
        
        ## Get z value
        q_zi = q[i]
        
        ## Sum over x
        for j in range(len(p[0])):
            
            ## Get conditional probabilities p(x|z)
            p_x_given_z = (p.T/p_z).T
            
            ## The conditional probability of THIS value of x given THIS value of z.
            p_cond_x_z = p_x_given_z[i][j]
            
            ## This conditional probability of THIS value of z given THIS value of x.
            # p_cond_z_x = p_z_given_x[i][j]
            
            ## The logarithm.
            logger = np.log(1/p_x[j])
            
            if units == 'b':
                logger /= np.log(2)
            
            ## Add this component of the sum to the total.
            total += q_zi * p_cond_x_z * logger
            
    return total

"""
    Testing theorems
"""

def test_entropy_pragmatic(p,q,tolerance=1e-8):
    """
    Test whether G_pragmatic(p,q) == entropy(p_x)

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    p = np.array(p)
    q = np.array(q)
    
    p_x = p.sum(axis=0)
    
    difference = G_pragmatic(p, q) - entropy(p_x)
    
    return abs(difference) < tolerance

def test_entropy_pragmatic_many(p,grain=0.05):
    """
    Holds coincidentally whenever p_x == [0.5,0.5].

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    grain : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    for q1 in np.arange(0+grain,1,grain):
        
        q = np.array([q1,1-q1])
        
        if not test_entropy_pragmatic(p,q):
            return q
        
    return True
        

def test_G_decomposition(p,q,tolerance=1e-8):
    """
    Check that G = G_epistemic + G_pragmatic.
    
    Fails with p = [[0.05,0.2],[0.05,0.7]].
        TODO: check why this fails.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    difference = G(p,q) - G_epistemic(p,q) - G_pragmatic(p,q)
    
    return abs(difference) < tolerance

def test_conditional_entropy_theorem(
        p       = np.array([[0.4, 0.2],[0.1, 0.3]]),
        grain   = 0.05
        ):
    """
    
    Seems to fail with p = [[0.05,0.2],[0.05,0.7]].
    Best q is [0.2,0.8]

    Parameters
    ----------
    p : TYPE, optional
        DESCRIPTION. The default is np.array([[0.4, 0.2],[0.1, 0.3]]).
    grain : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    verified : bool.
        True if the theorem is satisfied to the stated level of grain.

    """
    
    p = np.array(p)
    
    assert p.sum() == 1
    
    p_z = p.sum(axis=1)
    
    ## sanity check
    assert sum(p_z) == 1
    
    # cond_ent = conditional_entropy(p)
    
    G=99999
    for q1 in np.arange(0+grain,1,grain):
        
        q = np.array([q1,1-q1])
        
        G_new = efe_discrete_two_variables(p,q)
        
        if G_new < G:
            G = G_new
            q_winner = q
    
    ## Is q_winner within rounding tolerance of marginal p_z?
    if sum(abs(p_z-q_winner))<1e-8:
        return True
    
    return q_winner,p_z
    

"""
=================
+ Plots +
=================
"""

def plot_efe_conditional_entropy(
        p,
        grain=0.05,
        units='b'
        ):
    """
    Plot expected free energy against conditional entropy 
     for a range of distributions q.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ## 1. PREPARE DATA
    
    ## Length of x-axis array.
    q1_values = np.arange(0+grain,1,grain)
    
    len_x = len(q1_values)
    
    ## Conditional entropy of p.
    cond_ent = conditional_entropy(p,units)
    
    ## Create y-axis values for conditional entropy.
    ## This is just the single value repeated <len_x> times.
    y_cond_ent = np.full((len_x,),cond_ent)
    
    ## Get expected free energy for various values of q
    y_G = []
    for q1 in q1_values:
        
        ## Create two-value distribution q
        q = np.array([q1,1-q1])
        
        ## Get expected free energy
        G = efe_discrete_two_variables(p,q)
        
        ## Add to series
        y_G.append(G)
    
    
    ## 2. PLOT DATA
    
    ## Refresh plt object
    # plt.gcf() # initial guess
    # plt.gca() # initial guess
    plt.clf() # initial guess
    
    ## Clear plot
    # plt.clf() # correct
    plt.cla() # correct
    plt.close() # correct
    
    ## Create new figure
    # fig,ax = plt.figure() # initial guess
    fig,ax=plt.subplots() # correct
    
    ## X-axis label
    ax.set_xlabel("Value of q1")
    
    ## Y-axis label
    ax.set_ylabel("Bits" if units=='b' else 'Nats')
    
    ## Y-axis limits
    plt.ylim(0,max(y_G))
    
    ## Plot conditional entropy
    plt.plot(q1_values, # x-axis
             y_cond_ent, # y-axis
             label= 'Conditional entropy' # forgot this
             )
    
    ## Plot expected free energy
    plt.plot(q1_values, # x-axis
             y_G, # y-axis
             label = 'Expected free energy' # forgot this
             )
    
    ## Create legend
    plt.legend()
    
    ## Display plot
    plt.show()