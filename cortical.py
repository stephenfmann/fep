# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:08:25 2021

@author: ste_m

Predictive processing model from Friston (2005): A theory of cortical responses

The external world is a hierarchy of causes. 
Sensory input to the system is defined as v_1. 
The causes of v_1 are defined as v_2. The causes of v_2 are defined as v_3. 
And so on.

External states (strictly, elements of the generative model that represent external states):
    v_i: value of state (cause) at level i
    g_i: function of v_{i+1} and θ_i, contributing to the value of v_i
    θ_i (Greek letter theta): parameter determining how the value v_{i+1} deterministically contributes to the value v_i
    ε_i (Greek letter epsilon): noise contributing to the value of v_i
    Σ_i (Greek letter capital sigma): covariance of ε_i (under Gaussian assumption)
    λ_i (Greek letter lambda): parametrises Σ_i (under Gaussian assumption) such that Σ_i = Σ(λ_i)

Internal states
    φ_i (Greek letter phi): representational unit at level i
    ξ_i (Greek letter Xi): error unit at level i

"""

def setup():
    """
    
    Container for model setup terms
     and translation of Friston's model
     into python idiom.

    Returns
    -------
    None.

    """
    
    ## p. 821 eq. 3.6
    ## The prior value of cause v
    ##  is its expectation v_p
    ##  plus covariance ε_p
    v = v_p + ε_p
    
    ## The observation u is caused by some function g
    ##  of causes v and their interrelationship θ,
    ##  plus some noise ε_u
    u = g(v,θ) + ε_u
    
    ## Does this mean ε_u is drawn from a distribution with
    ##  specified covariance? I guess it must be.
    Cov(ε_u) = Σ_u
    Cov(ε_p) = Σ_p
    
    