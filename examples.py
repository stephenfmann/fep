# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats,integrate
import matplotlib.pyplot as plt
from datetime import datetime

import calc as cl


def example_2x2(savefig=False,both=True):
    """
        savefig: (boolean) save the figure to an external PNG file?
    
        Imagine a world with two unobservable states w1, w2 and two observable states x1, x2.
        Your system has a generative model p(w,x), an estimate q(w), and can observe x.
        
        The INFERENCE problem: Given a SINGLE observed value of x, 
                                what is the current value of w?
                                That is, what is p(w|x)?
        The LEARNING problem: Given a SEQUENCE of observed values of x, 
                               what is the distribution of values of w?
                               That is, what is p(w)?
        
        Break the LEARNING problem into two stages:
            1. How to update q as a result of observations of x?
            2. How to update p as a result of the changes in q that took place in stage 1?
        
        For the first stage, we choose q so that it minimise the function F:
            F = SUM(q(w)*log(1/p(x,w))) - SUM(q(w)*log(1/q(w)))
            
        This example function plots the values of F as a function of the first entry in q,
         for different inputs x1 and x2.
        Because q only has two values (the estimated probabilities of w1 and w2),
         it is possible to graph F as a function of its first value.
    """
    
    ## 1. Choose the range of values of q to be plotted.
    q_range = np.arange(0.1,1.,0.01) # q1 ranges from 0.1 to 0.9 at 0.1 increments
    
    ## 2. Choose a generative model p(w,x)
    ##  joint probability of w1 and x1: 0.4
    ##  joint probability of w1 and x2: 0.2
    ##  joint probability of w2 and x1: 0.1
    ##  joint probability of w2 and x2: 0.3
    ##  the implied marginal distributions are p(w) = (0.6,0.4) and p(x) = (0.5,0.5)
    p = np.array([[0.4,0.2],[0.1,0.3]])
    
    ## 3. Initialise
    F_0_series = [] # values of F when x=0
    F_1_series = [] # values of F when x=1
    
    ## Calculate free energy for various estimates q
    for q0 in q_range:
        ## Create the estimated distribution across world states
        q = np.array([q0,1-q0])
        
        F_0 = cl.vfe_discrete(p,q,0) # free energy when x=0
        F_1 = cl.vfe_discrete(p,q,1) # free energy when x=1
        
        F_0_series.append(F_0)
        F_1_series.append(F_1)
    
    ## 5. Plot
    '''
    plt.plot(q_range,F_0_series,label="free energy when x=meow")
    plt.plot(q_range,F_1_series,label="free energy when x=purr")
    plt.legend()
    '''
    
    ## 5a. Data and data labels
    fig = plt.figure()
    
    ax = plt.axes()
    ax.plot(q_range,F_0_series,label="Free energy when $x=$meow")
    if both: ax.plot(q_range,F_1_series,label="Free energy when $x=$purr")
    ax.legend()
    
    ## 5b. Axis labels
    plt.xlabel('Degree of belief $q($kitchen$)$')
    plt.ylabel('Free energy')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "vfe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)

def ex_cont(
        x=2,            # observed light intensity
        σ_x=1,          # variance of light intensity (i.e. the noise)
        w_p=3,          # mean of prior over food item diameter
        σ_p=1,          # variance of prior over food item diameter
        w_start=0.01,   # compute posterior from
        w_end=5,        # compute posterior to
        w_grain=0.01,   # level of detail for integrating and plotting
        normalise=True  # include normalisation term in the denominator
        ):
    """
        Example adapted from Bogacz, exercise 1, page 200.
        
        An organism infers the diameter (w) of a food item from the light intensity (x) it observes.
        
        Compute posterior using Bayes' rule: p(w|x) = p(x|w).p(w) / p(x)
        
        x: light intensity
        w: diameter
        p(w): normal with mean v_p and variance σ_p
        p(x|w): normal with mean g(w) = v^2 and variance σ_x.
        p(x): int(p(x|w).p(w)) over the range 0.01 to 5.
        
        "Assume that our animal observed the light intensity x = 2, 
          the level of noise in its receptor is σ_x = 1, 
          and the mean and variance of its prior expectation of size are w_p = 3 and σ_p = 1. 
          Write a computer program that computes the posterior probabilities 
          of sizes from 0.01 to 5, and plots them."
    """
    
    ## 1. Prior distribution over w
    p_w = stats.norm(loc=w_p,scale=np.sqrt(σ_p))
    
    ## 2. Likelihood function, which is the probability of the observation x 
    ##     given the state w.
    ## Generator function that receives values of w.
    ## Assume light intensity is the square of diameter (see text).
    def p_x_given_w_func(w):
        return stats.norm(loc=w**2,scale=np.sqrt(σ_x))
    
    ## The approximate Free Energy way
    ## Need p(w), p(x|w), x, q(w)
    ## We already have the first 3
    ##  now pick an estimate q(w) of the posterior 
    ##  and calculate the free energy.
    q_w = stats.norm(loc=1.4,scale=0.3)
    
    vfe = cl.vfe_cont(p=p_w, 
                      p_cond=p_x_given_w_func, 
                      q=q_w, 
                      x=x,
                      debug=False)
    print(vfe)
    
    
    ## The exact Bayesian way
    
    ## 3. Prior distribution over x
    ## Integrate p(x|w).p(w) over the range
    ## First, define a function that sp.integrate can work with
    def integral_component_func(w): # w is supplied below, x is known
        return p_x_given_w_func(w).pdf(x)*p_w.pdf(w)
    
    ## Now compute the definite integral
    p_x,error = integrate.quad(integral_component_func,w_start,w_end)
    print(f"Calculated normalisation prior with error {error}")
    
    ## 4. Do the bayes sum for values at each level of grain
    x_axis = np.arange(w_start,w_end,w_grain)
    y_bayes = []
    y_fep = []
    
    for w in x_axis: # the x axis is values of w
        
        ## Bayes rule: p(w|x) = p(x|w).p(w) / p(x)
        p_w_given_x = integral_component_func(w)
        if normalise:
            p_w_given_x = p_w_given_x / p_x
        
        y_bayes.append(p_w_given_x)
        y_fep.append(q_w.pdf(w))
    
    plt.plot(x_axis,y_bayes)
    plt.plot(x_axis,y_fep)
    plt.show()

if __name__=="__main__":
    ex_cont()