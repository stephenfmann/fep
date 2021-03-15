# -*- coding: utf-8 -*-

import numpy as np
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


if __name__=="__main__":
    example_2x2()