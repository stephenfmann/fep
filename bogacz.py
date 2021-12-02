# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:30:37 2020

@author: Ste

Exercises and concepts from Bogacz (2017): A tutorial on the free-energy framework
 for modelling perception and learning. 
 https://www.sciencedirect.com/science/article/pii/S0022249615000759
 
See also Laurent Perrinet's notebook https://laurentperrinet.github.io/sciblog/posts/2017-01-15-bogacz-2017-a-tutorial-on-free-energy.html
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats,integrate
from tqdm import tqdm # progress bar


def ex1(
        u=2,            # observed light intensity
        σ_u=1,          # variance of light intensity (i.e. the noise)
        v_p=3,          # mean of prior over food item diameter
        σ_p=1,          # variance of prior over food item diameter
        v_start=0.01,   # compute posterior from
        v_end=5,        # compute posterior to
        v_grain=0.01,   # level of detail for integrating and plotting
        normalise=True  # include normalisation term in the denominator
        ):
    """
        Exercise 1, page 200.
        
        An organism infers the diameter (v) of a food item from the light intensity (u) it observes.
        
        Compute posterior using Bayes' rule: p(v|u) = p(u|v).p(v) / p(u)
        
        u: light intensity
        v: diameter
        p(v): normal with mean v_p and variance σ_p
        p(u|v): normal with mean g(v) = v^2 and variance σ_u.
        p(u): int(p(u|v).p(v)) over the range 0.01 to 5.
        
        "Assume that our animal observed the light intensity u = 2, 
          the level of noise in its receptor is σ_u = 1, 
          and the mean and variance of its prior expectation of size are v_p = 3 and σ_p = 1. 
          Write a computer program that computes the posterior probabilities 
          of sizes from 0.01 to 5, and plots them."
    """
    
    ## 1. Prior distribution over v
    p_v = stats.norm(loc=v_p,scale=np.sqrt(σ_p))
    
    ## 2. Likelihood function, which is the probability of the observation u given the state v.
    ## Generator function that receives values of v from 0.01 to 5.
    ## Assume light intensity is the square of diameter (see text).
    def p_u_given_v_func(v):
        return stats.norm(loc=v**2,             # mean
                          scale=np.sqrt(σ_u)    # standard deviation
                          )
    
    ## 3. Prior distribution over u
    ## Integrate p(u|v).p(v) over the range
    ## First, define a function that sp.integrate can work with
    def integral_component_func(v): # v is supplied below, u is known
        return p_u_given_v_func(v).pdf(u)*p_v.pdf(v)
    
    ## Now compute the definite integral
    p_u,error = integrate.quad(integral_component_func,v_start,v_end)
    print(f"Calculated normalisation prior with error {error}")
    
    ## 4. Do the bayes sum for values at each level of grain
    x_axis = np.arange(v_start,v_end,v_grain)
    y_axis = []
    for v in x_axis: # the x axis is values of v
        
        ## Bayes rule: p(v|u) = p(u|v).p(v) / p(u)
        p_v_given_u = integral_component_func(v)
        if normalise:
            p_v_given_u = p_v_given_u / p_u
        
        y_axis.append(p_v_given_u)
    
    ## 5. Plot results
    plt.plot(x_axis,y_axis)
    plt.show()


def ex2(
        u=2,            # observed light intensity
        σ_u=1,          # variance of light intensity (i.e. the noise)
        v_p=3,          # mean of prior over food item diameter
        σ_p=1,          # variance of prior over food item diameter
        timestep=0.01,  # delta(t) in the exercise
        time=5          # number of time units of gradient ascent
        ):
    """
        Exercise 2, page 201.
        
        Determine the most likely value of v by gradient ascent on the 
         numerator of the Bayesian inference equation.
        
        "Write a computer program finding the most likely size of the food item φ 
         for the situation described in Exercise 1. 
        Initialize φ = v_p, and then find its values in the next 5 time units 
         (you can use Euler’s method, 
         i.e. update φ(t + delta(t)) = φ(t) + delta(t)∂F/∂φ with delta(t) = 0.01)."
    """
    
    ## 1. Initialise
    φ = v_p  # will be updated with the most likely value of v
    steps = int(time/timestep)
    x_axis = [0]
    y_axis = [φ]
    
    ## 2. Do gradient ascent
    for step in range(steps):
        ## a. Find ∂F/∂φ
        ## Equation (8), page 200 of Bogacz (2017)
        differential = ((v_p-φ)/σ_p) + ((u-φ**2)*2*φ/σ_u)
        
        ## b. New φ is φ(t) + delta(t)∂F/∂φ 
        φ = φ + (timestep*differential)
        
        ## c. Add to list for plotting
        x_axis.append(timestep*step)
        y_axis.append(φ)
    
    ## 3. Plot
    axes = plt.gca()
    axes.set_xlim([0,time])
    axes.set_ylim([-v_p,v_p])
    plt.plot(x_axis,y_axis)
    plt.show()
    
    ## 4. Return best guess for v
    return φ


def ex3(
        u=2,            # observed light intensity
        σ_u=1,          # variance of light intensity (i.e. the noise)
        v_p=3,          # mean of prior over food item diameter
        σ_p=1,          # variance of prior over food item diameter
        timestep=0.01,  # delta(t) in the exercise
        time=5          # number of time units of gradient ascent
        ):
    """
        Exercise 3, page 201. See also figure 3, page 202.
        
        "Simulate the model from Fig. 3 for the problem from Exercise 1. 
        In particular, initialize φ = v_p, ε_p = ε_u = 0, 
         and find their values for the next 5 units of time."
    """
    
    ## 1. Initialise
    φ = v_p
    ε_p = ε_u = 0
    steps = int(time/timestep)
    x_axis = [0]
    y1 = [φ]
    y2 = [ε_p]
    y3 = [ε_u]
    
    ## 2. Loop through updates according to equations (12-14)
    for step in tqdm(range(steps)):     # tqdm adds a progress bar
        ## a. Update timestep
        x_axis.append(timestep*step)
        
        ## b. Update φ. Equation (12), page 201.
        φ_dot = ε_u*2*φ - ε_p
        φ = φ + timestep*φ_dot
        y1.append(φ)
    
        ## c. Update ε_p. Equation (13), page 201.
        ε_p_dot = φ - v_p - σ_p*ε_p
        ε_p = ε_p + timestep*ε_p_dot
        y2.append(ε_p)
    
        ## d. Update ε_u. Equation (14), page 201.
        ε_u_dot = u - φ**2 - σ_u*ε_u
        ε_u = ε_u + timestep*ε_u_dot
        y3.append(ε_u)
        
    
    
    ## 3. Plot
    axes = plt.gca()
    axes.set_xlim([0,time])
    axes.set_ylim([-2,v_p+0.5])
    plt.plot(x_axis,y1,color="k",label = "φ")
    plt.plot(x_axis,y2,color="green",linestyle='dashed',label="ε_p")
    plt.plot(x_axis,y3,color="blue",linestyle='dashed',label="ε_u")
    axes.legend()
    plt.show()
    
    