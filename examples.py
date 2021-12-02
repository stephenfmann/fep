# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats,integrate
import matplotlib.pyplot as plt
from datetime import datetime
import logging # error reporting

import calc as cl


def example_2x2_vfe(savefig=False):
    """
        savefig: (boolean) save the figure to an external PNG file
        
        Figure 2 of Free Energy: A User's Guide.
    
        Imagine a world with two unobservable states w1, w2 and two observable states x1, x2.
        Your system has a generative model p(w,x), an estimate q(w), and can observe x.
        
        The INFERENCE problem: Given a SINGLE observed value of x, 
                                what is the current value of w?
                                That is, what is p(w|x)?
        The LEARNING problem: Given a SEQUENCE of observed values of x, 
                               what is the distribution of values of w?
                               That is, what is p(w)?
        
        Here we are just doing INFERENCE.
        And we are pretending that we cannot calculate p(w|x) directly,
         but must find an approximation to it: q(w).
        
        Q: How to choose q as a result of observations of x?
        A: Choose the q that minimizes F(p,q,x).
        
            F(p,q,x) = SUM(q(w)*log(q(w)/p(w))) + SUM(q(w)*log(1/p(x|w)))
            
        The first term is a penalty for overfitting.
        The second term is a penalty for failing to explain the data.
        See https://stephenmann.isaphilosopher.com/posts/fep_expln/ for more.
            
        This method plots values of F as a function of the first entry in q.
        Because q only has two values (the estimated probabilities of w1 and w2),
         it is possible to graph F as a function of its first value.
         Such a graph will contain all the relevant information.
        
        We also plot the two component terms of F individually.
        Call these the 'overfitting penalty' and the 'explaining penalty'.
    """
    
    ## 1. Choose the range of values of q to be plotted.
    q_range = np.arange(0.1,1.,0.01) # q1 ranges from 0.1 to 0.9 at 0.01 increments
    
    ## 2. Choose a generative model p(w,x)
    ##  joint probability of w1 and x1: 0.4
    ##  joint probability of w1 and x2: 0.2
    ##  joint probability of w2 and x1: 0.1
    ##  joint probability of w2 and x2: 0.3
    ##  the implied marginal distributions are p(w) = (0.6,0.4) and p(x) = (0.5,0.5)
    p = np.array([[0.4,0.2],[0.1,0.3]])
    
    ## 3. Initialise
    F_0_series = [] # values of F when x=0
    
    ## SFM 2021-06-07: plot overfitting penalty (KLD) and explaining penalty
    p_w = p.sum(axis=1)
    p_x_w = p.T/p_w
    p_x_w = p_x_w.T
    D_0_series = []
    E_0_series = []
    
    F_check_series = [] # confirm F = D+E
    
    ## Calculate free energy for various estimates q
    for q0 in q_range:
        ## Create the estimated distribution across world states
        q = np.array([q0,1-q0])
        
        F_0 = cl.vfe_discrete(p,q,0) # free energy when x=0; default units are nats
        
        F_0_series.append(F_0)
        
        D_0_series.append(stats.entropy(q,p_w))
        
        ## Penalty-for-explaining sum
        e_sum = 0
        i=0
        for q_value in q:
            e_sum+=q_value*np.log(1/p_x_w[i,0]) # 0th value of x, ith value of w
            i+=1
        E_0_series.append(e_sum)
        
        F_check_series.append(stats.entropy(q,p_w)+e_sum)
    
    ## 5. Plot
    
    ## 5a. Data and data labels
    fig = plt.figure()
    
    ax = plt.axes()
    ax.plot(q_range,F_0_series,color='k',linestyle='-',label="Variational free energy")
    
    ## SFM 2021-06-07: penalties
    ax.plot(q_range,D_0_series,color='k',linestyle='-.',label="Penalty for overfitting")
    ax.plot(q_range,E_0_series,color='k',linestyle='--',label="Penalty for failing to explain data")
    #ax.plot(q_range,F_check_series,label="VFE check")
    
    ax.legend()
    
    ## 5b. Axis labels
    plt.xlabel('Degree of belief $q($kitchen$)$')
    plt.ylabel('Nats')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "vfe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)


def ex_vfe_both(savefig=False):
    '''
        Like example_2x2_vfe(), but plots F against q for both values of x.
        Generates the line graphs in https://stephenmann.isaphilosopher.com/posts/fep/.

    Parameters
    ----------
    savefig : boolean, optional
        DESCRIPTION. Save plot to external PNG file. The default is False.

    Returns
    -------
    None.

    '''
    
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
    
    ## 5a. Data and data labels
    fig = plt.figure()
    
    ax = plt.axes()
    ax.plot(q_range,F_0_series,color='k',linestyle='-',label="Free energy when $x=$meow")
    ax.plot(q_range,F_1_series,label="Free energy when $x=$purr")
    
    ax.legend()
    
    ## 5b. Axis labels
    plt.xlabel('Degree of belief $q($kitchen$)$')
    plt.ylabel('Variational free energy (nats)')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "vfe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)
    
    

def ex_efe_bar(savefig=False):
    """
        Expected free energy on a simple bar chart.
        Figure 4 of Free Energy: A User's Guide.
    """
    
    ## Joint matrix
    p = np.array([[0.4,0.2],[0.1,0.3]])
    
    ## Conditional matrix
    q = np.array([[0.9, 0.1],[0.5,0.5]])
    
    ## Calculate expected free energy
    efe_0 = cl.efe_discrete(p,q,0)
    efe_1 = cl.efe_discrete(p,q,1)
    
    ## Components of efe
    p_w = p.sum(1)
    p_x_w = p.T/p_w
    p_x_w = p_x_w.T
    
    ## Normalise: float error fix
    p_x_w = p_x_w / p_x_w.sum(1)
    
    ## KLD preferences
    kld_0 = stats.entropy(q[0],p_w)
    kld_1 = stats.entropy(q[1],p_w)
    
    ## Conditional entropy
    cond_ent_0 = cl.cond_ent(p_x_w,q[0]) # expects q(w|z) and p(x|w)
    cond_ent_1 = cl.cond_ent(p_x_w,q[1]) # expects q(w|z) and p(x|w)
    
    ## 5. Plot
    ## See also https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
    ## Data and data labels
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    ## Alignment
    X = np.arange(2)
    
    ax.bar(X+0.00, [efe_0,efe_1], color='black', width=0.2)
    ax.bar(X+0.25, [kld_0,kld_1], color='silver', width=0.2, tick_label=["kitchen","bedroom"])
    ax.bar(X+0.50, [cond_ent_0,cond_ent_1], color='dimgrey', width=0.2)
    
    ## Labels
    for i, v in enumerate([efe_0,efe_1]):
        ax.text(i-0.07, v+0.005, str(v.round(3)), color='black', fontweight='bold')
        
    
    for i, v in enumerate([kld_0,kld_1]):
        ax.text(i+0.18, v+0.005, str(v.round(3)), color='black', fontweight='bold')
    
    for i, v in enumerate([cond_ent_0,cond_ent_1]):
        ax.text(i+0.43, v+0.005, str(v.round(3)), color='black', fontweight='bold')
    
    #ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    #ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    #ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
    
    ax.legend(labels=['Expected free energy', 'Preference penalty', 'Surprise penalty'])
    
    ## 5b. Axis labels
    plt.xlabel('Where you put the cat')
    plt.ylabel('Nats')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "efe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600,bbox_inches='tight')



def example_2x2_efe(savefig=False,both=False):
    """
        savefig: (boolean) save the figure to an external PNG file
    
        Imagine a world with two unobservable states w1, w2
                                two observable states x1, x2
                                two available acts, z1, z2
        Your system has a generative model p(w,x), an estimate q(w|z), and can act z.
        
        We choose z to minimize expected free energy:
            G = sum_w[ q(w|z) . sum_x[ p(x|w) log(q(w|z)/p(w,x)) ] ]
            
        This example function plots the values of G as a function of the first entry in q,
         for different inputs z1 and z2.
        Because q only has two values (the expected probabilities of w1 and w2),
         it is possible to graph G as a function of its first value.
    """
    
    ## 1. Choose the range of values of q to be plotted.
    q_range = np.arange(0.1,1.,0.01) # q1 ranges from 0.1 to 0.9 at 0.1 increments
    
    ## 2. Choose a generative model p(w,x)
    ##  joint probability of w1 and x1: 0.4
    ##  joint probability of w1 and x2: 0.2
    ##  joint probability of w2 and x1: 0.1
    ##  joint probability of w2 and x2: 0.3
    ##  the implied marginal distributions are p(w) = (0.6,0.4) and p(x) = (0.5,0.5)
    p = np.array([[0.4,0.2],[0.2,0.3]])
    
    ## 3. Initialise
    G_0_series = [] # values of G when z=0
    G_1_series = [] # values of G when z=1
    
    ## Calculate free energy for various estimates q
    for q0 in q_range:
        ## Create the estimated distribution across world states.
        ## Because of the way the expected free energy function works,
        ##  q must be a matrix.
        ## The function will choose only one row of this matrix, though.
        q = np.array([[q0,1-q0],[q0,1-q0]]) 
        
        G_0 = cl.efe_discrete(p,q,0) # free energy when z=0
        G_1 = cl.efe_discrete(p,q,1) # free energy when z=1
        
        G_0_series.append(G_0)
        G_1_series.append(G_1)
    
    ## 5. Plot
    '''
    plt.plot(q_range,F_0_series,label="free energy when x=meow")
    plt.plot(q_range,F_1_series,label="free energy when x=purr")
    plt.legend()
    '''
    
    #print(np.argmin(G_0_series)) # debug
    
    ## 5a. Data and data labels
    fig = plt.figure()
    
    ax = plt.axes()
    ax.plot(q_range,G_0_series,label="Expected free energy when $z=$kitchen")
    if both: ax.plot(q_range,G_1_series,label="Expected free energy when $z=$bedroom")
    #ax.legend()
    
    ## 5b. Axis labels
    plt.xlabel('Conditional probability $q($kitchen$|z)$')
    plt.ylabel('Expected free energy')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "efe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)


def ex_cont(
        x=2,            # observed light intensity
        σ_x=1,          # variance of light intensity (i.e. the noise)
        w_p=3,          # mean of prior over food item diameter
        σ_p=1,          # variance of prior over food item diameter
        w_start=0.01,   # compute posterior from
        w_end=5,        # compute posterior to
        w_grain=0.01,   # level of detail for integrating and plotting
        normalise=True, # include normalisation term in the denominator
        debug=False
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
                      debug=debug)
    print(vfe)
    
    
    ## The exact Bayesian way
    
    ## 3. Prior distribution over x
    ## Integrate p(x|w).p(w) over the range
    ## First, define a function that sp.integrate can work with
    def integral_component_func(w): # w is supplied below, x is known
        return p_x_given_w_func(w).pdf(x)*p_w.pdf(w)
    
    ## Now compute the definite integral
    p_x,error = integrate.quad(integral_component_func,w_start,w_end)
    if debug: logging.info(f"Calculated normalisation prior with error {error}")
    
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


def ex_test_consistency():
    """
        ex_test_consistency()
        
        Check that the two different ways
        of calculating variational free energy
        for continuous distributions
        produce the same result.
        
        Initial testing: equal to 10 significant figures :)
        
        Now need to figure out which is faster.
    """
    
    x = 2
    
    ## 1. Prior distribution over w
    p_w = stats.norm(loc=3,scale=1)
    
    ## 2. Likelihood function, which is the probability of the observation x 
    ##     given the state w.
    ## Generator function that receives values of w.
    ## Assume light intensity is the square of diameter (see text).
    def p_x_given_w_func(w):
        return stats.norm(loc=w**2,scale=1)
    
    ## The approximate Free Energy way
    ## Need p(w), p(x|w), x, q(w)
    ## We already have the first 3
    ##  now pick an estimate q(w) of the posterior 
    ##  and calculate the free energy.
    q_w = stats.norm(loc=1.4,scale=0.3)
    
    vfe_1 = cl.vfe_cont(p=p_w, 
                      p_cond=p_x_given_w_func, 
                      q=q_w, 
                      x=x
                      )
    print(f"First VFE: {vfe_1}")
    
    vfe_2 = cl.vfe_cont_2(p=p_w, 
                          p_cond=p_x_given_w_func, 
                          q=q_w, 
                          x=x)
    print(f"Second VFE: {vfe_2}")


def ex_fractions():
    """
        For two distributions N_x, M_x over events X,
         compare SUM(N_x/M_x) with |X|.
    """
    
    N = 1000
    
    def cost(N,N_x,M_x):
        """
            The average cost per customer
             of assuming M_x when N_x is true.
        """
        total = 0
        
        ## Do the sum
        for i in range(len(N_x)):
            component = (N_x[i] / N) * ((1/M_x[i])-(1/N_x[i]))
            total+=component
        
        return total
    
    N_x = np.array([500,500])
    
    xvalues = []
    yvalues = []
    
    ## Get sum for range of M_x's
    for i in np.arange(5,1000,5):
        M_x = np.array([i,1000-i])
        
        avg_cost = cost(N,N_x,M_x)
        
        ## Data for plot
        xvalues.append(i)
        yvalues.append(avg_cost)
    
    plt.plot(xvalues,yvalues)
    plt.show()


def ex_rel_ent(debug=False):
    """
        Testing intuitions about relative entropy.
    """
    
    ## 1. Initialise
    ## Set up the first distribution
    ##  and report its entropy (in bits)
    p = np.array([1/2,1/4,1/8,1/8])
    ent_p = stats.entropy(p) / np.log(2)
    print(f"Entropy of p: {ent_p:.4f}") # precision: 4 decimal places
    
    ## Set up the second distribution
    ##  and report its entropy (in bits)
    q = np.array([1/4,1/4,1/4,1/4])
    ent_q = stats.entropy(q) / np.log(2)
    print(f"Entropy of Q: {ent_q:.4f}") # precision: 4 decimal places
    
    ######################################################
    ## 2. Component calculations
    
    ## Calculate the cross-entropy of p from q: defined as SIGMA(p.log(1/q))
    p_ent_q = 0
    for i in range(len(q)):
        
        ## Get the component of the sum for this element
        comp = p[i] * np.log(1/q[i])
        
        ## Convert to bits
        comp /= np.log(2)
        
        ## Add to total
        p_ent_q += comp
        
        if debug:print(f"Component {i}: {comp}")

    
    ## Report p-entropy of q
    print(f"P-entropy of Q: {p_ent_q:.4f}")
    
    ## Calculate the cross-entropy of q from p: defined as SIGMA(q.log(1/p))
    q_ent_p = 0
    for i in range(len(p)):
        
        ## Get the component of the sum for this element
        comp = q[i] * np.log(1/p[i])
        
        ## Convert to bits
        comp /= np.log(2)
        
        ## Add to total
        q_ent_p += comp
        
        if debug:print(f"Component {i}: {comp}")

    
    ## Report q-entropy of p
    print(f"Q-entropy of P: {q_ent_p:.4f}")
    
    ######################################################
    ## 3. Relative entropy calculations
    
    ## Calculate the relative entropy of P from Q
    rel_p = p_ent_q - ent_p 
    print(f"Relative entropy of P from Q: {rel_p:.4f}")
    
    ## Calculate the relative entropy of P from Q using scipy
    rel_p_sp = stats.entropy(p,q) / np.log(2)
    print(f"Relative entropy of P from Q (scipy): {rel_p_sp:.4f}")
    
    ## Calculate the relative entropy of Q from P
    rel_q = q_ent_p - ent_q 
    print(f"Relative entropy of Q from P: {rel_q:.4f}")
    
    ## Calculate the relative entropy of Q from P using scipy
    rel_q_sp = stats.entropy(q,p) / np.log(2)
    print(f"Relative entropy of Q from P (scipy): {rel_q_sp:.4f}")


def rel_ent_theorem(
        p=np.array([0.4,0.6]),
        q=np.array([0.75,0.25]),
        r=np.array([0.1,0.9]),
        debug=False):
    """
        The default arrays provide a counterexample
         to the claim that D(p||q) < D(p||r) ==> D(q||p) < D(r||p)

    Parameters
    ----------
    p, q, r: numpy arrays.
      Represent probability distributions.
    debug : boolean, optional
        Verbose reporting. The default is False.

    Returns
    -------
    None.

    """
    
    n = 2
    
    ## 1. Generate three arbitrary probability distributions.
    
    ## P
    if p is None:
        p = np.random.random(n)
    
    ## Normalise
    p /= p.sum()
    
    if debug: print(p)
    if debug: print(p.sum())
    
    ## Q
    if q is None:
        q = np.random.random(n)
    
    ## Normalise
    q /= q.sum()
    
    if debug: print(q)
    if debug: print(q.sum())
    
    ## R
    if r is None:
        r = np.random.random(n)
    
    ## Normalise
    r /= r.sum()
    
    if debug: print(r)
    if debug: print(r.sum())
    
    
    ## 2. Get D(p||q) and D(p||r), and see which is bigger
    
    dpq = stats.entropy(p,q) / np.log(2)
    if debug:print(f"D(P||Q): {dpq}")
    dpr = stats.entropy(p,r) / np.log(2)
    if debug:print(f"D(P||R): {dpr}")
    
    sign_1 = np.sign(dpq - dpr)
    if debug: print(f"Sign 1: {sign_1}")
    
    
    ## 3. Get D(q||p) and D(r||p) and see which is bigger
    dqp = stats.entropy(q,p) / np.log(2)
    if debug:print(f"D(Q||P): {dqp}")
    drp = stats.entropy(r,p) / np.log(2)
    if debug:print(f"D(R||P): {drp}")
    
    sign_2 = np.sign(dqp - drp)
    if debug: print(f"Sign 2: {sign_2}")
    
    ## 4. Check
    assert sign_1 == sign_2
    

def gauss_and_reciprocal(savefig=False):
    """
        Generating images for MPI talk 2021-04-06

    """
    
    ## 1. Initialise
    v_start=0.01   # plot from
    v_end=6       # plot to
    v_grain=0.01  # level of detail for plotting
    
    p_v = stats.norm(loc=3,scale=np.sqrt(1))
    
    ## 2. Data
    x_axis = np.arange(v_start,v_end,v_grain)
    y_axis = []
    for v in x_axis: # the x axis is values of v
        
        y_axis.append(np.log(1/p_v.pdf(v)))
    
    ## 3. Plotting
    fig = plt.figure()
    plt.plot(x_axis,y_axis)
    plt.show()
    
    ## 4. Output
    if savefig:
        fp = "gauss_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)



if __name__=="__main__":
    ex_vfe_both()
    
