# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:41:50 2021

@author: ste_m
"""

import matplotlib.pyplot as plt
from datetime import datetime



def plot_vfe(q_range,
             Fseries=[], # list of lists
             savefig=False):
    
    fig = plt.figure()
    
    ax = plt.axes()
    
    i=1
    for F in Fseries:
        ax.plot(q_range,F[0],label=F[1])
        i+=1
    
    ax.legend()
    
    ## 5b. Axis labels
    plt.xlabel('First component of degree of belief $q$')
    plt.ylabel('Free energy')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "vfe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)


def plot_agent_history(agent,savefig=False):
    """
        An agent from ft.py with its state histories

    Parameters
    ----------
    agent : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    fig = plt.figure()
    
    ax = plt.axes()
    ax.plot(range(len(agent.w_hist)),agent.w_hist,color='k',linestyle='-',label="w")
    ax.plot(range(len(agent.x_hist)),agent.x_hist,color='b',linestyle='-',label="x")
    ax.plot(range(len(agent.y_hist)),agent.y_hist,color='r',linestyle='-',label="y")
    ax.plot(range(len(agent.z_hist)),agent.z_hist,color='g',linestyle='-',label="z")
    
    ax.legend()
    
    ## 5b. Axis labels
    plt.xlabel('Time')
    plt.ylabel('System state values')
    
    ## 5c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "vfe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        fig.savefig(fp,dpi=600)