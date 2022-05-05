# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:41:50 2021

Plotting and output functions for FEP module.
"""

import matplotlib.pyplot as plt
from datetime import datetime



def plot_vfe(q_range,
             Fseries=[], # list of lists
             xlabel='First component of degree of belief $q$',
             savefig=False,
             range_of_states=0):
    
    fig = plt.figure()
    
    ax = plt.axes()
    
    i=1

    for F in Fseries:
        ax.plot(q_range,F[0],label=F[1],color='k')
        i+=1
    
    ax.legend()
    
    ## a. axis limits
    plt.ylim(0,5)
    
    ## b. Axis labels
    plt.xlabel(xlabel)
    plt.ylabel('Free energy')
    
    ## c. Display plot
    plt.show()
    
    ## 6. Output
    if savefig:
        fp = "out/vfe_fig_1_" + datetime.strftime(datetime.now(),"%Y%m%d-%H%M%S")
        if range_of_states>0: fp+=f'_{range_of_states}'
        fig.savefig(fp,dpi=600)

def plot_efe_time(timesteps,
             efe_list, # list
             xlabel='Timesteps',
             ylabel='Expected free energy',
             savefig=False):
    
    fig = plt.figure()
    
    ax = plt.axes()

    ax.plot(timesteps,efe_list,color='k')
    
    #ax.legend()
    
    ## a. axis limits
    plt.ylim(0,5)
    
    ## b. Axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    ## c. Display plot
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