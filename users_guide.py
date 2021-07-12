# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:26:01 2021

@author: ste_m

Figures 2, 4, 5 and 6 in the document `Free Energy: A User's Guide'
  can be generated by the relevant methods here.

fig_2()
fig_4()
fig_5()
fig_6()

"""

import examples as ex
import selection as sn


def fig_2():
    '''
        Figure 2 plots variational free energy against degree of belief
         in a certain external state.
    '''
    
    ex.example_2x2_vfe(savefig=False,both=False)


def fig_4():
    '''
        Figure 4 plots a bar chart of expected free energy 
          for two different courses of action.
    '''
    
    ex.ex_efe_bar(savefig=False)


def fig_5():
    '''
        Figure 5 plots variational free energy against time
          for a smart agent.
        
        NB this function generates a new simulation each time it is run.
        The graph in the paper will not match the output exactly.
        
        Run this function several times to get an idea of the general features
          of the simulation.
        Qualitatively, the figure's caption in the paper should roughly match
          the result of most runs of this function.
    '''
    
    agents = sn.run()
    
    ## First agent is smart, second agent acts randomly
    smart_agent = agents[0]
    sn.ex_vfe_time(smart_agent,savefig=False)
    
    
def fig_6():
    '''
        Figure 6 plots variational free energy against time
          for an agent that acts randomly.
        
        NB this function generates a new simulation each time it is run.
        The graph in the paper will not match the output exactly.
        
        Run this function several times to get an idea of the general features
          of the simulation.
        Qualitatively, the figure's caption in the paper should roughly match
          the result of most runs of this function; though time to death
          will differ on each run.
    '''
    
    agents = sn.run()
    
    ## First agent is smart, second agent acts randomly
    random_agent = agents[1]
    sn.ex_vfe_time(random_agent,savefig=False)
    
    