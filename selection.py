# -*- coding: utf-8 -*-
"""
Created on Wed Jun 9 14:58:41 2021

Investigating a fundamental theorem of active inference in simple systems

First of all, we don't yet have an explicit statement of the theorem.
So we will be trying to refine some intuitive concepts as we go.

Set-up:
    W: external state / hidden state
    X: sensory state / input
    Y: inner state / parametrizes distributions over external state / controls action state
    Z: action state

First pass:
    
    Every Markov blanket system will come to minimize F via inference and G via action.
    
Inference means: the strategy that connects X (sensory states) to Y (inner states)
Action means: the strategy that connects Y (inner states) to Z (active states)

So what we need are:
    Death condition: some combination of values <W,X,Y,Z> that destroys the system
    Strategies: input strategies p(Y|X) and output strategies p(Z|Y)
    World causes: p(X|W) and p(W|Z); fixed and known to the agent

Start with:
    W = Y = Z = {-1,1}
    X = {-3,...0,...3}

What we need is: surviving strategies can be cast as minimizing F and G
    AND dying strategies cannot.

"""

import numpy as np

import output as op # custom plotting
import calc as cl


"""
    What are some interesting death conditions?
    If [1,1,1,1], the agent can simply set p(Y|X) = p(Z|Y) [0,1] and always survive.
    It seems we need death conditions that require some kind of 
     responsive inference and action.
    If [0,0,0,0] and [1,1,1,1], simply let Y be the opposite of X.
    
    Try: X=3 or X=-3, and X can go up or down.
"""

"""
DEATH_CONDITIONS = [
    np.array([0,0,0,0]),
    np.array([1,1,1,1])
]"""

## Default prob dist is conceptually equivalent to a preference distribution.
## You most want X=0 because that is furthest from death condition.
## Rows are w, columns are x

P_WX_DEFAULT_3 = np.array([
    [0.,0.05,0.1,0.2,0.1,0.05,0.],
    [0.,0.05,0.1,0.2,0.1,0.05,0.]
    ])

P_WX_DEFAULT_4 = np.array([
    [0.,0.025,0.05,0.075,0.2,0.075,0.05,0.025,0.],
    [0.,0.025,0.05,0.075,0.2,0.075,0.05,0.025,0.]
    ])

P_WX_DEFAULT_5 = np.array([
    [0,0.01,0.02,0.06,0.1,0.12,0.1,0.06,0.02,0.01,0],
    [0,0.01,0.02,0.06,0.1,0.12,0.1,0.06,0.02,0.01,0]
    ])

## Epsilon to avoid zero division
EPS = 0.001


def check_death(agent,range_of_states):
    """
        Checks whether the agent's states meet the death condition
    """
    
    #print(f"Agent value of x: {agent.x}") # debug
    if (agent.x >= range_of_states) or (agent.x <= -1*range_of_states):return True
    return False
    
    """
    ## Get the system states for this agent
    system = agent.states
    
    ## Does the system state match a death condition?
    for dc in DEATH_CONDITIONS:
        if np.array_equal(system,dc): return True
    
    return False"""

def die(agent):
    """
        Output current states
    """
    
    agent.dead=True
    
    #print(agent.w)
    #print(agent.x)
    #print(agent.y)
    #print(agent.z)

def run(
        t=100,   # timesteps
        range_of_states=4
        ):
    """
        Main method
        
        1. Construct agents
        
        2. Loop:
        3. Check death
        4. Run world to input
        5. Run input to inner state strategy
        6. Run inner state to output strategy
        7. Run output to world
        8. Update probability distribution
        9. :Endloop
        
        10. Print probability distribution of remaining agents
        11. Print strategies of remaining agents

    """
    
    agents = [Agent("smart"),Agent("dumb")]
    
    results = [] # will be list of dicts
    for _ in range(len(agents)):
        results.append({"w":[],"x":[],"y":[],"z":[]})
        
    for t in np.arange(t):#2
        i=0
        for agent in agents:
            if agent.dead:continue
            if check_death(agent,range_of_states):
                die(agent)
                #return agent # test
                #agents.pop(i) # lose the info
                continue
            agent = do_agent(agent) #4,5,6,7
            i+=1
        if len(agents)==0:break
    
    return agents

"""
    A single timestep for a single agent
"""
def do_agent(agent):
    agent.w_x()
    agent.x_y()
    agent.y_z()
    agent.z_w()
    
    return agent


# def ex_vfe(agent,savefig=False):
#     """
#         To figure out if the agent is complying with variational inference
#          we need to get its historical frequency distribution.
#     """
    
#     ## Set p and q
#     p_wx = cl.get_p_wx_from_agent(agent)
#     q_range = np.arange(0.1,1.,0.01) # q1 ranges from 0.1 to 0.9 at 0.1 increments
    
#     ## Initialise
#     F_0_series = [] # values of F when x=-1
#     F_1_series = [] # values of F when x=0
#     F_2_series = [] # values of F when x=1
#     F_3_series = [] # values of F when x=2
    
#     for q0 in q_range:
#         ## Create the estimated distribution across world states
#         q = np.array([q0,1-q0])
        
#         ## Get free energy for each different possible input
#         F_0 = cl.vfe_discrete(p_wx,q,0) # free energy when x=-1
#         F_1 = cl.vfe_discrete(p_wx,q,1) # free energy when x=0
#         F_2 = cl.vfe_discrete(p_wx,q,2) # free energy when x=1
#         F_3 = cl.vfe_discrete(p_wx,q,3) # free energy when x=2
        
#         F_0_series.append(F_0)
#         F_1_series.append(F_1)
#         F_2_series.append(F_2)
#         F_3_series.append(F_3)
    
#     ## List [data,label]
#     series = [[F_0_series,"free energy when x=-1"],
#               [F_1_series,"free energy when x=0"],
#               [F_2_series,"free energy when x=1"],
#               [F_3_series,"free energy when x=2"]]
    
#     op.plot_vfe(q_range,series,savefig)


def ex_vfe_time(agent,range_of_states=3,savefig=False):
    """
        Plot variational free energy of agent over time.
        Figures 5 and 6 of Free Energy: A User's Guide.
    """
    
    p_wx = eval(f'P_WX_DEFAULT_{range_of_states}')
    timesteps = range(len(agent.w_hist)) # number of timesteps
    
    vfe_list = []
    
    for t in timesteps:
        ## NB it DOESN'T MATTER what w is because p(w,x) is the same.
        if agent.y_hist[t] == 1: # assume w=1
            q = np.array([EPS,1-EPS])
        if agent.y_hist[t] == -1: # assume w=-1
            q = np.array([1-EPS,EPS])
        
        ## Shift by <range_of_states> because the first value of X is e.g. -3
        ##  and this is indexed by 0 in the array.
        vfe = cl.vfe_discrete(p_wx,q,agent.x_hist[t]+range_of_states) 
        #if vfe > 10: 
            ## debug
            #print(p_wx)
            #print(q)
            #print(agent.x_hist[t])
            #print(t)
            #return
        vfe_list.append(vfe)
    
    label = "F when agent is smart" if agent.strategy == "smart" else "F when agent acts randomly"
    series = [[vfe_list,label]]
    
    ## Plot
    op.plot_vfe(timesteps,
                series,
                xlabel='Timestep',
                savefig=savefig,
                range_of_states=range_of_states)
    
    return vfe_list


# def ex_efe_time(agent,savefig=False):
#     """
#         Expected free energy at each timestep
#     """
    
#     ## 1. Get q(w|z)
#     ## Rows are z, columns are w
#     q = np.array([[0.95,0.05],[0.05,0.95]])
    
#     timesteps = range(len(agent.z_hist)) # number of timesteps
    
#     efe_list = []
    
#     for t in timesteps:
#         ## 2. Get current z
#         z = agent.z_hist[t]
    
#         ## 3. Get current p(w,x) from P_WX_DEFAULT
#         ## The possible values of w are (-1, 1)
#         ## The possible values of x are one step below and one step above the current value
#         x = agent.x_hist[t] # current value of x
#         p_wx_00 = P_WX_DEFAULT[0][x+5-1] # plus 5 for index, -1 for the value below current
#         p_wx_01 = P_WX_DEFAULT[0][x+5+1] # plus 5 for index, +1 for the value above current
#         p_wx_10 = P_WX_DEFAULT[1][x+5-1] # plus 5 for index, -1 for the value below current
#         p_wx_11 = P_WX_DEFAULT[1][x+5+1] # plus 5 for index, +1 for the value above current
#         p_wx = np.array([[p_wx_00,p_wx_01],[p_wx_10,p_wx_11]])
        
#         ## Normalize
#         p_wx = p_wx / p_wx.sum()
    
#         ## 4. Calculate EFE for this timestep (i.e. expectation of the next timestep)
#         efe = cl.efe_discrete(p=p_wx, q=q, z=z)
        
#         efe_list.append(efe)
    
#     ## 5. Plot
#     op.plot_efe_time(timesteps, efe_list)

"""
    Agent class
"""
class Agent():
    
    
    def __init__(self,strategy="smart",w=1,x=0,y=1,z=1):
        
        self.strategy = strategy
        
        self.states = [w,x,y,z]
        
        self.w_hist = [w]
        self.x_hist = [x]
        self.y_hist = [y]
        self.z_hist = [z]
        
        self.update_attrs()
        
        self.dead=False
    
    def update_attrs(self):
        """
            Update the shortcut attributes w,x,y,z
            Update the state histories
        """
        
        self.w = self.states[0]
        
        self.x = self.states[1]
        
        self.y = self.states[2]
        
        self.z = self.states[3]
    
    def w_x(self):
        """
            World to input cause
            
            Update self.states[1] on the basis of self.states[0]
        """
        
        ## input state will be increased if external state is 1,
        ##  will be decreased if external state is -1        
        # delta = np.random.choice([self.states[0],0],p=[0.95,0.05])
        delta = self.states[0]
        
        self.states[1]+=delta
        
        self.update_attrs()
        self.w_hist.append(self.w)
    
    def x_y(self):
        """
            Input to inner state strategy
            Update self.states[2] on the basis of self.states[1]
        """
        
        ## Set to negative unless input is >0
        
        if self.strategy == "smart":
        
            self.states[2] = -1
            
            if self.states[1]<=0: self.states[2] = 1
            
        if self.strategy == "dumb":
            
            ## do something stupid
            self.states[2] = np.random.choice([-1,1],p=[0.5,0.5])
            
        
        self.update_attrs()
        self.x_hist.append(self.x)
    
    def y_z(self):
        """
            Inner state to active state strategy
            Update self.states[3] on the basis of self.states[2]
        """
        
        ## inner state is just the conduit to action here
        self.states[3] = self.states[2]
        
        self.update_attrs()
        self.y_hist.append(self.y)
    
    def z_w(self):
        """
            Output to world cause
            Update self.states[0] on the basis of self.states[3]
        """
        
        ## For now, make this fixed
        choice_mat = [1,-1]
        
        if self.states[3] == -1: choice_mat = [-1,1]
        
        ## With 95% probability you can control the state
        self.states[0] = np.random.choice(choice_mat,p=[0.95,0.05])
        
        self.update_attrs()
        self.z_hist.append(self.z)
            
        

"""
    Run 
"""
if __name__ == "__main__":
    #run()
    pass