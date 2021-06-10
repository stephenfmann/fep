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

The two-level model:
    BASIC TERMS
    v_1 = u
    v_2 = v
    v_p = prior expectation of v
    g_1 = g
    g_2 is not used
    θ_1 = θ
    θ_2 is not used
    ε_1 = ε_u
    ε_2 = ε_p
    Σ_1 = Σ_u
    Σ_2 = Σ_p
    λ_1 = λ_u
    λ_2 = λ_p
    φ_1 = u
    φ_2 = φ
    
    ERROR TERMS
    Prediction error
    ξ_u = Σ^{-1/2}_u (u-g(φ,θ))
        = (u-g(φ,θ)) / (1 + λ_u)
    
    Prior constraint
    ξ_p = Σ^{-1/2}_p (φ - v_p)
        = (φ - v_p) / (1 + λ_p)
    
    OBJECTIVE FUNCTION
    
    L = -1/2 (ξ^T_uξ_u + ξ^T_pξ_p + log|Σ_u| + log|Σ_p|)
    
    F = <L>_u
    
    ASSUMPTION 1: Noise is Gaussian (p. 822)
    Σ_u = Σ(λ_u)
    Σ^{1/2}_u = 1 + λ_u
    Σ^{-1/2}_u = (1 + λ_u)^-1
    
    ASSUMPION 2 aka HEBBIAN ASSUMPTION: 
    g(φ,θ) = φθ
    
    DERIVATIVES w.r.t φ:
    dξ_u/dφ = θ / (1+λ_u)
    
    dξ_p/dφ = 1 / (1+λ_p)
    
    E-M ALGORITHM STEPS
    E-step
    
    φ'  = dF/dφ
        = -(dξ_u/dφ)ξ_u -(dξ_p/dφ)ξ_p
        = [ θ(θφ - u) / (1+λ_u)^2 ] + [ (v_p - φ) / (1+λ_p)^2 ]

    M-step
    
    θ'  = dF/dθ 
        = -<(dξ_u/dθ)ξ_u>_u     (In Friston (2005) eqn 3.11 the second ξ has no subscript.)
        = <φ^Tξ_u>_u / (1+λ_u) (Eqn 4.2)
        
    λ'_i = dF/dλ_i
         = -<((dξ_i/dλ_i)ξ_i)>_u - (1/(1+λ_i))
         = (<ξ_iξ^T_i>_u - 1) / (1+λ_i)

See also:
    + My (draft) post https://stephenmann.isaphilosopher.com/posts/friston2005/
    + Friston's paper https://royalsocietypublishing.org/doi/full/10.1098/rstb.2005.1622

"""

import numpy as np
from scipy import linalg
import logging


def setup():
    """
    
    Container for model setup terms
     and translation of Friston's model
     into python idiom.


    """
    
    ## p 821 eq 3.6
    ## The prior value of cause v
    ##  is its expectation v_p
    ##  plus covariance ε_p
    v = v_p + ε_p
    
    ## p 821 eq 3.6
    ## The observation u is caused by some function g
    ##  of causes v and their interrelationship θ,
    ##  plus some noise ε_u
    u = g(v,θ) + ε_u
    
    ## p 821, between eqs 3.6 and 3.7
    ## ε_u and ε_p are drawn from distributions with
    ##  specified covariance matrices Σ_u and Σ_p.
    Σ_u, Σ_p = cov_matrix()
    
    ## p 821 eq 3.7
    ## Get Σ_u^{-1/2}
    Σ_u_minus_half = cov_matrix_negative_half(Σ_u)
    
    ## p 821 eq 3.7
    ## ξ_u is the prediction error.
    ##  g(φ,θ) is what was predicted.
    ##  u is what was observed.
    ##  Both are vectors, so their difference is a vector.
    ## Σ_u_minus_half is a matrix.
    ## So we are doing a matrix operation on a vector.
    ξ_u = np.dot(Σ_u_minus_half,(u - g(φ,θ)))
    
    
def cov_matrix():
    """
        From Wikipedia:
            "In probability theory and statistics, a covariance matrix 
             is a square matrix giving the covariance between each 
             pair of elements of a given random vector. 
            Any covariance matrix is symmetric and positive semi-definite 
             and its main diagonal contains variances 
             (i.e., the covariance of each element with itself).
            Intuitively, the covariance matrix generalizes the notion of 
             variance to multiple dimensions."
            
        See also https://datascienceplus.com/understanding-the-covariance-matrix/
        
        Consider weight and height. There are three numbers that
         describe the shape of a distribution made of
         these two random variables:
             + The variance of the weight
             + The variance of the height
             + The covariance between weight and height
        The covariance matrix is a 2x2 matrix with the following entries:
            + 1,1: variance of weight
            + 2,2: variance of height
            + 1,2 AND 2,1: covariance of weight and height.
        
        From this you can see that covariance matrices are always symmetric.
        The size of a covariance matrix is NxN, where N is the number of
         random variables involved i.e. the number of dimensions.
    """
    
    ## Covariance matrices must be:
        ## triangular
        ## invertible
        ## positive semidefinite 
    ##  in order to have a unique square root.
    ## I have chosen these matrix values to ensure they satisfy those conditions.
    ## See also:
        ## https://www.math.drexel.edu/~tolya/301_spd_cholesky.pdf
        ## https://cpb-us-w2.wpmucdn.com/sites.wustl.edu/dist/3/2139/files/2019/09/definitematrices.pdf
    
    ## Covariance matrix for noise in generating sensory input
    ## Generative process generates sensory inputs u
    ##  hence the name of the matrix is Σ_u
    Σ_u = np.array([[1,1],[1,2]]) # variances both 1, covariances both 1
    
    ## Covariance matrix for noise in generating cause
    ##  from higher-level causes
    Σ_p = np.array([[1,0],[0,2]]) # variances 1 and 2, covariances both 0
    
    return Σ_u, Σ_p
    

def cov_matrix_negative_half(A):
    """
        Find A^{-1/2}.
        
        From StackExchange: https://math.stackexchange.com/a/1257734
            "If A is positive definite, then A^{1/2} denotes the unique 
             positive definite square root of A. 
            That is, A^{1/2} is the unique positive definite matrix M 
             satisfying M^2=A.
            Because A^{1/2} is positive definite, it is invertible. 
             A^{−1/2} denotes the inverse of A^{1/2}."
        
        Also SE: https://stats.stackexchange.com/a/256425
            "a covariance matrix must be positive semi-definite and 
             a positive semi-definite matrix has only one square root 
             that is also positive semi-definite. "
    """
    
    ## 1. Find M such that M^2 = A.
    M = linalg.sqrtm(A)
    
    ## 2. Find inverse of M.
    M_inv = np.linalg.inv(M)
    
    return M_inv
    

def is_pos_semidef(A):
    """
        Check if a matrix is positive semi-definite.    
        From Wikipedia:
            "In linear algebra, a symmetric nxn real matrix M is said to be 
             positive-definite if the scalar z^TMz is strictly positive 
             for every non-zero column vector z of n real numbers. 
            Positive semi-definite matrices are defined similarly, 
             except that the above scalars must be positive or zero 
             (i.e. non-negative)."
        From StackExchange: https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
            "You can also check if all the eigenvalues of matrix are positive, 
            if so the matrix is positive definite."
            [This assumes the matrix is symmetric.]
    """
    
    return np.all(np.linalg.eigvals(A) >= 0)


def calculate_ξ_u(
        g,
        u,
        Σ_u=None,
        λ_u=None
        ):
    """
        Calculate ξ_u or ξ_p from inputs using either Σ or λ.
        
        Sigma form:
            ξ_u = Σ^{-1/2}_u (u-g(φ,θ))
        
        Lambda form:
            ξ_u = (u-g(φ,θ)) / (1 + λ_u)
        
    """
    
    if Σ_u is None and λ_u is None: 
        ## Error!
        logging.error('Calculating ξ_u requires either Σ_u or λ_u')
        return False
    
    ## Prioritise Σ_u
    if Σ_u is not None:
        Σ_u_minus_half = cov_matrix_negative_half(Σ_u)
    
        ξ_u = np.dot(Σ_u_minus_half,u - g)
        
        return ξ_u
    
    ## λ_u is not None
    ξ_u = (u-g) / (1+λ_u)
    
    return ξ_u


def calculate_ξ_p():
    """
    
    TODO
    
    """
    
    pass

def objective_function():
    """
        The objective function L as defined on p 821 eq 3.7
        
        Deriving eq 3.7 from eq 3.4:
        + https://stephenmann.isaphilosopher.com/posts/friston2005/
        + Wikipedia: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        + Matrix cookbook: http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf
        + StackExchange 1: https://stats.stackexchange.com/questions/345784/conditional-probability-distribution-of-multivariate-gaussian
        + StackExchange 2: the intuition: https://stats.stackexchange.com/a/71303
        
        Even when the predictions exactly match the inputs
         and the representations exactly match the causes,
         you still get a non-zero error term.
        Mathematically that's because the covariance matrices
         have nonzero determinant.
        I guess we should interpret that as: there will always
         be some noise, no matter how good your predictions.
        But it seems strange that that ineradicable noise
         is manifesting as part of the error term.
        Wouldn't that promote an inopportune
         change of representation/prediction?
         Perhaps not, depending on the algorithm you end up using
         to minimise the error term.
    """
    
    ## 1. Prediction error
    Σ_u,Σ_p=cov_matrix()
    
    Σ_u_minus_half = cov_matrix_negative_half(Σ_u)
    
    ## Observation
    u = np.array([1,1])
    
    ## Prediction
    g = np.array([1,1])#([1.5,1.5])
    
    ## Prediction error
    ξ_u = np.dot(Σ_u_minus_half,u - g)#(φ,θ)))
    
    print("INPUTS")
    print(f"Covariance matrix: {str(Σ_u)}")
    print(f"Inverse root: {str(Σ_u_minus_half)}")
    print(f"Observed vector: {str(u)}")
    print(f"Predicted vector: {str(g)}")
    print(f"Prediction error: {str(ξ_u)}")
    
    
    ## 2. Prior constraint
    Σ_p_minus_half = cov_matrix_negative_half(Σ_p)
    
    ## Prior over cause
    v_p = np.array([6,6])
    
    ## Inferred representation of cause
    φ = np.array([6,6])#([7,5])
    
    ## Prior constraint
    ξ_p = np.dot(Σ_p_minus_half,φ - v_p)
    
    print("CAUSES")
    print(f"Covariance matrix: {str(Σ_p)}")
    print(f"Inverse root: {str(Σ_p_minus_half)}")
    print(f"Prior vector: {str(v_p)}")
    print(f"Inferred vector: {str(φ)}")
    print(f"Prior constraint: {str(ξ_p)}")
    
    
    ## 3. Objective function
    ## p 821 eq 3.7
    
    ## Determinant of covariance matrix over observations noise
    Σ_u_det = np.linalg.det(Σ_u)
    
    ## Determinant of covariance matrix over causes noise
    Σ_p_det = np.linalg.det(Σ_p)
    
    ## The objective function contains four terms.
    first_term  = (-1/2) * np.dot(ξ_u,ξ_u)
    
    second_term = (-1/2) * np.dot(ξ_p,ξ_p)
    
    third_term  = (-1/2) * np.log(Σ_u_det)
    
    fourth_term = (-1/2) * np.log(Σ_p_det)
    
    error = first_term + second_term + third_term + fourth_term
    
    print(f"ERROR: {str(error)}")
    
    return error


def e_step(φ,   # the neuron whose value is going to change
           θ,   # parameter determining how v changes u
           u,   # observation
           λ_u, # noise of observation
           v_p, # prior over cause
           λ_p  # noise of cause
           ):
    """
    
    First step of the expectation-maximisation algorithm.
    
    We specify how φ changes, φ', as a function of
     the objective function F = <L>_u.
    
    Friston gives eqn (3.11) in a form that can 
     generalise to a hierarchy of any depth:
         
         φ'_{i+1} = dF/dφ_{i+1} 
                  = -(dξ^T_i/dφ_{i+1}).ξ_i -(dξ^T_{i+1}/dφ_{i+1}).ξ_{i+1}
    
    See the initial comment in this script for how this equation looks
     in a two-level system.
    Upshot:
        
        φ' = [ θ(θφ - u) / (1+λ_u)^2 ] + [ (v_p - φ) / (1+λ_p)^2 ]

    """
    
    first_term = θ*(θ*φ - u) / (1+λ_u)**2
    
    second_term = (v_p - φ) / (1+λ_p)**2
    
    φ_prime = first_term + second_term
    
    ## return φ_prime or update φ here?
    
    return φ_prime


def m_step(φ,   # the neuron whose value is going to change
           θ,   # parameter determining how v changes u
           u,   # observation
           λ_u, # noise of observation
           v_p, # prior over cause
           λ_p  # noise of cause
           ):
    """
    
    See intro notes.
    
    How θ, λ_u and λ_p change.
    
    θ' = <φ^Tξ_u>_u / (1+λ_u) (Eqn 4.2)
    
    λ'_u = (<ξ_uξ^T_u>_u - 1) / (1+λ_u)
    λ'_p = (<ξ_pξ^T_p>_u - 1) / (1+λ_p)

    """
    
    pass
    


def TEST2(v=None,A=None):
    """
        As part of trying to understand Friston's derivation
         of eq 3.7 from eq 3.4,
         check to see if the following identity holds:
             
             v: n-vector
             A: nxn matrix
             
             v^T A^{-1} v = (A^{-1/2}v)^T . (A^{-1/2}v)
    """
    
    if A is None:
        A,B = cov_matrix()
    
    if v is None:
        v = np.random.random((2,))
    
    A_inv = np.linalg.inv(A)
    
    lhs = round(np.dot(v,np.dot(A_inv,v)),8)
    
    print(f"LHS: {str(lhs)}")
    
    A_inv_sqrt = cov_matrix_negative_half(A)
    
    rhs_component = np.dot(A_inv_sqrt,v)
    
    rhs = round(np.dot(rhs_component,rhs_component),8)
    
    print(f"RHS: {str(rhs)}")
    
    print(f"RHS equals LHS? {str(rhs==lhs)}")


if __name__ == "__main__":
    #TEST2()
    pass