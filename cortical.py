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

import numpy as np
from scipy import linalg


def setup():
    """
    
    Container for model setup terms
     and translation of Friston's model
     into python idiom.

    Returns
    -------
    None.

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
        The size of a covariance matrix is DxD, where D is the number of
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


def TEST():
    """
        For testing while developing.
        
        2021-04-16: Friston(2005:821) equation 3.7
        
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
    

if __name__ == "__main__":
    TEST()