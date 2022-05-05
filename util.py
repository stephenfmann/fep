# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:29:31 2021

Utility functions for FEP module.

"""

def check_dist(p):
    """
        Check p is a probability distribution
    """
    
    assert p.sum() == 1

def check_cond_mat(p):
    """
        Check p is a conditional probability matrix.
        Each row must sum to 1.
    """
    
    for row in p:
        
        assert row.sum() == 1