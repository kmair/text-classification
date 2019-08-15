# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:50:25 2019

DP - Matrix Chain Multiplication

@author: Longxiang Zhang
"""
import numpy as np


def matrix_chain_multiplication(p):
    """
    input ----
    p: a sequence (list, tuple, np.array, etc) of integers representing the chained matrix dimensions
    
    e.g. [1, 2, 3] means two matrices with dimension (1, 2) and (2, 3)

    output ----
    minimal number of multiplications needed
    """
    if len(p) == 1:
        raise ValueError('Invalid input, must contain at least 2 numbers')
    elif len(p) == 2:
        print('only one matrix, returns')
        return 0
    elif len(p) == 3:
        print('only two matrices, easy')
        return np.prod(p)
    
    n = len(p) - 1  # number of matrices
    m = np.zeros((n, n))

    print(m)  # print commands in the code shows how m evolves as algorithm carries on

    # edge case: chain of length 2, just two matrices product
    for i in range(n-1):
        m[i, i+1] = p[i]*p[i+1]*p[i+2]

    print(m)

    # L is chain length: from 3 to n
    for L in range(3, n+1): 
        # i is the starting matrix of the chain
        for i in range(0, n-L+1): 
            # j is the ending matrix of the chain
            j = i+L-1
            # first, set the result to very large values
            m[i, j] = np.inf
            # then, find the best point to break the chain
            for k in range(i, j): 
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1] 
                if q < m[i, j]: 
                    m[i, j] = q
                    print(m)
        
    return m[0, n-1]


if __name__ == '__main__':
    seq = [10, 30, 5, 60]
    print('Chained matrices of dimension {}'.format(seq))
    m = matrix_chain_multiplication(seq)
    print('Miminum no. of multiplication is {:d}'.format(int(m)))

