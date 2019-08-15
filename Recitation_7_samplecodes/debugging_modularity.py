# -*- coding: utf-8 -*-
"""
Modular coding example
"""
import numpy as np


#%% Module wide variable definition
"""
These variables are shared through the whole module (current file)
and are considered module attributes (e.g., when you import this file
into another python module, these can be accessed using [module_name].[variable_name])

try to limit the number of variables of this kind, this is not a good practice
just like using a lot of global variables in C-like languages; also, either use
capitalized letters or use double underscores (e.g., __N, __SHIFT) to name those
variables
"""
N = 100       # no. of sample points
SHIFT = 5     # correlated (not equal to) R, the bigger the SHIFT, the bigger R is
GAMMA = 0.5   # correlated with GAMMA. the larger the value, the larger the margin (GAMMA)
SEED = 10601  # SEED for random number generator


#%% Data generation
def gen_dataset(w, b):
    # generate linearly separable dataset
    np.random.seed(SEED)
    x = np.random.randn(N+!,1)
#    x = np.random.randn(N,1)
    y = np.random.randn(N,1)
    z = np.zeros_like(x)
    x = x + SHIFT  
    y = y + SHIFT
    z[w[0]*x+w[1]*y+b >= 0] = 1
    z[w[0]*x+w[1]*y+b < 0] = -1
    x[w[0]*x+w[1]*y+b < 0] -= GAMMA
    x = np.concatenate((x, y), axis=1)
    
    return x, y, z


#%% Main training function
def train(x, y, z, w0, b0):
    # perceptron learning without shifting to center
    ecount1 = 0  # count no. of updates learning makes
    w1 = w0
    b1 = b0
    changed = True
    # offline learning algorithm
    while changed:
        changed = False
        for (xi, zi) in zip(x, z):
            syhat = np.sign(w1.dot(xi[:, np.newaxis].T) + b1) 
#            syhat = np.sign(w1.dot(xi) + b1) 
            syhat = 1 if syhat >= 0 else -1
            if syhat != zi:
                ecount1 += 1
                w1 = w1 + xi*zi
                b1 += zi
                changed = True
    print('---- original data ----')
    print("error count is {}".format(ecount1))
    print("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))


#%% Training with shifted data
def train_with_shift(x, y, z, w0, b0):
    # weights and bias used to generate the sample point, not necessarily what the
    # perceptron algorithm learns

    xmean = x.mean(axis=0)
    x -= xmean
    ecount1 = 0
    w1 = w0
    b1 = b0
    changed = True
    # offline learning
    while changed:
        changed = False
        for (xi, zi) in zip(x, z):
#            syhat = np.sign(w1.dot(xi) + b1)
            syhat = np.sign(w1.dot(xi) - b1)
            syhat = 1 if syhat >= 0 else -1
            if syhat != zi:
                ecount1 += 1
                w1 = w1 + xi*zi
                b1 += zi
#                changed = True
    # restore the data
    x += xmean
    b1 -= w1.dot(xmean.T)
    print('--- after recentering ---')
    print("error count is {}".format(ecount1))
    print("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))


#%% Main routine
if __name__ == '__main__':
    # set your local constants here
    w = np.array([2, -1])
    b = -SHIFT
    w0 = np.array([0, 0])
    b0 = 0
    
    # strategically place print statements
    print('1. this is the start')
    
    x, y, z = gen_dataset(w, b)
    print('2. dataset generated')
    
    train(x, y, z, w0, b0)
    print('3. normal training done')
    
    train_with_shift(x, y, z, w0, b0)
    print('4. training with shift done')

