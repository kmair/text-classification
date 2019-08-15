# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:06:31 2019
Example code on how to debug
"""

import numpy as np
import logging


# ---- Log setting ---- #
#logger = logging.getLogger(__name__)  # get module name as the log file name
logger = logging.getLogger(__name__)    # create "logger" object
logger.setLevel(logging.DEBUG)    # set logging level at root
# Typical usage
#sysout = logging.StreamHandler()  # create streaming handler ---- print to console
#sysout.setLevel(logging.DEBUG)    # set debugging level DEBUG is the lowest level; the lower the level, the more types of messages will be saved
#logger.addHandler(sysout)         # add handler to "logger" object

# Can add more handlers with different level
filelog = logging.FileHandler('saved_on_disk.log', mode='w')  # mode can be 'a' for append or 'w' for overwrite
filelog.setLevel(logging.INFO)    # this file log will only save logging.INFO level of messages
formatter = logging.Formatter('%(levelname)s - %(message)s')
filelog.setFormatter(formatter)   # specify format of logging messages written to "filelog" ONLY
logger.addHandler(filelog)
#logger.removeHandler(filelog)     # remove handler from "logger" object


if __name__ == '__main__':
    # Modify those parameters to change experiments
    N = 100       # no. of sample points
    shift = 5     # correlated (not equal to) R, the bigger the shift, the bigger R is
    gamma = 0.5   # correlated with gamma. the larger the value, the larger the margin (gamma)
    seed = 10601  # seed for random number generator
    
    
    # weights and bias used to generate the sample point, not necessarily what the
    # perceptron algorithm learns
    w = np.array([2, -1])
    b = -shift
    w0 = np.array([0, 0])
    b0 = 0
    
    
    # generate linearly separable dataset
    np.random.seed(seed)
    x = np.random.randn(N,1)
    y = np.random.randn(N,1)
    z = np.zeros_like(x)
    x = x + shift  
    y = y + shift
    z[w[0]*x+w[1]*y+b >= 0] = 1
    z[w[0]*x+w[1]*y+b < 0] = -1
    x[w[0]*x+w[1]*y+b < 0] -= gamma
    x = np.concatenate((x, y), axis=1)
    
    logger.info('start training....')
    
    
    # perceptron learning without shifting to center
    ecount1 = 0  # count no. of updates learning makes
    w1 = w0
    b1 = b0
    changed = True
    # offline learning algorithm
    while changed:
        changed = False
        for (xi, zi) in zip(x, z):
            syhat = np.sign(w1.dot(xi) + b1) 
            syhat = 1 if syhat >= 0 else -1
            if syhat != zi:
                ecount1 += 1
                w1 = w1 + xi*zi
                b1 += zi
                changed = True
    logger.info('---- original data ----')
    logger.info("error count is {}".format(ecount1))
    logger.info("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))
    logger.info('---- original data ----')
    logger.info("error count is {}".format(ecount1))
    logger.info("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))
    
    
    # learning after shifting to zero mean
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
            syhat = np.sign(w1.dot(xi) + b1) 
            syhat = 1 if syhat >= 0 else -1
            if syhat != zi:
                ecount1 += 1
                w1 = w1 + xi*zi
                b1 += zi
                changed = True
    # restore the data
    x += xmean
    b1 -= w1.dot(xmean)
    logger.info('--- after recentering ---')
    logger.info("error count is {}".format(ecount1))
    logger.info("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))


