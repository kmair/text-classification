# -*- coding: utf-8 -*-
"""
Example using pdb library
"""
import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt

print(sys.argv)
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

#pdb.set_trace()
# generate linearly separable dataset
np.random.seed(seed)
x = np.random.randn(N+1,1)
y = np.random.randn(N,1)
z = np.zeros_like(x)
x = x + shift  
y = y + shift
z[w[0]*x+w[1]*y+b >= 0] = 1
z[w[0]*x+w[1]*y+b < 0] = -1
x[w[0]*x+w[1]*y+b < 0] -= gamma
plt.scatter(x, y, c=z)
x = np.concatenate((x, y), axis=1)

print('start training....')


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
print('---- original data ----')
print("error count is {}".format(ecount1))
print("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))
plt.plot(x[:, 0], -1/w1[1]*(w1[0]*x[:, 0]+b1), 'r')
plt.show()


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
print('--- after recentering ---')
print("error count is {}".format(ecount1))
print("estimated weights are ({},{}) and bias is {}".format(w1[0], w1[1], b1))
plt.scatter(x[:, 0], x[:, 1], c=z.ravel())
plt.plot(x[:, 0], -1/w1[1]*(w1[0]*x[:, 0]+b1), 'b')
plt.show()

