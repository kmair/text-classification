# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:04:56 2019

@author: Kanishk
"""

import sys
import numpy as np

def Indexing(index_file):
    with open(index_file, "r") as dictionary: 
        strin = dictionary.readlines()
        length = len(strin)
        D_W = {}
        for i in range(length-1):
            word = strin[i]
            w = word[:-1]
            D_W[w] = i+1
        
        D_W[strin[length-1]] = length
    return D_W

def Mx(filename):
    with open(filename, "r") as Mx_val:
        strin = Mx_val.readlines()
        X = []
        #length = len(strin)
        for data in strin:
            x = data.split()
            l = len(x)
            for i in range(l):
                x[i] = float(x[i])
            x_ = np.array(x)
                
            X.append(x_)
            
        Mx = np.array(X)
    return Mx

class Forward_Backward():
    def __init__(self, A, B, C, X = None, Y = None, Y_W = None):
        self.A = A
        self.B = B
        self.C = C
        self.X = X
        self.Y = Y
        self.Y_W = Y_W

    def Data_separate(self, test_ip):
        with open(test_ip, "r") as dictionary: 
            strin = dictionary.readlines()
            X = []
            Y = []
            Y_W = []
            for data in strin:
                xy = data.strip().split()
                l = len(xy)
                x = []
                y = []
                y_w = []
                for i in range(l):
                    x_, y_ = xy[i].split("_")
                    x_val = WInd.get(x_)
                    y_val = TInd.get(y_)
                    y_w.append(y_)
                    x.append(x_val)
                    y.append(y_val)
                    
                X.append(x)
                self.X = X
                Y.append(y)
                self.Y = Y
                Y_W.append(y_w)
                self.Y_W = Y_W
        return X, Y, Y_W
        
    def Alpha(self, Xi):
        pi = self.C[:, 0]
        T = len(Xi)
        J = len(self.B)
        Alpha = np.zeros((J, T)).T
        Alpha[0] = self.B.T[Xi[0]-1] + pi
        
        for t in range(1, T):
            At_1 = Alpha[t-1]
            Sum_alp_a = self.Dot_Alpha(At_1)
    
            Alpha[t] = Sum_alp_a + self.B.T[Xi[t]-1]
    
        return Alpha
    
    def Beta(self, Xi):
        T = len(Xi)
        J = len(self.B)
        Beta = np.zeros((J, T)).T
        Beta[T-1] = np.ones(J)
        for t in range(T-1, 0, -1):
            #print(B[:,Xi[t]-1])
            temp = Beta[t] * self.B[:,Xi[t]-1] * self.A
            Beta[t-1] = sum(temp.T)
    
        return Beta
    
    def FB(self):
         N = len(self.X)
         ALPHA = {}
         BETA = {}
         for i in range(N):
             Xi = X[i]
             
             alpha = self.Alpha(Xi)
             beta = self.Beta(Xi)
             
             ALPHA[i] = alpha
             BETA[i] = beta
             #ALPHA[i] = np.log(Alph)
         return ALPHA, BETA
    
    def Dot_Alpha(self, Alp_1):
        dotpdt = []
        J = len(self.B)
        for j in range(J):
            Aj = self.A[:,j]
            sum_a = Aj + Alp_1
            T1 = sum_a[0]
            for i in range(1, J):
                T2 = sum_a[i]
                temp = logsumexp(T1, T2)
                T1 = temp
            dotpdt.append(T1)
            
        return dotpdt         

#%% New SLATE
def logsumexp(x, y):
    if x == -np.inf:
        return y
    if y == -np.inf:
        return x
    s = max(x, y) + np.log(1 + np.exp(-abs(x-y)))
    return s

def prediction(Alpha, Beta): # MBRP
    Y_pred = {}
    N = len(Alpha)
    for i in range(N):      # len(Alpha)
        AB = Alpha[i].T * Beta[i].T
        Y = AB.T
        T = len(Y)
        y_hat = []
        for j in range(T):
            yh = np.argmax(Y[j])
            y_hat.append(yh)
            
        Y_pred[i] = y_hat
    return Y_pred

def get_key(a, Index):
    for key, value in Index.items():
        if a == value:
            return key

class output_files():
    def __init__(self, X, Y, Yhat, Y_W, avg_LL= None, Accuracy=None):
        self.X = X
        self.Y = Y
        self.Yhat =Yhat
        self.Y_W = Y_W
        self.avg_LL = avg_LL
        self.Accuracy = Accuracy

    def predict_file(self, predict):
        outfile=open(predict,"w+")
        Y_pred = list()
        for i, Xi in enumerate(X):
            yh_i = y_hat[i]
            T = len(Xi)
            y_pred = []
            for t in range(T-1):
                x = get_key(Xi[t], WInd)
                outfile.write("%s_" %(x))
                yh = get_key(yh_i[t]+1, TInd) # +1?
                outfile.write("%s " %(yh))
                y_pred.append(yh) 
                
            x = get_key(Xi[T-1], WInd)
            outfile.write("%s_" %(x))
            yh = get_key(yh_i[T-1]+1, TInd) # +1?
            outfile.write("%s" %(yh))
            y_pred.append(yh)         
            Y_pred.append(y_pred)
            outfile.write("\n")
        outfile.close()   

    # Metrics file of Avg log-likelihood
    def Metrics(self): #Y, Yhat
        N = len(self.Y)
        err = 0
        cor = 0
        LL = []
        for i in range(N):
            Yi = self.Y[i]
            Yhi = self.Yhat[i]
            T = len(Yi)
            # Log-Likelihood
            lAi = ALPHA[i].T         # Since Alpha is in ln terms
            Ai = np.exp(lAi)
            sumA = sum(Ai[:, T-1])
            LL.append(np.log(sumA))
            # Accuracy
            for j in range(T):
                if Yi[j] == Yhi[j]:
                    cor += 1
                else:
                    err += 1

        self.avg_LL = sum(LL) / N
        self.Accuracy = 1 - err / (cor + err)
        # return Accuracy, avg_LL

    def metrics_writer(self, metric_file):
        with open(metric_file, "w+") as outfile:
            outfile.write('Average Log-Likelihood: ' + '%0.11f' %(self.avg_LL) + '\n')
            outfile.write('Accuracy: ' + '%0.12f' %(self.Accuracy) + '\n')


if __name__ == "__main__":
    pass

test_ip = sys.argv[1]
ind_word = sys.argv[2]
ind_tag = sys.argv[3]
prior = sys.argv[4]
emission = sys.argv[5]
transition = sys.argv[6]
predict = sys.argv[7]
metric_file = sys.argv[8]

WInd = Indexing(ind_word)
TInd = Indexing(ind_tag)

A = Mx(transition)
B = Mx(emission)       
C = Mx(prior)

lA = np.log(A)
lB = np.log(B)
lC = np.log(C)

fb = Forward_Backward(lA, lB, lC)
X, Y, Y_W = fb.Data_separate(test_ip)
    
ALPHA, BETA = fb.FB()   # X, lA, lB, lC
y_hat = prediction(ALPHA, BETA)     

# Creating prediction and metrics files
output = output_files(X, Y, y_hat, Y_W)
output.predict_file(predict)
    
output.Metrics()
output.metrics_writer(metric_file)        