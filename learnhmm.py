
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:54:17 2019

@author: Kanishk
"""
import sys
import numpy as np


class HMM():
    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y

    @staticmethod
    def Indexing(index_file):
        with open(index_file, "r") as dictionary: 
            strin = dictionary.readlines()
            D_W = {}
            
            for i, data in enumerate(strin):
                string = data.strip()    
                D_W[string] = i+1
            
        return D_W
        
    def Data_separate(self, train_ip):
        with open(train_ip, "r") as dictionary: 
            strin = dictionary.readlines()
            X = []
            Y = []
            for data in strin:
                xy = data.strip().split()
                l = len(xy)
                x = []
                y = []
                for i in range(l):
                    x_, y_ = xy[i].split("_")
                    x_val = WInd.get(x_)
                    y_val = TInd.get(y_)
                    x.append(x_val)
                    y.append(y_val)
                    
                X.append(x)
                Y.append(y)
            
            self.X = X
            self.Y = Y

        # return self.X, self.Y

    def hmmLearner(self, WInd, TInd):

        x_len = len(WInd)
        y_len = len(TInd)
        A = np.ones((y_len,y_len))   # ones, becoz we have to add 1 to all
        B = np.ones((y_len,x_len))
        C = np.ones(y_len)  # size of C = # states of y (=3 here)

        for i in range(len(self.Y)):             
            yi = self.Y[i]
            xi = self.X[i]
            T = len(yi)
            
        # Finding A    
            for t in range(T-1):
                A[yi[t]-1, yi[t+1]-1] += 1
            
        # Finding B
            for t in range(T):
                B[yi[t]-1, xi[t]-1] += 1
                
        # Finding C
            C[yi[0]-1] += 1
                
        # Normalizing A    
        for i in range(y_len):
            A[i] = A[i] / sum(A[i])

        # Normalizing B
        for i in range(y_len):
            BSum = sum(B[i])
            B[i] = B[i] / BSum

        # Normalizing C
        C = C / sum(C)        
        C = C.reshape((len(C), 1))

        return A, B, C

    @staticmethod
    def op_writer(filename, Mx):
        outfile=open(filename,"w+")
        r, c = np.shape(Mx)
        for i in range(r):
            for j in range(c-1):
                outfile.write("%0.18e " %(Mx[i, j]))
            outfile.write("%0.18e" %(Mx[i, c-1]))    
            outfile.write("\n")
        outfile.close()       

if __name__ == "__main__":
    train_ip = sys.argv[1]
    ind_word = sys.argv[2]
    ind_tag = sys.argv[3]
    prior = sys.argv[4]
    emission = sys.argv[5]
    transition = sys.argv[6]
    
    hmm = HMM()
    
    # Creating the Indexed dictionaries
    WInd = hmm.Indexing(ind_word)
    TInd = hmm.Indexing(ind_tag)
    
    # Separate the X and Y labels for learning
    hmm.Data_separate(train_ip)

    # Create the prior, transition and emission Matrices
    A, B, C = hmm.hmmLearner(WInd, TInd)
        
    hmm.op_writer(transition, A)    
    hmm.op_writer(emission, B)    
    hmm.op_writer(prior, C)