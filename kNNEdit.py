# This module implements two case-base (training set) editing techniques for kNN
# Condensed Nearest Neighbour
# Conservative Redundancy Reduction

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from collections import Counter

# This class implements the Conservative Redundancy Reduction algorithm.  
# There is one method `fit` that takes the X & y dataset arrays and returns the reduced versions. 
# Author: Padraig Cunningham, March 2020. 

class kNNCRR():
    def __init__(self):
        self.cov_d = {}
        self.edist = None
        self.n = 0
        
    def setup(self, X_train, y_train):
        # set up distance matrix
        self.n = X_train.shape[0]
        self.edist = np.empty([self.n, self.n])
        self.nbrs = NearestNeighbors(n_neighbors = self.n).fit(X_train, y_train)
        for i in range(self.n):
            dists, inds = self.nbrs.kneighbors([X_train[i]],return_distance=True)
            for d, j in zip(dists,inds):
                self.edist[i,j] = d  
        
        # set up u_label dictionary - the class an instance is not.
        self.u_label = {}
        labels = list(Counter(y_train).keys()) # class labels
        for i in range(self.n):
            if y_train[i] == labels[0]:
                self.u_label[i] = labels[1]
            else:
                self.u_label[i] = labels[0]        
        
        # set up distance to nun array 
        self.NUNdist = np.empty([self.n])
        for i in range(self.n):
            dists, indices = self.nbrs.kneighbors([X_train[i]],return_distance=True)
            ys = list(y_train[indices][0])
            nun_i = ys.index(self.u_label[i])
            self.NUNdist[i] = dists[0,nun_i]
       
    def EDi(self,i1,i2):
        return self.edist[i1,i2]
   
    def fit(self, X_train, y_train):
        self.setup(X_train, y_train)
        
        if self.n < 2000:
            nn = self.n
        else:
            nn = 2000
        self.nbrs = NearestNeighbors(n_neighbors = nn)
       
        self.nbrs.fit(X_train, y_train)
        self.cov_d = {}
        ESet = []
        for q in range(len(y_train)):
            cs = self.coverage_set(X_train,y_train,q)
            self.cov_d[q]=cs
       
        Full_TSet = list(self.cov_d.keys()).copy()
        Full_TSet.sort(key=lambda k: len(self.cov_d[k]))
        #print('FTS',Full_TSet)
        ES = self.build_Ei(ESet, Full_TSet)
        
        #print(self.cov_d)
        return X_train[ES], y_train[ES]
    
    def coverage_set(self, X,y,qi):
        cs = []
        labels = list(Counter(y).keys()) # class labels
        q_label = y[qi]
        labels.remove(q_label)
        u_label = labels[0]
    
        same_class = np.where(y == y[qi])[0]
        for x_ind in same_class:
             if self.EDi(x_ind,qi) < self.NUNdist[x_ind]:
                cs.append(x_ind)
        return cs
    
    def build_Ei(self,E, T):
        for c in T:
            E.append(c)
            for n in self.cov_d[c]: 
                if n in T:
                    T.remove(n)
        return E


    
# Class for Condensed Nearest Neighbour
# Iterate through the training data and select all samples where the nearest neighbour 
# is an unlike neoghbour.

class kNNCNN():
    def __init__(self):
        self.cnn = []      
        
    def fit(self, X_train, y_train):
        self.cnn = [1]      
        nbr = NearestNeighbors(n_neighbors=1)
        
        for j in range(len(y_train)):
            Xj,yj = X_train[j],y_train[j]
            if not NN_match(X_train[self.cnn], y_train[self.cnn], nbr,Xj,yj):
                self.cnn.append(j)

        return X_train[self.cnn], y_train[self.cnn]
                                                                              
def NN_match(X,y,nbr,Xj,yj):
    nbr.fit(X)
    indx = nbr.kneighbors([Xj],return_distance=False)[0][0]
    if yj == y[indx]:
        return True
    else:
        return False