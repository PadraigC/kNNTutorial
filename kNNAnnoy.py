# This module wraps code from the annoy package into a classifier object that provides a sklearn-like interface.  
# i.e. a classifier with `fit` and `predict` methods. 
# Author: Padraig Cunningham,March 2020. 


import annoy
from collections import Counter 

class kNNAnnoyClassifier():
    def __init__(self,n_neighbors=5, metric='euclidean', n_trees=10):
        self.n_neighbors = n_neighbors
        self.Ametric = metric
        self.n_trees = n_trees
    
    def fit(self, X_train, y_train):
        self.N_feat = X_train.shape[1]
        self.N_train = X_train.shape[0]
        self.y_train = y_train
        self.t = annoy.AnnoyIndex(self.N_feat,metric=self.Ametric)
        for i, v in zip(range(self.N_train), X_train):
            self.t.add_item(i, v)
        self.t.build(self.n_trees)
        return self

    def predict(self,X_test):
        y_hat = []
        for tv in X_test:
            nn_inds = self.t.get_nns_by_vector(tv, self.n_neighbors)
            nn_classes =[self.y_train[nn] for nn in nn_inds]
            y_hat.append(most_frequent(nn_classes))
        return y_hat
    
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 