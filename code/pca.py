# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 18:02:55 2016
@author: priyanka

This code uses sklearn.decomposition.PCA from scikit-learn machine learning library for python.
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
The dimention is reduced from 23 to 9 
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def meanSubtraction(X):
    m=np.mean(X,axis=0)
    X_scaled=np.subtract(X,m)
    return X_scaled,m

def readFile(fileName):
    X=np.loadtxt(fileName+".txt",dtype=np.float)
    return X

    


''' pca '''
def pcaML(X_train):
    # n_components=0.95 means amount of variance that needs to be explained is greater than the 95%
    pca = PCA(n_components=0.95,whiten=True)
    reduced_X=pca.fit_transform(X_train)
    #reduced_X=pca.transform(X_train)
    print ("PCA done")
    #np.savetxt("reduced_train_data.txt",reduced_X,newline="\n",delimiter=" ")
    print ("shape after pca",reduced_X.shape)
    return reduced_X,pca
    #s=pca.explained_variance_ratio_
    #imp_eigen_val=s[0:reduced_X.shape[1]]
    #np.save("eigen_val",imp_eigen_val)
    #np.savetxt("combined_data_test.txt",reduced_X,newline="\n",delimiter=" ")

        

''' svd '''
#X_train_scaled,mean = meanSubtraction(X_train)
#print X_train_scaled.shape
#
#u,s,vt=np.linalg.svd(X_train_scaled.T,full_matrices=True)
#print s.shape
#
#cumsum=np.cumsum(s)
#cumsum=100*cumsum/np.max(cumsum)
#plt.grid(True)
#plt.plot(cumsum)
#plt.xlabel("joint angles")
#plt.ylabel("normalized cumulative sum of PC")
#plt.savefig('cumsum.png',dpi=1200)
#X_pc = u[:,0:MAJOR_COMPONENTS]
#X_train_reduced = np.dot(X_train_scaled,X_pc)
#print X_train_reduced.shape
#imp_eigen_val=s[0:MAJOR_COMPONENTS]
#np.savetxt("reduced_train_data.txt",X_train_reduced,newline="\n",delimiter=" ")
#np.save("eigen_val",imp_eigen_val)
#    




