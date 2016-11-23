
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:11:50 2016

This code converts each frame of data to timeseries data by sliding window approach. SVM Classification library is used from scikit-learn library. 
http://scikit-learn.org/stable/modules/svm.html
@author: priyanka
"""
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
import itertools
import os
import pca
#import autoEncoder


win_size=5
def plot_confusion_matrix(cm, label_names=False,title='Confusion matrix(log scale)', cmap=plt.cm.Blues):
    np.set_printoptions(precision=2)
    print('Confusion matrix(logscale)')
    print(cm)
    plt.figure()    
    label_names=[".rest","retract-opening",".grasp","reach-overopen-closing","reach-closing",
".grasp-curl-grasp",".grasp-adjust-grasp","transition","retract-close-opening",
"reach-hold-closing","retract-release-opening","put"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=90)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_svm_timeseries.png',dpi=1200)

def convertToTimeSeries(train,win_size,label):
    step_size=win_size/2
    new_train_data=np.zeros((0,train.shape[1]*win_size))
    new_label=np.zeros((0,1))
    data=[]
    for i in range(len(train)-win_size+1):
        for j in range(win_size):
            data.append(train[i+j,:].tolist())    
        data_merged = list(itertools.chain(*data))
        new_train_data=np.row_stack((new_train_data,data_merged))
        data=[]
        new_label=np.row_stack((new_label,label[i+step_size,]))
    new_label=np.reshape(new_label,(new_label.shape[0],))
    return new_train_data,new_label
    
def convertToTimeSeriesDataOnly(train,win_size):
    new_train_data=np.zeros((0,train.shape[1]*win_size))
    data=[]
    for i in range(len(train)-win_size+1):
        for j in range(win_size):
            data.append(train[i+j,:].tolist())    
        data_merged = list(itertools.chain(*data))
        new_train_data=np.row_stack((new_train_data,data_merged))
        data=[]
    return new_train_data
    
def readFile(dataFileName,labelFileName):
    X_train_reduced=np.loadtxt(dataFileName,dtype=np.float) 
    label=np.loadtxt(labelFileName,dtype=np.float)
    print ("Reading data and label files for training....")
    return X_train_reduced,label

def train(algo):
    print os.getcwd()
    os.chdir("../data/")
    # Training
    #dataFileName="encoded_test.txt" # Using autoencoder
    dataFileName="full_train_data.txt" # Using pca
    labelFileName="combined_annotation.txt"
    X_train,label=readFile(dataFileName,labelFileName)
    if algo=="pca":
        X_train_reduced,dim_red_object=pca.pcaML(X_train)
#    if algo=="autoencoder":
#        X_train_reduced,dim_red_object=autoEncoder.encode(X_train)  
    new_train_data,new_label=convertToTimeSeries(X_train_reduced,win_size,label)
    print ("Train data converted to time series, shape",new_train_data.shape)
#    X_train, X_test, y_train, y_test = train_test_split( new_train_data, new_label, test_size=0.20, random_state=42)  
#    new_train_data_list=new_train_data.tolist()
#    X_train_list=X_train.tolist()
#    index_list=[]
#    for ele in X_train_list:
#        index_list.append(new_train_data_list.index(ele))
#    print(index_list)
#    itemindex = np.where(new_train_data==X_train)
#    print ('item index: ',itemindex)
    label = np.ravel(label)
    clf = svm.SVC(C=15,gamma= 0.10000000000000001)
    clf.fit(new_train_data, new_label)
    
    clf.predict(new_train_data)    
    score=clf.score(new_train_data,new_label)
    print "Score of SVM on data with reduced dimentionality: ",score*100,"%" 
    
    print ("Training over....")
    return clf,dim_red_object
   
    
def test(clf,dim_red_object,X_test,algo):
    if algo==('pca'):
        reduced_X_test=dim_red_object.transform(X_test)
#    if algo==('autoencoder'):
#        reduced_X_test,_=dim_red_object.predict(X_test)
    X_test_new=convertToTimeSeriesDataOnly(reduced_X_test,win_size)
    print ("Test data converted to time series: shape ",X_test_new.shape )
    y_predict=clf.predict(X_test_new)
    last_ele=y_predict[len(y_predict)-1]
    remaining=X_test.shape[0]-len(y_predict)
    padding=[last_ele]*remaining
    y_predict=y_predict.tolist()
    for i in range(len(padding)):
        y_predict.append(padding[i])
    print (len(y_predict)  )  
    #score=clf.score(X_test,y_test)
    #print "Score of SVM on data with reduced dimentionality: ",score*100,"%" 
    print ("Testing over....")
    return y_predict
    