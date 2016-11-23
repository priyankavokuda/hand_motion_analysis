# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:31:18 2016

23 joint angles of hand recorded by a glove with sensors. 
There are around 250 frames in each recording and there are 1730 such recordings. 
Sampling rate of the frames in the recording is 60[Hz].
This code reads all the data from each recording and their annotations to generate combined data and annotation files.
@author: priyanka
"""
import numpy as np
import glob
import os

# file used to generate data
os.chdir("../transport-controlled/")
d_outfile = open("../data/full_train_data.txt", 'aw')
a_outfile= open("../data/combined_annotation.txt", 'aw')

def lookup(pose):
    if (pose == ".rest"): label=0
    elif(pose == "retract-opening"): label=1
    elif(    pose == ".grasp"): label=2
    elif(    pose == "reach-overopen-closing"): label=3
    elif(    pose == "reach-closing"): label=4
    elif(    pose == ".grasp-curl-grasp"): label=5
    elif(    pose == ".grasp-adjust-grasp"): label=6
    elif(    pose == "transition"): label=7
    elif(    pose == "retract-close-opening"): label=8
    elif(    pose == "reach-hold-closing"): label=9
    elif(    pose == "retract-release-opening"): label=10
    elif(    pose == "put"): label=11
    return label
    
for dataFileName in glob.glob("*-su??.txt"):
    print dataFileName
    rFileName=dataFileName.replace(".txt","")
    with open(dataFileName,"r") as dataReadfp:
        dataRow= dataReadfp.readlines()
        dataReadfp.close()
    with open(rFileName+"_annot.txt","r") as annotReadfp:
        annotRow= annotReadfp.readlines()
    annotReadfp.close()
    # Process data and combine all data to one long file
    for i in range(len(dataRow)):        
        dataRowList=dataRow[i].split()
        x= dataRowList[7:]
        for data in x:
            d_outfile.write(data+" ")
        d_outfile.write("\n") 
    # Process the annotation and combine all the annotation to one file
    for i in range(len(annotRow)):
        rowList=annotRow[i].split()
        if rowList[0].isdigit():
            diff=int(rowList[1])-int(rowList[0])
            # lookup annotation label
            for i in range(diff+1):
                label=lookup(rowList[2])
                a_outfile.write(str(label)+"\n")
d_outfile.close()  
a_outfile.close()  