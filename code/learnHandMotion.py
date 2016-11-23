# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:34:54 2016
@author: priyanka

This code predicts the grasp type given the joint angles as input.
For dimentionality reduction PCA is used then the data is converted timeseries by sliding window approach. The predection is done using SVM.
This code used modules from svm_timeseries.py

"""

import numpy as np
import svm_timeseries
import os
import glob
from PIL import Image
#import autoEncoder

size=5
def createBand(pixelVal):
    '''
    source: https://en.wikibooks.org/wiki/Python_Imaging_Library/Editing_Pixels
    '''
    img = Image.new( 'RGB', (size,size), "white") # create a new black image
    pixels = img.load() # create the pixel map
    for i in range(img.size[0]):    # for every pixel:
        for j in range(img.size[1]):
            pixels[i,j] = pixelVal # set the colour accordingly
    return img
    
def lookup(val):
    if (val == 0):  pixelVal=(255,255,255)
    elif(val == 1): pixelVal=(0,0,255)
    elif(val == 2): pixelVal=(255,255,0)
    elif(val == 3):pixelVal=(255,0,0)
    elif(val == 4):pixelVal=(0,255,0)
    elif(val == 5):pixelVal=(0,0,0)
    elif(val == 6):pixelVal=(0,255,255)
    elif(val == 7):pixelVal=(255,0,255)
    elif(val == 8):pixelVal=(128,0,128)
    elif(val == 9):pixelVal=(0,128,128)
    elif(val == 10):pixelVal=(178,34,34)
    elif(val == 11):pixelVal=(139,139,139)
    return pixelVal

# Write prediction into a file
def createAnnotFiles(y_predicted):
    entries=[]
    startIndex=0
    for i in range(len(y_predicted)-1):
            if y_predicted[i]!=y_predicted[i+1]:
                endIndex=i
                entry="   "+str(startIndex+1)+"   "+str(endIndex+1)+"    "+str(phaseDict[y_predicted[i]])
                entries.append(entry)
                startIndex=i+1    
                entry="   "+str(startIndex+1)+"   "+str(len(y_predicted))+"    "+str(phaseDict[y_predicted[i]])
                entries.append(entry)
                firstLine="#"+" "+"interval"+"   "+"annotation"
                writefp = open(fileName+"_annot.txt", "w")   
                writefp.write(firstLine+"\n")
                for i in range(len(entries)):
                    writefp.write(str(entries[i])+"\n") 
    writefp.close() 

# Create output with colour coding for each grasp    
def createOutputImage(y_predicted):
    images=[]
    for i in range(len(y_predicted)):
        val=lookup(y_predicted[i])
        rect=createBand(val)
        images.append(rect)
    '''
    source: http://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    '''
    ##############################################
    widths, heights = zip(*(i.size for i in images)) 
    total_width = sum(widths)
    max_height = max(heights) 
    new_im = Image.new('RGB', (total_width, max_height)) 
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    ##############################################
    new_im.save(fileName+".png")

phaseDict={0:".rest",1:"retract-opening",2:".grasp",3:"reach-overopen-closing",4:"reach-closing",5:".grasp-curl-grasp",6:".grasp-adjust-grasp",7:"transition",8:"retract-close-opening",9:"reach-hold-closing",10:"retract-release-opening",11:"put"}


############################################## 
# Training
clf,dim_red_object=svm_timeseries.train("pca") # Train the classifier
#clf,dim_red_object=svm_timeseries.train("autoencoder") # Train the classifier

##################################################3
# Path from where files are read

#os.chdir("../transport-controlled-notannotated/")
os.chdir("../transport-uncontrolled/")
#os.chdir("../test/")
##################################################


for fileName in glob.glob("*[0-9].txt"): 
    with open(fileName,"r") as readfp:
        fileName=fileName.replace(".txt","")
        row= readfp.readlines()
        row=np.array(row)
        xList=[]
        for i in range(len(row)):
            rowList=row[i].split()
            x= rowList[7:]
            xList.append(x)
        xList=np.array(xList)
        print ("Initial shape: ",xList.shape)
        xList=xList.astype(float)
        ###############################################
        #Testing
        y_predicted=svm_timeseries.test(clf,dim_red_object,xList,'pca')
        readfp.close()
        #os.chdir("../transport-controlled-notannotated/")
        os.chdir("../transport-uncontrolled/")
        #os.chdir("../test/")
        ###############################################
        # Create matlab readable annotation files
        createAnnotFiles(y_predicted)
        # Create output image files        
        createOutputImage(y_predicted)
        
        
       
        
    