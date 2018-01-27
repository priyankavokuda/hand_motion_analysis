# Hand motion analysis

This project was done to compare performance of computer vision methods and machine learning methods on same data in-order to find efficient methods to assign labels to unseen/unlabeled data .
 
In this project, classification of hand pose wass performed using supervised machine learning method on labeled data.

The input to the system were joint angles of a hand recorded by CyberGlove which was moved under controlled conditions.

The the number of joint angles in a hand were too high and the joint angles were interdependent. So dimensionality reduction was performed using Principle Component Analysis.

As the data was recorded at equally spaced time intervals,  it was treated as timeseries data by combining 'n' consecutive data recordings into one reading.

At the end classification was performed using Support Vector Machines. The method compared favorably against computer vision methods by  ∼9 % points with classification accuracy of 98.08 % and 10-way cross validation. It was found that dimensionality reduction and treating data as timeseries contributed to increase in performance.

# Dependencies

*Python 2.7

*scikit-learn: http://scikit-learn.org/stable/install.html

# Usage

Run "prepareTrainingData.py" when you have new training data. New training data should be in "transport-controlled" folder.

Run "learnHandMotion.py" for classification of data which are not annotated.

"prepareTrainingData.py" reads data and annotation files from folder "transport-controlled" and writes a combined file of data and annotation to folder "data" 

"learnHandMotion.py" reads training data from "data" folder and testing data from "transport-controlled-notannotated" or "transport-uncontrolled" folders. It performs dimentionlity reduction using pca, uses svm to train and get classification of the data. It lso converts data to timeseries.

PCA and SVM are done using scikit-learn machine learning library.

# Results


![](https://github.com/priyankavokuda/priyankavokuda.github.io/blob/master/images/handmotion.gif)


Credits to: 
Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12, no. Oct (2011): 2825-2830.




