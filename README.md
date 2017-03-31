# This program finds the hand pose given joint angles of the hand.

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




