# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:39:27 2021

@author: ayomy
"""

import numpy as np
# This function helps to compute the euclidean distance
def euclidean_distance(point1,point2):
    euc = np.sum((point1-point2)**2)
    e_distance = np.sqrt(euc)
    return e_distance

# This function helps to compute the manhattan distance    
def manhattan_distance(point1,point2):
    m_distance = np.sum(np.abs(point1-point2))
    return m_distance      

# K indicates the number of the nearest neighbors
# This function helps to predict the label of a single point            
def KNN(distancetype,X_train,unknown,y_train,K):
    distance = []
    for row in X_train:
        distance = np.append(distance,distancetype(row,unknown))
    distance = np.array(distance)
    sort_dist = np.argsort(distance)
    sort_dist = sort_dist[:K]
    
    #labels of the k datapoints
    labels_k = y_train[sort_dist].tolist()
    predicted_label = max(set(labels_k),key=labels_k.count)
    return predicted_label

# This function helps to predict all the data in the test data
def prediction(distancetype,X_train,X_test,y_train,K):
    predicted_labels = []
    for unknown in X_test:
        predicted_labels.append(KNN(distancetype,X_train,unknown,y_train,K))
    return predicted_labels

# This function compute the prediction accuracy based on a particular method
def prediction_accuracy(distancetype,X_train,X_test,y_train,y_test,K):
    predicted_labels = prediction(distancetype,X_train,X_test,y_train,K)
    correctcount = 0
    wrongcount = 0
    testlabel_and_predictedlabel = list(zip(y_test,predicted_labels))
    for i in range(len(testlabel_and_predictedlabel)):
        if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
            correctcount += 1
        else:
            wrongcount += 1
    accuracyratio = (correctcount/(correctcount+wrongcount))
    return accuracyratio
    
            

import numpy as np
np.random.seed (0)                             
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
X, y = load_svmlight_file("C:/Users/ayomy/Documents/ML Mini Project/datatrain1.txt")
X_test, y_test = load_svmlight_file("C:/Users/ayomy/Documents/ML Mini Project/datatest.txt")
X_train,X_test,y_train,y_test = train_test_split(X.toarray(),y,test_size = 0.2)

print(prediction_accuracy(euclidean_distance,X_train,X_test,y_train,y_test,11))
print(prediction_accuracy(manhattan_distance,X_train,X_test,y_train,y_test,11))











    