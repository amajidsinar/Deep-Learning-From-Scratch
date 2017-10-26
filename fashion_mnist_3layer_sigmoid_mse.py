#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:03:17 2017

@author: amajidsinar
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("Dataset/fashion-mnist_train.csv")

y = dataset.iloc[:,0].values
X = dataset.iloc[:,1:].values

X = X / 255 

# one hot encoding
# label = set of y
label = set(y)
# one hot encoder is a big diagonal matrix
label = np.eye((len(label)))
# convert y into one hot encoding
y = label[y]

# training!

# sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

alpha = 0.4
batch = 1000
hidden_neuron = 5

a0 = X
w1 = np.random.random((a0.shape[1], hidden_neuron))
w2 = np.random.random((hidden_neuron, y.shape[1]))

for i in range(batch):
    mini = np.random.randint(0,a0.shape[0],minibatch)
    a0=a0[mini]
    y=y[mini]
    
    z1 = np.dot(a0,w1)
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1,w2)
    a2 = sigmoid(z2)
    
    # update w2 first
    err2 = ((a2-y)/minibatch)*sigmoid_prime(z2)
    w2 -= alpha * np.dot(a1.T,err2)
    
    # update w1
    err1 = np.dot(w2.T,err2)*sigmoid_prime(z1)
    w1 -= alpha * np.dot(a0.T,err1)


# testing!
    
y_predict = np.dot(a0,w1)