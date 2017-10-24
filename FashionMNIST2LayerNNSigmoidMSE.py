#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:26:59 2017

@author: amajidsinar
"""

# This is my attempt to create a 2 layer NN (only input and output layer) from scratch 
# Output neuron is sigmoid
# Loss is very simple loss, not even MSE

import numpy as np
import pandas as pd

np.random.seed(1)

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
minibatch = 64

a0 = X 



# make sure the mean of weight is close to zero and std is close to one
w1 = np.random.randn(a0.shape[1],y.shape[1])

# naive batch
#for i in range(batch):
#    z1 = np.dot(a0,w1)
#    a1 = sigmoid(z1)
#    C=(a1-y)**2/(2*batch)
#    err1 = C*sigmoid_prime(z1)
#    w1 -= alpha * np.dot(a0.T,err1)
    
# minibatch
for i in range(batch):
    z1 = np.dot(a0,w1)
    a1 = sigmoid(z1)
    C=(a1-y)**2/(2*batch)
    err1 = C*sigmoid_prime(z1)
    w1 -= alpha * np.dot(a0.T,err1)


# testing!
    

