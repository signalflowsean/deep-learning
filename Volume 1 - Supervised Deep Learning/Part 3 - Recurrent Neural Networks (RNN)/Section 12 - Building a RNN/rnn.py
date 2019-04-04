#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:28:33 2019

@author: signalflowsean
"""

#Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

# Part 1 - Data Preprocesing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# Numpy array of one column
training_set = dataset_train.iloc[:, 1:2].values

# Two types of Feature Scaling : Standardization, Normalization 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data strucutre with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
 
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))

# Stacked LSTM with dropout regularization to prevent overfitting
# Part 2 - Building the RNN
# nerual network object representing a sequense of layers
from keras.models import Sequential
# dense class to add the output layer
from keras.layers import Dense
# to add the LSTM layers
from keras.layers import LSTM
# to add dropout regularization
from keras.layers import Dropout

# Initializing the RNN
# named regressor instead of classifier becuase now we are doing regression (continuous value)
regressor = Sequential()
# Part 3 - Making the predictions and vizualizing the results
