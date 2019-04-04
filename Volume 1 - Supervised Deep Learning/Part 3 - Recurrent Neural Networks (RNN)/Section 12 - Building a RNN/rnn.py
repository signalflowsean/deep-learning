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

# Adding the first LSTM layer and some Dropout regularization
# return sequences is true when building a stacked LSTM (multiple layers)
# input shape is the shape of the inputs we created in X_train (3 dimensions)
# X_train dimensions 1. batch size, 2. time steps, 3. indicators (we only have to include the last two)
# We want our model to have a high dimensionality (having a large number of neurons with multiple LSTM layers)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1 )))
# Dropping out %20 of the neurons (%20 of the neurons with be ignored)
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
# Dropping out %20 of the neurons (%20 of the neurons with be ignored)
regressor.add(Dropout(0.2))

#Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
# Dropping out %20 of the neurons (%20 of the neurons with be ignored)
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and some Dropout regulariztion
regressor.add(LSTM(units = 50))
# Dropping out %20 of the neurons (%20 of the neurons with be ignored)
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

# Part 3 - Making the predictions and vizualizing the results
