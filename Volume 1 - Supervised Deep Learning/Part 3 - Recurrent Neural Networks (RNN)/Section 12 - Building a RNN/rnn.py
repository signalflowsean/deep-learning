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

# Compiling the RNN
# RMSprop (stochastic gradient desent optimizer), is recommended by Keras for RNNs
# We are going to use the Adam optimizer
# Loss for classification is binary cross entropy 
# Loss for regression is mean squared error
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and vizualizing the results
# Getting the real stock price of 2017 - test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# Numpy array of one column
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# training set + test set
# test set must contain the previous 60 timesteps which are inclued in out train set
# we just need the open column
# vertical concat = 0, horizontal concat = 1
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# need to get the previous prices of test size
# .values to make a numpy array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# need the reshape the data so its the same as training data
inputs = inputs.reshape(-1,1)
# scale the inputs
inputs = sc.transform(inputs)

X_test = []

# 60 previous inputs of the test set
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
 
X_test = np.array(X_test)
# needs to be in the three dimensions: batch_size, timesteps, indicators
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
predicted_stock_price = regressor.predict(X_test)

#trained with scaled values - inversing scale values: back to normal values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', lavel = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xLabel('Time')
plt.yLabel('Google Stock Price')
plt.legend()
plt.show()