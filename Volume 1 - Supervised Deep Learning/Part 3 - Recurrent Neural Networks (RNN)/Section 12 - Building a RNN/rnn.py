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


# Part 2 - Building the RNN
# Part 3 - Making the predictions and vizualizing the results
