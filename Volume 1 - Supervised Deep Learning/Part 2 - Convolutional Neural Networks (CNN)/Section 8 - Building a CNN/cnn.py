#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:31:18 2019

@author: signalflowsean
"""
# Part 1 - Building the CNN
# sequential initializes the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
# Image is a matrix of pixels
# Feature Detector is a 3X3 Matrix of pixels
# When the two are multipled a Feature Map is outputted
# We do this with many feature detectors
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

