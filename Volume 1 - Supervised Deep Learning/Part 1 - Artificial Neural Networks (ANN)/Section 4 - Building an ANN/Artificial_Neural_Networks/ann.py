# Artificial Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Grabs 3-12 columns
X = dataset.iloc[:, 3:13].values
#Grabs 13 column
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#fit_transform centers the data by standard deviation

#Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Since categorical values are nominal put binary values into multiple columns
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# Only puts 20% of tain data in test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - make the independent variable have the same range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
# Importing the Kera libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense

#Initialize the ANN
#Sequential model is a linear stack of layers
classifier = Sequential() 

#Relu is the rectifier function
#using the rectifier function we will use for the hidden layer

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim = 11))

#Adding the second hidden layer - already made the first input layer (don't need input_dim param)
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

#Compiling the ANN - using gradient descent
#For logistic loss function is logarithmic loss
#Adam a method for stochastic optimization
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the ANN to the Training set
#Batch Size: the number of samples processed before the model is updated. 
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


#Part 3 - Making the predictions and evaluating the model
#Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


#Making the confusion matrix
#A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)