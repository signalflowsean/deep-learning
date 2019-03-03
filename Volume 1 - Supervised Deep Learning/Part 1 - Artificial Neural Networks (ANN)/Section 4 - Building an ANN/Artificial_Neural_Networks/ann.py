# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!
# Importing the Kera libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense
#from keras.backend.tensorflow_backend import set_session



#1. Randomly initialize the weights to small numbers close to (but not 0)
#2. Input the first observation of your dataset in the input layer, each feature in one input node
#3. Forward-Propagation
    # The neuron applies the activation function to this sum. 
    # The closer the activation function value is to 1, tge more the neuron passes on the signal.
#4. Compare the predicted result to the actual result. Measure generated error. 
#5. Back-Propagation. 
    #The error is back-propagated. Update the weights accortding to how much they are responsible for the error. 
    #The learning rate decides by how much we update the weights
#6. Reapeat steps 1 to 5 and update the weights after each observation
    #Or only after a batch observations 
#7. This makes an epoch. Redo more epochs.  
    
#Paramater tuning    

#Initialize the ANN
classifier = Sequential() 

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
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
