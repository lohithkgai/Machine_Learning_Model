# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Encoding categorical data
dummyCountry=pd.get_dummies(dataset.Geography,drop_first=True)
dummyGender=pd.get_dummies(dataset.Gender,drop_first=True)
clean=pd.concat([dataset,dummyCountry,dummyGender],axis=1)
features=clean.drop(columns=["RowNumber","CustomerId","Surname","Geography","Gender","Exited"])
X=features.iloc[:,:].values
y=clean.Exited.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
#Sequential is used to initialize ANN
from keras.models import Sequential
#Dense is used to build the layers of ANN
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#output_dim is how many nodes in hidden layers,no right of thumb, average of the number of nodes in the input layers and output layers (1+11)/2=6
#init is , activation is recifier (only use in hidden layer).input-dim is the number of features in X only need to define once, very beginning
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = len(features.columns)))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# We only need one ouput, activation is propbability
# If the dependent variabes have multiclass classification (more than 2) , change output-dim to the number of classifications, and activation change to softmax
#if we are interested in predicting numeric values, we do not need activation function for outputlayer, and all init change to normal.
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
#if we are interested in predicting numeric values, loss change to "mean_squared_error", metrics change to ["mse"]
# if there are more than 2 classifications in the dependent variable, loss="categorical_crossentropy"
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))