import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
#from tensorflow import keras
#from tensorflow.keras import layers
#.\tf\Scripts\activate

filePath = 'Iris/iris.data' #getting data path
newFile = 'Iris/iris-mixed.data' #path of new mixed data

#creating a iris-mixed.data file
"""
mixData = []
with open(filePath) as fP:
    for line in fP:
        mixData.append(line)
with open(newFile, 'a') as nF:
    for each in range(50):
        nF.write(mixData[each])
        nF.write(mixData[each+50])
        nF.write(mixData[each+100])
#print(mixData)
"""

filePath = 'Iris/iris-mixed.data' #renaming filepath

data = pd.read_csv(filePath, names=["sl", "sw", "pl", "pw", "label"]) #read data from pandas
#print(data, data.shape)

features = data.drop(['label'], axis=1) #getting the training features
labels = data['label'] #getting the training labels
#print(labels, features)

encoder = LabelEncoder() #instantiate encoder class to encode labels
lEncoded = encoder.fit_transform(labels) #fitting and transforming data
labelsE = pd.get_dummies(lEncoded).values #encoding labels
#print(labelsE, features)

fd = int(.7*len(data)) #Getting first split index point of division 
sd = fd+int(.15*len(data)) #Getting second split index point of division

trainF, validationF, testF = np.split(features, [fd, sd]) #split data in train, test and validation sets
trainLE, validationLE, testLE = np.split(labelsE, [fd, sd]) #split data in train, test and validation sets
#print(trainF.shape, validationF.shape, testF.shape)
#print(trainLE.shape, validationLE.shape, testLE.shape)

model = Sequential() #instantiate the Sequential module to create the model
model.add(Dense(4, input_shape=(4,), activation='relu')) #adding an input layer of 4 perceptrons and as a activation function relu
model.add(Dense(3, activation='softmax')) #adding the output layer
model.compile(Adam(lr=0.2), 'categorical_crossentropy', metrics=['accuracy']) #definning the learning rate
print(model.summary())

model.fit(trainF, trainLE, epochs=100) #training the model

labelPred = model.predict(testF)
#print(labelPred)

labelTestClass = np.argmax(testLE, axis=1)
labelPredClass = np.argmax(labelTestClass, axis=1)