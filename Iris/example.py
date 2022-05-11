import os
import numpy as np
import pandas as pd
#import tensorflow as tf

filePath = 'Iris/iris.data' #getting data path
newFile = 'Iris/iris-mixed.data' #path of new mixed data

#creating a mixed data file
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

filePath = 'Iris/iris-mixed.data' #renamefilepath

data = pd.read_csv(filePath, names=["sl", "sw", "pl", "pw", "label"]) #read data with pandas

#split data in train, test and validation sets
#dataSet = tf.data.Dataset(data) 


print(data)
print(data.shape)