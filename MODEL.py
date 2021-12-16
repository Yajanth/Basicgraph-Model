#--------Importing Packages

import numpy as np
#from numpy.lib.financial import ipmt
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

#-------------------------Importing Datasets

#_____________Training Data

Train_Data = pd.read_csv('train.csv')
Train = Train_Data.dropna() # Training Data

#------------Variables of Training

X_Train = np.array(Train.iloc[:,:-1].values)
Y_Train = np.array(Train.iloc[:,1].values)

#__________Test Data
Test_Data = pd.read_csv('test.csv')
Test = Test_Data.dropna() #Test Data

#---------------Variables of Testing

X_test = np.array(Test.iloc[:,:-1].values)
Y_test = np.array(Test.iloc[:,1].values)

#_______________Model

Model = LinearRegression()
Model.fit(X_Train,Y_Train)

Prediction = Model.predict(X_test)
print(Prediction)

#----------------Accuracy
Accuracy = Model.score(X_test,Y_test)

#------Graph
# 
plt.plot(X_Train,Model.predict(X_Train),color = 'green')

plt.show()
print(Accuracy)

#____________Done
