import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = ['MSSubClass', 'LotArea', 'OverallQual', 
'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea','FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

xTrain = pd.read_csv('/home/aryan/Desktop/Kaggle/Housing Price Prediction/train.csv')

yTrain = xTrain['SalePrice']
xTrain = xTrain[FEATURES]
xTrain = pd.get_dummies(xTrain)

# model
model = LinearRegression()
model.fit(xTrain, yTrain)
print('Model trained')

xTest = pd.read_csv('/home/aryan/Desktop/Kaggle/Housing Price Prediction/test.csv')
xTest = xTest[FEATURES]
xTest = pd.get_dummies(xTest)

predictions = model.predict(xTest)
print(model.score(xTrain, yTrain))

r = pd.DataFrame(predictions)
r.index+=1461
r.to_csv('/home/aryan/Desktop/Kaggle/Housing Price Prediction/myCSV1.csv', index_label=['Id','SalePrice'])
