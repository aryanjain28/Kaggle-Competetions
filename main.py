import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = ['MSSubClass', 'LotArea', 'OverallQual', 
'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea','FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

xTrain = pd.read_csv('/home/aryan/Desktop/Kaggle/Housing Price Prediction/train.csv')

yTrain = xTrain['SalePrice']
xTrain = xTrain[FEATURES]
xTrain = pd.get_dummies(xTrain)

#   0.9794138105186126 1        --random state

# model
best = 0
# for i in range(1, 10):
model = RandomForestRegressor(max_depth=18, random_state=1, n_estimators=83, ccp_alpha=8.0, )
model.fit(xTrain, yTrain)
# if (best < model.score(xTrain, yTrain)):
best = model.score(xTrain, yTrain)
print(best)



xTest = pd.read_csv('/home/aryan/Desktop/Kaggle/Housing Price Prediction/test.csv')
xTest = xTest[FEATURES]
xTest = pd.get_dummies(xTest)

predictions = model.predict(xTest)

r = pd.DataFrame(predictions)
r.index+=1461
r.to_csv('/home/aryan/Desktop/Kaggle/Housing Price Prediction/myCSV2.csv', index_label=['Id','SalePrice'])

