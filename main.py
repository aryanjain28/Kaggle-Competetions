from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn

FEATURES = ['Sex', 'SibSp', 'Parch', 'Pclass']

xTrain = pd.read_csv('/home/aryan/Desktop/Kaggle/input/titanic/train.csv')

yTrain = xTrain['Survived']
xTrain = xTrain[FEATURES]

xTrain = pd.get_dummies(xTrain)

# model
model = RandomForestClassifier(n_estimators=100, random_state=1, max_depth=5)
model.fit(xTrain, yTrain)

xTest = pd.read_csv('/home/aryan/Desktop/Kaggle/input/titanic/test.csv')
xTest = xTest[FEATURES]
xTest = pd.get_dummies(xTest)

predictions = model.predict(xTest)
yTest = pd.DataFrame(predictions)
yTest.index += 892

yTest.to_csv('/home/aryan/Desktop/Kaggle/myCSV.csv', index_label=['PassengerId', 'Survived'])

print('Model Trained')