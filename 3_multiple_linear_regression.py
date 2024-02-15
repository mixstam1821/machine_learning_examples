import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the cars.csv file
cars = pd.read_csv('cars.csv')
print(cars)

# Print first 5 data
cars.head()
# Print last 5 data
cars.tail()
# Print mathematical description
cars.describe()
# Print information of Dataset
cars.info()


# Divide the data into “Attributes” and “labels”
X = cars[['Weight', 'Volume']]
y = cars['CO2']

# Split 80% of the data to the training set while 20% of the data to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model and fit it
regressor =LinearRegression(fit_intercept=True)
regressor.fit(X_train,y_train)


# Getting Results
print('Linear Model Coeff (m) =' , regressor.coef_)
print('Linear Model Coeff (b) =' , regressor.intercept_)
# Linear Model Coeff (m) = [0.00728963 0.0076251 ]
# Linear Model Coeff (b) = 80.5710979169092

# Predicting the data
y_predict=regressor.predict(X_test)
print(y_predict)
# [108.54900223 104.31804036 102.72161109 108.2836746  106.53416307
# 102.46647399  96.10255102  94.96826943]


# Prediction the CO2 emission of car using Weight and Volume Value of the car directly
print('---------[700,900]---------')
wg = 700
vol = 900
co2 = regressor.predict([[wg,vol]])
print(co2)
print('--------[1100,1500]----------')
wg = 1100
vol = 1500
co2 = regressor.predict([[wg,vol]])
print(co2)
print('--------[1500,2500]----------')
wg = 1500
vol = 2500
co2 = regressor.predict([[wg,vol]])
print(co2)
