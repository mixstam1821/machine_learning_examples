import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2]   # Using 1:2 as indices will give us np array of dim (10, 1)
y = df.iloc[:, 2]

df.head()

from  sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1000, random_state=0).fit(X, y)



plt.scatter(X, y)
X_grid = np.arange(min(X.values), max(X.values), 0.01).reshape(-1, 1)
plt.plot(X_grid, regressor.predict(X_grid), color='r')
plt.show()
