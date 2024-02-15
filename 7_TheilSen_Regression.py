import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(1, 100, 100)
y = X * 2

y[10:30] = np.random.rand(20) * 120 + 100

# Needed since X, and y has only 1 feature
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


# visualizing data
def plot(clf=None, clf_name="", color=None):
    fig = plt.figure(figsize=(15, 8))
    plt.scatter(X, y, label="Samples")
    plt.title("Made up data with outliers")
    if clf is not None:
        y_pred = clf.predict(X)
        plt.plot(X, y_pred, label=clf_name, color=color)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
plot()



from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X, y)
plot(lr, "OLS", "green")



from sklearn.linear_model import TheilSenRegressor

tr = TheilSenRegressor().fit(X, y)
plot(tr, "TheilSenRegressor", "red")




