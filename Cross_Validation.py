#K-Fold Cross Validation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, 2:4]   # Using 1:2 as indices will give us np array of dim (10, 1)
y = df.iloc[:, 4]

df.head()



# Scale
from sklearn.preprocessing import StandardScaler
X_sca = StandardScaler()
X = X_sca.fit_transform(X)




from __future__ import division
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


kfold_cv = KFold(n_splits=10)
correct = 0
total = 0
for train_indices, test_indices in kfold_cv.split(X):
    X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], \
                                        y[train_indices], y[test_indices]
    clf = SVC(kernel='linear', random_state=0).fit(X_train, y_train)
    correct += accuracy_score(y_test, clf.predict(X_test))
    total += 1
print("Accuracy: {0:.2f}".format(correct/total))






from sklearn.svm import SVC #support vector classifier
clf = SVC(kernel='linear', random_state=0).fit(X_train, y_train)





# applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(clf, X_train, y_train, cv=10)
print accuracies
print accuracies.mean()
print accuracies.std()


# Leave one out cross validation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, 2:4]   # Using 1:2 as indices will give us np array of dim (10, 1)
y = df.iloc[:, 4]

df.head()



# Scale
from sklearn.preprocessing import StandardScaler
X_sca = StandardScaler()
X = X_sca.fit_transform(X)



from __future__ import division
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

loo_cv = LeaveOneOut()
correct = 0
total = 0
for train_indices, test_indices in loo_cv.split(X):
#     uncomment these lines to print splits
#     print("Train Indices: {}...".format(train_indices[:4]))
#     print("Test Indices: {}...".format(test_indices[:4]))
#     print("Training SVC model using this configuration")
    X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], \
                                        y[train_indices], y[test_indices]
    clf = SVC(kernel='linear', random_state=0).fit(X_train, y_train)
    correct += accuracy_score(y_test, clf.predict(X_test))
    total += 1
print("Accuracy: {0:.2f}".format(correct/total))



#Stratified KFold



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, 2:4]   # Using 1:2 as indices will give us np array of dim (10, 1)
y = df.iloc[:, 4]

df.head()




# Scale
from sklearn.preprocessing import StandardScaler
X_sca = StandardScaler()
X = X_sca.fit_transform(X)




from sklearn.model_selection import TimeSeriesSplit
import numpy as np

X = np.random.rand(10, 2)
y = np.random.rand(10)
print(X)
print(y)




tss = TimeSeriesSplit(n_splits=7)

for train_indices, test_indices in tss.split(X):
    print("Train indices: {0} Test indices: {1}".format(train_indices, test_indices))
