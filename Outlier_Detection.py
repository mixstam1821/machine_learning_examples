# create a fake dataset 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs


X, y = make_blobs(100, centers=1)

# visualize the data

fig = plt.figure(figsize=(10, 8))
def plot_clusters(X, y):
    for label in np.unique(y):
        plt.scatter(X[y==label][:, 0], X[y==label][:, 1], label=label)
    plt.legend()
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 1")
    plt.title("Fake Data")
plot_clusters(X, y)


from sklearn.cluster import KMeans

clf = KMeans(n_clusters=1)
clf.fit(X)
centroids = clf.cluster_centers_



fig = plt.figure(figsize=(10, 8))
plt.scatter(centroids[0][0], centroids[0][1], label="Centroid", color="magenta")
plot_clusters(X, y)

distances = clf.transform(X)
fig = plt.figure(figsize=(10, 8))

plt.scatter(centroids[0][0], centroids[0][1], label="Centroid", color="magenta")

sorted_indices = list(reversed(np.argsort(distances.ravel())))[:5]
plot_clusters(X, y)

plt.scatter(X[sorted_indices][:, 0], X[sorted_indices][:, 1], color="red", edgecolor="green", 
            s=100, label="Extreme Values")
plt.legend()
plt.show()




X = np.delete(X, sorted_indices, axis=0) # important to mention axis=0
y = np.delete(y, sorted_indices, axis=0)





clf = KMeans(n_clusters=1)
clf.fit(X)

fig = plt.figure(figsize=(10, 8))
centroids = clf.cluster_centers_
plt.scatter(centroids[0][0], centroids[0][1], label="Centroid", color="magenta")
plot_clusters(X, y)



