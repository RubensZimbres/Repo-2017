from sklearn.cluster import KMeans
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import manifold
import itertools

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
kmeans.predict(X)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)

kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X_tsne)
kmeans2.labels_
kmeans2.predict(X_tsne)

fig = plt.figure(figsize=(10,4))
plt.subplot2grid((1,2), (0,0))
plt.title('ONLY K-MEANS: 2 CENTROIDS')
plt.scatter(kmeans.cluster_centers_[0],kmeans.cluster_centers_[1],color='red')
plt.subplot2grid((1,2), (0,1))
plt.title('K-MEANS WITH PREVIOUS t-SNE: 2 CENTROIDS')
plt.scatter(kmeans2.cluster_centers_[0],kmeans2.cluster_centers_[1],color='red')
plt.show()
