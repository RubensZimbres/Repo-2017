from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
from sklearn.decomposition import FactorAnalysis

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

X=norm(X)
y=norm(y)

fa=decomposition.FactorAnalysis(n_components=2, tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method='randomized', iterated_power=3, random_state=0)
fa2=fa.fit_transform(X)

pca=decomposition.TruncatedSVD(n_components=2)
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
pca2=pca.fit(X)
pca3=pca2.fit_transform(X)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=1,n_iter=1500)
X_tsne = tsne.fit_transform(X)

fig = plt.figure(figsize=(10,5))
plt.subplot2grid((1,2), (0,0))
plt.title('PRINCIPAL COMPONENTS ANALYSIS')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target)
plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=2)
plt.title('t-SNE')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target)
plt.show()


## ORIGINAL DATA DIMENSIONS
print('ORIGINAL DATA DIMENSION:',np.array(X).shape)

## DIMENSIONS AFTER t-SNE
print('DIMENSIONS AFTER t-SNE',np.array(X_tsne).shape)

from sklearn.cluster import KMeans
model3=KMeans(n_clusters=10,random_state=0)
model3.fit(X_tsne[0:1500],y[0:1500])
pred3=model3.predict(X_tsne[1501:1740])

model4=KMeans(n_clusters=10,random_state=0)
model4.fit(pca3[0:1500],y[0:1500])
pred4=model4.predict(pca3[1501:1740])

model4.fit(fa2[0:1500],digits.target[0:1500])
pred7=model4.predict(fa2[1501:1740])

print('Accuracy Test Set t-SNE + K-Means',1-[int(i) for i in (pred3-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
print('\n')
print('Accuracy Test Set PCA + K-Means',1-[int(i) for i in (pred4-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
print('\n')
print('Accuracy Test Set Factor Analysis + K-Means',1-[int(i) for i in (pred7-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
