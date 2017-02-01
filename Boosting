from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

xx=[]
for i in range(0,len(X)):
    xx.append((X[i] - np.min(X)) / (np.max(X) - np.min(X)))
X=xx

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


'''BOOSTING'''

from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(
init= None,
learning_rate= 0.6,
loss= 'deviance',
max_depth= 5,
max_features= pca3.shape[1],
max_leaf_nodes= 4,
min_samples_leaf= 1,
min_samples_split= 2,
min_weight_fraction_leaf= 0.0,
n_estimators= 1000,
presort= 'auto',
random_state= None,
subsample= 1.0,
verbose=1,
warm_start= False)

model.fit(pca3,y)
model.score(pca3,y)
pred3=model.predict(pca3)
t3=sum(x==0 for x in pred3-y)/len(pred3)
print('Accuracy PCA + Boosting:',t3)

model.fit(X_tsne,y)
model.score(X_tsne,y)
pred4=model.predict(X_tsne)
t4=sum(x==0 for x in pred4-y)/len(pred4)
print('\n','Accuracy t-SNE + Boosting:',t4)

conf = confusion_matrix(y, pred4)
plt.matshow(conf)
plt.title('CONFUSION MATRIX')
plt.colorbar()
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.show()
