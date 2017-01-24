from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

X=norm(X)
y=norm(y)

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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.interpolate import spline
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

y=np.array(pd.get_dummies(digits.target))
y=y[0:1500]
y_test=y[1501:1740]

### NEURAL NET FOR t-SNE
epochs = 20
learning_rate = 0.03
decay_rate = 5e-6
momentum = 0.9

model=Sequential()
model.add(Dense(12, input_dim=2, init='uniform'))
model.add(Dense(10, init='uniform'))
model.add(Activation('sigmoid'))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        
model.fit(X_tsne[0:1500],y,nb_epoch=epochs,verbose=2)
pred1=model.predict_classes(X_tsne[1501:1740])

### NEURAL NET FOR PCA

model2=Sequential()
model2.add(Dense(12, input_dim=2, init='uniform'))
model2.add(Dense(10, init='uniform'))
model2.add(Activation('sigmoid'))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model2.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        
model2.fit(pca3[0:1500],y,nb_epoch=epochs,verbose=2)
pred2=model.predict_classes(pca3[1501:1740])


from sklearn.cluster import KMeans
model3=KMeans(n_clusters=10,random_state=0)
model3.fit(X_tsne[0:1500],y[0:1500])
pred3=model3.predict(X_tsne[1501:1740])

model4=KMeans(n_clusters=10,random_state=0)
model4.fit(pca3[0:1500],y[0:1500])
pred4=model4.predict(pca3[1501:1740])

print('\n')
print('Accuracy Test Set t-SNE + Neural Network 20 epochs:',1-[int(i) for i in (pred1-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
print('\n')
print('Accuracy Test Set PCA + Neural Network 20 epochs',1-[int(i) for i in (pred2-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
print('\n')
print('Accuracy Test Set t-SNE + K-Means',1-[int(i) for i in (pred3-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
print('\n')
print('Accuracy Test Set PCA + K-Means',1-[int(i) for i in (pred4-digits.target[1501:1740])].count(0)/len(X_tsne[1501:1740]))
