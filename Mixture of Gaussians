import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math
import scipy.stats as stats
import pylab as pl

iris = datasets.load_iris()
x = iris.data[:, 2:][0:140]
y = iris.target[0:140]
x_test = iris.data[:, 2:][141:150]
y_test = iris.target[141:150]

a1=[int(i) for i in np.where(y==0)[0]]
a2=[int(i) for i in np.where(y==1)[0]]
a3=[int(i) for i in np.where(y==2)[0]]
normal1 = stats.norm.pdf(x[a1,:1].ravel(), np.mean(x[a1,:1].ravel()), np.std(x[a1,:1].ravel()))
normal2 = stats.norm.pdf(x[a2,:1].ravel(), np.mean(x[a2,:1].ravel()), np.std(x[a2,:1].ravel()))
normal3 = stats.norm.pdf(x[a3,:1].ravel(), np.mean(x[a3,:1].ravel()), np.std(x[a3,:1].ravel()))

pl.figure(figsize=(10,6))
pl.plot(x[a1,:1],normal1,'o')
pl.plot(x[a2,:1],normal2,'o')
pl.plot(x[a3,:1],normal3,'o')
pl.hist(x[a1,:1].ravel(),normed=True) 
pl.hist(x[a2,:1].ravel(),normed=True) 
pl.hist(x[a3,:1].ravel(),normed=True) 
pl.title('MIXTURE OF GAUSSIANS CLASSIFICATION - IRIS DATASET - Petal Width')
pl.show()

gauss=[]
for i in range(0,len(x)):
    gauss.append(1/(np.sqrt(2*math.pi*np.std(x)**2))*math.exp(-.5*(x[i][1]-np.mean(x))/np.std(x)))

classe=[]
for i in range(0,len(gauss)):
    if gauss[i]<.26:
        classe.append(2)
    elif gauss[i]>.33:
        classe.append(0)
    else:
        classe.append(1)

print('Accuracy=',len(np.where(np.array(classe)-y==0)[0])/len(y))
