import pandas as pd
import numpy as np
from sklearn import manifold
from matplotlib import pyplot as plt
from sklearn.svm import SVR

np.random.seed(222)
X = np.sort(5 * np.random.rand(60, 1), axis=0)
y = np.sin(X).ravel()
y[::4] += 2 * (0.5 - np.random.rand(15))

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(X, y).predict(X)

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='data')
plt.plot(X, y_rbf, color='red', lw=3, label='Radial Basis Function model')
plt.xlabel('X')
plt.ylabel('Target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

print('Performance:',1-np.mean(abs(y_rbf-y)))
