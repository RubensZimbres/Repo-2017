import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=100)
svr_lin = SVR(kernel='linear', C=1e3)
svr_rbf2 = SVR(kernel='rbf', C=1e3, gamma=.1)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_rbf2.fit(X, y).predict(X)

lw = 3
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', label='Overfitted')
plt.plot(X, y_lin, color='red', lw=lw, label='Underfitted')
plt.plot(X, y_poly, color='green', lw=lw, label='Best model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

print('Error RBF ovefitted:', np.mean(abs(y_rbf-y)))
print('Error RBF right:', np.mean(abs(y_poly-y)))
print('Error Linear:', np.mean(abs(y_lin-y)))
