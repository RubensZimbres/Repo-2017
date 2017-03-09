import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn import datasets

iris = datasets.load_iris()
X_train = iris.data[:,2:4]
y_train = iris.target

### COVARIANVE TYPE = full, spherical,diag,tied
mix = mixture.GMM(n_components=3, covariance_type='spherical')
mix.fit(X_train)
print(mix.means_)

cc=mix.predict(X_train)

from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] 
cm = LinearSegmentedColormap.from_list(
        cc, colors, N=3)

colors2 = [(1, 0, 0), (0, 0, 1), (0, 1, 0)] 
cm2 = LinearSegmentedColormap.from_list(
        y_train, colors2, N=3)

cc2=y_train
fig = plt.figure(figsize=(10,5))
plt.subplot2grid((1,2), (0,0))
plt.title('CLASSES')
plt.scatter(X_train[:,0], X_train[:,1], c=cc,cmap=cm, alpha=0.8)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=2)
plt.title('PREDICTED CLASSES MIXTURE MODEL')
plt.scatter(X_train[:,0], X_train[:,1], c=cc2,cmap=cm2, alpha=0.8)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
