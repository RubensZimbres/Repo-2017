import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.core import Reshape
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

zero=np.where(y_train==0)

x_train=x_train[zero][0:20]

batch_size = 30
nb_classes = 10
img_rows, img_cols = shape, shape
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape=(shape,shape,1)
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epsilon_std = 1.0

learning_rate = 0.028
decay_rate = 5e-5
momentum = 0.9
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

part=8
thre=1

# START VAE
recog=Sequential()
recog.add(Dense(64,activation='relu',input_shape=(784,),init='glorot_uniform'))
get_0_layer_output=K.function([recog.layers[0].input, 
                                 K.learning_phase()],[recog.layers[0].output])
c=get_0_layer_output([x_train[0].reshape((1,784)), 0])[0][0]

recog_left=recog
recog_right.add(Lambda(lambda x: x + np.mean(c), output_shape=(64,)))

recog_right=recog
recog_right.add(Lambda(lambda x: x + K.exp(x / 2) * K.random_normal(shape=(1, 64), mean=0.,
                              std=epsilon_std), output_shape=(64,)))

recog1=Sequential()
recog1.add(Merge([recog_left,recog_right],mode = 'ave'))
recog1.add(Dense(64, activation='relu',init='glorot_uniform'))
recog1.add(Dense(784, activation='relu',init='glorot_uniform'))
### END FIRST MODEL VAE

### START DAE
recog1.add(Reshape((28,28,1)))
recog1.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
recog1.add(BatchNormalization(mode=2))
recog1.add(Activation('relu'))
recog1.add(UpSampling2D(size=(2, 2)))
recog1.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
recog1.add(BatchNormalization(mode=2))
recog1.add(Activation('relu'))
recog1.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
recog1.add(BatchNormalization(mode=2))
recog1.add(Activation('relu'))
recog1.add(MaxPooling2D(pool_size=(3,3)))
recog1.add(Convolution2D(4, 3, 3,init='glorot_uniform'))
recog1.add(BatchNormalization(mode=2))
recog1.add(Activation('relu'))
recog1.add(Reshape((28,28,1)))
recog1.add(Reshape((784,)))
recog1.add(Dense(784, activation='sigmoid',init='glorot_uniform'))

recog1.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])

### VANISHING GRADIENT w/ SIGMOID TANH
recog1.fit(x_train[0].reshape((1,784)), x_train[0].reshape((1,784)),
                nb_epoch=150,
                batch_size=30,verbose=1)

n = 4
plt.figure(figsize=(10, 2))
for i in range(0,n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

pred0=7
a=recog1.predict(x_train[pred0].reshape((1,784)))

from matplotlib.colors import LinearSegmentedColormap
n_classes=2
colors = [(0, 0, 0), (0, 1, 0), (0, 0, 1),(1,1,0)] 
cm = LinearSegmentedColormap.from_list(
        a, colors, N=2)

colors2 = [(0, 0, 0), (0, 1, 0), (0, 0, 1),(1,0,0)] 
cm2 = LinearSegmentedColormap.from_list(
        a, colors2, N=2)

plt.figure(figsize=(10,10))
ax = plt.subplot(1, 3, 1)
plt.imshow(x_train[pred0].reshape((28,28)))
plt.gray()
plt.title('Original')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 3, 2)
plt.imshow(a.reshape((shape,shape)),cmap=cm2)
plt.title('Prediction')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 3, 3)
plt.imshow(x_train[pred0].reshape((28,28)))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 3, 3)
plt.imshow(a.reshape((shape,shape)),cmap=cm,alpha=0.35,interpolation='nearest')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('MASK')
plt.show()

import pydot
import graphviz
import pydot_ng as pydot

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
SVG(model_to_dot(recog_right).create(prog='dot', format='svg'))
