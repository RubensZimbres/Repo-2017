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
from keras.layers import Convolution2D, MaxPooling2D,Highway
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

zero=np.where(y_train==0)

x_train_orig=x_train[zero][0:20]
x_train=x_train_noisy[zero][0:20]

shape=28
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

learning_rate = 0.07
decay_rate = 5e-5
momentum = 0.9
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=True)

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

part=8
thre=1

recog=Sequential()
recog.add(Dense(64,activation='relu',input_shape=(784,),init='glorot_uniform'))

recog_left=recog
recog_left.add(Dense(64,input_shape=(64,),activation='relu'))

recog_right=recog
recog_right.add(Dense(64,input_shape=(64,),activation='relu'))
recog_right.add(Lambda(lambda x: x + K.exp(x / 2) * K.random_normal(shape=(1, 64), mean=0.,
                              std=epsilon_std), output_shape=(64,)))
recog_right.add(Highway())
recog_right.add(Activation('sigmoid'))

recog1=Sequential()
recog1.add(Merge([recog_left,recog_right],mode = 'ave'))
recog1.add(Dense(784))
recog1.add(Activation('relu'))

#### GATE***
recog11=Sequential()
layer=Dense(2,init='glorot_uniform',input_shape=(784,))
layer.trainable=False
recog11.add(layer)
layer2=Dense(784, activation='sigmoid',init='glorot_uniform')
layer2.trainable=True
recog11.add(layer2)
recog11.layers[0].W.set_value(np.ones((784,2)).astype(np.float32))

recog11.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])

recog11.get_weights()[0].shape

gan_input = Input(batch_shape=(1,784))

gan_level2 = recog11(recog1(gan_input))

GAN = Model(gan_input, gan_level2)
GAN.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mae'])

GAN.fit(x_train_orig[0].reshape(1,784), x_train_orig[0].reshape((1,784)), 
        batch_size=30, nb_epoch=100,verbose=1)

### UNIQUE BLUEPRINT
a=GAN.predict(x_train_orig[0].reshape(1,784),verbose=1)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train_orig[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(a.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

