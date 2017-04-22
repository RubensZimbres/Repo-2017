import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Merge
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.core import Reshape
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
bb=np.where(y_train==0)[0][0:10]
cc=np.where(y_train==1)[0][0:10]

x_train0=np.array([x_train[i] for i in bb])
x_train1=np.array([x_train[i] for i in cc])

x_train0=x_train0.reshape((10,28,28,1))
x_train1=x_train1.reshape((10,28,28,1))

x_train_parallel_left=np.array([x_train0[7]])
x_train_parallel_right=np.array([x_train1[3]])
x_train_CNN=np.array([x_train0[7]])


n = 10
plt.figure(figsize=(10, 2))
for i in range(0,n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train0[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

batch_size = 30
nb_classes = 10
img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape=(28,28,1)

epochs=80
learning_rate = 0.027
decay_rate = 5e-5
momentum = 0.6

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)


denoise_left = Sequential()
denoise_left.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
denoise_left.add(BatchNormalization(mode=2))
denoise_left.add(Activation('relu'))
denoise_left.add(MaxPooling2D(pool_size=(2,2)))
denoise_left.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
denoise_left.add(BatchNormalization(mode=2))
denoise_left.add(Activation('relu'))
denoise_left.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise_left.add(BatchNormalization(mode=2))
denoise_left.add(Activation('relu'))
denoise_left.add(UpSampling2D(size=(2, 2)))
denoise_left.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise_left.add(BatchNormalization(mode=2))
denoise_left.add(Activation('relu'))
denoise_left.add(UpSampling2D(size=(2, 2)))
denoise_left.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise_left.add(BatchNormalization(mode=2))
denoise_left.add(Activation('relu'))
denoise_left.add(Convolution2D(1, 3, 3,init='glorot_uniform'))
denoise_left.add(BatchNormalization(mode=2))
denoise_left.add(Activation('sigmoid'))

denoise_right = Sequential()
denoise_right.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
denoise_right.add(BatchNormalization(mode=2))
denoise_right.add(Activation('relu'))
denoise_right.add(MaxPooling2D(pool_size=(2,2)))
denoise_right.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
denoise_right.add(BatchNormalization(mode=2))
denoise_right.add(Activation('relu'))
denoise_right.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise_right.add(BatchNormalization(mode=2))
denoise_right.add(Activation('relu'))
denoise_right.add(UpSampling2D(size=(2, 2)))
denoise_right.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise_right.add(BatchNormalization(mode=2))
denoise_right.add(Activation('relu'))
denoise_right.add(UpSampling2D(size=(2, 2)))
denoise_right.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise_right.add(BatchNormalization(mode=2))
denoise_right.add(Activation('relu'))
denoise_right.add(Convolution2D(1, 3, 3,init='glorot_uniform'))
denoise_right.add(BatchNormalization(mode=2))
denoise_right.add(Activation('sigmoid'))

denoise0 = Sequential()
denoise0.add(Merge([denoise_left,denoise_right],mode = 'ave'))
denoise0.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['accuracy'])

denoise = Sequential()
denoise.add(Convolution2D(20, 3,3,
                          border_mode='valid',
                        input_shape=input_shape))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(MaxPooling2D(pool_size=(2,2)))
denoise.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(UpSampling2D(size=(2, 2)))
denoise.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(UpSampling2D(size=(2, 2)))
denoise.add(Convolution2D(8, 3, 3,init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(Convolution2D(1, 3, 3,init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('sigmoid'))
denoise.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['accuracy'])
denoise.summary()

denoise.fit(x_train_CNN, x_train_CNN,
                nb_epoch=epochs,
                batch_size=30,verbose=1)

a1=denoise.predict(x_train_CNN,verbose=1)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train_CNN.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(a1.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()


################## GAN

def not_train(net, val):
    net.trainable = val
    for k in net.layers:
       k.trainable = val
not_train(denoise0, False)

gan_input = Input(batch_shape=(1, 28,28,1))

gan_level2 = denoise(denoise0([gan_input,gan_input]))

GAN = Model(gan_input, gan_level2)
GAN.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])

GAN.fit(x_train_parallel_left, x_train_parallel_right, 
        batch_size=30, nb_epoch=epochs,verbose=1)

a=GAN.predict(x_train_parallel_right,verbose=1)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train_parallel_right.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(a.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
