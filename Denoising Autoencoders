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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
from keras.optimizers import SGD
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train=x_train[1:2]
x_test=x_test[1:2]

noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_train=x_train.reshape((1,28,28,1))
x_test=x_test.reshape((1,28,28,1))
x_train_noisy = x_train_noisy.reshape((1,28,28,1))
x_test_noisy = x_test_noisy.reshape((1,28,28,1))


n = 1
plt.figure(figsize=(10, 2))
for i in range(0,n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

batch_size = 1
nb_classes = 10

img_rows, img_cols = 28, 28

nb_filters = 32

pool_size = (2, 2)

kernel_size = (3, 3)
input_shape=(28,28,1)

learning_rate = 0.07
decay_rate = 5e-5
momentum = 0.9


denoise = Sequential()
denoise.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(UpSampling2D(size=(2, 2)))
denoise.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(MaxPooling2D(pool_size=(3,3)))
denoise.add(Convolution2D(4, 3, 3,init='glorot_uniform'))
denoise.add(BatchNormalization(mode=2))
denoise.add(Activation('relu'))
denoise.add(Reshape((28,28,1)))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

denoise.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['accuracy'])
denoise.summary()

denoise.fit(x_train_noisy, x_train,
                nb_epoch=50,
                batch_size=30,verbose=1)
                
a=denoise.predict(x_train_noisy)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train_noisy.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(a.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
