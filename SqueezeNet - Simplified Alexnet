'''adapted from https://arxiv.org/pdf/1602.07360.pdf'''

import keras
import numpy as np
from keras.layers import Input, Dense, Lambda, Merge
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import GlobalAveragePooling2D
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_CNN=x_train[1:2]

y_train1=pd.get_dummies(y_train[1:2]).T
y_train2=y_train1

epochs=3
learning_rate = 0.07
decay_rate = 5e-5
momentum = 0.6

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

input_shape=(28,28,1)

squeeze = Sequential()
squeeze.add(Lambda(lambda x: x ** 2,input_shape=(784,),output_shape=(1,784)))
squeeze.add(Reshape((28,28,1)))
squeeze.add(Convolution2D(2, 3,3,
                          border_mode='valid',
                        input_shape=input_shape))
squeeze.add(BatchNormalization(mode=2))
squeeze.add(ELU(alpha=1.0))
squeeze.add(MaxPooling2D(pool_size=(2,2)))
squeeze.add(Convolution2D(1, 1, 1,
                            init='glorot_uniform'))
squeeze.add(BatchNormalization(mode=2))
squeeze.add(ELU(alpha=1.0))

squeeze_left=squeeze
squeeze_left.add(Convolution2D(2, 3,3,
                          border_mode='valid',
                        input_shape=input_shape))
squeeze_left.add(ELU(alpha=1.0))

squeeze_right=squeeze
squeeze_right.add(Convolution2D(2, 3,3,
                          border_mode='valid',
                        input_shape=input_shape))
squeeze_right.add(ELU(alpha=1.0))

squeeze0 = Sequential()
squeeze0.add(Merge([squeeze_left,squeeze_right],mode = 'concat'))
squeeze0.add(Dropout(0.2))
squeeze0.add(GlobalAveragePooling2D((1, 28, 28, 1)))
squeeze0.add(Dense(1))
squeeze0.add(Activation('sigmoid'))
squeeze0.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['accuracy'])
squeeze0.summary()

squeeze0.fit(x_train_CNN,np.array(y_train2),
                nb_epoch=15,
                batch_size=30,verbose=1)

a1=squeeze0.predict_classes(x_train_CNN,verbose=1)
a1
