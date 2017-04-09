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
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import os
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train=x_train[1:2]
x_test=x_test[1:2]
y_train=y_train[1:2]
y_test=y_test[1:2]

batch_size = 1
nb_classes = 10
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows,img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')[1:2]
X_test = X_test.astype('float32')[1:2]
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

generator = Sequential()
generator.add(Convolution2D(28, 1,1,
                        border_mode='valid',
                        input_shape=input_shape))
generator.add(Activation('relu'))
generator.add(Convolution2D(28, 1, 1,
                            init='glorot_uniform'))
generator.add(Activation('relu'))
generator.add(Convolution2D(1, 1, 1,init='glorot_uniform'))
generator.add(Activation('relu'))
generator.compile(loss='binary_crossentropy', optimizer='adam')

generator.summary()


############# DISCRIMINATOR

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows,img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')[1:2]
X_test = X_test.astype('float32')[1:2]
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[1:2]
Y_test = np_utils.to_categorical(y_test, nb_classes)[1:2]

discriminator = Sequential()
discriminator.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
discriminator.add(Activation('relu'))
discriminator.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
discriminator.add(Activation('relu'))
discriminator.add(MaxPooling2D(pool_size=pool_size))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(64))
discriminator.add(Activation('relu'))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(nb_classes))
discriminator.add(Activation('sigmoid'))

discriminator.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

discriminator.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
          verbose=1, validation_data=(X_test, Y_test))

score = discriminator.evaluate(X_test, Y_test, verbose=2)
print('Test score:', score[0])
print('Test accuracy:', score[1])


def not_train(net, val):
    net.trainable = val
    for k in net.layers:
       k.trainable = val
not_train(discriminator, False)

gan_input = Input(batch_shape=(1, 28,28,1))

generator.summary()
discriminator.summary()

gan_level2 = discriminator(generator(gan_input))

GAN = Model(gan_input, gan_level2)
GAN.compile(loss='categorical_crossentropy', optimizer='adam')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train=x_train[1:2]
x_test=x_test[1:2]
y_train=y_train[1:2]
y_test=y_test[1:2]

GAN.fit(x_train.reshape((1,28,28,1)), Y_train.reshape((1,10)), batch_size=batch_size, nb_epoch=2,
          verbose=1)

bb=GAN.predict(x_test.reshape((1,28,28,1)))
c=np.where(bb[0]==np.max(bb))[0]
Y_train[c]

plt.imshow(x_train[c].reshape((28,28)),cmap='Greys_r')

