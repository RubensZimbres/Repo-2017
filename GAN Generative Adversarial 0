import cv2
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
import os
from keras.optimizers import SGD

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_cascade)
img = cv2.imread('JenniferGroup.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

a=[]
for i in range(0,faces.shape[0]):
    a.append(gray[faces[i][1]:faces[i][1]+faces[i][3],faces[i][0]:faces[i][0]+faces[i][2]])
    
import matplotlib.pyplot as plt
plt.imshow(a[1],cmap=plt.get_cmap('gray'))

for k in range(0,faces.shape[0]):    
    print(a[k].shape)

import scipy.misc
shape=28

img1=[]
img2=[]
for i in range(0,faces.shape[0]):    
    scipy.misc.imsave('face{}.jpg'.format(i), a[i])
    img1.append(cv2.cvtColor(cv2.imread('face{}.jpg'.format(i)), cv2.COLOR_BGR2GRAY))
    img2.append(cv2.resize(img1[i], (shape, shape)))

img2=np.array(img2)

for k in range(0,faces.shape[0]):    
    print(img2[k].shape)

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

### START GENERATOR
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
recog1.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])
### END FIRST MODEL

### START DISCRIMINATOR
recog12=Sequential()
recog12.add(Reshape((28,28,1),input_shape=(784,)))
recog12.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
recog12.add(BatchNormalization(mode=2))
recog12.add(Activation('relu'))
recog12.add(UpSampling2D(size=(2, 2)))
recog12.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
recog12.add(BatchNormalization(mode=2))
recog12.add(Activation('relu'))
recog12.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
recog12.add(BatchNormalization(mode=2))
recog12.add(Activation('relu'))
recog12.add(MaxPooling2D(pool_size=(3,3)))
recog12.add(Convolution2D(4, 3, 3,init='glorot_uniform'))
recog12.add(BatchNormalization(mode=2))
recog12.add(Activation('relu'))
recog12.add(Reshape((28,28,1)))
recog12.add(Reshape((784,)))
recog12.add(Dense(784, activation='sigmoid',init='glorot_uniform'))

recog12.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])

recog12.fit(x_train[0].reshape((1,784)), x_train[0].reshape((1,784)),
                nb_epoch=1,
                batch_size=30,verbose=1)

################## GAN

def not_train(net, val):
    net.trainable = val
    for k in net.layers:
       k.trainable = val
not_train(recog1, False)

gan_input = Input(batch_shape=(1,784))

gan_level2 = recog12(recog1(gan_input))

GAN = Model(gan_input, gan_level2)
GAN.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mae'])

GAN.fit(x_train[0].reshape(1,784), x_train[0].reshape((1,784)), 
        batch_size=30, nb_epoch=1,verbose=1)

x_train_GAN=x_train[0].reshape(1,784)

a=GAN.predict(x_train[0].reshape(1,784),verbose=1)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train_GAN.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 2, 2)
plt.imshow(a.reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()


import pydot
import graphviz
import pydot_ng as pydot

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
SVG(model_to_dot(recog_right).create(prog='dot', format='svg'))
