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
from keras.layers import Dense, Dropout, Activation, Flatten
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
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train=x_train[0:4]

img2=x_train.reshape((4,28,28,1))



batch_size = 30
nb_classes = 10
img_rows, img_cols = shape, shape
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape=(shape,shape,1)

learning_rate = 0.023
decay_rate = 5e-5
momentum = 0.9
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

part=1
thre=1
recog = Sequential()
recog.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
recog.add(BatchNormalization(mode=1))
c=get_0_layer_output=K.function([recog.layers[0].input, 
                                 K.learning_phase()],[recog.layers[part].output]);c=get_0_layer_output([img2[0].reshape(1,shape,shape,1), 0])[0][0];c[c>.01*np.min(c)]=1
recog.add(BatchNormalization(mode=2))
recog.add(Activation('sigmoid'))
recog.add(Lambda(lambda x: x+thre*c,output_shape=(26,26,20)))
recog.add(BatchNormalization(mode=2))
recog.add(MaxPooling2D(pool_size=(3,3)))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(UpSampling2D(size=(3, 3)))
recog.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))
recog.add(UpSampling2D(size=(2, 2)))
recog.add(Convolution2D(1, 5, 5,init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))

recog.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])
recog.summary()

recog.fit(img2[1].reshape(1,shape,shape,1), img2[1].reshape(1,shape,shape,1),
                nb_epoch=100,
                batch_size=30,verbose=1)

a=recog.predict(img2[1].reshape(1,shape,shape,1))

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
plt.imshow(img2[1].reshape((28,28)))
plt.gray()
plt.title('Original')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 3, 2)
plt.imshow(a.reshape((shape,shape)),cmap=cm2)
plt.title('Attention')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 3, 3)
plt.imshow(img2[1].reshape((28,28)))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(1, 3, 3)
plt.imshow(a.reshape((shape,shape)),cmap=cm,alpha=0.35,interpolation='nearest')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('MASK')
plt.show()

print('threshold=',thre)
