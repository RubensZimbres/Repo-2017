import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Merge
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.layers.core import Reshape
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Highway
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import os
from keras.optimizers import SGD

img = cv2.imread('Brain_CT.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import matplotlib.pyplot as plt
plt.imshow(gray,cmap=plt.get_cmap('gray'))

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))    
    
x_train0=cv2.resize(gray, (60,60), interpolation = cv2.INTER_AREA)
x_train01=norm(x_train0)
x_train=x_train01.reshape((1,60,60,1))

shape2=54
x_train20=cv2.resize(gray, (shape2,shape2), interpolation = cv2.INTER_AREA)
x_train21=norm(x_train20)
x_train2=x_train21.reshape((1,shape2,shape2,1))

plt.imshow(x_train.reshape((60,60)),cmap=plt.get_cmap('gray'))


shape=60
batch_size = 30
nb_classes = 10
img_rows, img_cols = shape, shape
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape=(shape,shape,1)

reg=0.001
learning_rate = 0.013
decay_rate = 5e-5
momentum = 0.9

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=True)
shape2

recog0 = Sequential()
recog0.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
recog0.add(BatchNormalization(mode=2))

recog=recog0
recog.add(Activation('relu'))
recog.add(MaxPooling2D(pool_size=(2,2)))
recog.add(UpSampling2D(size=(2, 2)))
recog.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
recog.add(BatchNormalization(mode=2))
recog.add(Activation('relu'))

for i in range(0,2):
    print(i,recog0.layers[i].name)

recog_res=recog0
part=1
recog0.layers[part].name
get_0_layer_output = K.function([recog0.layers[0].input, K.learning_phase()],[recog0.layers[part].output])

get_0_layer_output([x_train, 0])[0][0]

pred=[np.argmax(get_0_layer_output([x_train, 0])[0][i]) for i in range(0,len(x_train))]

loss=x_train-pred
loss=loss.astype('float32')

recog_res.add(Lambda(lambda x: x,input_shape=(56,56,20),output_shape=(56,56,20)))

recog2=Sequential()
recog2.add(Merge([recog,recog_res],mode='ave'))
recog2.add(Activation('relu'))
recog2.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
recog2.add(BatchNormalization(mode=2))
recog2.add(Activation('relu'))
recog2.add(Convolution2D(1, 1, 1,init='glorot_uniform'))

recog2.add(Reshape((shape2,shape2,1)))
## SOFTMAX P/ RGB
recog2.add(Activation('relu'))

recog2.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])
recog2.summary()

x_train3=x_train2.reshape((1,shape2,shape2,1))

recog2.fit(x_train,x_train3,
                nb_epoch=25,
                batch_size=30,verbose=1)


filename = "U-Net_Cells_MinusPool_MinusUp_Brain_Ave_CopyLayer.hdf5"
recog2.load_weights(filename)
recog2.compile(loss='mean_squared_error', optimizer='adam')


a=norm(recog2.predict(x_train))

plt.hist(a.reshape((54*54,)),bins=20)

b=a.reshape((54*54,))
b[b>.5]=1
b[b<.25]=0

from matplotlib.colors import LinearSegmentedColormap
n_classes=4
colors = [(0, 0, 0), (1, 0, 0), (1,1 ,0),(1,1,1)] 
cm = LinearSegmentedColormap.from_list(
        a, colors, N=4)

plt.figure(figsize=(10,10))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train.reshape((60,60)),cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('ORIGINAL\n'+'BRAIN\n'+'TOMOGRAPHY')

ax = plt.subplot(1, 2, 2)
plt.imshow(b.reshape((shape2,shape2)),cmap=cm)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('UNSUPERVISED LABELS\n'+'1 - Brain Stem\n'+'2 - Bone\n'+'3 - Brain')
plt.annotate('1',(25,20),color='y',fontsize=20,fontweight='bold')
plt.annotate('2',(25,5),color='b',fontsize=20,fontweight='bold')
plt.annotate('3',(15,20),color='b',fontsize=20,fontweight='bold')
plt.show()
