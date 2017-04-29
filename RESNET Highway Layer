import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Merge,Lambda,GlobalAveragePooling1D,GlobalAveragePooling2D,UpSampling1D,UpSampling2D,Highway
from keras.optimizers import SGD
from scipy.interpolate import spline
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import datasets
import keras.backend as K
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

iris = datasets.load_iris()

learning_rate = 0.023
decay_rate = 5e-6
momentum = 0.9
epochs=50

scaler = MinMaxScaler(feature_range=(0, 1))

X_train2=scaler.fit_transform(iris.data[:,0:4])
Y_train = np.array(iris.target).reshape((150,1))

data=pd.DataFrame(np.concatenate((X_train2,Y_train),axis=1))
data2=shuffle(data)

X_train2=np.array(data2.ix[:,0:3])
Y_train=np.array(pd.get_dummies(data2.ix[:,4]))

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

model_left=Sequential()
model_left.add(Dense(5, input_dim=4, init='glorot_uniform'))
model_left.add(BatchNormalization(mode=2))
model_left.add(Activation('relu'))
model_left.add(Dense(5))
model_left.add(BatchNormalization(mode=2))
model_left.add(Activation('relu'))
model_left.add(Dense(3))
model_left.add(Activation('relu'))
model_left.add(Dense(4))

for i in range(0,6):
    print(i,model_left.layers[i].name)

model_right=Sequential()
model_right.add(Dense(4,input_shape=(4,)))
model_right.add(Highway())
model_right.add(BatchNormalization(mode=2))

model2=Sequential()
model2.add(Merge([model_left,model_right],mode = 'concat'))
model2.add(Activation('relu'))
model2.add(Reshape((8,)))
model2.add(Dense(5))
model2.add(BatchNormalization(mode=2))
model2.add(Activation('relu'))
model2.add(Dense(3))
model2.add(Activation('relu'))
model2.add(Dense(4))

for i in range(0,6):
    print(i,model2.layers[i].name)

model_right2=Sequential()
model_right2.add(Dense(4,input_shape=(4,)))
model_right2.add(Highway())
model_right2.add(BatchNormalization(mode=2))

model22=Sequential()
model22.add(Merge([model2,model_right2],mode = 'concat'))
model22.add(Activation('relu'))
model22.add(Reshape((8,)))
model22.add(Dense(5))
model22.add(BatchNormalization(mode=2))
model22.add(Activation('relu'))
model22.add(Dense(3))
model22.add(Activation('sigmoid'))

model22.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model22.summary()

model22.fit([X_train2,X_train2,X_train2], Y_train, 
           batch_size = 30, nb_epoch = 300, verbose = 1,validation_split=0.9)

res22 = model22.predict_classes([X_train2,X_train2,X_train2],batch_size = 30)
acc22=((res22-data2.ix[:,4])==0).sum()/len(res22)
acc22






import pydot
import graphviz
import pydot_ng as pydot

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
SVG(model_to_dot(model22).create(prog='dot', format='svg'))
