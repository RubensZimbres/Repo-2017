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

learning_rate = 0.015
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

np.var(X_train2.T)

model = Sequential()
model.add(Dense(7, input_dim=4, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dense(7, init='glorot_uniform'))
model.add(Highway())
model.add(Dense(3, init='glorot_uniform'))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,metrics=['accuracy'])


model.fit(X_train2, Y_train, 
           batch_size = 30, nb_epoch = 1000, verbose = 1,validation_split=0.9)

res22 = model.predict_classes([X_train2,X_train2,X_train2],batch_size = 30)
acc22=((res22-data2.ix[:,4])==0).sum()/len(res22)
acc22






import pydot
import graphviz
import pydot_ng as pydot

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
SVG(model_to_dot(model22).create(prog='dot', format='svg'))
