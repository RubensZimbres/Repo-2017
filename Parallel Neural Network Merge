import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Merge
from keras.optimizers import SGD
from scipy.interpolate import spline
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from sklearn import datasets

iris = datasets.load_iris()


sd=[]
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.23:
        lrate=0.06
        momentum=0.3
        decay_rate=2e-6
        return lrate
    else:
        lrate=0.06
        return lrate

learning_rate = 0.07
decay_rate = 5e-6
momentum = 0.7

scaler = MinMaxScaler(feature_range=(0, 1))

X_train_right = scaler.fit_transform(iris.data[:,2:4])
X_train_left = scaler.fit_transform(iris.data[:,0:2])
Y_train = np.array(pd.get_dummies(iris.target))

model_left=Sequential()
model_left.add(Dense(3, input_dim=2, init='uniform'))
model_left.add(Dense(3))

model_right=Sequential()
model_right.add(Dense(3, input_dim=2, init='uniform'))
model_right.add(Dense(3))

# mode concat, dot, sum
model3=Sequential()
model3.add(Merge([model_left,model_right],mode = 'sum'))
model3.add(Dense(3, init = 'uniform'))
model3.add(Dense(3))
model3.add(Activation('sigmoid'))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model3.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model3.fit([X_train_left, X_train_right], Y_train, 
           batch_size = 16, nb_epoch = 300, verbose = 1)

# 1
res = model3.predict_classes([X_train_left,X_train_right])
print('Accuracy:',((res-iris.target)==0).sum()/len(res))
