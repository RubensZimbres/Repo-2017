### USE RESET TO START A BRAND NEW ENVIRONMENT EACH RUN (Variables will be cleared)

%reset

### MADE WITH KERAS 1.1.0

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import theano
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint
from keras import backend as K
import time
from sklearn.datasets import load_boston
import sys

### OUT OF LOOP
reward=0
### Number of initial hidden layers
hid=6
### Derivative value before adjusting learning rate
deriv=.23
### Threshold to stop the reinforcement learning
threshold=.924
### Initial random accuracy
acc=0.7
pp=[0,0]

dataset=load_boston()
x_train,y_train=dataset.data,dataset.target
a=pd.DataFrame(x_train)
a.dropna()

### Feature selection
c=[]
for i in range(0,x_train.shape[1]):
    c.append(np.corrcoef(x_train.T[i],y_train)[0][1])
d=np.where(abs(np.array(c))>.5)
d
e=[]
for i in d:
    e.append(x_train.T[i])
x=e[0].T

## Remove outliers
z2=[]
z2.append(np.where(y_train.T<np.mean(y_train.T)+2*np.std(y_train.T)))
z2[0][0]

def norm(x):
    return (x-min(x))/(max(x)-min(x))
f=norm(x[z2[0][0]].T[0])
g=norm(1/x[z2[0][0]].T[1])
h=norm(1/x[z2[0][0]].T[2])
X_train=np.array([f[0:370],g[0:370],h[0:370]]).T
y20=np.array(norm(y_train[z2[0][0]]))
y_train=y20[0:370]
X_test=np.array([f[371:475],g[371:475],h[371:475]]).T
y_test=y20[371:475]

### Loss history, Learning Rate and Derivative of loss
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('learning rate:', step_decay(len(self.losses)))
        print('derivative of loss:', 2*np.sqrt((self.losses[-1])))

### Random Initial weights
def my_init(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)

### Scheduled Learning Rate
def step_decay(losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<deriv:
        lrate=0.001
        momentum=0.07
        decay_rate=0.0
        return lrate
    else:
        lrate=learning_rate
        return lrate

###### LOOP based on improvement of accuracy (Reward)
while acc-pp[-2]>0.0001:    
    sd=[]
    if reward==0:
        def base_model(nodes):
            model=Sequential()
            model.add(Dense(nodes, input_dim=3, init=init0))
            model.add(Activation('sigmoid'))
            model.add(Dense(1, init='uniform'))
            model.add(Activation('sigmoid'))
            model.add(Dense(1, init='uniform'))
            model.add(Activation('sigmoid'))
            sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
            model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['mean_absolute_error'])
            return model
        hidden_layers=2            

    else:
        def base_model(nodes):
            model=Sequential()
            model.add(Dense(nodes, input_dim=3, init=init0))
            model.add(Dense(1, init=init0))
            model.add(Activation('sigmoid'))
            sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
            model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['mean_absolute_error'])
            return model
        hidden_layers=1            

    history=LossHistory()
    lrate=LearningRateScheduler(step_decay)

##### NEURAL NETWORK INITIAL SETTINGS

## If configuration is OK, train for more epochs to converge

    if reward>.87:
        epochs = 800
    else:
        epochs=100
    if reward==0:
        learning_rate = 0.1
    else:
        learning_rate=0.04
    decay_rate = 5e-6
    momentum = 0.9
    batch=50
    if reward==0:
        hidden_nodes=hid
        hidden_nodes=hidden_nodes-1
        init0='uniform'
    else: 
        hidden_nodes=hidden_nodes-1
        if hidden_nodes<5:
            hidden_nodes=4
            init0=my_init
    model=base_model(hidden_nodes)
    time.sleep(6)
    model.fit(X_train, y_train,nb_epoch=epochs,batch_size=batch,callbacks=[history,lrate],verbose=2)
    d=model.evaluate(X_train, y_train, batch_size=batch)
    reward=1-d[-1]
    y_pred=model.predict(X_test, batch_size=batch, verbose=0)
    acc=1-np.mean(abs(np.array([float(i) for i in y_pred])-y_test))
    pp.append(acc)
    sys.stdout.flush()
    for i in range(1):
        time.sleep(2)
        print('\n',flush=True)
        print('\n','Number of Hidden Layers:',hidden_layers,flush=True)        
        print('\n','Number of Hidden Nodes:',hidden_nodes,flush=True)        
        print('\n','Accuracy on Training:',reward,flush=True)
        print('\n','Improvement:',acc-pp[-2],flush=True)
        print('\n','Accuracy on test set:',acc,flush=True)
        print('\n','New configuration on its way', end='')
        sys.stdout.flush()
        for i in range(7):
            time.sleep(1)
            print('.', end='', flush=True)  
model.summary()

### OUTPUT

learning rate: 0.001
derivative of loss: 0.220770262037
0s - loss: 0.0122 - mean_absolute_error: 0.0825
Epoch 299/300
learning rate: 0.001
derivative of loss: 0.220748450555
0s - loss: 0.0122 - mean_absolute_error: 0.0825
Epoch 300/300
learning rate: 0.001
derivative of loss: 0.22072300209
0s - loss: 0.0122 - mean_absolute_error: 0.0825
300/370 [=======================>......] - ETA: 0s

 Number of Hidden Layers: 1

 Number of Hidden Nodes: 4

 Accuracy on Training: 0.917522253116

 Improvement: -0.00210045750898

 Accuracy on test set: 0.879107174837

 New configuration on its way.......
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_431 (Dense)                (None, 4)             16          dense_input_212[0][0]            
____________________________________________________________________________________________________
dense_432 (Dense)                (None, 1)             5           dense_431[0][0]                  
____________________________________________________________________________________________________
activation_116 (Activation)      (None, 1)             0           dense_432[0][0]                  
====================================================================================================
Total params: 21
____________________________________________________________________________________________________
