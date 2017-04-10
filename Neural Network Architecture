import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Merge
from keras.optimizers import SGD
from scipy.interpolate import spline
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler

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

X_train=scaler.fit_transform(iris.data)
X_train2=scaler.fit_transform(iris.data[:,2:4])
X_train_right = scaler.fit_transform(iris.data[:,2:4])
X_train_left = scaler.fit_transform(iris.data[:,0:2])
Y_train = np.array(pd.get_dummies(iris.target))

### SIMPLE NEURAL NETS
model0=Sequential()
model0.add(Dense(5, input_dim=4, init='glorot_uniform'))
model0.add(Dense(3))
model0.add(Activation('sigmoid'))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model0.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model0.fit(X_train, Y_train, 
           batch_size = 16, nb_epoch = 100, verbose = 1)

res0 = model0.predict_classes([X_train])
acc0=((res0-iris.target)==0).sum()/len(res0)

### SIMPLE NEURAL NETS WITH VARIABLE SELECTION
model2=Sequential()
model2.add(Dense(5, input_dim=2, init='glorot_uniform'))
model2.add(Dense(3))
model2.add(Activation('sigmoid'))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model2.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model2.fit(X_train2, Y_train, 
           batch_size = 16, nb_epoch = 100, verbose = 1)

res2 = model2.predict_classes([X_train2])
acc2=((res2-iris.target)==0).sum()/len(res2)

### PARALLEL NEURAL NETS
model_left=Sequential()
model_left.add(Dense(5, input_dim=2, init='glorot_uniform'))
model_left.add(Dropout(0.5))
model_left.add(Dense(3))

model_right=Sequential()
model_right.add(Dense(5, input_dim=2, init='glorot_uniform'))
model_right.add(Dense(3))

# mode concat, dot, sum
model3=Sequential()
model3.add(Merge([model_left,model_right],mode = 'sum'))
model3.add(Dense(3, init = 'glorot_uniform'))
model3.add(Activation('sigmoid'))
model3.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model3.fit([X_train_left, X_train_right], Y_train, 
           batch_size = 16, nb_epoch = 100, verbose = 1)

# 1
res = model3.predict_classes([X_train_left,X_train_right])
acc3=((res-iris.target)==0).sum()/len(res)

print('Accuracy Simple Neural Net (4 features):',acc0)
print('Accuracy Simple Neural Net + Variable Selection (2 features):',acc2)
print('Accuracy Parallel Neural Net (4 features):',acc3)

ss2=[acc0,acc2,acc3]
algos=[" Simple NN 4 features", "NN + Var Select 2 features", "Parallel NN 4 features"]
s2 = pd.Series(
    ss2,
    index = [" Simple NN 4 features", "NN + Variable Selection 2 features", "Parallel NN 4 features"]
)

plt.figure(figsize=(9,6))
plt.title("NEURAL NET ARCHITECTURE COMPARISON - IRIS CLASSIFICATION TASK"+'\n'+"100 EPOCHS - LR=0.07 - MOMENTUM=0.7")
plt.ylabel('MODEL SCORE')
plt.xlabel('MODEL TYPE')
ax = plt.gca()
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
for i in range(0,len(algos)):
    ax.text(i,ss2[i],round(ss2[i],4),ha='center', va='bottom')
my_colors = 'rgb'
s2.plot( kind='bar', color=my_colors)
plt.axhline(0, color='black')
plt.xticks(rotation=0)
plt.ylim(.85,1)
plt.show()
