import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K
import keras.callbacks
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2, activity_l2
from scipy.interpolate import spline

epochs = 50
learning_rate = 0.01
decay_rate = 5e-6
momentum = 0.9
reg=0.0001
look_back = 23

dataframe = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
##### PERIODO DEFINE MELHOR A PREVISÃO
numpy.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
	
def my_init(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)

def step_decay(losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.15:
        lrate=0.01*1/(1+0.1*len(history.losses))
        momentum=0.2
        decay_rate=0.0
        return lrate
    else:
        lrate=0.01
        return lrate
sd=[]
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('learning rate:', step_decay(len(self.losses)))
        print('derivative of loss:', 2*np.sqrt((self.losses[-1])))

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX.shape
# reshape input to be [samples, time steps, features]
trainY = trainY.reshape(len(trainY), 1)
testY = testY.reshape(len(testY), 1)

model = Sequential()
model.add(Dense(4,input_dim=look_back,init=my_init))
model.add(Dense(1, W_regularizer=l2(reg), activity_regularizer=activity_l2(reg)))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False,)

model.compile(loss='mean_squared_error', optimizer=sgd)

history=LossHistory()
lrate=LearningRateScheduler(step_decay)

model.fit(trainX, trainY, nb_epoch=epochs, batch_size=1, verbose=2,callbacks=[history,lrate])

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict-trainY
sd[0]
c3=range(0,len(sd))

threshold=1.2
error0=np.mean(abs(trainPredict-trainY))
error=error0+error0*threshold
est_min=np.reshape([trainPredict[i]-error for i in range(0,len(trainPredict))],(92,))
est_max=np.reshape([trainPredict[i]+error for i in range(0,len(trainPredict))],(92,))

x = np.arange(0, len(trainPredict), 1)
color0='lawngreen'


plt.figure(figsize=(9,6))
plt.plot(est_min,color=color0,linewidth=1)
plt.plot(est_max,color=color0,linewidth=1)
plt.fill_between(x,est_min,est_max,facecolor=color0)
plt.plot(trainY,linewidth=2,color='b',label='TIME SERIES')
plt.title("Probabilistic Neural Networks\n"+"TIME SERIES PREDICTION\n"+'Error Margin: {}'.format(round(error,3)),fontweight='bold')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.show()

print('Accuracy Train:',1-np.mean(abs(trainPredict-trainY)))
print('Accuracy Test:',1-np.mean(abs(testPredict-testY)))
