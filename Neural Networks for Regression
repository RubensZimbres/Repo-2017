### LINEAR REGRESSION WITH SKLEARN LIBRARY

import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.formula.api as sm
from scipy.interpolate import spline
from keras.regularizers import l2, activity_l2
dataset=load_boston()

x_train,y_train=dataset.data,dataset.target
a=pd.DataFrame(x_train)
a.dropna()

c=[]
for i in range(0,x_train.shape[1]):
    c.append(np.corrcoef(x_train.T[i],y_train)[0][1])
d=np.where(abs(np.array(c))>.5)
d
e=[]
for i in d:
    e.append(x_train.T[i])
x=e[0].T

z=[]
for i in range(0,3):
    z.append(np.where(x.T[i]<2*np.std(x.T[i])))

z2=[]
z2.append(np.where(y_train.T<np.mean(y_train.T)+2*np.std(y_train.T)))
z2[0][0]

def norm(x):
    return (x-min(x))/(max(x)-min(x))
f=norm(x[z2[0][0]].T[0])
g=norm(1/x[z2[0][0]].T[1])
h=norm(1/x[z2[0][0]].T[2])
x2=np.array([f,g,h]).T
y2=np.array(norm(y_train[z2[0][0]]))

plt.scatter(f,y2)
plt.scatter(g,y2)
plt.scatter(h,y2)

linear=sm.OLS(y2,x2).fit()
linear.summary()
1-np.mean(abs(linear.predict(x2)-y2))

residuos=linear.predict(x2)-y2
plt.hist(residuos)

### HOMOSCEDASTICITY
plt.scatter(residuos,linear.predict(x2))

T=np.array(list(range(0,len(y2))))
plt.figure(figsize=(10,4))
plt.plot(T,y2,'-')
plt.plot(linear.predict(x2),'.',color='r')

linn=linear_model.LinearRegression()
linn.fit(x2,y2)
linn.score(x2,y2)
pred=linn.intercept_+linn.coef_[0]*x2.T[0]+linn.coef_[1]*x2.T[1]+linn.coef_[2]*x2.T[2]
1-np.mean(abs(y2-pred))


##### LINEAR REGRESSION USING KERAS


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.wrappers.scikit_learn import KerasRegressor
sd=[]
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))
        print('derivative of loss:',float(2*np.sqrt(np.array(history.losses[-1]))))

epochs = 100
learning_rate = 0.06
decay_rate = 5e-6
momentum = 0.9
reg=0.0002

model=Sequential()
model.add(Dense(4, input_dim=3, init='uniform',W_regularizer=l2(reg), activity_regularizer=activity_l2(reg)))
model.add(Dense(1, init='uniform',W_regularizer=l2(reg), activity_regularizer=activity_l2(reg)))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['mean_absolute_error'])

def step_decay(losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.28:
        lrate=0.01
        momentum=0.3
        decay_rate=2e-6
        return lrate
    else:
        lrate=learning_rate
        return lrate
ll=['1*1/(1+0.1*len(history.losses))']

y_train=y2

X_train=x2

history=LossHistory()
lrate=LearningRateScheduler(step_decay)

model.fit(X_train,y_train,batch_size=130,nb_epoch=epochs,callbacks=[history,lrate],verbose=2)

res = model.predict(X_train)

T=np.array(list(range(0,len(y_train))))
xnew = np.linspace(T.min(),T.max(),300)
smooth = spline(T,res,xnew)

plt.figure(figsize=(10,4))
plt.plot(T,y_train,'o')
plt.plot(xnew,smooth,'-',color='r',linewidth=3)
plt.show()


error=np.mean(abs([float(i) for i in res]-y_train))

print('Error',error,'Keras Accuracy=',1-error)
print('Linear Regression Accuracy:',1-np.mean(abs(linear.predict(x2)-y2))
)


#### LINEAR REGRESSION USING THEANO


import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors

y_train=y2

X_train=x2

x = T.dscalar()
fx = T.exp(T.sin(x**2))
f = theano.function(inputs=[x], outputs=[fx])
fp = T.grad(fx, wrt=x)
fprime = theano.function([x], fp)
x = T.dvector()
y = T.dscalar()
def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h
def grad_desc(cost, theta):
    alpha = 0.003 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))
theta1 = theano.shared(np.array(np.random.rand(4,4), dtype=theano.config.floatX)) 
theta2 = theano.shared(np.array(np.random.rand(5,1), dtype=theano.config.floatX))
hid1 = layer(x, theta1) #hidden layer
out1 = T.sum(layer(hid1, theta2)) #output layer
fc = (out1 - y)**2 #cost expression
cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),
        (theta2, grad_desc(fc, theta2))])
run_forward = theano.function(inputs=[x], outputs=out1)

inputs = X_train 
exp_y = y_train
cur_cost = 0
z=[]
for i in range(100):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k])
    if i % 1 == 0: 
        z.append(cur_cost,)
        print('Epoch:',i,'| Accuracy=',1-cur_cost)
np.array(z).T
z[:40]
plt.plot(z[:100],marker='o',linestyle='-',color='r')
plt.xlabel('Iterations')
plt.ylabel('ERROR')
plt.title('Neural Network Cost')
plt.show()

w0=[]
for i in range (0,len(y_train)):
    w0.append(run_forward(X_train[i]))

st=np.mean(abs([float(i) for i in w0]-y_train))
print('Theano Accuracy=',1-st)





### LINEAR REGRESSION USING THEANO+LASAGNE

import numpy as np
import pandas as pd
np.random.seed(1882) 
import lasagne
from lasagne import layers
from lasagne.layers import ReshapeLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import theano.tensor as T
import theano
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

y=y2

train=x2

target=y
target = np.array(y).astype(np.float32)
target.shape

train = np.array(x2).astype(np.float32)
test = np.array(x2).astype(np.float32)
target=target[20:98]
train=train[20:98]
test=test[0:19]
target2=target[0:19]
target=target.reshape(78,1)

def norm(x):
    return (x-min(x))/(max(x)-min(x))

target=norm(target)
aa=np.array([norm(train.T[0]),norm(train.T[1]),norm(train.T[2])])
train=aa.T

def build_mlp(input_var):
    l_in=lasagne.layers.InputLayer(shape=(78,3),input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=4,nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.DenseLayer(l_hid1, num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_out
    
input_var = T.matrix('inputs')
target_var = T.matrix('targets')

network = build_mlp(input_var)
prediction = lasagne.layers.get_output(network,deterministic=False)

loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()
layers = {build_mlp(input_var): 0.002}
l2_penalty = regularize_layer_params_weighted(layers, l2)
loss=loss-l2_penalty

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.03, momentum=0.9)

test_prediction = lasagne.layers.get_output(network)
test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
test_loss=test_loss-l2_penalty

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

pred=T.eq(T.argmax(test_prediction, axis=1), target_var)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_acc])
predict = theano.function([input_var, target_var], prediction, updates=updates)

pred1=theano.function([input_var, target_var],pred)

pars = theano.function([input_var,target_var], params,updates=updates)

z=[]
for i in range(0,1200):
    z.append(train_fn(train,target))
    print('Epoch:',i,'## Loss=',z[-1])

print('Accuracy:',1-float(z[-1]))

##### OUTPUT

Epoch: 1165 ## Loss= 0.013356180861592293
Epoch: 1166 ## Loss= 0.013340073637664318
Epoch: 1167 ## Loss= 0.01332398783415556
Epoch: 1168 ## Loss= 0.013307924382388592
Epoch: 1169 ## Loss= 0.013291879557073116
Epoch: 1170 ## Loss= 0.013275855220854282
Epoch: 1171 ## Loss= 0.01325985323637724
Epoch: 1172 ## Loss= 0.013243871740996838
Epoch: 1173 ## Loss= 0.013227911666035652
Epoch: 1174 ## Loss= 0.013211971148848534
Epoch: 1175 ## Loss= 0.013196051120758057
Epoch: 1176 ## Loss= 0.013180151581764221
Epoch: 1177 ## Loss= 0.013164273463189602
Epoch: 1178 ## Loss= 0.013148417696356773
Epoch: 1179 ## Loss= 0.013132580555975437
Epoch: 1180 ## Loss= 0.013116763904690742
Epoch: 1181 ## Loss= 0.013100968673825264
Epoch: 1182 ## Loss= 0.013085193932056427
Epoch: 1183 ## Loss= 0.013069439679384232
Epoch: 1184 ## Loss= 0.013053706847131252
Epoch: 1185 ## Loss= 0.013037994503974915
Epoch: 1186 ## Loss= 0.013022303581237793
Epoch: 1187 ## Loss= 0.013006633147597313
Epoch: 1188 ## Loss= 0.012990981340408325
Epoch: 1189 ## Loss= 0.012975352816283703
Epoch: 1190 ## Loss= 0.012959743849933147
Epoch: 1191 ## Loss= 0.012944155372679234
Epoch: 1192 ## Loss= 0.012928587384521961
Epoch: 1193 ## Loss= 0.012913043610751629
Epoch: 1194 ## Loss= 0.01289751660078764
Epoch: 1195 ## Loss= 0.01288201380521059
Epoch: 1196 ## Loss= 0.01286652684211731
Epoch: 1197 ## Loss= 0.012851063162088394
Epoch: 1198 ## Loss= 0.01283562183380127
Epoch: 1199 ## Loss= 0.012820199131965637
Accuracy: 0.9871798008680344
