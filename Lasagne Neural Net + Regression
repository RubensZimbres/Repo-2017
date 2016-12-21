import numpy as np
import pandas as pd
np.random.seed(1883)
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

df=pd.read_csv('DadosTeseLogit.csv',sep=',',header=0)
y=np.array(df[[29]])
y=[item for sublist in y for item in sublist]
x=np.array(df).T
x2=[]
for i in range (0,98):
    x2.append([x[18][i],x[30][i]])

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
aa=np.array([norm(train.T[0]),norm(train.T[1])])
train=aa.T

def build_mlp(input_var):
    l_in=lasagne.layers.InputLayer(shape=(78,2),input_var=input_var)
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

#### OUTPUT

Epoch: 1169 ## Loss= 0.006011609453707933
Epoch: 1170 ## Loss= 0.00600651279091835
Epoch: 1171 ## Loss= 0.006001438479870558
Epoch: 1172 ## Loss= 0.005996381398290396
Epoch: 1173 ## Loss= 0.005991342477500439
Epoch: 1174 ## Loss= 0.005986322648823261
Epoch: 1175 ## Loss= 0.005981317721307278
Epoch: 1176 ## Loss= 0.005976332351565361
Epoch: 1177 ## Loss= 0.005971361882984638
Epoch: 1178 ## Loss= 0.005966410040855408
Epoch: 1179 ## Loss= 0.005961474496871233
Epoch: 1180 ## Loss= 0.005956560373306274
Epoch: 1181 ## Loss= 0.005951664410531521
Epoch: 1182 ## Loss= 0.005946781020611525
Epoch: 1183 ## Loss= 0.005941916722804308
Epoch: 1184 ## Loss= 0.005937070120126009
Epoch: 1185 ## Loss= 0.005932238418608904
Epoch: 1186 ## Loss= 0.00592742208391428
Epoch: 1187 ## Loss= 0.005922625306993723
Epoch: 1188 ## Loss= 0.005917845293879509
Epoch: 1189 ## Loss= 0.005913081578910351
Epoch: 1190 ## Loss= 0.005908333230763674
Epoch: 1191 ## Loss= 0.005903602112084627
Epoch: 1192 ## Loss= 0.005898888222873211
Epoch: 1193 ## Loss= 0.00589419063180685
Epoch: 1194 ## Loss= 0.005889509338885546
Epoch: 1195 ## Loss= 0.005884844344109297
Epoch: 1196 ## Loss= 0.005880196578800678
Epoch: 1197 ## Loss= 0.005875561852008104
Epoch: 1198 ## Loss= 0.005870942026376724
Epoch: 1199 ## Loss= 0.005866343155503273
Accuracy: 0.9941336568444967
