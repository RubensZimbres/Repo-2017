import numpy as np
import pandas as pd
np.random.seed(1337) 
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
import matplotlib.pyplot as plt

df=pd.read_csv('DadosTeseLogit.csv',sep=',',header=0)
y=np.array(df[[30]])
y=[item for sublist in y for item in sublist]
x=np.array(df).T


x2=[]
for i in range (0,98):
    x2.append([x[18][i],x[19][i]])

target=y
target = np.array(y).astype(np.uint8)
train = np.array(x2).astype(np.float32)
test = np.array(x2).astype(np.float32)
target=target[20:98]
train=train[20:98]
test=test[0:19]
target2=target[0:19]

def build_mlp(input_var):
    l_in=lasagne.layers.InputLayer(shape=(None,2),input_var=input_var,W=theano.shared(np.random.normal(0, 0.01, (50, 100))))
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=4,nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.DenseLayer(l_hid1, num_units=2,nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_out
    
input_var = T.fmatrix('inputs')
target_var = T.ivector('targets')
network = build_mlp(input_var)
prediction = lasagne.layers.get_output(network,deterministic=True)

loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
layers = {build_mlp(input_var): 0.002}
l2_penalty = regularize_layer_params_weighted(layers, l2)
loss=loss-l2_penalty

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.07, momentum=0.9)

test_prediction = lasagne.layers.get_output(network)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
test_loss=test_loss-l2_penalty

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

pred=T.eq(T.argmax(test_prediction, axis=1), target_var)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_acc])
predict = theano.function([input_var, target_var], prediction, updates=updates)

pred1=theano.function([input_var, target_var],pred)

pars = theano.function([input_var,target_var], params,updates=updates)

z=[]
for i in range(0,100):
    z.append(train_fn(train,target))
    print('Epoch:',i,'## Loss=',z[-1])

b0=np.array(z[-1]).tolist()
b=np.array(val_fn(train,target)).tolist()

plt.plot(z[0:200],color='r',linewidth=2)
plt.title("LASAGNE TRAINING LOSS")
plt.show()
print('Train Accuracy:',1-np.sqrt(b0))
print('Test Accuracy:',float(np.array(b)))
print('              Weights below')
target2
pred1(test,target2)

def get_params(self, **kwargs):
    result = [self.W]
    if not kwargs.get('no_biases'):
        result += [self.b]
    if not kwargs.get('trainable_only'):
        result += [self.asdf]
    return result
    
params = build_mlp(input_var).get_params()
params
ss=params[0].get_value().T[0]

s = pd.Series(
    ss,
    index = ["Node 1", "Node 2", "Node 3", "Node 4"]
)

plt.title("Weights after Backpropagation Variable X1")
plt.ylabel('Weigth Value')
plt.xlabel('Hidden Layer Node')
ax = plt.gca()
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
my_colors = 'rgby'  
s.plot( kind='bar', color=my_colors)
plt.axhline(0, color='black')
plt.xticks(rotation=0)
plt.show()

ss2=params[0].get_value().T[1]

s2 = pd.Series(
    ss2,
    index = ["Node 1", "Node 2", "Node 3", "Node 4"]
)

plt.title("Weights after Backpropagation Variable X2")
plt.ylabel('Weigth Value')
plt.xlabel('Hidden Layer Node')
ax = plt.gca()
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
my_colors = 'rgby' 
s2.plot( kind='bar', color=my_colors)
plt.axhline(0, color='black')
plt.xticks(rotation=0)
plt.show()


print('Weights Hidden Nodes','\n',params[0].get_value(),'\n')
print('Weights Bias',params[1].get_value())

## OUTPUT

Weights Hidden Nodes 
 [[ 0.49984368  0.64220988]
 [ 0.13120215  0.19968873]
 [ 0.56122197  0.04298479]
 [-0.42349583 -0.19503986]] 
