### Convolutional Neural Network in Lasagne for MNIST

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm

df=pd.read_csv('mnist_test_2k.csv',sep=',',header=1)
xx=np.transpose(df)
x0=np.array(xx[:785])
y_train=x0[0].astype(np.uint8)

a=np.delete(x0,(0),axis=0)
X=a.T
X_train0=X.reshape(2053,1,28,28)
X_train=X_train0.astype(np.uint8)

df=pd.read_csv('mnist_train_100.csv',sep=',',header=1)
xx=np.transpose(df)
x=np.array(xx[:785])
y_test=x[0].astype(np.uint8)

a2=np.delete(x,(0),axis=0)
X1=a2.T
X_test0=X1.reshape(98,1,28,28)
X_test=X_test0.astype(np.uint8)

CNN=NeuralNet(
    layers=[('input',layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),],
    input_shape=(None,1,28, 28),
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    maxpool1_pool_size=(2, 2),    
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    maxpool2_pool_size=(2, 2),
    dropout1_p=0.5,    
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    dropout2_p=0.5,    
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,)

nn = CNN.fit(X_train, y_train)

prediction = CNN.predict(X_test)

visualize.plot_conv_weights(CNN.layers_['conv2d1'])
