import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Merge,Lambda,GlobalAveragePooling1D,GlobalAveragePooling2D,UpSampling1D,UpSampling2D
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

iris = datasets.load_iris()

learning_rate = 0.028
decay_rate = 5e-6
momentum = 0.9
epochs=50

scaler = MinMaxScaler(feature_range=(0, 1))

X_train=scaler.fit_transform(iris.data[:,1:4])
X_train2=scaler.fit_transform(iris.data[:,0:4])

Y_train = np.array(pd.get_dummies(iris.target))

### SIMPLE NEURAL NETS WITH VARIABLE SELECTION
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

model_left=Sequential()
model_left.add(Dense(5, input_dim=4, init='glorot_uniform'))
model_left.add(Activation('relu'))
model_left.add(Dense(5))
model_left.add(Activation('relu'))
model_left.add(Dense(3))
model_left.add(Activation('sigmoid'))
model_left.add(Dense(4))

for i in range(0,6):
    print(i,model_left.layers[i].name)

model_right=Sequential()
part=5
model_left.layers[part].name
get_0_layer_output = K.function([model_left.layers[0].input, K.learning_phase()],[model_left.layers[part].output])

get_0_layer_output([X_train2, 0])[0][0]

pred=[np.argmax(get_0_layer_output([X_train2, 0])[0][i]) for i in range(0,len(X_train2))]

loss=iris.target-pred
loss=loss.astype('float32')

model_right.add(Lambda(lambda x: x-np.mean(loss),input_shape=(4,),output_shape=(4,)))

model2=Sequential()
model2.add(Merge([model_left,model_right],mode = 'concat'))
model2.add(Activation('relu'))
model2.add(Reshape((8,)))
model2.add(Dense(3))
model2.add(Activation('sigmoid'))

model2.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model2.summary()

model2.fit([X_train2,X_train2], Y_train, 
           batch_size = 30, nb_epoch = 1000, verbose = 1)

res2 = model2.predict_classes([X_train2,X_train2])
acc2=((res2-iris.target)==0).sum()/len(res2)
acc2

150/150 [==============================] - 0s     
Out[449]: 0.98666666666666669

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_536 (Dense)                (None, 5)             25                                           
____________________________________________________________________________________________________
activation_568 (Activation)      (None, 5)             0                                            
____________________________________________________________________________________________________
dense_537 (Dense)                (None, 5)             30                                           
____________________________________________________________________________________________________
activation_569 (Activation)      (None, 5)             0                                            
____________________________________________________________________________________________________
dense_538 (Dense)                (None, 3)             18                                           
____________________________________________________________________________________________________
activation_570 (Activation)      (None, 3)             0                                            
____________________________________________________________________________________________________
dense_539 (Dense)                (None, 4)             16                                           
____________________________________________________________________________________________________
lambda_150 (Lambda)              (None, 4)             0                                            
____________________________________________________________________________________________________
activation_573 (Activation)      (None, 8)             0           merge_104[0][0]                  
____________________________________________________________________________________________________
reshape_38 (Reshape)             (None, 8)             0           activation_573[0][0]             
____________________________________________________________________________________________________
dense_540 (Dense)                (None, 3)             27          reshape_38[0][0]                 
____________________________________________________________________________________________________
activation_574 (Activation)      (None, 3)             0           dense_540[0][0]                  
====================================================================================================
Total params: 116
____________________________________________________________________________________________________
