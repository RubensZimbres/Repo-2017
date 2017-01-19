import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l2, activity_l2
from scipy.interpolate import spline

reg=0.02
reg2=0.02

model=Sequential()
model.add(Dense(4, input_dim=2, init='uniform',W_regularizer=l2(reg), activity_regularizer=activity_l2(reg2)))
model.add(Dense(1, init='uniform',W_regularizer=l2(reg), activity_regularizer=activity_l2(reg2)))
sgd = SGD(lr=0.06, decay=2e-5, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['mean_absolute_error'])

aa=pd.read_csv('GameR.csv',sep=',',header=0)
df=aa[0:2100]
df
y=np.array(df[[2]])
y_train=[item for sublist in y for item in sublist]
y_train=np.array(y_train)
y_train.shape

x=np.array(df)
x1=x.T
x2=[x1[0],x1[1]]
x3=np.array(x2).T
X_train=x3
X_train.shape

y2=np.array(df[[2]])
y_train2=[item for sublist in y2 for item in sublist]
y_test=np.array(y_train2)

x0=np.array(df)
x10=x0.T
x20=[x10[0],x10[1]]
x30=np.array(x20).T
X_test=x30
x30

seed = 7
np.random.seed(seed)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
model.fit(X_train,y_train,batch_size=10)
res = model.predict(X_test)

print('\n','TEST Mean Absolute Error',abs(np.mean(abs(res)-abs(y_test)))/len(y_test),
'Error',abs(np.mean(abs(res)-abs(y_test))),'Accuracy=',1-abs(np.mean(abs(res)-abs(y_test))))

T=np.array(list(range(0,len(y_test[100:140]))))
xnew = np.linspace(T.min(),T.max(),300)
smooth = spline(T,res[100:140],xnew)

plt.figure(figsize=(10,4))
plt.plot(T,y_test[100:140],'o')
plt.plot(xnew,smooth,'-',color='r',linewidth=3)
plt.show()
