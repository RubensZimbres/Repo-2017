import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.interpolate import spline
from sklearn.svm import SVR
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


dataframe = pd.read_csv('Apple_Data_300.csv').ix[2000:2555,:]
dataframe.head()
autocorrelation_plot(dataframe.ix[:,4])
look_back=50

### AVALIAR V3 LINHAS
model00 = ARIMA(np.array(dataframe.ix[:,4]), dates=None,order=(5,2,2))
model11 = model00.fit(disp=1)
model11.summary()
model11.forecast()
resid9=model11.resid
np.mean(abs(resid9))/max(np.array(dataframe.ix[:,4]))

x3 = resid9
x3 = x3[numpy.logical_not(numpy.isnan(x3))]
dftest13 = adfuller(x3, autolag='AIC')
dfoutput1 = pd.Series(dftest13[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print('Dickey Fuller Test:\n',dfoutput1)

start=0
end=len(resid9)
lag=look_back
xx=np.array(resid9[start+lag:end])
yy=np.array(resid9[start:end-lag])
autocorrelation=np.corrcoef(xx,yy)
print('Autocorrelation of Residuals=',round(autocorrelation[0][1],3))

plt.plot(resid9)
plt.title('Residuals ARIMA')
plt.ylim(-50,50)
plt.show()

### FIX
plt.plot(resid9/np.array(dataframe.ix[2002:2555,4]))
plt.title('Residuals/Stock Value - ARIMA')
plt.ylim(-.2,.2)
plt.show()

print(pd.DataFrame(resid9).describe())

plt.hist(resid9)

density = gaussian_kde(resid9)
xs = np.linspace(-50,50,len(resid9))
density.covariance_factor = lambda : .25
density._compute_covariance()

plt.plot(xs,density(xs))
plt.show()

### DELETE OUTLIERS
delete=np.concatenate([np.where(resid9<np.mean(resid9)-2*np.std(resid9))[0],np.where(resid9>np.mean(resid9)+2*np.std(resid9))[0]])

train0=np.delete(np.array(dataframe.ix[:,4]),delete)
train=np.sqrt(train0)

rollmean = pd.rolling_mean(train, window=20)
rollstd = pd.rolling_std(train, window=20)

ts_log0 = np.log(train)
ts_log=pd.DataFrame(ts_log0).dropna()
decomposition = seasonal_decompose(np.array(ts_log).reshape(len(ts_log),),freq=100)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

z=np.where(seasonal==min(seasonal))[0]
period=z[2]-z[1]

look_back = period

plt.figure(figsize=(8,8))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend',color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality',color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals',color='black')
plt.legend(loc='upper left')
plt.tight_layout()

from statsmodels.tsa.stattools import adfuller
dftest = adfuller(train, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
dfoutput
'''Not Stationary'''

x = seasonal
x = x[numpy.logical_not(numpy.isnan(x))]
dftest1 = adfuller(x, autolag='AIC')
dfoutput1 = pd.Series(dftest1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print('Dickey Fuller Test:\n',dfoutput1)

train=np.sqrt(train0)
for i in range(0,3):
    modelP2= ARIMA(np.array(train)[0:-2], order=(2,1,0))
    model_fit2 = modelP2.fit(disp=-1,tol=1e-20,maxiter=20000)
    pred71 = model_fit2.forecast()[0]
    new=np.concatenate((train,pred71),axis=0)
    train=new

model_fit2.summary()

print('Precision=',round(float((pred71[-1]**2)/train0[-1]),3))
print('Error=',round(100*(1-float((pred71[-1]**2)/train0[-1])),3),'percent')
print('Real Stock Value',train0[-1])
print('Predicted Stock Value',pred71[-1]**2)

predicted=train**2
predicted_ok=predicted[-4:]
dataframe3 = pd.read_csv('Apple_Data_Comparison.csv')
real_data=np.array(dataframe3.ix[2554:2557,4])

plt.plot(predicted_ok,marker='o',linewidth=2,color='red')
plt.plot(real_data,marker='o',linewidth=2,color='blue')
plt.ylim(130,150)
plt.title('ARIMA PREDICTION')
plt.ylabel('Stock Value')
plt.xlabel('Future Predictions')
plt.show()

predicted_ok-real_data

plt.figure(figsize=(10,6))
line1,=plt.plot(train,color='blue',label='Time Series AAPL')
line2,=plt.plot(rollmean,color='red',label='Rolling Mean',linewidth=2)
line3,=plt.plot(rollstd,color='green',label='Standand Deviation',linewidth=2)
plt.legend([line1,line2,line3],loc='upper left')
plt.show()

pred = []
for i in range(period,len(train)-1):
    modelP= ARIMA(np.array(train)[0:i], order=(2,1,0))
    model_fit = modelP.fit(disp=0,tol=1e-20,transparams=True,trend='c')
    pred7 = model_fit.forecast()[0]
    pred.append(pred7)

print('Error=',1-float((pred[-1]**2)/train0[-1]))

plt.plot(np.array(train[period:]),color='blue')
plt.plot(np.array(pred).reshape(len(pred),),color='red')


############### SVMs

dataframe2 = pd.read_csv('Apple_Data_300_SVM.csv')[2000:2555]

look_back=20
train=np.sqrt(np.delete(np.array(dataframe.ix[2000:2555,4]),delete))

dataset0 = dataframe2.values
dataset1 = dataset0.astype('float32')

numpy.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(train)
train_size = int(len(dataset) * .99)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test))
	
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)


trainX, trainY = create_dataset(train, look_back)

# reshape input to be [samples, time steps, features]
trainY = trainY.reshape(len(trainY), 1)

svr_rbf = SVR(kernel='linear', C=1e3, gamma=0.002)
model = svr_rbf.fit(trainX,trainY.ravel())
model.get_params()
trainPredict = model.predict(trainX)

plt.plot(trainPredict,linewidth=2,color='red')
plt.plot(trainY,linewidth=2,color='blue')
plt.show()

print('Accuracy Train:',1-np.mean(abs(trainPredict-trainY)))
print('Difference Last:',float(trainPredict[-1]-trainY[-1]))
