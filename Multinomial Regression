import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

df=pd.read_csv('DadosTeseLogit3.csv',sep=',',header=0)
data=np.array(df)
x=data
x0=np.transpose(data)

#### COLOCAR TODAS AS VARIAVEIS AQUI

zz=np.corrcoef(data.T)[30]
wi=[i for i,x in enumerate(zz) if x>0.5]

m=[]
for i in wi:
    m.append([float(i) for i in x0[i]])
x=np.array(m)
x=np.transpose(x)

y = np.array(df[[30]])
y=[float(i) for i in y]

model=LogisticRegression(solver='newton-cg', max_iter=50000, multi_class='multinomial',n_jobs=2,penalty='l2')

model.fit(x,y)
model.score(x,y)

print('Coefficient: \n',model.coef_)
print('Intercept: \n',model.intercept_)

x_test=x[50:98]

pred=model.predict(x_test)

# ACCURACY
sum(x==0 for x in pred-y[50:98])/len(pred)

