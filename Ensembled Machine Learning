'''ENSEMBLED MACHINE LEARNING WITH HYPERPARAMETER TUNING FOR CLASSIFICATION TASK 
IRIS DATASET - 3 CLASSES - 7 MACHINE LEARNING ALGORITHMS'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data[:, 2:][0:140]
y = iris.target[0:140]
x_test = iris.data[:, 2:][141:150]
y_test = iris.target[141:150]

'''NAIVE BAYES'''

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x,y)
nb=model.score(x,y)

pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

'''DECISION TREES'''

from sklearn import tree
model=tree.DecisionTreeClassifier(
class_weight= None,
criterion= 'entropy',
max_depth= 20,
max_features= x.shape[1],
max_leaf_nodes= 4,
min_samples_leaf= 1,
min_samples_split= 1,
min_weight_fraction_leaf= 0.0,
presort= False,
random_state= None,
splitter= 'best')

model.fit(x,y)
dt=model.score(x,y)
pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

'''BOOSTING'''

from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(
init= None,
learning_rate= 0.6,
loss= 'deviance',
max_depth= 5,
max_features= x.shape[1],
max_leaf_nodes= 4,
min_samples_leaf= 1,
min_samples_split= 1,
min_weight_fraction_leaf= 0.0,
n_estimators= 1000,
presort= 'auto',
random_state= None,
subsample= 1.0,
verbose=1,
warm_start= False)

model.fit(x,y)
boo=model.score(x,y)
pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

'''K NEAREST NEIGHBOR'''

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(
n_neighbors=3,
algorithm= 'auto',
leaf_size= 30,
metric= 'minkowski',
metric_params= None,
n_jobs= 1,
p= 2,
weights= 'uniform')

model.fit(x,y)
knn=model.score(x,y)
pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

'''MULTINOMIAL LOGISTIC REGRESSION'''

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(
C= 1.0,
class_weight= None,
dual= False,
fit_intercept= True,
intercept_scaling= 1,
max_iter= 50000,
multi_class= 'multinomial',
n_jobs= 2,
penalty= 'l2',
random_state= None,
solver= 'newton-cg',
tol= 0.0001,
verbose= 1,
warm_start= False)

model.fit(x,y)
log=model.score(x,y)
pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

'''SUPPORT VECTOR MACHINES'''

from sklearn import svm
model=svm.SVC(
C= 1.0,
cache_size= 10,
class_weight= None,
coef0= 0.0,
decision_function_shape= None,
degree= 3,
gamma= 100,
kernel= 'rbf',
max_iter= -1,
probability= True,
random_state= None,
shrinking= True,
tol= 0.001,
verbose=1)

model.fit(x,y)
svm_=model.score(x,y)
pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

'''RANDOM FORESTS'''

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(
bootstrap= True,
class_weight= None,
criterion= 'gini',
max_depth= 100,
max_features=x.shape[1],
max_leaf_nodes=100,
min_samples_leaf= 1,
min_samples_split= 1,
min_weight_fraction_leaf= 0,
n_estimators= 100,
n_jobs= 1,
oob_score= False,
random_state= None,
verbose= 1,
warm_start=True)
 
model.fit(x,y)
rand=model.score(x,y)
pred=model.predict(x_test)
sum(x==0 for x in pred-y_test)/len(pred)

ss2=[nb,dt,boo,knn,log,svm_,rand]
algos=["NaiveB", "DecisionTree", "Boost", "KNN","Logist","SVM","Rand Forest"]

best=[int(i) for i in np.where(ss2==max(ss2))[0]]
best2=[]
for i in best:
    best2.append([algos[i],round(ss2[i],4)])

s2 = pd.Series(
    ss2,
    index = ["NaiveB", "DecisionTree", "Boost", "KNN","Logist","SVM","Rand Forest"]
)
plt.figure(figsize=(9,6))
plt.title("ENSEMBLED MACHINE LEARNING - IRIS CLASSIFICATION TASK")
plt.ylabel('MODEL SCORE')
plt.xlabel('MODEL TYPE')
ax = plt.gca()
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
for i in range(0,len(algos)):
    ax.text(i,ss2[i],round(ss2[i],4),ha='center', va='bottom')
my_colors = 'rgbcmyk'
s2.plot( kind='bar', color=my_colors)
plt.axhline(0, color='black')
plt.xticks(rotation=0)
plt.ylim(.85,1)
plt.show()

print('BEST MODELS:',best2) 
