### DATA FROM KERAS.NLP.zip

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.interpolate import spline
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from bs4 import BeautifulSoup
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
        
sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS', 'test-pos.txt':'TEST_POS','test-neg2.txt':'TEST_NEG'}

sentences = LabeledLineSentence(sources)
sentences
model = Doc2Vec(min_count=1, window=5, size=10, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())

for epoch in range(10):
    model.train(sentences.sentences_perm())
    
model.save('./imdb.d2v')

model = Doc2Vec.load('./imdb.d2v')
model.most_similar('good')
model.syn0

sentences.to_array()
model.docvecs['TRAIN_POS_0']

train_arrays = numpy.zeros((14, 10))
train_labels = numpy.zeros(14)

for i in range(7):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[1 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[1 + i] = 0

test_arrays = numpy.zeros((14, 10))
test_labels = numpy.zeros(14)

for i in range(7):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs['TEST_POS_0']
    test_arrays[1 + i] = model.docvecs['TEST_NEG_0']
    test_labels[i] = 1
    test_labels[1 + i] = 0

sd=[]
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

epochs = 4000
learning_rate = 0.06
decay_rate = 5e-6
momentum = 0.9

model=Sequential()
model.add(Dense(12, input_dim=10, init='uniform'))
model.add(Dense(1, init='uniform'))
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['mean_absolute_error'])

def step_decay(losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.23:
        lrate=0.06
        momentum=0.3
        decay_rate=2e-6
        return lrate
    else:
        lrate=0.06
        return lrate

X_train=train_arrays
y_train=train_labels
X_test=test_arrays
y_test=test_labels        

history=LossHistory()
lrate=LearningRateScheduler(step_decay)

model.fit(X_train,y_train,nb_epoch=epochs,callbacks=[history,lrate],verbose=1)

DERIVATIVE_MIN=min(np.array(2*np.sqrt(history.losses)))

res = model.predict(X_test)

for i in range(0,len(res)):
    if res[i]>.5:
        res[i]=1
    else:
        res[i]=0

p=[]
for i in range(0,len(res)):
    if res[i]==0:
        p.append('Negative')
    else:
        p.append('Positive')

p2=[]
for i in range(0,len(res)):
    if y_test[i]==0:
        p2.append('Negative')
    else:
        p2.append('Positive')

filename = "test-pos.txt"
raw_text2 = open(filename).read()
html2=raw_text2
soup = BeautifulSoup(html2,"lxml")

###### SEMANTIC
texto=[]
for string in soup.stripped_strings:
    texto.append(repr(string))

texto

for script in soup(["script", "style"]):
    script.extract()    # rip it out

text2 = soup.get_text()
text2

sentences_pos = sent_tokenize(text2)
sentences_pos

filename = "test-neg2.txt"
raw_text2 = open(filename).read()
html=raw_text2
soup2 = BeautifulSoup(html,"lxml")
html

text2 = soup2.get_text()
text2
sentences_neg = sent_tokenize(text2)
sentences_neg

sentences=sentences_neg
sentences

acc=1-len(np.where([int(i) for i in res]-y_test==0))/len(y_test)

print('ACTUAL SENTIMENT:','\n',p2,'\n')
print('PREDICTED SENTIMENT:','\n',p,'\n')
print('Accuracy=',acc)

