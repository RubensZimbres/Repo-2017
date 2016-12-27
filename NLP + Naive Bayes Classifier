from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
import numpy as np
from random import shuffle
from sklearn.linear_model import LogisticRegression

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
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
model
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

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
classifier.predict(train_arrays)

classifier.predict(test_arrays)
classifier.score(test_arrays, test_labels)

### DECISION TREES
from sklearn import tree
model4=tree.DecisionTreeClassifier(criterion='entropy',max_depth=20,max_features=4,max_leaf_nodes=3,min_samples_leaf=1,min_samples_split=1,splitter='best')
a=model4.fit(train_arrays, train_labels)
model4.score(train_arrays, train_labels)
pred=model4.predict(test_arrays)
sum(x==0 for x in pred-test_labels)/len(pred)

### NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
model4=GaussianNB()
a=model4.fit(train_arrays, train_labels)
model4.score(train_arrays, train_labels)
pred=model4.predict(test_arrays)
sum(x==0 for x in pred-test_labels)/len(pred)

res = pred

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
    if test_labels[i]==0:
        p2.append('Negative')
    else:
        p2.append('Positive')

from bs4 import BeautifulSoup
import nltk
import numpy as np
from nltk import sent_tokenize, word_tokenize, pos_tag

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
acc=1-len(np.where([int(i) for i in res]-test_labels==0))/len(test_labels)

sentences_pos
sentences_neg
print('\n','NAIVE BAYES CLASSIFIER','\n')
print('Accuracy=',acc,'\n')
print('ACTUAL SENTIMENT:','\n',p2,'\n')
print('PREDICTED SENTIMENT:','\n',p,'\n')

#### OUTPUT
Out[39]: 
['Next was a very nice and interesting movie.',
 'Matrix was awesome.',
 'Paycheck was excellent.',
 'Ronin was great.',
 'Pulp Fiction was very good.',
 'Lion King was cute.',
 'Madagascar was extremely interesting.\n']
 
 Out[40]: 
["I wish I could say that I loved it but I'd be lying.",
 'There were feeble attempts at humor and historical accuracy, but alas both failed to rise to the occasion.',
 'Swordfish was definitely not good.',
 'Rambo was an awful movie.',
 'Gamer was a bad movie.',
 'I did not like Jumper.',
 'Lake House was a bad one.\n']
 
NAIVE BAYES CLASSIFIER 

Accuracy= 0.9285714285714286 

ACTUAL SENTIMENT: 
 ['Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative'] 

PREDICTED SENTIMENT: 
 ['Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative'] 
