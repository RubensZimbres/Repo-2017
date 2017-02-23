import nltk
import numpy as np
from nltk import sent_tokenize, word_tokenize, pos_tag
import matplotlib.pyplot as plt
from pylab import *
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, pos_tag
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import re
 
consumer_key = '12345'
consumer_secret = '12345'
access_token = '123-12345'
access_secret = '12345'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

number_tweets=100
data=[]
for status in tweepy.Cursor(api.search,q="trump").items(number_tweets):
    try:
        URLless_string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', status.text)
        data.append(URLless_string)
    except:
        pass

lemmatizer = WordNetLemmatizer()

text=data

sentences = sent_tokenize(str(text))
sentences2=sentences
sentences2

tokens = word_tokenize(str(text))
tokens=[lemmatizer.lemmatize(tokens[i]) for i in range(0,len(tokens))]

len(tokens)

tagged_tokens = pos_tag(tokens)
tagged_tokens

## NOUNS
text2 = word_tokenize(str(text))
is_noun = lambda pos: pos[:2] == 'NN'
b=nltk.pos_tag(text2)
b
nouns = [word for (word, pos) in nltk.pos_tag(text2) if is_noun(pos)] 
nouns
V = set(nouns)
long_words1 = [w for w in tokens if 4<len(w) < 10]
sorted(long_words1)
fdist01 = nltk.FreqDist(long_words1)
fdist01
a1=fdist01.most_common(40)
a1

names0=[]
value0=[]
for i in range(0,len(a1)):
    names0.append(a1[i][0])
    value0.append(a1[i][1])
names0.reverse()
value0.reverse()
val = value0    # the bar lengths
pos = arange(len(a1))+.5    # the bar centers on the y axis
pos
val
plt.figure(figsize=(9,9))
barh(pos,val, align='center',alpha=0.7,color='blue')
yticks(pos, names0)
xlabel('Mentions')
title(['Nouns'])
grid(True)


def lexical_diversity(text):
    return len(set(text)) / len(text)
lexical_diversity(text)


vocab = set(text)
vocab_size = len(vocab)
vocab_size


V = set(text)
long_words = [w for w in tokens if 4<len(w) < 13]
sorted(long_words)

text2 = nltk.Text(word.lower() for word in long_words)
print(text2.similar('wound'))


fdist1 = nltk.FreqDist(long_words)
fdist1
a=fdist1.most_common(15)
a

names=[]
value=[]
for i in range(0,len(a)):
    names.append(a[i][0])
    value.append(a[i][1])
names.reverse()
value.reverse()
val = value    # the bar lengths
pos = arange(15)+.5    # the bar centers on the y axis
pos

plt.figure(figsize=(9,9))
barh(pos,val, align='center',alpha=0.7,color='rgbcmyk')
yticks(pos, names)
xlabel('Mentions')
grid(True)


list(nltk.bigrams(tokens))

list(nltk.trigrams(tokens))


sorted(w for w in set(tokens) if w.endswith('ing'))

[w.upper() for w in tokens]

for token in tokens:
    if token.islower():
        print(token, 'is a lowercase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')

########################################################

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import matplotlib.pyplot as plt
from gensim import corpora
from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

documents=[strip_punctuation(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',sentences2[i])) for i in range(0,len(sentences2))]

import pandas as pd 
df = pd.DataFrame(documents)
df.to_csv("Russian_Hacking.csv")


# remove common words and tokenize
stoplist = set('for a of the and to in is the he she on i will it its us as that at who be '.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
    for document in long_words]
texts
# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1
frequency

texts = [[token for token in text if frequency[token] > 1]
    for text in texts]
from pprint import pprint  # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester4.dict')

print(dictionary.token2id)


## VETOR DAS FRASES
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester4.mm', corpus)  # store to disk, for later use
print(corpus)

from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model


corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi.print_topics(2)

## COORDENADAS DOS TEXTOS
todas=[]
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    todas.append(doc)
todas

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('/tmp/deerwester4.dict')
corpus = corpora.MmCorpus('/tmp/deerwester4.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

np.array(corpus).shape

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)


p=[]
for i in range(0,len(documents)):
    doc1 = documents[i]
    vec_bow2 = dictionary.doc2bow(doc1.lower().split())
    vec_lsi2 = lsi[vec_bow2] # convert the query to LSI space
    p.append(vec_lsi2)
    
p
    
index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

index.save('/tmp/deerwester4.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester4.index')

#################

import gensim
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl

matrix1 = gensim.matutils.corpus2dense(p, num_terms=4)
matrix3=matrix1.T
matrix3

from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

X=norm(matrix3)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=1,n_iter=1500)
X_tsne = tsne.fit_transform(X)

### WORK HERE - COMO DESCOBRI QUE TINHA 3 CLUSTERS ???? SORT X_tsne
## DEFINE K-MEANS
plt.hist(X_tsne)

from sklearn.cluster import KMeans
model3=KMeans(n_clusters=4,random_state=0)
model3.fit(X_tsne)
cc=model3.predict(X_tsne)

## ALSO TRY COM X PARA VER QUE TOPICO SELECIONA

tokens2 = word_tokenize(str(sentences2))
tokens2

tokens2=[lemmatizer.lemmatize(tokens2[i]) for i in range(0,len(tokens2))]

long_words12 = [w for w in tokens2 if len(w) > 5]
sorted(long_words12)
fdist012 = nltk.FreqDist(long_words12)
a12=fdist012.most_common(5)

from matplotlib.colors import LinearSegmentedColormap

print('TOPIC 1\n')

print(a12,'\n')

for i in np.where(cc==2)[0][2:10]:
    print(i,sentences2[i])

n_classes=4
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),(0,0,0)] 
cm = LinearSegmentedColormap.from_list(
        cc, colors, N=4)
cor=[colors[cc[i]] for i in range(0,len(cc))]

h=[]
label=[]
fig = plt.figure(figsize=(8,4))
plt.title('NATURAL LANGUAGE PROCESSING\n\n'+'TOPIC MODELING at TWITTER HASHTAG: '+'#trump',fontweight="bold")
for i in range(0,4):
    label.append('Topic {}'.format([0,1,2,3][i]))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=cc,cmap=cm,marker='o',s=100)
    h1,=plt.plot(1,1,color=colors[i],linewidth=3)
    h.append(h1)
plt.legend(h,label,loc="upper left")
plt.show()
model = models.LdaModel(corpus, id2word=dictionary, num_topics=4)
model.print_topics(4)

### ACCUMULATE FEELINGS

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
sentim=sia()

cc0=[]
for sentence in documents:
    cc0.append(sentim.polarity_scores(sentence))

neu=[]
neg=[]
for sentence in documents:
        ss = sentim.polarity_scores(sentence)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='')
            neg.append(ss[k])
            neu.append(k)
        print()
        print('\n')

f=int(len(neg)/4)
sent0=np.array(neu).reshape(f,4)
sent=np.array(neg).reshape(f,4)
comp=sent.T[0]


positivos=len(np.where(np.array(comp)>0)[0])
neutros=len(np.where(np.array(comp)==0)[0])
negativos=len(np.where(np.array(comp)<0)[0])

from matplotlib import style
print(plt.style.available)

style.use("seaborn-darkgrid")

x = np.arange(0, len(comp), 1)
plt.figure(figsize=(9,6))
plt.plot(np.cumsum(comp),linewidth=3,color='blue')
plt.fill_between(x,np.cumsum(comp),0,where=np.cumsum(comp)<0,facecolor='red',alpha=.7)
plt.fill_between(x,np.cumsum(comp),0,where=np.cumsum(comp)>0,facecolor='lawngreen',alpha=.7)
plt.annotate('POSITIVE',(140,1.5),fontweight='bold')
plt.annotate('NEGATIVE',(140,-3),fontweight='bold')
plt.title('Natural Language Processing\n'+'\n'+'Mood in Twitter Streaming #trump Feb 23, 2017 - 5 Minutes',fontweight='bold')
plt.show()
