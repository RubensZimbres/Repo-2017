import urllib
import json
import datetime
import csv
import urllib
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, pos_tag
import nltk
import numpy as np
import matplotlib.pyplot as plt
import codecs

reader = codecs.getreader("utf-8")

app_id = "12345" # YOUR APP ID HERE
app_secret = "12345" # YOUR PASSWORD HERE

access_token = app_id + "|" + app_secret

page_id = 'foxnews'

def feedFacebook(page_id, access_token,num_statuses):
    base = "https://graph.facebook.com/v2.8"
    node = "/" + page_id + "/feed"
    parameters = "/?fields=message,link,likes.limit(1).summary(true),comments.limit(1).summary(true),shares&limit=%s&access_token=%s" % (num_statuses, access_token) # changed    url = base + node +parameters
    url = base + node + parameters
    print(url)
    response = urllib.request.urlopen(url)
    data = json.load(reader(response))
    print(json.dumps(data, indent=4, sort_keys=True))
    b=json.dumps(data, indent=4, sort_keys=True)
    return data
a=feedFacebook(page_id, access_token,10)
a

a['data'][0]['shares']['count']

for i in range(0,10):
    print(a['data'][i]['message'])
    print('Shares:',a['data'][i]['shares']['count'])

txt=[]
share=[]
for i in range(0,10):
    txt.append(a['data'][i]['message'])
    share.append(a['data'][i]['shares']['count'])

txt

tokens = word_tokenize(str(a))
tokens

long_words1 = [w for w in tokens if 7<len(w)<9]
sorted(long_words1)
fdist01 = nltk.FreqDist(long_words1)
fdist01
a1=fdist01.most_common(20)
a1

names0=[]
value0=[]
for i in range(5,len(a1)):
    names0.append(a1[i][0])
    value0.append(a1[i][1])
names0.reverse()
value0.reverse()
val = value0    # the bar lengths
pos = np.arange(len(a1)-5)+.5    # the bar centers on the y axis
pos
val
plt.figure(figsize=(9,4))
plt.barh(pos,val, align='center',alpha=0.7,color='rgbcmyk')
plt.yticks(pos, names0)
plt.xlabel('Mentions')
plt.title('FACEBOOK ANALYSIS\n'+page_id)


sentences = sent_tokenize(str(txt))


##### LDA

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import matplotlib.pyplot as plt
from gensim import corpora
documents = sentences

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
    for document in documents]
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

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
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

matrix1 = gensim.matutils.corpus2dense(p, num_terms=2)
matrix3=matrix1.T
matrix3

from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

X=norm(matrix3)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=1,n_iter=1500)
X_tsne = tsne.fit_transform(X)

### WORK HERE - COMO DESCOBRI QUE TINHA 3 CLUSTERS ???? SORT X_tsne

from sklearn.cluster import KMeans
model3=KMeans(n_clusters=3,random_state=0)
model3.fit(X_tsne)
cc=model3.predict(X_tsne)

## ALSO TRY COM X PARA VER QUE TOPICO SELECIONA

tokens2 = word_tokenize(str(sentences[0:10]))
tokens2

long_words12 = [w for w in tokens2 if 7<len(w)<10]
sorted(long_words12)
fdist012 = nltk.FreqDist(long_words12)
a12=fdist012.most_common(50)

from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] 
cm = LinearSegmentedColormap.from_list(
        cc, colors, N=3)

print('TOPIC 1\n')

print(a12,'\n')

for i in np.where(cc==2)[0][2:10]:
    print(i,sentences[i])

fig = plt.figure(figsize=(8,4))
plt.title('NATURAL LANGUAGE PROCESSING\n\n'+'TOPIC MODELLING - LDA at page:',fontweight="bold")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cc,cmap=cm,marker='o', s=200)
plt.show()

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in long_words12]  

import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

plt.figure(figsize=(8,3))
plt.barh(pos,val, align='center',alpha=0.7,color='rgbcmyk')
plt.yticks(pos, names0)
plt.xlabel('Mentions')
plt.title('FACEBOOK ANALYSIS   '+page_id+'   Word Frequency',fontweight="bold")

fig = plt.figure(figsize=(8,3))
plt.title('FACEBOOK ANALYSIS\n\n'+'TOPIC MODELLING - Latent Dirichlet at Fox News page:',fontweight="bold")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cc,cmap=cm,marker='o', s=200)
plt.annotate('Topic 0',(-400,600),fontweight="bold")
plt.annotate('Topic 1',(-200,-400),fontweight="bold")
plt.annotate('Topic 2',(300,400),fontweight="bold")
plt.show()

print('TOPICS','\n')
ldamodel.print_topics(num_topics=3, num_words=3)
