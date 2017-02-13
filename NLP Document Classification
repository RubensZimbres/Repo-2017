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
from string import punctuation
import logging
import matplotlib.pyplot as plt
from gensim import corpora
from collections import Counter

reader = codecs.getreader("utf-8")

app_id = "12345"
app_secret = "12345"

access_token = app_id + "|" + app_secret

page_id = 'foxnews'

def feedFacebook(page_id, access_token,num_statuses):
    base = "https://graph.facebook.com/v2.8"
    node = "/" + page_id + "/feed"
    parameters = "/?fields=message,link,likes.limit(1).summary(true),comments.limit(1).summary(true),shares&limit=%s&access_token=%s" % (num_statuses, access_token)
    url = base + node + parameters
    print(url)
    response = urllib.request.urlopen(url)
    data = json.load(reader(response))
    print(json.dumps(data, indent=4, sort_keys=True))
    return data
a=feedFacebook(page_id, access_token,100)

for k in range(0,100):
    print(a['data'][k]['message'])

txt=[]
share=[]
for i in range(0,100):
    txt.append(a['data'][i]['message'])

txt

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
import re

txt=[strip_punctuation(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',txt[i])) for i in range(0,len(txt))]


tokens = word_tokenize(str(txt))
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

txt
sentences = txt


##### LDA

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi.print_topics(5)

ag=lsi.show_topics(num_topics=5, num_words=10)

cd=[''.join([i for i in str(ag[x]) if not i.isdigit()]) for x in range(0,4)]

bc=[strip_punctuation(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',str(cd[i]))) for i in range(0,4)]

txt2=[str(txt[i]).lower() for i in range(0,len(txt))]

tokens_lsi=[word_tokenize(bc[i]) for i in range(0,len(bc))]
tokens_txt=[word_tokenize(txt2[i]) for i in range(0,len(txt))]

for i in range(0,len(tokens_txt)):
    wordcounts = Counter(tokens_txt[i])
    print(wordcounts['donald'])


def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


'''TOPIC 0'''
a0=[]
for i in range(0,len(tokens_txt)):
    a0.append(np.sum([Counter(tokens_txt[i])[x] for x in tokens_lsi[0]]))
topic1=norm(a0)

'''TOPIC 1'''
a1=[]
for i in range(0,len(tokens_txt)):
    a1.append(np.sum([Counter(tokens_txt[i])[x] for x in tokens_lsi[1]]))
topic2=norm(a1)

'''TOPIC 2'''
a2=[]
for i in range(0,len(tokens_txt)):
    a2.append(np.sum([Counter(tokens_txt[i])[x] for x in tokens_lsi[2]]))
topic3=norm(a2)

threshold=0.5

[print(i,documents[i],'|| Match={}'.format(topic1[i]),'\n') for i in np.where(topic1>threshold)[0]]

[print(i,documents[i],'|| Match={}'.format(topic2[i]),'\n') for i in np.where(topic2>threshold)[0]]

[print(i,documents[i],'|| Match={}'.format(topic3[i]),'\n') for i in np.where(topic3>threshold)[0]]
