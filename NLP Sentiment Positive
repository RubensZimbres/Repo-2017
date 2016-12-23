import urllib
from bs4 import BeautifulSoup
import nltk
import numpy as np
from nltk import sent_tokenize, word_tokenize, pos_tag
import matplotlib.pyplot as plt
from pylab import *
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

url="http://www.infowars.com"
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html)

texto=[]
for string in soup.stripped_strings:
    texto.append(repr(string))

texto

text = soup.get_text()
text

chars_to_remove = ["\t","\n"]
sc = set(chars_to_remove)
text=''.join([c for c in text if c not in sc])
text

sentences = sent_tokenize(text)
sentences2=sentences
sentences2

tokens = word_tokenize(text)
tokens

from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
sentim=sia()

cc=[]
for sentence in sentences2:
    cc.append(sentim.polarity_scores(sentence))
len(cc)
len(sentences2)
cc[0]

neu=[]
neg=[]
for sentence in sentences2:
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

neutro=len(np.where(comp==0)[0])
positivo=len(np.where(comp>0)[0])
negativo=len(np.where(comp<0)[0])

data=[negativo,neutro,positivo]

pos = arange(3)
plt.figure(figsize=(7,4))
barh(pos,data, align='center',alpha=0.7,color='rgy')
yticks(pos, ['NEGATIVE','NEUTRAL','POSITIVE'])
xlabel('TONE OF SENTENCES')
title(soup.title.string)

####################### OUTPUT #########################

What's Ahead For Civil Liberty in a Trump Administration Monday: The Alex Jones Show.
compound: 0.5267, neg: 0.0, neu: 0.779, pos: 0.221, 


EPIC: Global Gov't In Complete CollapseSunday: The Alex Jones Show.
compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 


Democrats Dream of Censoring Alex JonesFriday: The Infowars Nightly News.
compound: 0.25, neg: 0.0, neu: 0.818, pos: 0.182, 


Fake News Fails To Report Truth About Trump AttacksFriday: The Alex Jones Show.
compound: -0.5574, neg: 0.324, neu: 0.549, pos: 0.126, 


Ford / Apple Coming Back To Us, Trump DeliversThursday: The Infowars Nightly News.
compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 


Obama Lame Duck: Crapping Regulations & Squawking About DemocracyThursday: The Alex Jones Show.
compound: -0.4215, neg: 0.203, neu: 0.797, pos: 0.0, 


Desperate Establishment Launches Massive Blacklist.
compound: -0.3182, neg: 0.365, neu: 0.635, pos: 0.0, 


Free Speech In PerilTuesday: The Infowars Nightly News: Is Trump Trolling Everybody?Tuesday: The Alex Jones Show.
compound: 0.5106, neg: 0.0, neu: 0.82, pos: 0.18, 


Left Attack Trump As He Trolls Dinosaur MediaMonday.
compound: -0.4767, neg: 0.307, neu: 0.693, pos: 0.0, 
