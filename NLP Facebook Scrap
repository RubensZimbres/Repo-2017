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

app_id = "12345"
app_secret = "12345"

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
    return data
a=feedFacebook(page_id, access_token,100)

tokens = word_tokenize(str(a))
tokens

long_words1 = [w for w in tokens if 7<len(w)<9]
sorted(long_words1)
fdist01 = nltk.FreqDist(long_words1)
fdist01
a1=fdist01.most_common(40)
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
plt.figure(figsize=(9,9))
plt.barh(pos,val, align='center',alpha=0.7,color='rgbcmyk')
plt.yticks(pos, names0)
plt.xlabel('Mentions')
plt.title('FACEBOOK ANALYSIS\n'+page_id)
