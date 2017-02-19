## SENTIMENT ANALYSIS WITH 760 POSSIBLE FEELINGS ###

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

app_id = "1235"
app_secret = "12345"

access_token = app_id + "|" + app_secret

page_id = 'washingtonpost'

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
a=feedFacebook(page_id, access_token,100)
a

txt=[]
share=[]
for i in range(0,50):
    txt.append(a['data'][i]['message'])

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
from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
sentences=[strip_punctuation(sentences[i]) for i in range(0,len(sentences))]

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

texts
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
model3=KMeans(n_clusters=5,random_state=0)
model3.fit(X)
cc=model3.predict(X)

## ALSO TRY COM X PARA VER QUE TOPICO SELECIONA

tokens2 = word_tokenize(str(sentences[0:10]))
tokens2

## ADJUST HERE
long_words12 = [w for w in tokens2 if 5<len(w)<12]
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
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)

plt.figure(figsize=(8,3))
plt.barh(pos,val, align='center',alpha=0.7,color='rgbcmyk')
plt.yticks(pos, names0)
plt.xlabel('Mentions')
plt.title('FACEBOOK ANALYSIS   '+page_id+'   Word Frequency',fontweight="bold")

fig = plt.figure(figsize=(8,3))
plt.title('CONSUMER COMPLAINT AFTER COMPUTER PURCHASE at Costco\n\n'+'ANALYIS USING FACEBOOK API and Natural Language Processing\n\n'+'Arguments used (clusters) obtained via K-Means',fontweight="bold")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='',cmap=cm,marker='o', s=200)
ff=np.arange(10)
plt.show()
print('WEIGHTS OF ARGUMENTS:\n')
ldamodel.print_topics(num_topics=10, num_words=3)

anger=['assured bright buoyant cheerful cheering confident encouraged expectant happy high hopeful hoping idealistic keeping the faith merry positive promising rose-colored rosy sanguine sunny trusting utopian pleased appreciative contented happy satisfied charmed content amused entertained Annoyed Apathetic Bored Certain Cold Crabby Cranky Critical Cross Detached Displeased Frustrated Impatient Indifferent Irritated Peeved Rankled Medium (or Mood-State) Anger Affronted Aggravated Anger Antagonized Arrogant Bristlin Exasperated Incensed Indignant Inflamed Mad Offended Resentful Riled up Sarcastic Intense Anger Hatred Aggressive Appalled Belligerent Bitter Contemptuous Disgusted Furious Hateful Hostile Irate Livid Menacing Outraged Ranting Raving Seething Spiteful Vengeful Vicious Vindictive Violent Abashed Awkward Discomfited Flushed Flustered Hesitant Humble Reticent Self-conscious Speechless Withdrawn Shame Guilt Ashamed Chagrined Contrite Culpable Embarrassed Guilty Humbled Intimidated Penitent Regretful Remorseful Reproachful Rueful Sheepish Intense Sham Belittled Degraded Demeaned Disgraced Guilt-ridden Guilt-stricken Humiliated Mortified Ostracized Self-condemning Self-flagellating Shamefaced Stigmatized Fear Anxiety Alert Apprehensive Cautious Concerne Confused Curiou Disconcerted Disoriented Disquieted Doubtful Edgy Fidgety Hesitant Indecisive Insecure Instinctive Intuitiv Leery Pensive Shy Timid Uneasy Watchful FearAnxiety Afraid Alarmed Anxious Aversive Distrustful Fearful Jumpy Nervous Perturbed Rattled Shaky Startled Suspicious Unnerved Unsettled Wary Worried Intense Fear Panic Dread Horrified Panicked Paralyzed Petrified Phobic Shocked Terrorized JealousyEnvy Disbelieving Distrustful Insecure Protective Suspicious Vulnerable Medium JealousyEnvy Covetous Demanding Desirous Envious Jealous Threatened Intense Jealousy Avaricious Gluttonous Grasping Greedy Envy Persistently Jealous Possessive Resentful Happiness Amused Calm Encouraged Friendly Hopeful Inspired Jovial Open Peaceful Smiling Upbeat Medium Happiness Contentment Cheerful Contented Delighted Excited Fulfilled Glad Gleeful Gratified Happy Healthy Self-esteem Joyful Lively Merry Optimistic Playful Pleased Proud Rejuvenated Satisfied Intense Happiness Contentment Joy Awe-filled Blissful Ecstatic Egocentric Elated Enthralled Euphoric Exhilarated Giddy Jubilant Manic Overconfident Overjoyed Radiant Rapturous Self-aggrandized Thrilled Sadness Contemplative Disappointed Disconnected Distracted Grounded Listless Low Steady Regretful Wistful Medium Sadness Grief Depression Dejected Discouraged Dispirited Down Downtrodden Drained Forlorn Gloomy Grieving Heavy-hearted Melancholy Mournful Sad Sorrowful Weepy World-weary Intense Sadness Grief Depression Anguished Bereaved Bleak Depressed Despairing Despondent Grief-stricken Heartbroken Hopeless Inconsolable Morose Depression Suicidal Urges Apathetic Constantly Irritated Angry Enraged Depressed Discouraged Disinterested Dispirited Feeling Worthless Flat Helpless Humorless Impulsive Indifferent Isolated Lethargic Listless Melancholy Pessimistic Purposeless Withdrawn World-weary Integration Depression Suicidal Urges Crushed Desolate Despairing Desperate Drained Empty Fatalistic Hopeless Joyless Miserable Morbid Overwhelmed Passionless Pleasureless Sullen Intense Suicidal Urges Agonized Anguished Bleak Death-seeking Devastated Doomed Gutted Nihilistic Numbed Reckless Self-destructive Suicidal Tormented Tortured delighted happy sad acrimony animosity annoyance antagonism displeasure exasperation fury hatred impatience indignation ire irritation outrage passion rage resentment temper violence chagrin cholera conniption dander disapprobation distemper gall huff infuriation irascibility irritability miff peevishness petulance pique rankling soreness stew storm tantrum tiff umbrage vexation blow fit hissy humor ill temper mad slow burn bliss contentment delight elation enjoyment euphoria exhilaration glee joy jubilation laughter optimism peace pleasure prosperity well-being beatitude blessedness cheer cheerfulness content delectation delirium ecstasy enchantment exuberance felicity gaiety geniality gladness hilarity hopefulness joviality lightheartedness merriment mirth paradise playfulness rejoicing sanctity vivacity cheeriness good cheer good humor good spirits seventh heaven ache agony burn cramp discomfort fever illness injury irritation misery sickness soreness spasm strain tenderness torment trouble twinge wound affliction catch convulsion crick distress gripe hurt laceration malady pang paroxysm prick smarting sting stitch throb throe tingle torture anguish grief heartache heartbreak hopelessness melancholy misery mourning poignancy sorrow blahs bleakness bummer cheerlessness dejection despondency disconsolateness dispiritedness distress dolefulness dolor downer dysphoria forlornness funny funk gloominess letdown listlessness moodiness mopes mournfulness sorrowfulness tribulation woe blue devils blue funk broken heart dismals downcastness grieving heavy heart the blues the dumps balked crabbed cramped crimped defeated discontented discouraged disheartened embittered foiled irked stonewalled stymied fouled hung resentful through ungratified unsated unslaked the wall angst anxiety concern despair dismay doubt dread horror jitters panic scare suspicion terror unease uneasiness worry abhorrence agitation aversion awe consternation cowardice creeps discomposure disquietude distress faintheartedness foreboding fright funk misgiving nightmare phobia presentiment qualm reverence revulsion timidity trembling tremor trepidation bête noire cold feet cold sweat recreancy bothered clutched concerned distracted distressed disturbed frightened perturbed tense tormented upset afraid apprehensive beside oneself distraught fearful fretful ease nervous edge needles overwrought solicitous uneasy uptight worried cheerful contented delighted ecstatic elated glad joyful joyous jubilant lively merry overjoyed peaceful pleasant pleased thrilled upbeat blessed blest blissful blithe captivated chipper chirpy content convivial exultant flying high gay gleeful gratified intoxicated jolly laughing light looking good mirthful on cloud nine peppy perky playful sparkling sunny tickled tickled pink amusing comical entertaining humorous laughable lively priceless uproarious convivial exhilarated frolicsome gay gleeful gut-busting happy jocular jolly jovial joyful joyous merry mirthful noisy riot rollicking scream side-splitting witty comfortable contented fulfilled satisfied willing appeased gratified complacent happy pleased']



## VETOR DAS FRASES
tokens23 = word_tokenize(str(anger))
tokens23=tokens23[1:-2]
texts2 = [[str(i)] for i in tokens23[1:-2]]

dictionary2 = corpora.Dictionary(texts2)
dictionary2.save('/tmp/deerwester5.dict')

corpus1 = [dictionary2.doc2bow(text) for text in texts2]
corpora.MmCorpus.serialize('/tmp/deerwester5.mm', corpus1)  # store to disk, for later use
print(corpus1)

from gensim import corpora, models, similarities
tfidf2 = models.TfidfModel(corpus1) # step 1 -- initialize a model


corpus_tfidf2 = tfidf2[corpus1]
for doc in corpus_tfidf2:
    print(doc)

lsi2 = models.LsiModel(corpus_tfidf2, id2word=dictionary2, num_topics=2) # initialize an LSI transformation
corpus_lsi2 = lsi[corpus_tfidf2] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi2.print_topics(1)

## COORDENADAS DOS TEXTOS
todas2=[]
for doc in corpus_lsi2: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    todas2.append(doc)
todas2

import gensim
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl

matrix1 = gensim.matutils.corpus2dense(todas2, num_terms=2)
matrix31=matrix1.T
DATA2=matrix31[0:53]
matrix3
DATA2.shape[0]
from scipy.spatial.distance import cosine

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
sentim=sia()

cc=[]
for sentence in tokens23:
    cc.append(sentim.polarity_scores(sentence))

neu=[]
neg=[]
for sentence in tokens23:
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

aj=[]
for i in range(0,len(comp)):
    if comp[i]<0:
        aj.append('Negative')
    elif comp[i]>0:
        aj.append('Positive')
    else:
        aj.append('Neutral')


for k in range(0,len(sentences)):
    try:
        print('SENTENCE:',sentences[k],'SENTIMENT:',aj[np.where(np.array([round(cosine(matrix3[k],DATA2[i]),6) for i in range(0,DATA2.shape[0])])==min(np.array([round(cosine(matrix3[k],DATA2[i]),6) for i in range(0,DATA2.shape[0])])))[0][0]],
              tokens23[np.where(np.array([round(cosine(matrix3[k],DATA2[i]),6) for i in range(0,DATA2.shape[0])])==min(np.array([round(cosine(matrix3[k],DATA2[i]),6) for i in range(0,DATA2.shape[0])])))[0][0]],'\n')
    except:
        pass
