from gensim import corpora, models, similarities
from collections import defaultdict
import csv
from nltk import sent_tokenize, word_tokenize, pos_tag
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import gensim
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from string import punctuation
import logging
from pprint import pprint  # pretty-printer
text=["The purpose of the current study was to examine the painful elbow, and in particular enthesitis, in psoriatic arthritis PsA and rheumatoid arthritis RA using clinical examination, ultrasonography US and magnetic resonance imaging MRI. Patients with elbow pain 11 with PsA and 9 with RA were recruited. Clinical examination, US and MRI studies were performed on the same day. For enthesitis, the common extensor and flexor insertions and the triceps insertion were imaged 20 patients, giving a total of 60 sites with comparative data. Imaging was performed with the radiologists blinded to the diagnosis and clinical findings. US was used to assess inflammatory activity Power Doppler signal, oedema, tendon thickening and bursal swelling and damage erosions, cortical roughening and enthesophytes. MRI was used to assess inflammation fluid in paratenon, peri-entheseal soft-tissue oedema, entheseal enhancement with gadolinium, entheseal oedema and bone oedema and damage erosion, cortical roughening and enthesophyte. Complete scan data were not available for all patients as one patient could not tolerate the MRI examination. No significant differences in imaging scores were found between PsA an d RA. Analysis of damage scores revealed complete agreement between US and MRI data in 43/55 78% comparisons; in 10/55 18% cases the US data were abnormal but the MRI data normal; in 2/55 4% cases, the MRI data were abnormal and the US data normal. Analysis of the inflammation scores revealed complete agreement between US and MRI data in 33/55 60% comparisons; in 3/55 5% cases US data were abnormal but MRI data normal; in 19/55 35% cases the MRI data were abnormal and the US data normal. There was a poor relationship between assessments based on clinical examination and imaging studies. Readers could not accurately identify the disease from imaging findings."]

# kill all script and style elements
txt=sent_tokenize(str(text))

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

txt=[strip_punctuation(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',txt[i])) for i in range(0,len(txt))][0:110]

txt

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
plt.title('WIKIPEDIA ANALYSIS\n')

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
frequency = defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1
frequency

texts = [[token for token in text if frequency[token] > 1]
    for text in texts]
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester4.dict')

print(dictionary.token2id)


## VETOR DAS FRASES
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester4.mm', corpus)  # store to disk, for later use
print(corpus)

from gensim import corpora, models, similarities
import logging
import matplotlib.pyplot as plt
from gensim import corpora
from collections import defaultdict
from pprint import pprint  # pretty-printer
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model


corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

p=[]
for i in range(0,len(documents)):
    doc1 = documents[i]
    vec_bow2 = dictionary.doc2bow(doc1.lower().split())
    vec_lsi2 = lsi[vec_bow2] # convert the query to LSI space
    p.append(vec_lsi2)

p

matrix1 = gensim.matutils.corpus2dense(p, num_terms=2)
matrix3=matrix1.T
matrix3

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

X=norm(matrix3)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=50,verbose=1,n_iter=1500)
X_tsne = tsne.fit_transform(X)

model3=KMeans(n_clusters=2,random_state=0)
model3.fit(X)
cc=model3.predict(X)


fig = plt.figure(figsize=(9,4))
plt.title('TEXT RATIONALITY - Structure of Arguments (0,1)\n'+'WIKIPEDIA webpage Statistical Inference\n'+'Gensim Only 1 Topic',fontweight="bold")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cc,marker='.', s=2,label='Topic {0})'.format(cc))
for i in range(0,len(cc)):
    plt.annotate('{}'.format(cc[i]),(X_tsne[:, 0][i], X_tsne[:, 1][i]),fontweight="bold",color='rgbck'[cc[i]])
plt.show()

ag=lsi.show_topics(num_topics=1, num_words=100)

cd=[''.join([i for i in str(ag[x]) if not i.isdigit()]) for x in range(0,1)]

bc=[strip_punctuation(re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',str(cd[i]))) for i in range(0,1)]

txt2=[str(txt[i]).lower() for i in range(0,len(txt))]

tokens_lsi=[word_tokenize(bc[i]) for i in range(0,len(bc))]
tokens_txt=[word_tokenize(txt2[i]) for i in range(0,len(txt))]

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


'''TOPIC 0'''
a0=[]
for i in range(0,len(tokens_txt)):
    a0.append(np.sum([Counter(tokens_txt[i])[x] for x in tokens_lsi[0]]))
topic1=norm(a0)

threshold=0.3
[print(topic1[i],documents[i]) for i in np.where(topic1>threshold)[0]]
lsi.print_topics(1)

from scipy.cluster.hierarchy import dendrogram, linkage

P = linkage(matrix3, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

corr, coph_distances = cophenet(P, pdist(matrix3))
corr

plt.figure(figsize=(9,4))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('PARAGRAPH')
plt.ylabel('DISTANCE')
dendrogram(P,
    leaf_rotation=0.,
    leaf_font_size=12.,)
plt.show()

model00=[]
for i in range(0,len(sentences)):
    tokens = word_tokenize(str(sentences[i]))
    
    long_words = [w for w in tokens if len(w)>6]
    
    texts = [[word for word in document.lower().split() if word not in stoplist]
        for document in long_words]
    texts
    
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester4.mm', corpus)  # store to disk, for later use
    from gensim import corpora, models, similarities
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    
    
    corpus_tfidf = tfidf[corpus]
    
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    
    ## COORDENADAS DOS TEXTOS
    todas=[]
    for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
        todas.append(doc)
    
    from gensim import corpora, models, similarities
    dictionary = corpora.Dictionary.load('/tmp/deerwester4.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester4.mm') # comes from the first tutorial, "From strings to vectors"
    
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=1)
    model00.append(model.print_topics(1))

model00[0]

import re
sp=[]
for i in range(0,len(sentences)):
    word1 = " ".join(re.findall("[a-zA-Z]+", str(model00[i])))
    sp.append(word_tokenize(word1)[0])
    
P
x = np.array(range(len(sp)))

plt.figure(figsize=(8,4))
plt.title('Hierarchical Clustering Dendrogram\n'+'Obtained via LDA of Paragraphs in Text Summarization')
plt.xlabel('\nPARAGRAPHS KEYWORDS')
plt.ylabel('DISTANCE - SUMMARIZATION')
dendrogram(P,
    leaf_rotation=40.,
    leaf_font_size=10.,color_threshold=1,)
plt.xticks(3+x*10,sp)
plt.show()

plt.figure(figsize=(8,4))
plt.title('Hierarchical Clustering Dendrogram\n'+'Obtained via LDA of Paragraphs in Text Summarization')
plt.xlabel('\nPARAGRAPHS KEYWORDS')
plt.ylabel('DISTANCE - SUMMARIZATION')
dendrogram(P,
    leaf_rotation=0.,
    leaf_font_size=10.,color_threshold=12,)
plt.show()

#clusters
from scipy.cluster.hierarchy import fcluster
max_d = 1
clusters = fcluster(P, max_d, criterion='distance')
clusters

plt.figure(figsize=(9, 3.5))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=clusters, cmap='prism',s=400)
for i in range(0,len(sentences)):
    plt.annotate(sp[i],xy=(X_tsne[:,0][i],X_tsne[:,1][i]),xytext=(X_tsne[:,0][i]+8,X_tsne[:,1][i]+20))
plt.title('Cluster of topics via Dendrogram + LDA')
plt.show()

sa=[]
for i in range(1,max(clusters)+1):
    sa.append(np.where(clusters==i)[0])
sa
for x in range(len(sa)):[print('Cluster {0}:'.format(x),sentences[i]) for i in sa[x]]
