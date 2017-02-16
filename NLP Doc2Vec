import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import matplotlib.pyplot as plt
from gensim import corpora
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

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
dictionary.save('/tmp/deerwester.dict')

print(dictionary.token2id)

# VETOR DA FRASE
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

## VETOR DAS FRASES
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)

from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

### COORDENADA DA FRASE
frase=tfidf[new_vec]
print(frase)

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi.print_topics(1)

## COORDENADAS DOS TEXTOS
todas=[]
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    todas.append(doc)
todas

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

p=[]
for i in range(0,len(documents)):
    doc1 = documents[i]
    vec_bow2 = dictionary.doc2bow(doc1.lower().split())
    vec_lsi2 = lsi[vec_bow2] # convert the query to LSI space
    p.append(vec_lsi2)
    
p
    
index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples

 # print sorted (document number, similarity score) 2-tuples
print('Human computer interaction\n')
for i in range(0,len(sims)):
    print('Similarity:', sims[i],documents[i])

#################

import gensim
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl

matrix1 = gensim.matutils.corpus2dense(p, num_terms=2)
matrix3=matrix1.T
matrix3[0]
ss=[]
for i in range(0,9):
    ss.append(np.insert(matrix3[i],0,[0,0]))
matrix4=ss
matrix4

matrix2 = gensim.matutils.corpus2dense([vec_lsi], num_terms=2)
matrix2=np.insert(matrix2,0,[0,0])
matrix2

DATA=np.insert(matrix4,0,matrix2)
DATA=DATA.reshape(10,4)
DATA

#### AVALIAR DAQUI
names=np.array(documents)
names=np.insert(names,0,new_doc)
new_doc
cmap = plt.cm.jet

cNorm  = colors.Normalize(vmin=np.min(DATA[:,3])+.2, vmax=np.max(DATA[:,3]))

scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
len(DATA[:,1])

plt.subplots()
plt.figure(figsize=(9,6))
plt.scatter(matrix1[0],matrix1[1],s=40)
plt.scatter(matrix2[2],matrix2[3],color='r',s=80)
for idx in range(0,len(DATA[:,1])):
    colorVal = scalarMap.to_rgba(DATA[idx,3])
    plt.arrow(DATA[idx,0],  #x1
              DATA[idx,1],  # y1
              DATA[idx,2], # x2 - x1
              DATA[idx,3], # y2 - y1
              color=colorVal,head_width=0.002, head_length=0.001)
for i,names in enumerate (names):
    plt.annotate(names, (DATA[i][2],DATA[i][3]),va='top')
plt.title("PHRASE SIMILARITY - DOC2VEC with GENSIM library")
plt.xlim(-.1,9)
plt.ylim(-1.8,1)
plt.show()
print('Human computer interaction\n')
for i in range(0,len(sims)):
    print('Cosine Similarity:', sims[i],documents[i])
