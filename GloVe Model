from keras.layers import Embedding
import os
texts = []
labels_index = {}
labels = [] 
for name in sorted(os.listdir('/Volumes/16 DOS/Python/text')):
    path = os.path.join('/Volumes/16 DOS/Python/text', name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            f = open(fpath, encoding='latin-1')
            t = f.read()
            i = t.find('\n')  # skip header
            if 0 < i:
                t = t[i:]
            texts.append(t)
            f.close()
            labels.append(label_id)


print('Found %s texts.' % len(texts))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np

MAX_NB_WORDS=50
MAX_SEQUENCE_LENGTH=20
VALIDATION_SPLIT=0.9

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np_utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open('/Volumes/16 DOS/Python/text/text2/ZarathustraSmall.2.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=20
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
        
MAX_SEQUENCE_LENGTH=20

from keras.layers import Embedding, Conv1D,MaxPooling1D,Dense,Flatten
from keras.models import Sequential

from keras import backend as K
K.set_image_dim_ordering('th')

model=Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels_index), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)

model.predict_classes(x_train)

'''Adapted from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html'''
