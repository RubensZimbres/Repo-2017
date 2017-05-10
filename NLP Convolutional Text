from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
import numpy as np
from random import shuffle
from sklearn.linear_model import LogisticRegression
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Highway
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
K.set_image_dim_ordering('th')

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
model = Doc2Vec(min_count=1, window=5, size=16, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')
model.most_similar('good')

sentences.to_array()

train_arrays = numpy.zeros((8, 16))

for i in range(7):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[1 + i] = model.docvecs[prefix_train_neg]

test_arrays = numpy.zeros((8, 16))

for i in range(7):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]

batch_size = 1
nb_classes = 2
nb_epoch = 10
nb_filters = 12
pool_size = (2, 2)
kernel_size = (2, 2)

X_train= train_arrays.astype('float32')
X_train=X_train.reshape(2,1,8,8)
X_test = test_arrays.astype('float32')
X_test=X_test.reshape(2,1,8,8)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = [[1,0],[0,1]]
test_labels = [[1,0],[0,1]]

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(1,8,8)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2))
model.add(Highway())
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.01, decay=5e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.fit(X_train, Y_train, 
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)

print('Test accuracy:', np.sum(model.predict_classes(X_test)-test_labels==0)/len(test_labels))
print(model.predict_classes(X_test))
print(test_labels)
