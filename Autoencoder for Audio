import wave
import struct
from struct import *
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
a=wave.open('pare.wav')
leng = a.getnframes()

c=[]
for i in range(70000,90000):
    data=a.readframes(1)
    d=struct.unpack("f", data)
    c.append(d)

e=pd.DataFrame(c)
f=e.dropna()

g=[]
for i in range(5000,10000,25):
    g.append(np.mean(f[i:i+500]))

x_train2= MinMaxScaler().fit_transform(g)
x_train1=x_train2[0:200]
x_train0=[float(i) for i in x_train1]
x_train=np.array([x_train0]).astype('float32')
plt.figure(figsize=(10, 5))
plt.plot([float(i) for i in x_train[0]])

encoding_dim = 150 
input_img = Input(shape=(200,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(200, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

filepath="audio.compress-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='binary_crossentropy', verbose=1, save_best_only=False)

autoencoder.summary()
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=40,
                shuffle=False,
                validation_data=(x_train, x_train),callbacks=[checkpoint],verbose=0)

filename = "audio.compress-0.5576.hdf5"
autoencoder.load_weights(filename)
autoencoder.compile(loss='mean_squared_error', optimizer='adam')
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)

n = 1  # how many digits we will display
plt.figure(figsize=(10, 5))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(10, 20))
    plt.title("ORIGINAL AUDIO")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.figure(figsize=(8, .5))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs.reshape(75,2).T)
    plt.title('ENCODED AUDIO')    
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.figure(figsize=(10, 5))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(10, 20))
    plt.title("RECONSTRUCTED AUDIO")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("Accuracy=",1-np.mean(abs(x_train-decoded_imgs)),'\n')

plt.figure(figsize=(10, 2))
plt.plot(x_train[0],color='r',linewidth=3)
plt.plot(decoded_imgs[0])
plt.title('ORIGINAL (blue) and RECONSTRUCTED AUDIO (red)')
plt.show()
