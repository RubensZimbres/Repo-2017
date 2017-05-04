from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('Dogs.png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/Volumes/16 DOS/Python/temp/Dogs', save_prefix='dog', save_format='jpeg'):
    i += 1
    if i > 10:
        break
        
img = load_img('cats.png') 
x = img_to_array(img) 
x = x.reshape((1,) + x.shape) 

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/Volumes/16 DOS/Python/temp/cats', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 10:
        break 

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

model = Sequential()
model.add(Conv2D(20, 3, 3, input_shape=(3, 30, 30)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
batch_size = 10

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/Volumes/16 DOS/Python/temp',  # this is the target directory
        target_size=(30, 30),  # all images will be resized to 30x30
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/Volumes/16 DOS/Python/temp',
        target_size=(30, 30),
        batch_size=batch_size,
        class_mode='binary')
        
model.fit_generator(
        train_generator,
        nb_epoch=10,
        validation_data=validation_generator,nb_val_samples=20
        ,samples_per_epoch=20)        

model.save_weights('first_try.hdf5')

import numpy as np
import cv2

image = cv2.imread('cats.png')
cv2.imshow("Original", image)
dim = (30,30)

resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

X_data=resized.reshape(1,3,30,30)

filename = "first_try.hdf5"
model.load_weights(filename)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.predict_classes(X_data)

'''Adapted from Keras blog: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html'''
