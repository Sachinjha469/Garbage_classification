# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:25:16 2019

@author: Dark-Lord
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
classifier=Sequential()

#step1-Convoluion
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation='relu'))

#step 2 -Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))


classifier.add(Convolution2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size= (2,2)))

#step 3- Flattening
classifier.add(Flatten())

#step 4 - Full connection
classifier.add(Dense(128,activation = 'relu'))
classifier.add(Dense(1,activation = 'sigmoid'))

#compiling the Cnn
classifier.compile(optimizer='adam' , loss= 'binary_crossentropy', metrics=['accuracy'])

#part 2 - FItting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
#from keras import backend as K

train_datagen= ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set=test_datagen.flow_from_directory('Dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=2527,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=214)
classifier.save('model.h5')
                                                                 





