# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(16, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'relu'))

model.summary()

model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(
        'dataset/training_data/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_data = test_datagen.flow_from_directory(
        'dataset/validation_data/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# model.fit_generator(
#         training_data,
#         steps_per_epoch='<< samples >>',
#         epochs=50,
#         validation_data=validation_data,
#         validation_steps='<< samples >>')