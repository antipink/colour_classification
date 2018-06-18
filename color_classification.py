# -*- coding: utf-8 -*-

__author__ = 'R.Li'
'''
# Copyright (C) 2017 Publicis Groupe Media UK
# Data Sciences UK
#
# This file is generated for an internal project within ZO and
# should not be copied and/or distributed in part or as whole.
#
# In case of any queries, please contact
#        AttributionModelling@zenithoptimedia.co.uk;
#        ray.li@publicismedia.com
#
'''
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt
from keras.preprocessing import image
# from tensorflow.contrib.keras import backend as cond
import keras.backend as K
from keras.models import load_model
import numpy as np
import os

import resnet
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.models import Model

batch_size = 32
nb_classes = 12
nb_epoch = 500
# data_augmentation = False
img_rows, img_cols = 128, 128
img_channels = 3
file_name = 'Colour/Colour/'

def balanced_loss(y_true, y_pred):
    dog = K.cast(y_true, K.floatx())
    cat = K.cast(K.equal(y_true, 0), K.floatx())
    n_dog = K.sum(dog)
    n_cat = K.sum(cat)
    n_dog = K.tf.cond(n_dog >= 1.0, lambda: n_dog, lambda: 1.0)
    n_cat = K.tf.cond(n_cat >= 1.0, lambda: n_cat, lambda: 1.0)

    scale_factor = K.tf.divide(n_cat, n_dog)

    factors = dog
    factors = K.tf.multiply(factors, scale_factor)
    factors = K.tf.add(factors, cat)

    error = K.binary_crossentropy(y_true, y_pred)
    modified_error = K.tf.multiply(error, factors)
    modified_error = K.mean(modified_error, axis=1)

    return modified_error

# rgb_mean = [0.485, 0.456, 0.406]
# rgb_std = [0.229,0.224,0.225]

train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
#train_datagen = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization = True, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split = 0.2)

training_set = train_datagen.flow_from_directory(file_name, target_size=(img_rows, img_rows), batch_size=batch_size, class_mode = 'categorical', subset ='training')

print(training_set.class_indices)

test_set = train_datagen.flow_from_directory(file_name, target_size=(img_rows, img_rows), batch_size=batch_size, class_mode = 'categorical', subset ='validation')

#base_model = VGG16(weights= 'imagenet',include_top=False, input_shape= (img_rows, img_cols, 3))
#x = Flatten()(base_model.output)
#x = Dense(nb_classes, activation= 'softmax')(x)
#model = Model(inputs= base_model.input,outputs= x)
#model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), activation='relu', padding ='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding ='same'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
model.add(Conv2D(64, (3, 3), activation='relu', padding ='same'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding ='same'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, (3, 3), activation='relu', padding ='same'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# Uses the Dense() to add the hidden layer and defines the number of nodes
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
# Initialise Output layer, only one unit because its a binary classification
model.add(Dense(12, activation='softmax'))
model.summary()



model = resnet.ResnetBuilder.build_resnet_18((3, img_rows, img_cols), 12)

opt = optimizers.adam(lr=0.01, decay = 1e-3)
model.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])

# tbCallBack = TensorBoard(log_dir='./Graph')


checkpointer = ModelCheckpoint(filepath='./weights/weights.h5', verbose=1, save_best_only=True)
model.fit_generator(training_set,
                    # steps_per_epoch=X_train.shape[0] // batch_size,
                    steps_per_epoch = training_set.samples // training_set.batch_size,
                    validation_data = test_set,
                    validation_steps= test_set.samples // test_set.batch_size,
                    epochs=nb_epoch, verbose=1,
                    callbacks=[checkpointer])



