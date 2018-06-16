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
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.contrib.keras import backend as cond
import keras.backend as K
from tensorflow.contrib.keras.api.keras.models import load_model
import numpy as np
import os
import resnet
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.applications.vgg16 import VGG16


batch_size = 32
nb_classes = 13
nb_epoch = 100
# data_augmentation = False
img_rows, img_cols = 128, 128
img_channels = 3
file_name = 'Colour/'

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

train_datagen = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization = True, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split = 0.2)

training_set = train_datagen.flow_from_directory(file_name, target_size=(img_rows, img_rows), batch_size=batch_size, class_mode = 'categorical', subset ='training')

test_set = train_datagen.flow_from_directory(file_name, target_size=(img_rows, img_rows), batch_size=batch_size, class_mode = 'categorical', subset ='validation')

basemodel = VGG16(weights= 'imagenet',include_top=False, input_shape= (96, 96, 3))
x = Flatten()(basemodel.output)
x = Dense(nb_classes, activation= 'softmax')
#model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model = Model(model.input, x)
model.summary()

opt = optimizers.adam(lr=0.01)
model.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./Graph')



checkpointer = ModelCheckpoint(filepath='./weights/weights.h5', verbose=1, save_best_only=True)
model.fit_generator(training_set,
                    # steps_per_epoch=X_train.shape[0] // batch_size,
                    steps_per_epoch = training_set.samples // training_set.batch_size,
                    validation_data = test_set,
                    validation_steps= test_set.samples // test_set.batch_size,
                    epochs=nb_epoch, verbose=1,
                    callbacks=[tbCallBack, checkpointer])


# seq = Sequential()
# seq.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
#
# # Takes our Objects and adds it to the pooling layers
# # Also gathers the features of the photo
# seq.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Adding a second layer
# seq.add(Conv2D(32, (3, 3), activation='relu'))
# seq.add(MaxPooling2D(pool_size=(2, 2)))
#
# seq.add(Conv2D(32, (3, 3), activation='relu'))
# seq.add(MaxPooling2D(pool_size=(2, 2)))
#
# seq.add(Conv2D(32, (3, 3), activation='relu'))
# seq.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Converting the 2D array to a 1D
# seq.add(Flatten())
# # Uses the Dense() to add the hidden layer and defines the number of nodes
# seq.add(Dense(units=128, activation='relu'))
# # Initialise Output layer, only one unit because its a binary classification
#
# seq.add(Dense(units=1, activation='sigmoid'))
#
# seq.compile(optimizer='adam', loss=balanced_loss, metrics=['accuracy', dog_acc, cat_acc])
#
# seq.load_weights('model.h5')
#
# # Preproccessing images
# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# training_set = train_datagen.flow_from_directory(
#     '/Users/rootuser/Sync/AllData/Personal/MachineLearning/DataSets/DogCatDataset/training_set', target_size=(64, 64),
#     batch_size=32, class_mode='binary')
#
# test_set = test_datagen.flow_from_directory(
#     '/Users/rootuser/Sync/AllData/Personal/MachineLearning/DataSets/DogCatDataset/test_set', target_size=(64, 64),
#     batch_size=32, class_mode='binary')
#
# # seq.fit_generator(training_set, samples_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)
#
# test_acc = []
# train_acc = []
# for i in range(25):
#     seq.fit_generator(training_set, samples_per_epoch=8000, epochs=1)
#     test_acc.append(seq.evaluate_generator(test_set))
#     train_acc.append(seq.evaluate_generator(training_set))
#     print(i, "test", test_acc[i], "train", train_acc[i])
#
#     seq.save_weights('model.h5')
#     test = np.array([1, 2, 3, 4])
#     plt.plot(test)
#     plt.show()
#
# # test_image = image.load_img('/Users/rootuser/Documents/DataSets/DogCatDataset/pics', target_size = (64, 64))
# # test_image= image.img_to_array(test_image)
# # test_image = np.expand_dims(test_image, axis = 0)
# # result = seq.predict(test_image)
# # print(result)
# #
# # for prediction, image_path in zip(predictions, test_images_path):
# #    if predictions == 1:
# #        prediction = 'dog'
# #    else:
# #        prediction = 'cat'
# #        print("Predicted {}  for file {}".format(prediction, image_path.split("/")[-1]))
# # backend.clear_session()


