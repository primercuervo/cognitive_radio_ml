#!/usr/bin/env python
# pylint: disable=C0103
"""
Training various Deep Learning Algorithms using Keras with Tensorboard backend

This module takes paths to the location of images to be used for Image
classification. As of now it is used for cognitive radio spectrum awareness
based on the DySpan2017 Challenge spectrum setup, but it can be used for any
multiclass image recognition.

Example:
    To run this script just type:
        $ python2 cnn_k_1_2.py

    The name stands for:
        - CNN - Convolutional Neural Network
        - k_1_2 - Keras-v1.2 which had to be used because of hardware
        restrictions

    **is written for python2**

TODO:
    Urgent:
        * Revise regularization
          TODO: so far this regularization is applied to all convolution layers.
          Better to determine if there is an optimum way or layer for this to be
          applied to
    Optional:
        * API update - Keras and tensorflow
        * py3k
"""


from argparse import ArgumentParser
import math
import os.path
import sys

from keras.callbacks import (LearningRateScheduler, TensorBoard, EarlyStopping,
                             CSVLogger, ModelCheckpoint)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2

##################################################
# Parser setup
##################################################
parser = ArgumentParser()
parser.add_argument("-n", # Alias
                    "--name",
                    help="Set the basename for the generated files",
                    default=None)
args = parser.parse_args()
##################################################
# Constants and Variables
##################################################
BASEPATH = '../trained_models/keras'
if args.name is None:
    print("[ERROR]: Please provide a basename for the generated files.")
    sys.exit()

basename = args.name
# Set the filenames based on the basename
names = {'model': basename + "_model.h5",
         'weights': basename + "_weights.h5",
         'checkpoint': basename + '_ckpt.h5',
         'csv': basename + '.csv',
        }

# dimensions of our images.
img_width, img_height = 64, 64
# This is my relative location, please change accordingly
train_data_dir = '../../data/pic_set/train'
test_data_dir = '../../data/pic_set/test'
nb_train_samples = 76300
nb_test_samples = 8400
# Currently each iteration takes 7 seconds in this machine, so I set 3000
# Epochs only based on time constraints. Based on my results, more iterations
# Will *not* give me better results anyways
EPOCHS = 3000 #
BATCH_SIZE = 50
MOMENTUM = 0.9
# Apply L2 regularization
L2 = 0.0005

##################################################
# Procedural Start
##################################################
# Check if files already exist to avoid accidental
# Overwrite
for folder, name in names.items():
    my_file = os.path.join(BASEPATH, folder, name)
    if os.path.isfile(my_file):
        print("[ERROR]: " + my_file + " already exists.")
        print("Please choose another basename")
        sys.exit()

##################################################
# Model definition
##################################################

# Define CNN based on
# http://ieeexplore.ieee.org/document/8017499/
model = Sequential()

# The last dimension is 1 because of gray-scale
model.add(Conv2D(48, 2, 2,
                 input_shape=(img_width, img_height, 1)
                )
         )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024,
                W_regularizer=l2(L2),
                activity_regularizer=activity_l2(L2)
               )
         )
model.add(Dense(1024,
                W_regularizer=l2(L2),
                activity_regularizer=activity_l2(L2)
               )
         )
model.add(Dense(10))
model.add(Activation('softmax'))
# Print the model summary to terminal
model.summary()

# Compile the model
# Need to be categorical because this is not a binary classification problem
# Compile stochastic gradient descent model with step decreasing learning rate

sgd = SGD(lr=0.001, momentum=MOMENTUM, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Add a learning rate schedule
# References
# ---------
# https://goo.gl/4vQhdj
# https://goo.gl/VrrciJ

def step_decay(epoch):
    """
    Define a step decay, which takes the form
    lr = lr0 * drop^floor(epoch/epocs_drop)
    """
    initial_lrate = 0.001
    gamma = 0.1
    epochs_drop = EPOCHS / 4
    lrate = initial_lrate * math.pow(gamma, math.floor((1+epoch) / epochs_drop))
    return lrate

# learning schedule callback
lschedule = LearningRateScheduler(step_decay)

# Set the callback for Tensorboard (visualization purposes)
tbCallBack = TensorBoard(log_dir=os.path.join(BASEPATH,
                                              'tensorboard'),
                         histogram_freq=0,
                         write_graph=True, write_images=False)

#Setting EarlyStopping for rmsprop
early_stop = EarlyStopping(monitor='val_acc',
                           min_delta=0,
                           patience=60,
                           verbose=2,
                           mode='auto')

# CVS logger callback
csv = CSVLogger(os.path.join(BASEPATH,
                             'csv',
                             names['csv']),
                separator=',', append=False)

# Checkpointer callback: Saves the model weights after each epoch if
# the validation loss decreased
checkpointer = ModelCheckpoint(filepath=os.path.join(BASEPATH,
                                                     'checkpoint',
                                                     names['checkpoint']),
                               verbose=1,
                               save_best_only=True)

# Set the callback list
# callbacks_list = [lschedule, tbCallBack]
callbacks_list = [tbCallBack, csv, checkpointer]

# Create Image generators
# Not using rescaling or any other data augmentation technique
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True,
    class_mode='categorical')

# Fit the model
model.fit_generator(
    generator=train_generator,
    samples_per_epoch=1550,
    nb_epoch=EPOCHS,
    callbacks=callbacks_list,
    validation_data=test_generator,
    nb_val_samples=nb_test_samples // BATCH_SIZE)

# Save the models
model.save_weights(os.path.join(BASEPATH, 'weights', names['weights']))
model.save(os.path.join(BASEPATH, 'model', names['model']))
