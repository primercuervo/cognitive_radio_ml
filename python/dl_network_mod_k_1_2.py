'''
Directory structure:
```
data/
    train/
        scn_0/
            image_1.jpg
            image_2.jpg
            ...
        scn_1/
            image_1.jpg
            image_2.jpg
            ...
    validation/
        scn_0/
            image_1.jpg
            image_2.jpg
            ...
        scn_1/
            image_1.jpg
            image_2.jpg
            ...
```
'''

from keras import backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2

import math

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = '../../data/pic_set/train'
test_data_dir = '../../data/pic_set/test'
nb_train_samples = 76300
nb_test_samples = 8400
#Iterations
#EPOCHS = 4000 # TC / 10
EPOCHS = 2000 # TC / 10
# EPOCHS = 400 # Test
BATCH_SIZE = 50 # TC
MOMENTUM = 0.9 # TC

# Apply L2 regularization
# TODO: so far this regularization is applied to all convolution layers.
# Better to determine if there is an optimum way or layer for this to be
# applied to
L2 = 0.0005 # TC

model = Sequential()

# The last dimension is 1 because of gray-scale
# model.add(Conv2D(48, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Conv2D(48, 2, 2,
          # W_regularizer=l2(L2),
          # activity_regularizer=activity_l2(L2),
          input_shape=(img_width, img_height, 1)
          ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 2, 2,
          # W_regularizer=l2(L2),
          # activity_regularizer=activity_l2(L2)
          ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, 2, 2,
          # W_regularizer=l2(L2),
          # activity_regularizer=activity_l2(L2)
          ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, 2, 2,
          # W_regularizer=l2(L2),
          # activity_regularizer=activity_l2(L2)
          ))
model.add(Activation('relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 2, 2,
          # W_regularizer=l2(L2),
          # activity_regularizer=activity_l2(L2)
          ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024,
          W_regularizer=l2(L2),
          activity_regularizer=activity_l2(L2)
          ))
# model.add(Activation('relu'))
model.add(Dense(1024,
          W_regularizer=l2(L2),
          activity_regularizer=activity_l2(L2)
          ))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax')) # TC
model.summary()

# Add a learning rate schedule alike to the TC
# References
# ---------
# https://goo.gl/4vQhdj
# https://goo.gl/VrrciJ

# "The training is performed using the stochastic gradient descent algorithm
# Learning rate starts with alpha=0.001 and is multiplied by gamma=0.1 every
# 10K iterations, where alpha is the weight of the negative gradient
# The momentum miu is set to 0.9 (the weight of the previous update)
# L2 regularization is used with weight decay of 0.0005 to prevent over-fitting"

# This means a step decay, which takes the form
# lr = lr0 * drop^floor(epoch/epocs_drop)

def step_decay(epoch):
    initial_lrate = 0.001
    gamma = 0.1
    epochs_drop = EPOCHS / 4
    lrate = initial_lrate * math.pow(gamma, math.floor((1+epoch) / epochs_drop))
    return lrate

# model.compile(loss='binary_crossentropy',
# Need to be categorical because this is not a binary classification problem
# Dont use crossentropy. See https://stackoverflow.com/a/37046330
# model.compile(loss='categorical_crossentropy',

# Compile stochastic gradient descent model with step decreasing learning rate

# sgd = SGD(lr=0.001, momentum=MOMENTUM, decay=0.0, nesterov=False)
# sgd = SGD(lr=1e-9)
# sgd = SGD()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              # optimizer=sgd,
              # optimizer='rmsprop',
              # optimizer='adam',
              metrics=['accuracy'])

# Set the callback for Tensorboard (visualization purposes)
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0,
          write_graph=True, write_images=False)

#Setting EarlyStopping for rmsprop
early_stop = EarlyStopping(monitor='val_acc',
                           min_delta=0,
                           patience=60,
                           verbose=2,
                           mode='auto')

# learning schedule callback
# lrate = LearningRateScheduler(step_decay)

# CVS logger callback
csv = CSVLogger("adadelta_default_es.csv", separator=',', append=False)

# Checkpointer callback: Saves the model weights after each epoch if
# the validation loss decreased
checkpointer = ModelCheckpoint(filepath='adadelta_default_es_checkpoint.h5',
                               verbose=1,
                               save_best_only=True)

# Set the callback list
# callbacks_list = [lrate, tbCallBack]
callbacks_list = [tbCallBack, early_stop, csv, checkpointer]

# this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True)

train_datagen = ImageDataGenerator()
# this is the augmentation configuration we will use for testing:
# only rescaling
# test_datagen = ImageDataGenerator(rescale=1. / 255)
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

# print len(train_generator.filenames))
# Fit the model
model.fit_generator(
    generator=train_generator,
    # samples_per_epoch=nb_train_samples // BATCH_SIZE,
    samples_per_epoch=1550,
    nb_epoch=EPOCHS,
    callbacks=callbacks_list,
    validation_data=test_generator,
    nb_val_samples=nb_test_samples // BATCH_SIZE)

model.save_weights('adadelta_default_es.h5')
