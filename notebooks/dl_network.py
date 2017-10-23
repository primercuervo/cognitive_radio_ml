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

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import math

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = '../../data/pic_set/train'
test_data_dir = '../../data/pic_set/test'
nb_train_samples = 76300
nb_test_samples = 8400
#Iterations
# epochs = 40000 # TC
EPOCHS = 400 # Test
BATCH_SIZE = 50 # TC
MOMENTUM = 0.9 # TC
L2 = 0.0005 # TC TODO: confirm that this is the decay of SGD


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(48, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, (3, 3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax')) # TC


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

sgd = SGD(lr=0.0, momentum=MOMENTUM, decay=L2, nesterov=False)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Fit the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    validation_data=test_generator,
    validation_steps=nb_test_samples // BATCH_SIZE)

model.save_weights('first_try.h5')
