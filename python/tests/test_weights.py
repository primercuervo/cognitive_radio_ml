from keras import backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2
import numpy as np

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

sgd = SGD(lr=0.001, momentum=MOMENTUM, decay=0.0, nesterov=False)

model.compile(loss='categorical_crossentropy',
              # optimizer='adadelta',
              # optimizer=sgd,
              optimizer='rmsprop',
              # optimizer='adam',
              metrics=['accuracy'])
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
# model.save_weights('adadelta_default_es.h5')
from keras.preprocessing.image import img_to_array, load_img


model.load_weights('../weights/rmsprop_default_es.h5')

im = load_img('/home/cuervo/thesis/data/pic_set/test/scn_8/image_757.jpg', grayscale=True)
x = img_to_array(im)
x = np.expand_dims(x, axis=0)

print(model.predict(x)*100)

model.save('../weights/rmsprop_default_es_model.h5')
