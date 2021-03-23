# Importing Libraries and Packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD, Adam
from pathlib import Path
from efficientnet.keras import EfficientNetB0
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, BatchNormalization
from keras import initializers, regularizers
from pathlib import Path
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, History, LearningRateScheduler
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import os
from time import time
import sys
sys.path.insert(1, 'src/')
import config

size = config.SIZE
epochs = config.EPOCHS
steps = config.STEPS


train_data_generation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True
)
validation_data_generation = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_data_generation.flow_from_directory(
        config.TRAIN_PATH,
        target_size=(size, size),
        class_mode='categorical',
        batch_size = 64
    )

validation_generator = validation_data_generation.flow_from_directory(
    config.TEST_PATH,
    target_size=(size, size),
    class_mode='categorical',
    batch_size = 64
)

filepath = "bestweight.h5"
metric = 'val_accuracy'
checkpoint = ModelCheckpoint("models/weights{epoch:05d}.h5", monitor=metric, verbose=1, save_best_only=True, mode='max')
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, cooldown=0, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reduce]


conv_m = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(size, size, 3))
conv_m.trainable = False
model = Sequential()
model.add(conv_m)
model.add(AveragePooling2D(pool_size=(7, 7)))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr = 0.1, momentum = 0.9),
    metrics=['accuracy']
)

history = model.fit_generator(
    train_generator,
    callbacks=callbacks,
    epochs=100,
    steps_per_epoch=10,
    validation_data=validation_generator,
    validation_steps=10,
    initial_epoch = 30
)