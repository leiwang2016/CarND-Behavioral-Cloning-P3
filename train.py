import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port, using the secret
ptvsd.enable_attach("my_secret", address = ('105.128.124.74', 3000))
#ptvsd.enable_attach("my_secret", address = ('192.168.1.10', 3000))

# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()

import os
import csv

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

with open('../data/driving_log_1.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

with open('../data/driving_log_2.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '../data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                    images.append(cv2.flip(image,1))      
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angles.append(center_angle*-1.0)
                angles.append(center_angle+correction)
                angles.append((center_angle+correction)*-1.0)
                angles.append(center_angle-correction)
                angles.append((center_angle-correction)*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

BATCH_SIZE=16
DROP_RATE=0.2 

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

row, col, ch = 160, 320, 3

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch))) 
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(DROP_RATE))

model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(DROP_RATE))

model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(DROP_RATE))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(DROP_RATE))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(DROP_RATE))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(DROP_RATE))

model.add(Dense(50))
model.add(Dropout(DROP_RATE))

model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples)/BATCH_SIZE, epochs=5)

model.save('model.h5')