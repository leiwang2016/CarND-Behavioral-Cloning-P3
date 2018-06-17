import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port, using the secret
ptvsd.enable_attach("my_secret", address = ('105.128.124.74', 3000))

# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

# train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1:
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset::offset+batch_size]

#             images = []
#             angles = []
#             for batch_sample in batch_samples:
#                 name = './IMG/'+batch_sample[0].split('/')[-1]
#                 center_image = cv2.imread(name)
#                 center_angle = float(batch_sample[3])
#                 image.append(center_image)
#                 angles.append(center_angle)


#             X_train = np.array(images)
#             y_train = np.array(angles)

#             yield sklearn.utils.shuffle(X_train, y_train) 

# train_generator = generator(train_samples, batch_size=32)
# validation_samples = generator(validation_samples, batch_size=32)

# from keras.models import Sequential
# from keras

# ch, row, col = 3, 80, 320

# model = Sequential()

# model.add(lambda(lambda x: x/127.5-1.,
#     input_shape=(ch, row, col),
#     output_shape=(ch, row, col)
#     ))

# model.add

# model.compile(loss='mse', optimizer='adam')
# model.fit_generator(train_generator, samples_per_epoch=/
#     len(train_samples), validation_data=validation_generator, /
#     nb_val_samples=len(validation_samples), nb_epoch=3
#     )


