from scipy import ndimage
import csv
import numpy as np
import cv2
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Configuration
training_valid_dataset = ['/root/opt/track1_3lap_new', '/root/opt/track1_1lap_clockwise']
correction_factor = 0.2
batch_size = 32
epochs = 10

# training infos loaded from csv-file
lines = []
for training_data_path in training_valid_dataset:
    with open(training_data_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# mutiple camera images extraction
measurements = []
camera_images = []
for line in lines:
    # center image
    img_path = line[0]
    measurement = float(line[3])
    measurements.append(measurement)
    center_img = ndimage.imread(img_path)
    camera_images.append(center_img)

    # Horizontal flip center image for data augmentation
    camera_images.append(cv2.flip(center_img, 1))
    measurements.append(measurement * -1.0)

    # left image
    img_path = line[1]
    left_img = ndimage.imread(img_path)
    camera_images.append(left_img)
    measurements.append(min(measurement+0.2, 1))

    # right image
    img_path = line[2]
    right_img = ndimage.imread(img_path)
    camera_images.append(right_img)
    measurements.append(max(measurement-0.2, -1.0))

train_samples, val_samples = train_test_split(list(zip(camera_images, measurements)), test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = [sample_tuple[0] for sample_tuple in batch_samples]
            angles = [sample_tuple[1] for sample_tuple in batch_samples]

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
val_generator = generator(val_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout, Cropping2D

model = Sequential()
# Cropping
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalization
model.add(Lambda(lambda x: (x / 127.5) - 1.0))
# NVIDIA network architecture
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))  # FC-layer with 50 neurons replaced by a dropout layer
model.add(Dense(25, activation='relu'))  # FC-layer with 10 neurons increased to 25
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=15)
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / batch_size),
                    validation_data=val_generator, validation_steps=math.ceil(len(val_samples) / batch_size),
                    epochs=epochs, verbose=1)

model.save('model.h5')
print('Model saved!')
        
