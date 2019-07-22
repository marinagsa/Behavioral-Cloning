import os
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
import math

# Udacity data
data_dir = 'data'
data_csv = '/driving_log.csv'

# My recorded data
# data_dir = 'data_marina'
# data_csv = '/driving_log.csv'

samples = []
with open(data_dir + data_csv) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
        

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size = 32):
    
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                
                filename = batch_sample[0].split('/')[-1]

                if data_dir == 'data_marina':
                    current_path = 'data_marina/IMG/' + filename
                else:
                    current_path = 'data/IMG/' + filename
                
                center_image = cv2.imread(filename)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
#print(train_generator)

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, Dropout, Flatten, Dense, Lambda, Cropping2D
from keras.models import Model

# Nvidia model     
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = (160, 320, 3) ))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu')) # conv2d_1
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu')) # conv2d_2
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu')) # conv2d_3
model.add(Convolution2D(64, 3, 3, activation='elu'))                   # conv2d_4
model.add(Convolution2D(64, 3, 3, activation='elu'))                   # conv2d_5
model.add(Dropout(.9))                                                 # dropout at 0.9
model.add(Flatten())                                                   # flatten
model.add(Dense(100, activation = 'elu')) # dense_1
model.add(Dense(50, activation = 'elu'))  # dense_2
model.add(Dense(10, activation = 'elu'))  # dense_3
model.add(Dense(1))                       # dense_4

optimizer = Adam(lr=1e-3) # learning rate 0.001
model.compile(loss='mse', optimizer=optimizer)

model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)   

model.save('model.h5')
print('Model saved.')
