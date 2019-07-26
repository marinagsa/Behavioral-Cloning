import os
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt


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

steering_correction = {0 : 0, 1: 0.0, 2: 0.0} 

def generator(samples, batch_size = 32, is_train = 1 ):
    
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[ offset : offset + batch_size ]

            images = []
            angles = []
            
            for batch_sample in batch_samples: # batch_sample = line
                for i in range(3):
                    filename = batch_sample[i].split('/')[-1]

                    if data_dir == 'data_marina':
                        current_path = 'data_marina/IMG/' + filename
                    else:
                        current_path = 'data/IMG/' + filename
                    
                    image = plt.imread(current_path)
                    image = image[50:140,:,:] # trim image to only see section with road
                    #image = cv2.resize(image, (200, 66))  # trim image to be faster

                    if image is not None:    
                        
                        angle = float(batch_sample[3]) + steering_correction[i]
                        images.append(image)
                        angles.append(angle)
                    
                        if is_train:
                            images.append(np.fliplr(image))
                            angles.append(angle * -1.0)                    

            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

            
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=batch_size, is_train = 1 )
validation_generator = generator(validation_samples, batch_size=batch_size, is_train = 0 )


from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, Dropout, Flatten, Dense, Lambda
from keras.models import Model

# Nvidia model     
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (90, 320, 3) )) #50,20

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


print(model.summary())

model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)   

# 1 epoch: ~ 25 min
# model.fit_generator(train_generator, steps_per_epoch=len(train_samples) * 6, validation_data=validation_generator, validation_steps=validation_samples,epochs=1, verbose=1)   


# history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)   

# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()



model.save('model.h5')
print('Model saved.')
