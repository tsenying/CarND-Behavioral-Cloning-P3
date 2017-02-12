# input images shape (64,64,3) width, height, channels
# output 1 value steering angle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda, Convolution2D, Cropping2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import ELU
from keras import backend as K

#
# Model Definition
#

# model = Sequential([
#     Dense(32, input_dim=784, init='uniform'),
#     Activation('elu'),
#     Dense(10),
#     Activation('softmax'),
# ])

# NVIDIA model
model = Sequential([
	# Add image preprocessing layers in model means preprocessing doesn't have to be done prior to 
	# feeding data to model.
	# Such would be the case of running drive.py
	
	# Preprocessing includes
	# - cropping images to remove upper horizontal band above horizon (road) and lower horizontal band that includes the hood of car
	# - resize image to reduce number of parameters
	# - normalize image data
	
	# Crop image: remove 64 pixels from top and 36 pixels from the bottom.
	# should precalculate before model to reduce computation load?
	# output shape (64, 320, 3)
	# Cropping2D( cropping=( (60,36), (0,0) ), input_shape=(160,320,3) ),
	
	# Resize images contained in a 4D tensor of shape [batch, height, width, channels] (for 'tf' dim_ordering) 
	# this doesn't deal with dimensions correctly, should precalculate before model to reduce computation load anyways?
	#Lambda(lambda x: K.resize_images(x, 64, 64, dim_ordering='tf')),
	
	# # Scale to range with magnitude of 1.0 and Normalize to mean of zero
	# # resulting range -0.5 to 0.5
	# Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)),
	
	# Scale to range with magnitude of 2.0 and Normalize to mean of zero
	# resulting range -1.0 to 1.0
	Lambda(lambda x: (x/127.5) - 1.0, input_shape=(64,64,3)),
	
	# layer 1: filters 24, kernel 5x5, stride (subsample) 2x2 
	Convolution2D(24, 5, 5, border_mode='valid', input_shape=(64, 64, 3), subsample=(2,2)),
	
	# Use drop out if there is overfitting
	# model.add(Dropout(fractional value)), NOTE: need to be able to remove drop out when doing prediction
	
	# Activation ELU better than ReLU?
	# (without activation layer, the linear result of previous layer is used. "linear" activation: a(x) = x)
	Activation( ELU(alpha=1.0) ),
	
	# layer 2: filters 36, kernel 5x5, stride (subsample) 2x2, activation ELU
	Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation=ELU(alpha=1.0) ),
	
	# layer 3: filters 48, kernel 5x5, stride (subsample) 2x2, activation ELU
	Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation=ELU(alpha=1.0)),

	# layer 4: filters 64, kernel 3x3, activation ELU
	Convolution2D(64, 3, 3, border_mode='valid', activation=ELU(alpha=1.0)),
	
	# layer 5: filters 64, kernel 3x3, activation ELU
	# Convolution2D(64, 3, 3, border_mode='valid', activation=ELU(alpha=1.0)),
	
	# Flatten
	#   output dim is 64
	Flatten(),
	
	# Fully-connected layer 1
	# Dense(100, activation=ELU(alpha=1.0)),
	
	# Fully-connected layer 2
	Dense(50, activation=ELU(alpha=1.0)),
	
	# Fully-connected layer 3
	Dense(10, activation=ELU(alpha=1.0)),
	
	Dense(1)
	
])

#
# Model Compilation
#

learning_rate = 0.001
# for a mean squared error regression problem
adam = Adam(lr = learning_rate)
model.compile(optimizer = adam, loss = 'mean_squared_error')

#
# Model Diagram
#
# For keras plot to work, need:
# sudo pip install pydot-ng
# brew install graphviz
#
# generates model diagram in file specified by plot(to_file='')
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#
# Model Training
#

# Start with example from "How to Use Generators"
import os
import csv

samples = []
with open('../ud-sim-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# remove headings row
samples.pop(0)

# shuffle and split into train and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=88)

import cv2
import numpy as np
import sklearn
import PIL
from scipy.ndimage.interpolation import zoom

def generator(samples, batch_size=32, images_dir = '../ud-sim-data/'):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # name = './IMG/'+batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name) 
				# use PIL instead to read as RGB not cv2 BGR? as drive.py uses PIL
                center_image = PIL.Image.open( images_dir + batch_sample[0] )
                center_angle = float(batch_sample[3])
                images.append( np.asarray( center_image ) )
                angles.append(center_angle)
			
            X_train = np.array(images)
            y_train = np.array(angles)
			
			# crop images to remove upper horizontal band above horizon (road) and lower horizontal band that includes the hood of car
            X_train = X_train[:, 60:124, 0:320, :]
			
			# resize to 64x64
			# X_resized = [cv2.resize(img,(64, 64), interpolation = cv2.INTER_CUBIC) for img in X]
			#http://stackoverflow.com/questions/40201846/resizing-ndarray-of-images-efficiently
            X_train=zoom(X_train,zoom=(1,1,64./320,1),order=1)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)

# K.resize_images(x, 64, 64, dim_ordering='tf')

# train the model, iterating on the data in batches
# of 32 samples
### model.fit(data, labels, nb_epoch=10, batch_size=128)

# Save your trained model architecture as model.h5 using model.save('model.h5').
### model.save('model.h5')