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
	# Advanced activation layers are not activation functions. https://github.com/fchollet/keras/issues/2272
	# model.add(LeakyReLU(params['a'])) not model.add(Activation(LeakyReLU(params['a'])))
	ELU(alpha=1.0),
	
	# layer 2: filters 36, kernel 5x5, stride (subsample) 2x2, activation ELU
	Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2) ),
	ELU(alpha=1.0),
	
	# layer 3: filters 48, kernel 5x5, stride (subsample) 2x2, activation ELU
	Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)),
	ELU(alpha=1.0),

	# layer 4: filters 64, kernel 3x3, activation ELU
	Convolution2D(64, 3, 3, border_mode='valid'),
	ELU(alpha=1.0),
	
	# layer 5: filters 64, kernel 3x3, activation ELU
	# Convolution2D(64, 3, 3, border_mode='valid', activation=ELU(alpha=1.0)),
	
	# Flatten
	#   output dim is 64
	Flatten(),
	
	# Fully-connected layer 1
	# Dense(100),
	# ELU(alpha=1.0),
	
	# Fully-connected layer 2
	Dense(50),
	ELU(alpha=1.0),
	
	# Fully-connected layer 3
	Dense(10),
	ELU(alpha=1.0),
	
	Dense(1)
	
])

print ("Model defined")

#
# Model Compilation
#

learning_rate = 0.001
# for a mean squared error regression problem
adam = Adam(lr = learning_rate)
model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['mean_absolute_percentage_error'])

print ("Model compiled")

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

print("Model diagram")

#
# Model Training
#

### Start with example from "How to Use Generators"
import os
import csv

samples = []
with open('../ud-sim-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# remove headings row
samples.pop(0)

print("Samples CSV loaded len={}".format(len(samples)))

# shuffle and split into train and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=88)

print("Samples split train size={}, validation size={}".format(len(train_samples), len(validation_samples)))

# image plot library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib # uncomment for ipython

import cv2
import numpy as np
import sklearn
import PIL
from scipy.ndimage.interpolation import zoom

### Set up generator
print("Generator setup ...")

def generator(samples, batch_size=32, images_dir = '../ud-sim-data/', left_right=False, flip=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            angle_adjust = 0.15 # amount of correction for left and right cameras
            center_index = 0
            left_index=1
            right_index=2
            angle_index=3
            
            for batch_sample in batch_samples:
                # name = './IMG/'+batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name) 
                # use PIL instead to read as RGB not cv2 BGR? as drive.py uses PIL
                center_image = PIL.Image.open( images_dir + batch_sample[center_index] )
                center_angle = float(batch_sample[angle_index])
                images.append( np.asarray( center_image ) )
                angles.append(center_angle)
                
                # add in left and right images
                if left_right:
                    # left image, adjust angle to center
                    left_image = PIL.Image.open( images_dir + batch_sample[ left_index ].strip() )
                    left_angle = float(batch_sample[angle_index]) + angle_adjust # if car on left side, want to go more right
                    images.append( np.asarray( left_image ) )
                    angles.append( left_angle )
                    
                    # right image, adjust angle to center
                    right_image = PIL.Image.open( images_dir + batch_sample[ right_index ].strip() )
                    right_angle = float(batch_sample[angle_index]) - angle_adjust # if car on right side, want to go more left
                    images.append( np.asarray( right_image ) )
                    angles.append( right_angle )
                    
                # flip image vertically
                if flip:
                    image_flipped = np.fliplr(center_image)
                    angle_flipped = -center_angle
                    images.append( np.asarray( image_flipped ) )
                    angles.append( angle_flipped )
                
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # crop images to remove upper horizontal band above horizon (road) and lower horizontal band that includes the hood of car
            X_train = X_train[:, 60:124, 0:320, :]
            
            # resize to 64x64
            # X_resized = [cv2.resize(img,(64, 64), interpolation = cv2.INTER_CUBIC) for img in X]
            #http://stackoverflow.com/questions/40201846/resizing-ndarray-of-images-efficiently
            X_train=zoom(X_train,zoom=(1,1,64./320,1),order=1)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
# no left/right, no flip
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

# flip images
train_generator = generator(train_samples, batch_size=128, flip=True, left_right=True)
validation_generator = generator(validation_samples, batch_size=128, flip=True, left_right=True)



print("Start training ...")

# *2 to add flipped images
history = model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples)*4, 
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples)*4, 
    nb_epoch=6)
    
# center only
# history = model.fit_generator(train_generator, 
#     samples_per_epoch=len(train_samples), 
#     validation_data=validation_generator, 
#     nb_val_samples=len(validation_samples), 
#     nb_epoch=5)
	
#
# plot history for loss
#
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# train the model, iterating on the data in batches
# of 32 samples
### model.fit(data, labels, nb_epoch=10, batch_size=128)

# Save your trained model architecture as model.h5 using model.save('model.h5').
model.save('model.h5')

print('model saved')