#
# Behavioral Cloning Project
#
# input images shape (64,64,3) width, height, channels
# output 1 value steering angle
#
# Usage: python model.py
#
# Diagram of model is output into file 'model.png'
# Training and validation loss is plotted.
# Saves resulting model into file 'model.h5'
#

# process command line args
import sys, getopt
def process_args(argv):
    imagesdir = ''
    modelfile = ''
    epochs = 6
    try:
        opts, args = getopt.getopt(argv,"hm:i:e:",["mfile=","idir=","epochs="])
    except getopt.GetoptError:
        print( 'test.py -m <modelfile> -i <imagesdir>' )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -m <modelfile> -i <imagesdir>' ) 
            sys.exit()
        elif opt in ("-i", "--idir"):
            imagesdir = arg
        elif opt in ("-m", "--mfile"):
            modelfile = arg
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
    return modelfile, imagesdir, epochs
   
model_file, images_dir, epochs = process_args(sys.argv[1:])
print ('Model file is:', model_file)
print ('Images dir is:', images_dir)
print ('Epochs is:', epochs)


# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras.regularizers import l2

#
# Model Definition
#

# Uses NVIDIA model as the starting reference, but simplified down as input image dimensions used are 64x64 not 66x200
model = Sequential([
	# Add image preprocessing layers in model means preprocessing doesn't have to be done prior to 
	# feeding data to model.
	
	# Preprocessing can include:
	# - cropping images to remove upper horizontal band above horizon (road) and lower horizontal band that includes the hood of car
	# - resize image to reduce number of parameters
	# - normalize image data
	
	# NOTE: we could crop images in the model with Keras layer Cropping2D,
	# but we also want to resize images which is more compute intensize,
	# so cropping and resizing is done before the model,
	# this means corresponding image preprocessing has to be done when feeding the simulator (drive.py) as well
	# Crop image: remove 64 pixels from top and 36 pixels from the bottom.
	# Cropping2D( cropping=( (60,36), (0,0) ) ),
	
	# Scale to range with magnitude of 2.0 and Normalize to mean of zero
	# resulting range -1.0 to 1.0
	Lambda(lambda x: (x/127.5) - 1.0, input_shape=(64,64,3)),
	
	# layer 1: filters 24, kernel 5x5, stride (subsample) 2x2 
	Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)),
	
	# Dropout layer here did not improve driving performance
	# Use drop out if there is overfitting
	## Dropout( 0.2 ),
	
	# Activation ELU better than ReLU?
	# (without activation layer, the linear result of previous layer is used. "linear" activation: a(x) = x)
	# Advanced activation layers are not activation functions. https://github.com/fchollet/keras/issues/2272
	# model.add(LeakyReLU(params['a'])) not model.add(Activation(LeakyReLU(params['a'])))
	ELU(alpha=1.0),
	
	# Maxpool layer here did not improve driving performance
	# stride size same as pool size
	## MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'),
	
	# layer 2: filters 36, kernel 5x5, stride (subsample) 2x2, activation ELU
	Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)),
	ELU(alpha=1.0),
	
	# layer 3: filters 48, kernel 5x5, stride (subsample) 2x2, activation ELU
	Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)),
	ELU(alpha=1.0),
	
	# layer 4: filters 64, kernel 3x3, activation ELU
	Convolution2D(64, 3, 3, border_mode='valid'),
	ELU(alpha=1.0),
	
	# layer 5: filters 64, kernel 3x3, activation ELU
	# NOTE: this layer reduced dimensions down to 1x1, didn't seem necessary
    # Convolution2D(64, 3, 3, border_mode='valid'),
    # ELU(alpha=1.0),
	
	# Flatten
	#   output dim is 64
	Flatten(),
	
	# Fully-connected layer 1
	# NOTE: having a fully connected layer > 64 would expand dimensions not reduce it as is usual with Convolutional Networks
	#   so this layer was removed
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

# Optimizer: Adam (adaptive moment estimation) https://arxiv.org/abs/1412.6980v8
# Adam "computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients"
# Configure for a mean squared error regression problem
# Use default parameter values

# if model file is input, load and use previously saved model
# else compile new model
if ( len(model_file) == 0 ):
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
    model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])
    print ("Model compiled")
else:
    # returns a compiled model
    # identical to the previous one
    # NOTE: either reload previous model, or compile model from scratch, not both
    from keras.models import load_model
    model = load_model(model_file)
    print ("Previous model loaded:", model_file)

#
# Model Diagram
#
# For keras plot to work, need:
#   sudo pip install pydot-ng
#   brew install graphviz
#
# generates model diagram in file specified by plot(to_file='')
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print("Model diagram output to file")

#
# Model Training
#

### Start with example from course material "How to Use Generators"
# Use generators to avoid entire data set, load by batch instead.
import os
import csv

samples = []
with open( images_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# remove headings row
samples.pop(0)

print("Samples CSV loaded len={}".format(len(samples)))

# filter out near 0 steering angle samples
# ( Don't filter zeros, car went off road. )
# import random
# filtered_samples = [s for s in samples if not ( (-0.001 < float(s[3])  < 0.001) and (random.randint(1,5) != 1) ) ]
# print("Samples CSV some near zero angles filtered len={}".format(len(filtered_samples)))

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

# image preprocessing function
# - convert to numpy array
# - convert colorspace to YUV
def image_preprocess(image):
    return cv2.cvtColor( np.asarray( image ), cv2.COLOR_BGR2HSV)

### Set up generator
print("Generator setup ...")

def image_load(images_dir, image_file):
    image_file = image_file.strip()
    if ( image_file[0] == '/' ):
        image_path = image_file
    else:
        image_path = images_dir + '/' + image_file
    return PIL.Image.open( image_path )
    
#
# flip=True to flip images
# left_right=True to use left and right camera images with angle adjustment
#
def generator(samples, batch_size=32, images_dir = images_dir, left_right=False, flip=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            angle_adjust = 0.2 # amount of correction for left and right cameras
            center_index = 0
            left_index=1
            right_index=2
            angle_index=3
            
            for batch_sample in batch_samples:
                # use PIL instead of OpenCV cv2.imread to read as RGB not cv2 BGR. This is consistent with drive.py which uses PIL
                
                center_image = image_load( images_dir, batch_sample[center_index] )
                center_angle = float(batch_sample[angle_index])
                    
                images.append( image_preprocess( center_image ) )
                angles.append(center_angle)
                
                # flip center image vertically
                if flip:
                    image_flipped = np.fliplr(center_image)
                    angle_flipped = -center_angle
                    images.append( image_preprocess( image_flipped ) )
                    angles.append( angle_flipped )      
                              
                # add in left and right images
                if left_right:
                    # left image, adjust angle to center
                    left_image = image_load( images_dir, batch_sample[ left_index ] )
                    left_angle = float(batch_sample[angle_index]) + angle_adjust # if car on left side, want to go more right
                    images.append( image_preprocess( left_image ) )
                    angles.append( left_angle )
                    
                    # right image, adjust angle to center
                    right_image = image_load( images_dir, batch_sample[ right_index ] )
                    right_angle = float(batch_sample[angle_index]) - angle_adjust # if car on right side, want to go more left
                    images.append( image_preprocess( right_image ) )
                    angles.append( right_angle )
                    
                    # flip left and right images
                    if flip:
                        image_flipped = np.fliplr(left_image)
                        angle_flipped = -left_angle
                        images.append( image_preprocess( image_flipped ) )
                        angles.append( angle_flipped )
                        
                        image_flipped = np.fliplr(right_image)
                        angle_flipped = -right_angle
                        images.append( image_preprocess( image_flipped ) )
                        angles.append( angle_flipped )
                
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # crop images to remove upper horizontal band above horizon (road) and lower horizontal band that includes the hood of car
            # results in 64x128
            X_train = X_train[:, 60:124, 0:320, :]
            
            # resize to 64x64
            #http://stackoverflow.com/questions/40201846/resizing-ndarray-of-images-efficiently
            X_train=zoom(X_train,zoom=(1,1,64./320,1),order=1)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            

# Setup generators for valid and train data sets
train_generator = generator(train_samples, batch_size=128, flip=True, left_right=True)
validation_generator = generator(validation_samples, batch_size=128, flip=True, left_right=True)

print("Start training ...")

# augmentation_factor to add augmented images including flipped and left/right images
augmentation_factor = 6

history = model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples)*augmentation_factor, 
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples)*augmentation_factor, 
    nb_epoch=epochs)
	
#
# plot history for loss
#
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.draw()

# Save trained model architecture as model.h5
model.save('model.h5')

print('model saved')
plt.show()