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
	
	# Crop image: remove 64 pixels from top and 124 pixels from the bottom.
	# output shape (64, 320, 3)
	###Cropping2D( cropping=( (60,124), (0,0) ), input_shape=(160,320,3) ),
	Cropping2D( cropping=( (60,124), (96,96) ), input_shape=(160,320,3) ),
	
	# Resize images contained in a 4D tensor of shape [batch, height, width, channels] (for 'tf' dim_ordering) 
	# Lambda(lambda x: K.resize_images(x, 64, 64, dim_ordering='tf')),
	
	# Scale to range with magnitude of 1.0 and Normalize to mean of zero
	# resulting range -0.5 to 0.5
	Lambda(lambda x: (x / 255.0) - 0.5),
	
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

# For keras plot to work, need:
# sudo pip install pydot-ng
# brew install graphviz
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#
# Model Training
#

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=128)

# Save your trained model architecture as model.h5 using model.save('model.h5').
model.save('model.h5')