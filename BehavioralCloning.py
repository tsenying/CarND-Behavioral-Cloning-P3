
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')


# In[3]:

# Load data
driving_log_csv = pd.read_csv("../ud-sim-data/driving_log.csv")

print("Number of rows: %d" % len(driving_log_csv))

driving_log_csv.head()


# In[4]:

# Paths of center images
X_path = [driving_log_csv.loc[i]["center"] for i in range(len(driving_log_csv))]

# steering angle for images
y = [driving_log_csv.loc[i]["steering"] for i in range(len(driving_log_csv))]


# In[7]:

print("X_path samples={}".format(X_path[0:3]))
print("Steering angle samples={}".format(y[0:3]))


# In[8]:

# Load images
X_images = [mpimg.imread("../ud-sim-data/" + path) for path in X_path]


# In[11]:

# Examine image data
print("X_images len={}".format( len(X_images)))

print("Image shape: ", X_images[0].shape)

plt.imshow(X_images[0])


# In[30]:

img_0 = X_images[0]
crop_img = img_0[60:124, 0:320] # Crop from x, y, w, h -> 100, 200, 300, 400
print("crop_img.shape={}".format(crop_img.shape))
plt.imshow( crop_img )


# In[33]:

import cv2
resize_img = cv2.resize(crop_img,(64, 64), interpolation = cv2.INTER_CUBIC)
plt.imshow( resize_img )


# In[94]:

print(type(X_images))
print(len(X_images))
print(type(X_images[0]))
print(X_images[0].shape)

print(X_images[11].shape)
X = np.array(X_images[0:])
print(type(X))
print(X.shape)


# In[59]:

crops = np.array(X_images[0:3])
crops= crops[:, 60:124, 0:320, :]
print(type(crops))
print(crops.shape)
plt.imshow( crops[2] )


# In[ ]:

# save training data into pickle file
# import pickle

# train_data = { 'center_images': X_images, 'steering_angle': y}

# pickle.dump( train_data, open( "train.p", "wb" ) )

# loaded_train_data = pickle.load( open( "train.p", "rb" ) )
# print(loaded_train_data)

# plt.imshow(loaded_train_data['center_images'][2])

