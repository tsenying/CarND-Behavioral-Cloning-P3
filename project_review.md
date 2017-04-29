# Project Review: Use Deep Learning to Clone Driving Behavior
A part of the Self Driving Car Engineer Nanodegree Program

 
**Meets Specifications**

Great job! Congratulations and good luck in your next project!

## Required Files

*The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.*

## Quality of Code

*The model provided can be used to successfully operate the simulation.*

Good job here! Provided model can be used to successfully run the simulation!

*The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.*

Good job using yield generators to generate data for training rather than storing all data in memory!
Awesome comments throughout all project!

## Model Architecture and Training Strategy

*The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.*

- (+)multiple 2D convolution layers are used
- (+)nonlinearity applied using ELU.
  Here is a good article about it if you are interested:
  https://arxiv.org/pdf/1511.07289v1.pdf  
  https://arxiv.org/pdf/1605.09332v3.pdf  
- (+)data is normalized

As an enhancement you can also normalize layers:  
  https://keras.io/layers/normalization/  
  https://arxiv.org/abs/1502.03167  
This technique can increase learning time and overall performance.

*Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.*

Good job with training/validation/testing data! Nice discussion about dropout and overfitting!

Here is also more info about dropout if you are interested:  
http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/  
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf  

Here are more information about train/validation/test splits:  
http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set  
http://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio  

*Learning rate parameters are chosen with explanation, or an Adam optimizer is used.*
Adam optimizer is used!

Here is an excellent article about different gradient descent optimization algorithms:  
http://sebastianruder.com/optimizing-gradient-descent/index.html  

*Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).*

## Architecture and Training Documentation

*The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.*

*The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.*

Good start with README!

*The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.*

## Simulation

*No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).*

Everything is good with your model on my hardware - car drive well all the track!
