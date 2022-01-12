import numpy as np
import pandas as pd
from PIL import Image
from keras.datasets import mnist
import cv2
import os
import math

# // CNN Program Structure Overview  //
# A structure similar to the VGGNet [Simonyan and Zisserman, 2014] will be used
# The layer order will be as follows:
# Conv1, Conv2, Pool1, Conv3, Conv4, Pool2, Conv5, Conv6, Conv7, Pool3, FC1, FC2, Output
# Both Mini - Batch SGD and a model ensemble (N = 5) will be used


def main():
    # Load in data from Keras datasets (type() = ndarray)
    # Note: train_X = 60000 images, test_X = 10000 images
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    # Convert from int arrays to float arrays
    train_X = train_X.astype(float)
    test_X = test_X.astype(float)
    
    # Pre - Processing data
    train_X = preprocess(train_X)
    test_X = preprocess(test_X)
    
   
    # Model Ensemble Loop
    for i in range (5):
        # Initialize filter, weight, and bias matrices
        
        # Loop through mini batches until the loss plateaus
        #while (loss_improvement > 0.05):
        while (True):
            # Randomly sample 64 images (i.e. mini batch) from training data
            data_batch = sample_data (train_X, 64)
            for j in data_batch:
                input = train_X[j]
            
                # Forward Pass
                # - Input Layer: 28 x 28 image
                # - Going to use a stride size, S = 1
                # - The number of filters is going to follow a 8,16,32 pattern (modeled after the
                #   VGG model, doubling between convolutions) so that K = 8 then K = 16, etc.
                # - There will be (F-1)/2 or P = 1 amount of zero padding
                # - The spacial extent of each filter will be 3x3, i.e. F = 3
                # - ReLU will be the activation function applied throughout
                # - Max Pooling will be the pooling function used with a 2x2 spacial extent
            

        
                # Backward Pass
        
                # Update parameters
        
        # Store filter, weight, and bias matrices in an array
        # Add array to a dictionary with index i
        
def sample_data (data, size):
    # Generate size integers randomly between 0 and len(data) - 1
    arr_random = np.random.randint(0,len(data),size)
    return arr_random
    
def preprocess(data):
    # Normalize the pixel values across images
    for x in range(len(data)):
        data[x] = data[x] / 255
    
    # Zero center data by subtracting the mean pixel value from all pixels
    pixel_sum = 0
    for image in data:
        pixel_sum = pixel_sum + np.average(image)
        
    pixel_mean = pixel_sum / len(data)
    
    for x in range(len(data)):
        data[x] = data[x] - pixel_mean
    
    return data
        
    
if (__name__ == "__main__"):
    main()
