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
    # train_X = 60000 images, test_X = 10000 images
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print("train_X", train_X[0])
    
    # Pre - processing
    # Normalize data by dividing all pixel values by 255
    # Zero center data by subtracting the mean pixel value from all pixels
    
    # Model Ensemble Loop
    for i in range (5):
        # Initialize filter, weight, and bias matrices
        
        # Loop through mini batches until the loss plateaus
        #while (loss_improvement > 0.05):
        while (True):
            # Randomly sample 64 images (i.e. mini batch) from training data
            data_batch = sample_data (train_X, 64)
            for j in data_batch:
                print(j)
    
            # Forward Pass
            # - Input Layer: 28 x 28 image
            # - Going to use a stride size, S = 1
            # - The number of filters is going to follow a 8,16,32 pattern (modeled after the
            #   VGG model, doubling between convolutions) so that K = 8 then K = 16, etc.
            # - There will be (F-1)/2 or P = 1 amount of zero padding
            # - The spacial extent of each filter will be 3x3, i.e. F = 3
            # - ReLU will be the activation function applied throughout
            # - Max Pooling will be the pooling function used with a 2x2 spacial extent
            
            
            #weights_grad = evaluate_gradient (loss_fun, data_batch, weights)
            #weights += - step_size * weights_grad # perform parameter update

        
            # Backward Pass
        
            # Update parameters
        
        # Store filter, weight, and bias matrices in an array
        # Add array to a dictionary with index i
        
def sample_data (data, size):
    arr_random = np.random.randint(1,len(data) + 1,size)
    return arr_random
    
   
    
if (__name__ == "__main__"):
    main()
