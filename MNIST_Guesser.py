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
# Conv1, Conv2, Pool1, Conv3, Conv4, Conv5, Pool2, FC1, FC2, Output
# Both Mini - Batch SGD and a model ensemble (N = 5) will be used


def main():
    
    # Load in data from Keras datasets
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    # Pre - processing
    # Normalize data by dividing all pixel values by 255
    # Zero center data by subtracting the mean pixel value from all pixels
    
    # Model Ensemble Loop
    for i in range (5):
        # Initialize filter, weight, and bias matrices
        
        # Loop through mini batches until the loss plateaus
        while (loss_improvement > 0.05):
            # Generate mini batch
    
            # Forward Pass
        
            # Backward Pass
        
            # Update parameters
        
        # Store filter, weight, and bias matrices in an array
        # Add array to a dictionary with index i
        
    
        
    
    

    
    
   
    
if (__name__ == "__main__"):
    main()
