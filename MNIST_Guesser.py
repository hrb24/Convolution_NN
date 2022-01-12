import numpy as np
import pandas as pd
from keras.datasets import mnist
from scipy import signal
from scipy import misc
import cv2
import os
import math

# // CNN Program Structure Overview  //
# A structure similar to the VGGNet [Simonyan and Zisserman, 2014] will be used
# The layer order will be as follows:
# Conv1, Conv2, Pool1, Conv3, Conv4, Conv5, Pool2, FC1, FC2, Output
# Both Mini - Batch SGD and a model ensemble (N = 5) will be used to improve accuracy
# A ReLU activation function will be used as will Max Pooling
# Note: This program assumes the images are 28x28 and BW scale


def main():
    
    
    A = np.ones((3,3))
    B = np.pad(A,((1,1),(1,1)), 'constant')
    print(A)
    print(B)
    
    print("")
    C = B[0:3,0:3]
    print(C)


    # Load in data from Keras datasets (type() = ndarray)
    # Note: train_x = 60000 images, test_x = 10000 images
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    
    # Convert from int arrays to float arrays
    train_x = train_x.astype(float)
    test_x = test_x.astype(float)
    
    # Pre - Processing data
    train_x = preprocess(train_x)
    test_x = preprocess(test_x)
    
    # Hyper Parameter Initialization
    # Stride size, S
    # The number of filters involved in each convolution, Ki, is going to follow a doubling
    # patterning, modeled after the VGG model, doubling between convolutions
    # The spacial extent of each filter, F
    # There will be (F-1)/2 amount of zero padding
    # For fully connected layers, decide on lengths of X (input), H1, H2, and Out
    S = 1
    K1 = 8
    K2 = 16
    F = 3
    P = (F-1)/2
    num_Nodes_X = 784
    num_Nodes_H1 = 128
    num_Nodes_H2 = 64
    num_Nodes_Out = 10
    
    # Model Ensemble Loop
    for i in range (5):
        # Initialize filters, weights, and bias matrices
        FL1 = {}
        FL2 = {}
        FL3 = {}
        FL4 = {}
        FL5 = {}
        
        for i in range(K1):
            FL1[i] = np.random.randn(F, F) / np.sqrt((F*F)/2)
            FL2[i] = np.random.randn(F, F) / np.sqrt((F*F)/2)
        
        for j in range(K2):
            FL3[j] = np.random.randn(F, F) / np.sqrt((F*F)/2)
            FL4[j] = np.random.randn(F, F) / np.sqrt((F*F)/2)
            FL5[j] = np.random.randn(F, F) / np.sqrt((F*F)/2)
            
        W1 = np.random.randn(X, num_Nodes_H1) / np.sqrt(X/2)
        W2 = np.random.randn(num_Nodes_H1, num_Nodes_H2) / np.sqrt(num_Nodes_H1/2)
        W3 = np.random.randn(num_Nodes_H2, num_Nodes_Out) / np.sqrt(num_Nodes_H2/2)
        b1 = np.array(np.random.uniform(-0.5,0.5,num_Nodes_H1)).reshape(1, num_Nodes_H1)
        b2 = np.array(np.random.uniform(-0.5,0.5,num_Nodes_H2)).reshape(1, num_Nodes_H2)
        b3 = np.array(np.random.uniform(-0.5,0.5,num_Nodes_Out)).reshape(1, num_Nodes_Out)
        
        # Loop through mini batches until the loss converges
        #while (loss_improvement > 0.05):
        while (True):
            # Randomly sample 64 images (i.e. mini batch) from training data
            data_batch = sample_data (train_x, 64)
            for j in data_batch:
                input = train_x[j]
                # Forward Pass
                # Conv1
                
                
            

        
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
    
def max_pooling ():
    # Max Pooling will be the pooling function used with a 2x2 spacial extent
    return 0
    
def convolve (input, filter, filter_size, stride, pad_amount, pad_value):
    # Save the original input dimensions
    dim = len(input)
    print("28 = ",dim)
    
    # Zero pad the image
    input = np.pad(input, ((pad_amount, pad_amount), (pad_amount, pad_amount)), 'constant')

    # Convolve the image using filter
    convolution = np.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            convolution[i,j] = scipy.signal.convolve2d(input[0+i:filter_size+i, 0+j:filter_size+j], filter, mode='full', boundary='fill', fillvalue=0)
    A = np.ones((3,3))
    B = np.pad(A,((1,1),(1,1)), 'constant')
    print(A)
    print(B)
    
    print("")
    C = B[0:3,0:3]
    print(C)
    scipy.signal.convolve2d(input, in2, mode='full', boundary='fill', fillvalue=0)
    
    # Pass convolution to ReLU
    

    return convolution

def ReLU(x):
    x = (x > 0) * x
    return(x)

        
    
if (__name__ == "__main__"):
    main()
