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
    # For printing, change precision to 3 decimals and suppress scientific notation
    np.set_printoptions(precision =3, linewidth = 250, suppress = True, threshold = np.inf)

    A =  np.array([[1, 2, 1], [4, 5, 0], [0,1,1]])
    A = np.array([A])
    B =  np.array([[0, 1,3 ], [6, 7,8],[4, 5, 0]])
  
    F1 = np.array([[0,1,0 ], [0,0,1],[1, 1, 1]])
    F2 = np.array([[1,1,0 ], [1,0,1],[1, 0, 0]])
    C = np.array([F1,F2])
    
    FM1 = np.empty([len(C), len(A[0]), len(A[0])])
    for i in range(len(C)):
        print("i: ",i)
        print("C[i]: ",C[i])
        print("C[i].shape: ", C[i].shape)
        H = convolve (A, C[i], 1)
        print("H",H)
        FM1[i] = H
    
    print("FM1: ",FM1)
    
    M1 = np.array([[1, 1, 1, 0], [1,1, 5, 0], [6,0,1,1], [7,0,2,1]])
    M2 = np.array([[1, 3, 4, 0], [1,2, 5, 9], [0,0,0,1], [5,0,0,3]])
    M3 = np.array([[0, 2, 9, 0], [1,1, 3, 2], [5,4,0,0], [0,0,3,0]])
    print("M1: ",M1)
    print("M2: ",M2)
    print("M3: ",M3)
    
    TP = np.array([M1,M2,M3])
    print("TP: ",TP)
    
    PI = max_pooling(TP)
    print("PI: ",PI)
    
    
    

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
    # Stride size, S = 1, is assumed throughout
    # The number of filters involved in each convolution, Ki, is going to follow a doubling
    # patterning, modeled after the VGG model, doubling between convolutions
    # The spacial extent of each filter, F
    # There will be (F-1)/2 amount of zero padding
    # For fully connected layers, decide on lengths of X (input), H1, H2, and Out
    K1 = 8
    K2 = 16
    F = 3
    P = int((F-1)/2)
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
            
        W1 = np.random.randn(num_Nodes_X, num_Nodes_H1) / np.sqrt(num_Nodes_X/2)
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
                print("j: ",j)
                print("train_y[j]: ",train_y[j])
                input = np.array([train_x[j]])
                print("input: ",input)
                # Forward Pass
                # Conv1
                FM1 = np.empty([len(FL1), len(input[0]), len(input[0])])
                for i in range(len(FL1)):
                    FM1[i] = convolve (input, FL1[i], P)
                print("FM1: ",FM1)
                # Conv2
                FM2 = np.empty([len(FL2), len(FM1[0]), len(FM1[0])])
                for i in range(len(FL2)):
                    FM2[i] = convolve (FM1, FL2[i], P)
                print("FM2: ",FM2)
                # Pool1
                PL1 = max_pooling(FM2)
                print("PL1: ",PL1)
                # Conv3
                FM3 = np.empty([len(FL3), len(PL1[0]), len(PL1[0])])
                for i in range(len(FL3)):
                    FM3[i] = convolve (PL1, FL3[i], P)
                print("FM3: ",FM3)
                # Conv4
                FM4 = np.empty([len(FL4), len(FM3[0]), len(FM3[0])])
                for i in range(len(FL4)):
                    FM4[i] = convolve (FM3, FL4[i], P)
                print("FM4: ",FM4)
                # Conv5
                FM5 = np.empty([len(FL5), len(FM4[0]), len(FM4[0])])
                for i in range(len(FL5)):
                    FM5[i] = convolve (FM4, FL5[i], P)
                print("FM5: ",FM5)
                # Pool2
                PL2 = max_pooling(FM5)
                
                print("PL2: ",PL2)
                print("PL2.shape: ",PL2.shape)
                
                
                
                    
                    
                            
                    
                
                
                
                
                
            

        
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
    # Calculate mean pixel value
    pixel_sum = 0
    for image in data:
        pixel_sum = pixel_sum + np.average(image)
    pixel_mean = pixel_sum / len(data)
    # Subtract
    for x in range(len(data)):
        data[x] = data[x] - pixel_mean
    
    return data
    
    
def max_pooling (input):
    # Max Pooling will be the pooling function used with a 2x2 spacial extent
    # Calculate the dimension of the new, pooled, array
    dim = int(len(input[0])/2)
    pooled_arrays = np.empty([len(input), dim, dim])
    for i in range(len(input)):
        pooled_array = np.zeros((dim, dim))
        for j in range(dim):
            for k in range(dim):
                pooled_array[j,k] = np.max(input[i][0+(2*j):2+(2*j), 0+(2*k):2+(2*k)])
        pooled_arrays[i] = pooled_array
        
    return pooled_arrays
    
    
def convolve (input, filter, pad_amount):
    # Note: The parameter input is an array of one or more arrays
    # Save the original input array dimensions (height and width)
    dim = len(input[0])
    # Zero pad the images by first creating an empty resized array and then populating it
    pad_input = np.empty([len(input), dim+(2*pad_amount), dim+(2*pad_amount)])
    for i in range(len(input)):
        pad_input[i] = np.pad(input[i], ((pad_amount, pad_amount), (pad_amount, pad_amount)), 'constant')
    # Convolve the image using filter
    convolution = np.zeros((dim, dim))
    for i in range(len(pad_input)):
        for j in range(dim):
            for k in range(dim):
                convolution[j,k] = convolution[j,k] + np.sum(pad_input[i][0+j:len(filter)+j, 0+k:len(filter)+k] * filter)
    # Add the bias
    convolution = convolution + 0.05
    # Pass convolution to ReLU and return
    return Leaky_ReLU(convolution)


def Leaky_ReLU(x):
    x = np.maximum(0.1* x, 0)
    return x

        
    
if (__name__ == "__main__"):
    main()
