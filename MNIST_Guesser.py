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
# A Leaky ReLU activation function will be used as will Max Pooling
# Note: This program assumes the images are 28x28 and BW scale


def main():
    # For printing, change precision to 3 decimals and suppress scientific notation
    np.set_printoptions(precision =3, linewidth = 250, suppress = True, threshold = np.inf)
    
    A = np.array([[2],[2],[0],[2]])
    print("A.shape: ",A.shape," A: ",A)
    

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
            
        W1 = np.random.randn(num_Nodes_H1, num_Nodes_X) / np.sqrt(num_Nodes_X/2)
        W2 = np.random.randn(num_Nodes_H2, num_Nodes_H1, ) / np.sqrt(num_Nodes_H1/2)
        W3 = np.random.randn(num_Nodes_Out, num_Nodes_H2) / np.sqrt(num_Nodes_H2/2)
        b1 = np.array(np.random.uniform(-0.5,0.5,num_Nodes_H1)).reshape(num_Nodes_H1, 1)
        b2 = np.array(np.random.uniform(-0.5,0.5,num_Nodes_H2)).reshape(num_Nodes_H2, 1)
        b3 = np.array(np.random.uniform(-0.5,0.5,num_Nodes_Out)).reshape(num_Nodes_Out, 1)
        
        
        # Loop through mini batches until the loss converges
        #while (loss_improvement > 0.05):
        while (True):
            # Randomly sample 64 images (i.e. mini batch) from training data
            data_batch = sample_data (train_x, 64)
            for j in data_batch:
                input = np.array([train_x[j]])
                # Forward Pass
                # Conv1
                FM1 = np.empty([len(FL1), len(input[0]), len(input[0])])
                for i in range(len(FL1)):
                    FM1[i] = convolve (input, FL1[i], P)
                # Conv2
                FM2 = np.empty([len(FL2), len(FM1[0]), len(FM1[0])])
                for i in range(len(FL2)):
                    FM2[i] = convolve (FM1, FL2[i], P)
                # Pool1
                PL1 = max_pooling(FM2)
                # Conv3
                FM3 = np.empty([len(FL3), len(PL1[0]), len(PL1[0])])
                for i in range(len(FL3)):
                    FM3[i] = convolve (PL1, FL3[i], P)
                # Conv4
                FM4 = np.empty([len(FL4), len(FM3[0]), len(FM3[0])])
                for i in range(len(FL4)):
                    FM4[i] = convolve (FM3, FL4[i], P)
                # Conv5
                FM5 = np.empty([len(FL5), len(FM4[0]), len(FM4[0])])
                for i in range(len(FL5)):
                    FM5[i] = convolve (FM4, FL5[i], P)
                # Pool2
                PL2 = max_pooling(FM5)
                # Flatten and reshape PL2 for input to the FC layer
                FC_input = PL2.flatten()
                FC_input = np.reshape(FC_input, (len(FC_input), 1))
                print("FC_input.shape: ", FC_input.shape)
                
                # Forward pass using Leaky_ReLU activation function and dropout
                p = 0.5 # Probability of keeping a unit active. Higher = less dropout
                H1 = Leaky_ReLU(np.dot(W1,FC_input) + b1)
                print("W1.shape: ",W1.shape)
                print("H1.shape: ",H1.shape)
                U1 = (np.random.rand(*H1.shape) < p) / p
                H2 = Leaky_ReLU(np.dot(W2,H1) + b2)
                print("W2.shape: ", W2.shape)
                print("H2.shape: ",H2.shape)
                U2 = (np.random.rand(*H2.shape) < p)
                out = Leaky_ReLU(np.dot(W3, H2) + b3)
                print("out.shape: ",out.shape)
                print("W3.shape: ", W3.shape)
                # Pass output to soft max loss function to generate error
                #error = soft_max(out)
                #print(error)
    
                # Back Propagate
                # Create a "true" array
                expected_arr = np.zeros(out.shape)
                expected_arr[train_y[j]] = 1
                # Calculate error of output layer
                err_output = np.empty(out.shape)
                for i in range(len(out)):
                    if (out[i] < 0):
                        err_output[i] = 0.1 * expected_arr[i]
                    else:
                        err_output[i] = 1 * expected_arr[i]

                print("err_output.shape: ",err_output.shape)
                #errOutput = outputLayerOutput * (1 - outputLayerOutput) * (row['tag'] - outputLayerOutput)
                
                # Calculate error of hidden layer
                #I = np.identity(numNodesHidden)
                #D = np.identity(numNodesHidden) * np.outer(np.ones(numNodesHidden), hiddenLayerOutput)
                
                
                #errHidden = hiddenLayerOutput @ (I-D) @ (np.identity(numNodesHidden) * np.outer(np.ones(numNodesHidden), errOutput * W2))
                
                
                # Calculate change in weights and bias
                #deltaW1 = learnRate * np.outer(inputLayer, errHidden)
                #deltaW2 = learnRate * np.outer(hiddenLayerOutput, errOutput)
                #deltaB1 = learnRate * errHidden
                #deltaB2 = learnRate * errOutput
                
                # Update weight and bias matrices
               # W1 = W1 + deltaW1
                #W2 = W2 + deltaW2
                #B1 = B1 + deltaB1
                #B2 = B2 + deltaB2

    # Backward pass: compute gradients (not shown)
    # Perform parameter update (not shown)
                
                
                
                
                
                
                    
                    
                            
                    
                
                
                
                
                
            

        
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
    # Pass convolution to Leaky ReLU and return
    return Leaky_ReLU(convolution)


def Leaky_ReLU(x):
    x = np.maximum(0.1*x, x)
    return x

#def soft_max(input):
    #exp = np.exp(input)
    #a = (exp / np.sum(exp))
    #print("a: ",a)
    #return(exp / np.sum(exp))
        
    
if (__name__ == "__main__"):
    main()
