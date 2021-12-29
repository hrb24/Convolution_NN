import numpy as np
import pandas as pd
from PIL import Image
from keras.datasets import mnist
import cv2
import os
import math

def main():
    
    # Load in data from Keras datasets
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
   
    # Randomly generate a kernal and bias
    kernelDim = 5
    maxPoolDim = 4
    #kernel = np.array(np.random.uniform(0,1,kernelDim * kernelDim).reshape(kernelDim, kernelDim))
    #bias = np.random.uniform(-10, 10)
    bias = -0.5
    kernel = np.array([[0,1,1,0,1],[1,1,1,1,0],[0,0,0,1,1],[1,0,1,1,0],[1,0,0,1,1]])
  

    # Initialize "parameters" here for constant access throughout program
    # Note: Images are assumed to be 28 x 28
    # Therefore, the input to the NN will have 28 - 5 + 1)^2 = 36 nodes
    numNodesHidden = 2
    numEpochs = 5
    learnRate = 0.9
    inputNodes = 36
    
    # Generate the initial arch weight and bias matrixes
    # Note: W1 will be a 36 x numNodesHidden matrix and W2 will be a numNodesHidden x 10 matrix
    # Note: B1 will be the bias for the hidden layer and B2 will be bias for output layer
    W1 = np.array(np.random.uniform(-0.5,0.5,numNodesHidden * inputNodes)).reshape(inputNodes, numNodesHidden)
    W2 = np.array(np.random.uniform(-0.5,0.5,numNodesHidden * 10)).reshape(numNodesHidden, 10)
    B1 = np.array(np.random.uniform(-0.5,0.5,numNodesHidden)).reshape(1, numNodesHidden)
    B2 = np.array(np.random.uniform(-0.5,0.5,10)).reshape(1, 10)
    
    # We will also create an empty (initialized to zero) np.array for the input layer
    inputLayer = np.zeros(inputNodes)
    
    # Loop through epochs
    
    # Loop through each of the training images
    
    # 1. Apply kernal to create convolution (also apply Relu and Sigmoid functions)
    inputImage = train_X[1] / 255

    convolution = convolution_forward(inputImage, kernel, bias)
    print(convolution)
    
    # 2. Use max pool to reduce size of convolution
    maxConvolution = maxPool_forward(convolution, maxPoolDim)
    print(maxConvolution)
    
    # 3. Flatten
    neuralInput = flatten_foward(maxConvolution)
    print(neuralInput)

    # 4. Pass into Neural Network. Feed forward and then back propogate
    # Feed Forward
    hiddenLayerInput = (neuralInput @ W1) + B1
    hiddenLayerOutput = 1/(1 + np.exp(-hiddenLayerInput))
    
    print("hiddenLayerOutput: ",hiddenLayerOutput)
    print("W2: ",W2)
    outputLayerInput = (hiddenLayerOutput @ W2) + B2
    outputLayerOutput = 1/(1 + np.exp(-outputLayerInput))
    
    # Back propagation
    # Start by converting the true value into a 1 x 10 matrix
    labelReshape = np.zeros(outputLayerOutput.shape[1])
    labelReshape[train_y[1]] = 1
    
    onesMatrix = np.ones(outputLayerOutput.shape[1])

    # Calculate error at output nodes
    errorOutput = outputLayerOutput * (onesMatrix - outputLayerOutput) * (labelReshape - outputLayerOutput)
    
    print("errorOutput: ",errorOutput)
    
    # Calculate error at hidden nodes
    print("hiddenLayerOutput: ",hiddenLayerOutput)
    
    errorHidden = hiddenLayerOutput * (np.ones((1, numNodesHidden)) - hiddenLayerOutput) * np.transpose((np.transpose(hiddenLayerOutput)@ np.ones((1, 10))) @ np.transpose(errorOutput))
    
    print("errorHidden: ",errorHidden)
    print("errorOutput: ", errorOutput)
    
    
    # Calculate change in weights and bias
    deltaW1 = learnRate * np.outer(inputLayer, errorHidden)
    deltaW2 = learnRate * np.outer(hiddenLayerOutput, errorOutput)
    deltaB1 = learnRate * errorHidden
    deltaB2 = learnRate * errorOutput
    
                    
    # Update weight and bias matrices
    W1 = W1 + deltaW1
    W2 = W2 + deltaW2
    B1 = B1 + deltaB1
    B2 = B2 + deltaB2
    
    
    
    
               
                
            
    
    
    
    
    
    
def ReLu(x):
    x = (x > 0) * x
    return(x)
    
def convolution_forward(inputImage, kernel, bias):
    # Calculate the convolution matrix, z, from the original image using the
    # kernal and bias matrices
    
    # Begin by passing a single image through the convolutional layer
    # Assume:
    # 1. Each image is black and white (i.e. not 3 (RGB) layers)
    # 2. There is only one kernel matrix
    z_row = inputImage.shape[0] - kernel.shape[0] + 1
    z_col = inputImage.shape[1] - kernel.shape[1] + 1
    z = np.empty((z_row,z_col))
    for i in range(0,z_row):
        for j in range(0,z_col):
            subImage = inputImage[i:i+kernel.shape[0],j:j+kernel.shape[1]]
            z[i,j] = np.sum(subImage * kernel) + bias
    
    a = ReLu(z)
    return(a)
    
def maxPool_forward(convolution, maxPoolDim):
    # Populate a new matrix with the max value of each unique 2x2 matrix in the convolution
    m_row = int(convolution.shape[0] / maxPoolDim)
    m_col = int(convolution.shape[1] / maxPoolDim)
    m = np.empty((m_row,m_col))
    for i in range(0,m_row):
        for j in range(0,m_col):
            subConvolution = convolution[i*maxPoolDim : (i*maxPoolDim) + maxPoolDim + 1, j*maxPoolDim : (j*maxPoolDim) + maxPoolDim + 1]
            m[i,j] = np.amax(subConvolution)

    return(m)

def flatten_foward(inputFlatten):
    # Flatten the max pool matrix for use as an input layer to the neural network
    array1D = inputFlatten.flatten()
    return(array1D)
    
if (__name__ == "__main__"):
    main()
