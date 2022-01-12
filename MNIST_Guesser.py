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
    
    # Pre - processing
    # Normalize data by dividing all pixel values by 255
    # Zero center data by subtracting the mean pixel value from all pixels
    
   
    
if (__name__ == "__main__"):
    main()
