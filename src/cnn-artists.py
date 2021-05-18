#!/usr/bin/env python
"""
Info: This script builds a deep learning model using LeNet as the convolutional neural network architecture. This network is used to classify impressionist paintings by their artists. 

Parameters:
    (optional) train_data: str <name-of-training-data-directory>, default = "training_subset"
    (optional) test_data: str <name-of-validation-data-directory>, default = "validation_subset"
    (optional) n_epochs: int <number-of-epochs>, default = 20
    (optional) batch_size: int <size-of-batches>, default = 32
    (optional) output_filename: str <name-of-output-file>, default = "cnn_classification_report.txt"

Usage:
    $ python cnn-artists.py
    
Output:
    - model_summary.txt: a summary of the model architecture.
    - LeNet_model.png: a visual overview of the model architecture.
    - model_loss_accuracy_history.png: a visual representation of the loss/accuracy performance of the model during training. 
    - cnn_classification_report.txt: Neural network classification metrics.
"""

### DEPENDENCIES ###

# Core libraries
import os
import sys
sys.path.append(os.path.join(".."))

# numpy, matplotlib, openCV, glob, contextlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
from contextlib import redirect_stdout

# argparse
import argparse

# Scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to training data
    ap.add_argument("-t", "--train_data",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Path to the training data",
                    default = "training") # default is a subset of the training dataset
    
    # Argument 2: Path to test data
    ap.add_argument("-te", "--test_data",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Path to the validation data",
                    default = "validation") # default is a subset of the validation dataset
    
    # Argument 3: Number of epochs
    ap.add_argument("-e", "--n_epochs",
                    type = int,
                    required = False, # the argument is not required 
                    help = "The number of epochs to train the model on",
                    default = 20) # default number of epochs is set to 20
    
    # Argument 4: Batch size
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False, # the argument is not required 
                    help = "The size of the batches on which to train the model",
                    default = 32) # default batch size is 32
    
    # Argument 5: Output filename
    ap.add_argument("-o", "--output_filename",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Define the name of the output file",
                    default = "cnn_classification_report.txt") # default output filename
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    train_data = os.path.join("..", "data", "impressionist_classifier_data", args["train_data"])
    test_data = os.path.join("..", "data", "impressionist_classifier_data", args["test_data"])
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    output_filename = args["output_filename"]
    
    # Create output directory if it does not already exist
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Start message
    print("\n[INFO] Initializing the construction of a LeNet convolutional neural network model...")
    
    # Instantiate the CNN_classifier class
    cnn_classifier = CNN_classifier(train_data, test_data)
    
    # Create list of label names from the directory names in the training data folder
    label_names = cnn_classifier.list_labels()

    # Find the minimum dimensions to resize images 
    print("\n[INFO] Estimating the minimum image dimensions to resize images...")
    min_dimension = cnn_classifier.find_min_dimensions(label_names)
    print(f"\n[INFO] Input images will be resized to dimensions of height = {min_dimension} and width = {min_dimension}...")
    
    # Create trainX and trainY
    print("\n[INFO] Resizing training images and creating trainX and trainY...")
    trainX, trainY = cnn_classifier.create_XY(train_data, min_dimension, label_names)
    
    # Create testX and testY
    print("\n[INFO] Resizing validation images and creating testX and testY...")
    testX, testY = cnn_classifier.create_XY(test_data, min_dimension, label_names)
    
    # Normalize data and binarize labels
    print("\n[INFO] Normalize training and validation images and binarizing training and validation labels...")
    trainX_norm, trainY_bin, testX_norm, testY_bin = cnn_classifier.normalize_binarize(trainX, trainY, testX, testY)
    
    # Perform data augmentation to create artificial data
    print("\n[INFO] Performing data augmentation to create more data...")
    datagen = cnn_classifier.augment_data(trainX_norm)
    
    # Define model
    print("\n[INFO] Defining LeNet model architecture...")
    model = cnn_classifier.define_LeNet_model(min_dimension)
   
    # Train model
    print("\n[INFO] Training model...")
    model_history = cnn_classifier.train_LeNet_model(model, datagen, trainX_norm, trainY_bin, testX_norm, testY_bin, n_epochs, batch_size)
    
    # Plot loss/accuracy history of the model
    print("\n[INFO] Plotting loss/accuracy history of model and saving plot to 'output' directory...")
    cnn_classifier.plot_history(model_history, n_epochs)
    
    # Evaluate model
    print(f"\n[INFO] Evaluating model... Below is the classification report. These metrics can also be found as {output_filename} in the output folder.\n")
    cnn_classifier.evaluate_model(model, testX_norm, testY_bin, batch_size, label_names, n_epochs, output_filename)
  
    # User message
    print("\n[INFO] Done! Results can be found in the 'output/' folder.\n")
    
    
# Creating Neural network classifier class   
class CNN_classifier:
    
    def __init__(self, train_data, test_data):
        
        # Receive inputs: Image and labels 
        self.train_data = train_data
        self.test_data = test_data
   

    def list_labels(self):
        """
        Method that defines the label names by listing the names of the folders within training directory without listing hidden files. 
        """
        # Create empty list
        label_names = []
    
        # For every name in training directory
        for name in os.listdir(self.train_data):
            # If it does not start with . (which hidden files do)
            if not name.startswith('.'):
                label_names.append(name)
            
        return label_names

    
    def find_min_dimensions(self, label_names):
        """
        Method that estimates the dimensions (height and width) of the smallest image among all images within the training and validation datasets. These dimensions are later used to resize the images. Since all of the images are of different shapes and sizes, we need to resize them to be a uniform, smaller shape, which is exactly what this method does.
        """
        # Create empty list
        dimensions = []
    
        # Loop through directories for each artist
        for name in label_names:
        
            # Take images in both training and validation directories
            all_images = glob.glob(os.path.join(self.train_data, name, "*.jpg")) + glob.glob(os.path.join(self.test_data, name, "*.jpg"))
        
            # Loop through each image
            for image in all_images:
            
                # Load image
                loaded_img = cv2.imread(image)
            
                # Append to dimensions list the dimensions (height and width) of each image
                dimensions.append(loaded_img.shape[0]) # height
                dimensions.append(loaded_img.shape[1]) # width
            
        # Find the minimum value among all image dimensions
        min_dimension = min(dimensions)
    
        return min_dimension


    def create_XY(self, data, min_dimension, label_names):
        """
        Method creates trainX, trainY as well as testX and testY. It creates X, which is an array of images (corresponding to trainX and testX) and Y which is a list of the image labels (corresponding to trainY and testY). Hence, with this we can create the training and validation datasets. 
        """
        # Create empty array, X, for the images, and an empty list, y, for the image labels
        X = np.empty((0, min_dimension, min_dimension, 3))
        Y = []
    
        # For each artist name listed in label_names
        for name in label_names:
        
            # Get all images for each artist
            images = glob.glob(os.path.join(data, name, "*.jpg"))
        
            # For each image in images 
            for image in tqdm(images): # I use tqdm() to allow the user to follow along
        
                # Load image
                loaded_img = cv2.imread(image)
        
                # Resize image to the specified dimensions
                resized_img = cv2.resize(loaded_img, (min_dimension, min_dimension), interpolation = cv2.INTER_AREA) # INTER_AREA means that it is resizing using pixel-area relation which was a suggested method by Ross
        
                # Create array of image
                image_array = np.array([np.array(resized_img)])
        
                # Append to trainX array and trainY list
                X = np.vstack((X, image_array))
                Y.append(name)
        
        return X, Y


    def normalize_binarize(self, trainX, trainY, testX, testY):
        """
        Method that normalizes the training and validation data and binarizes the training and test labels. Normalizing is done by dividing by 255 to compress the pixel intensity values down between 0 and 1 rather than 0 and 255. Binarizing is performed using the LabelBinarizer function from sklearn. We binarize the labels to convert them into one-hot vectors. 
        """
        # Normalize training and test data
        trainX_norm = (trainX - trainX.min())/(trainX.max() - trainX.min()).astype("float")
        testX_norm = (testX - testX.min())/(testX.max() - testX.min()).astype("float")

        # Binarize training and test labels
        lb = LabelBinarizer() # intialize binarizer
        trainY_bin = lb.fit_transform(trainY) # binarizing training image labels
        testY_bin = lb.fit_transform(testY) # binarizing validation image labels
    
        return trainX_norm, trainY_bin, testX_norm, testY_bin
    
    
    def augment_data(self, trainX_norm):
        """
        This method performs data augmentation using the TensforFlow DataGenerator that generates artifical data from the original data in order to provide more data for the model to be trained on. I have tried to make the augmentation as "realistic" as possible to prevent generating data that the model would never encounter. 
        """
        # Initialize the data augmentation object
        datagen = ImageDataGenerator(zoom_range = 0.15, # zooming
                                     width_shift_range = 0.2, # horizontal shift
                                     height_shift_range = 0.2, # vertical shift
                                     horizontal_flip=True) # mirroring image
        
        # Perform the data augmentation
        datagen.fit(trainX_norm)

        return datagen

    
    def define_LeNet_model(self, min_dimension):
        """
        Method that defines the LeNet model architecture.
        """
        # Define sequantial model
        model = Sequential()

        # Add first set of convolutional layer, ReLu activation function, and pooling layer
        # Convolutional layer
        model.add(Conv2D(32, (3, 3), # 32 filters/kernels of size 3x3
                         padding="same", # "same" means that we pad the input images evenly to the left/right or up/down so that the output has the same height/width dimension as the input.
                         input_shape=(min_dimension, min_dimension, 3)))
    
        # Activation function
        model.add(Activation("relu")) # using ReLU as activation function
    
        # Max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2), # 2x2 pooling window
                           strides=(2, 2))) # stride of 2 horizontal, 2 vertical
    
        # Add second set of convolutional layer, ReLu activation function, and pooling layer
        # Convolutional layer
        model.add(Conv2D(50, (5, 5), # 50 filters/kernels of size 5x5
                         padding="same")) # once again we use "same" padding to ensure that the input and output have the same dimensions
    
        # Activation function
        model.add(Activation("relu"))
    
        # Max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2), # 2x2 pooling window
                               strides=(2, 2))) # stride = 2 horizontal and 2 vertical
    
        # Add fully-connected layer
        model.add(Flatten()) # flattening layer
        model.add(Dense(500)) # dense network with 500 nodes
        model.add(Activation("relu")) # activation function
    
        # Add output layer, i.e. softmax classifier
        model.add(Dense(10)) # dense layer of 10 nodes used to classify the images
        model.add(Activation("softmax")) # using softmax as the activaiton function since we are dealing with multiple classes (if we only had two classes we would have used the sigmoid function instead) 

        # Define optimizer 
        opt = SGD(lr=0.01)
    
        # Compile model
        model.compile(loss="categorical_crossentropy", # dealing with multiple classes
                      optimizer=opt, 
                      metrics=["accuracy"])
    
        # Save model summary
        output_path_1 = os.path.join("..", "output", "model_summary.txt")
        with open(output_path_1, 'w') as f:
            with redirect_stdout(f):
                model.summary()
    
        # Visualization of model
        output_path_2 = os.path.join("..", "output", "LeNet_model.png")
        plot_LeNet_model = plot_model(model,
                                      to_file = output_path_2,
                                      show_shapes=True,
                                      show_layer_names=True)
    
        return model

    
    def train_LeNet_model(self, model, datagen, trainX_norm, trainY_bin, testX_norm, testY_bin, n_epochs, batch_size):
        """
        Method that trains the LeNet model on the training data and validates it on the validation data.
        """
        # Train model
        model_history = model.fit(datagen.flow(trainX_norm, trainY_bin, batch_size=batch_size), # using a batch size defined by the user (or default batch size of 32)
                                  validation_data=(testX_norm, testY_bin),
                                  epochs=n_epochs, verbose=1) # using number of epochs specified by the user (or the default number which is 20)
    
        return model_history
    
    
    def plot_history(self, model_history, n_epochs):
        """
        Method that plots the loss/accuracy history of the model during training. The code was developed for use in class and modified for this project. 
        """
        # Visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, n_epochs), model_history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, n_epochs), model_history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, n_epochs), model_history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, n_epochs), model_history.history["val_accuracy"], label="val_acc")
        plt.title("Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("..", "output", "model_loss_accuracy_history.png"))
    
    
    def evaluate_model(self, model, testX_norm, testY_bin, batch_size, label_names, n_epochs, output_filename):
        """
        Method that evaluates the trained model and saves the classification report in output directory. 
        """
        # Predictions
        predictions = model.predict(testX_norm, batch_size=batch_size)
    
        # Classification report
        classification = classification_report(testY_bin.argmax(axis=1),
                                               predictions.argmax(axis=1),
                                               target_names=label_names)
            
        # Print classification report
        print(classification)
    
        # Save classification report
        output_path = os.path.join("..", "output", output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Below are the classification metrics for the trained model. Batch size = {batch_size} and number of epochs = {n_epochs}.\n\n {classification}")
            

# Define behaviour when called from command line
if __name__=="__main__":
    main()