# Verbose Console Output
DEBUG = True # Program Output | True: On, False: Off | Default: True
TF_DEBUG = 1 # TensorFlow Output | 0: Print all messages, 1: Print only warnings and errors, 2: Print only errors, 3: Off | Default: 1
KERAS_DEBUG = 1 # Keras Output | 1: On, 0: Off | Default: 1

'''Imports'''
import os # Interfacing with System
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_DEBUG)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import cv2 # Computer Vision

from imgaug import augmenters as iaa # Image Augmentation

import keras # Machine Learning
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense # Layers to construct Neural Network
from keras.models import Sequential # Particular stack of layers
from keras.optimizers import Adam # Optimisation algorithm, 'learning'

import matplotlib.pyplot as plt # Plotting
import matplotlib.image as mpimg # Image Operations

import ntpath # Path Manipulation
import numpy as np # Mathematical Operations, Data Analysis

import pandas as pd # Data Manipulation

import random # Random Number Generation

from sklearn.utils import shuffle # Data Balancing
from sklearn.model_selection import train_test_split # Data Splitting


'''Parameters'''
# Data Reading
data_dir = "dataset_directory" # Must be in the same directory as program file, omit './' | Input Parameter

# Data Balancing
num_bins = 25 # Number of steering angle groupings to make | Default: 25
max_samples_per_bin = 400 # Maximum number of datapoints in each grouping | Default: 400 | User-Modifiable

# Generating Labelled Data
validation_proportion = 0.2 # Proportion of dataset that will be set aside and used for validation throughout training | Default: 0.2 | User-Modifiable

# Augmenter
p = 0.5 # Probability of any image passed in to be given any of the augmentations | Default: 0.5 | User-Modifiable

# Batch Generator
batch_size = 100 # Size of training batches | Default: 100 | User-Modifiable

# Model
model_name = "model_name.h5" # Name of output model file | Input Parameter
learning_rate = 1e-3 # Step size, amount that weights are updated during training | Default: 1e-3 | User-Modifiable
epochs = 10 # Number of training epochs | Default: 10 | User-Modifiable
steps_per_epoch = 300 # Number of batch generator iterations before a training epoch is considered finished | Default: 300 | User-Modifiable
validation_steps = 200 # Similar to steps_per_epoch but for validation set, so lower | Default: 200 | User-Modifiable

'''Classes'''
class Augmenter():
    """Augmenter
    Object that can apply augmentation to images
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __zoom(self, image):
        zoom = iaa.Affine(scale=(1, 1.3)) # Zoom by up to 130% in about centre
        image = zoom.augment_image(image)
        return image
    
    def __pan(self, image):
        pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
        return image
    
    def __brightness_random(self, image):
        brightness = iaa.Multiply((0.2, 1.2))
        image = brightness.augment_image(image)
        return image
    
    def __flip_random(self, image, steering_angle):
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle # Steering angle needs to be flipped as well, since we are flipping horizontally
        return image, steering_angle
        
    def random_augment(self, image, steering_angle):
        image = mpimg.imread(image)
        if np.random.rand() < p:
            image = self.__pan(image)
        if np.random.rand() < p:
            image = self.__zoom(image)
        if np.random.rand() < p:
            image = self.__brightness_random(image)
        if np.random.rand() < p:
            image, steering_angle = self.__flip_random(image, steering_angle)
        return image, steering_angle


'''Functions'''
# Data Reading
def path_leaf(path):
    """Path Leaf

    Arguments:
        path (String): Full path to file

    Returns:
        String: File name and extension
    """
    _, tail = ntpath.split(path)
    return tail

# Generating Labelled Data
def load_training_data(data_dir, data):
    """Load Training Data

    Arguments:
        data_dir (String): Directory of dataset
        data (Pandas Dataframe): Imported data from driving_log.csv
        side_offset (Float): Amount of degrees

    Returns:
        numpy Array: Array of image paths (centre, left, right)
        numpy Array: Array of corresponding - by index - steering angle 'labels'
    """
    image_paths = []
    steering_angles = []

    side_cam_offset = 0.15
    
    for i in range(len(data)):
        row = data.iloc[i]
        centre_image_path, left_image_path, right_image_path = row[0], row[1], row[2]
        steering_angle = float(row[3])
        
        # Centre image
        image_paths.append(os.path.join(data_dir, centre_image_path.strip()))
        steering_angles.append(steering_angle)
        
        # Left image
        image_paths.append(os.path.join(data_dir, left_image_path.strip()))
        steering_angles.append(steering_angle + side_cam_offset)
        
        # Right image
        image_paths.append(os.path.join(data_dir, right_image_path.strip()))
        steering_angles.append(steering_angle - side_cam_offset)
        
    return np.asarray(image_paths), np.asarray(steering_angles)

# Image Preprocessing
def preprocess_image(image):
    """Preprocess Image

    Args:
        image (numpy Array): Image to be preprocessed

    Returns:
        numpy Array: Preprocessed Image
    """
    # image = mpimg.imread(image)
    image = image[60:135,:,:] # Crops out sky and bonnet of car
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) # Converting the channels to YUV colour space
    image = cv2.GaussianBlur(image, (3, 3), 0) # Gaussian Blur applied to image to reduce the effects of noise and smoothen the image, 3x3 is a small kernel, using no deviation
    image = cv2.resize(image, (200, 66)) # Reducing the size of the image to match the NVIDIA model input layer width
    image = image/255 # Normalisation of image to values between 0 and 1, but no visual impact on image
    return image

# Batch Generator
def batch_generator(images, steering_angle, batch_size, is_training):
    """Batch Generator

    Args:
        images (numpy Array): Images in dataset
        steering_angle (numpy Array): Labels in dataset
        batch_size (integer): Size of each batch
        is_training (integer): Augment or not

    Yields:
        tuple of numpy arrays: Batch images, Batch labels
    """
    while True:
        batch_images = []
        batch_steering_angles = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(images)-1)

            if is_training:
                image, steering = augmenter.random_augment(images[random_index], steering_angle[random_index]) # Randomly augment some images going into the batch
            else:
                image = mpimg.imread(images[random_index])
                steering = steering_angle[random_index]

            image = preprocess_image(image)
            batch_images.append(image)
            batch_steering_angles.append(steering)
        
        yield (np.asarray(batch_images), np.asarray(batch_steering_angles)) # Iterate the generator

# Model
def dave_2_model():
    """DAVE-2 Model

    Returns:
        Keras Model: Model with Architecture of NVIDIA's DAVE-2 Neural Network
    """
    model = Sequential()
    
    model.add(Convolution2D(24, (5, 5), input_shape=(66, 200, 3), activation="elu", strides=(2, 2))) # Input layer
    model.add(Convolution2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Convolution2D(48, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))

    model.add(Flatten()) # Converts the output of the Convolutional layers into a 1D array for input by the following fully connected layers

    model.add(Dense(100, activation = "elu"))
    model.add(Dense(50, activation = "elu"))
    model.add(Dense(10, activation = "elu"))
    model.add(Dense(1)) # Output layer

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

'''Program'''
# Data Reading
columns = ["centre_image",
           "left_image",
           "right_image",
           "steering_angle",
           "throttle",
           "reverse",
           "speed"]

data = pd.read_csv(os.path.join(data_dir, "driving_log.csv"), names=columns) # Reading data from comma separated value file into a variable
if DEBUG == True:
    print("[Data Reading]")
    print("Number of total entries in \"" + data_dir + "\" dataset: " + str(len(data)))
    print()

# Trimming image entries down from full paths
data["centre_image"] = data["centre_image"].apply(path_leaf)
data["left_image"] = data["left_image"].apply(path_leaf)
data["right_image"] = data["right_image"].apply(path_leaf)

# Data Balancing
_, bins = np.histogram(data["steering_angle"], num_bins) # Splitting data into num_bins groups of equal intervals

all_discard_indexes = []
for i in range(num_bins):
    bin_discard_indexes = []
    
    for j in range(len(data["steering_angle"])):
        if data["steering_angle"][j] >= bins[i] and data["steering_angle"][j] <= bins[i+1]:
            bin_discard_indexes.append(j)
    
    bin_discard_indexes = shuffle(bin_discard_indexes) # Non-random shuffle
    bin_discard_indexes = bin_discard_indexes[max_samples_per_bin:] # Leaves all indexes but those kept within the max_samples_per_bin region
    
    all_discard_indexes.extend(bin_discard_indexes) # Concatenating this bin's balanced list to the overall discard list

data.drop(data.index[all_discard_indexes], inplace=True) # Removing excess data from each bin from the overall dataset, now balanced
if DEBUG == True:
    print("[Dataset Balancing]")
    print("Number of discarded entries: " + str(len(all_discard_indexes)))
    print("Number of remaining entries: " + str(len(data)))
    print()

# Generating Labelled Data
image_paths, steering_angles = load_training_data(data_dir + "/IMG", data)

X_train, X_valid, y_train, y_valid = train_test_split(image_paths,
                                                      steering_angles,
                                                      test_size=validation_proportion)
if DEBUG == True:
    print("[Generating Labelled Data]")
    print("Number of training datapoints: " + str(len(X_train)))
    print("Number of validation datapoints: " + str(len(X_valid)))
    print()

# Augmenter
augmenter = Augmenter(p=p)

# Batch Generator
X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
X_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

# Model
dave_2_model = dave_2_model()
if DEBUG == True:
    print("[Model]")
    print("Model Summary:")
    print(dave_2_model.summary())
    print()
    print("Training:")

history = dave_2_model.fit(batch_generator(X_train, y_train, batch_size, 1),
                                steps_per_epoch=steps_per_epoch, 
                                epochs=epochs,
                                validation_data=batch_generator(X_valid, y_valid, batch_size, 0),
                                validation_steps=validation_steps,
                                verbose=KERAS_DEBUG,
                                shuffle=1)

dave_2_model.save(model_name)
