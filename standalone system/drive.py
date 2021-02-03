'''Imports'''
import socketio # Simulator interface
sio = socketio.Server()

import eventlet # Connection initiation wrapper

import numpy as np # Mathematical Operations

from flask import Flask # Eventlet backend
app = Flask(__name__)

from keras.models import load_model # Loading model

from io import BytesIO # Inputting image from simulator
from PIL import Image # Importing image from simulator
import base64 # Decoding image feed from simulator
import cv2 # Computer Vision


'''Parameters'''
model_name = "test_model.h5" # Name of model file on disk, in same directory as program
# (test_model.h5 in this directory has been trained on around 5 laps of track 1)

speed_limit = 32 # Maximum speed of vehicle


'''Functions'''
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

# Sending Steering/Throttle Data
def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        "steering_angle": steering_angle.__str__(),
        "throttle": throttle.__str__()
    })

# Received Image Data
@sio.on("telemetry")
def telemetry(sid, data):
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = preprocess_image(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

# Connected to simulator
@sio.on("connect")
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)

'''Program'''
if __name__ == "__main__":
    model = load_model(model_name)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
