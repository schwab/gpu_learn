"""
Chapter 12: Training your First CNN 12.2.1
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # intialize model and input shape to be "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # handle channels first if needed
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # define the CONV => RELU layer
        # with 32 filters of 3x3
        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
        

