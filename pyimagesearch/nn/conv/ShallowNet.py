# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:00:15 2020

@author: Administrador
"""

 # import the necessary packages
from keras.models import Sequential 
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes): 
        # initialize the model along with the input shape to be
        # "channels last" 
        model = Sequential()
        
        print('model 1: '+str(model))
        
        inputShape = (height, width, depth) 
        # if we are using "channels first", update the input shape 
        if K.image_data_format() == "channels_first": 
            inputShape = (depth, height, width)
            
            print('model 2: '+str(model))

            # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape)) 
        model.add(Activation("relu"))
        
        print('model 3: '+str(model))
        
        # softmax classifier
        model.add(Flatten()) 
        model.add(Dense(classes)) 
        model.add(Activation("softmax"))
        
        print('model 4: '+str(model))
        
        # return the constructed network architecture 
        return model

            
