# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:04:34 2020

@author: Administrador
"""

# import the necessary packages 
from pyimagesearch.nn.conv import LeNet 
from keras.utils import plot_model 
# initialize LeNet and then write the network architecture 
# visualization graph to disk 
model = LeNet.build(28, 28, 1, 10) 
plot_model(model, to_file="lenet.png", show_shapes=True)