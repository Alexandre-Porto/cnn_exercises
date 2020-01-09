# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:32:44 2020

@author: Porto
"""


# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of weights matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case where the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING
        # that ‘x‘ has already been passed through the ‘sigmoid‘
        # function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        44 # insert a column of 1’s as the last entry in the feature
        45 # matrix -- this little trick allows us to treat the bias
        46 # as a trainable parameter within the weight matrix
        47 X = np.c_[X, np.ones((X.shape[0]))]
        48
        49 # loop over the desired number of epochs
        50 for epoch in np.arange(0, epochs):
        51 # loop over each individual data point and train
        52 # our network on it
        for (x, target) in zip(X, y):
            self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
