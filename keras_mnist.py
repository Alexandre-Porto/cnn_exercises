# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:43:33 2020

@author: Administrador
"""
 # import the necessary packages
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.optimizers import SGD 
from sklearn import datasets 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser() 
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot") 
args = vars(ap.parse_args())
# grab the MNIST dataset (if this is your first time running this 
# script, the download may take a minute -- the 55MB MNIST dataset 
# will be downloaded) 


# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],
data.shape[1]))

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data,digits.target, test_size=0.25)

# convert the labels from integers to vectors 
lb = LabelBinarizer() 
trainY = lb.fit_transform(trainY) 
testY = lb.transform(testY)

# define the 784-256-128-10 architecture using Keras 
model = Sequential() 
model.add(Dense(256, input_shape=(784,), activation="sigmoid")) 
model.add(Dense(128, activation="sigmoid")) 
model.add(Dense(10, activation="softmax"))

# train the model using SGD 
print("[INFO] training network...") 
sgd = SGD(0.01) 
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

print('trainX: '+str(trainX))
print('testX: '+str(testX))
print('trainY: '+str(trainY))
print('testY: '+str(testY))

print('trainX shape: '+str(trainX.shape))
print('testX shape: '+str(testX.shape))
print('trainY shape: '+str(trainY.shape))
print('testY shape: '+str(testY.shape))

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# evaluate the network 51 print("[INFO] evaluating network...") 
predictions = model.predict(testX, batch_size=128) 
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy 
plt.style.use("ggplot")
plt.figure() 
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss") 
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss") 
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc") 
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])