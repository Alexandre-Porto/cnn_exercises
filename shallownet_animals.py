# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:38:40 2020

@author: Administrador
"""


# import the necessary packages 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from pyimagesearch.preprocessing import ImageToArrayPreprocessor 
from pyimagesearch.preprocessing import SimplePreprocessor 
from pyimagesearch.datasets import SimpleDatasetLoader 
from pyimagesearch.nn.conv import ShallowNet 
from keras.optimizers import SGD 
from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser() 
ap.add_argument("-d", "--dataset", required=True, 
                help="path to input dataset") 
args = vars(ap.parse_args()) 
# grab the list of images that we’ll be describing 
print("[INFO] loading images...") 
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors 
sp = SimplePreprocessor.SimplePreprocessor(32, 32) 
iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()
# load the dataset from disk then scale the raw pixel intensities 
# to the range [0, 1] 
sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500) 
data = data.astype("float") / 255.0

 # partition the data into training and testing splits using 75% of 
# the data for training and the remaining 25% for testing 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42) 

print('unknown label: '+str(trainY[664]))

#raise ValueError('label')

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

for i in range(len(trainY)):
    print('trainY: '+str(trainY[i]))
    if trainY[i][2] == 1:
        print('index: '+str(i))
        #raise ValueError('unknown label')
        
print('trainY shape before: '+str(trainY.shape))
        
trainY = np.delete(trainY, 2, 1)
        
print('trainY shape after: '+str(trainY.shape))

testY = np.delete(testY, 2, 1)
        
# initialize the optimizer and model 
print("[INFO] compiling model...") 
opt = SGD(lr=0.005) 
model = ShallowNet.ShallowNet.build(width=32, height=32, depth=3, classes=3)

print('model: '+str(model))

model.compile(loss="categorical_crossentropy", optimizer=opt, 
              metrics=["accuracy"]) 
# train the network 
print("[INFO] training network...")
print('trainX.shape: '+str(trainX.shape))
print('trainY.shape: '+str(trainY.shape))
print('testX.shape: '+str(testX.shape))
print('testY.shape: '+str(testY.shape))

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating network...") 
predictions = model.predict(testX, batch_size=32) 
print(classification_report(testY.argmax(axis=1), 
                            predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

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
plt.show()