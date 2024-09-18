# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:39:16 2023

@author: guido.bologna
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:54:46 2021

@author: guido.bologna
"""

###############################################################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

np.random.seed(seed=None)  
 
from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Activation, Flatten
from keras.layers     import Convolution1D, Convolution2D, DepthwiseConv2D, LocallyConnected2D, MaxPooling2D, LocallyConnected2D, BatchNormalization
from keras.models     import load_model

from keras import layers
from keras import activations

from keras.callbacks  import ModelCheckpoint

from keras import backend as K

###############################################################

doParam            = 0.2

nbIt               = 1
nbStairsPerUnit    = 30
size1D             = 28    # for images

nbStairsPerUnitInv = 1.0/nbStairsPerUnit

###############################################################
###############################################################

def staircaseUnbound(x):
   
   return (K.sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin(x):
   
   return (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################
   
def staircaseSemiLin2(x):
   
   a = (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))
   a = (a - 0.5)*6.0
   return a

###############################################################

def hardSigm2(x):
   
   a = tf.keras.activations.hard_sigmoid(x)
   a = (a - 0.5)*6.0
   return a

###############################################################

def staircaseBound(x):
   
   a = tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit)*0.5 * nbStairsPerUnitInv)
   a = (a - 0.5)*10.0
   return(K.sigmoid(a))

###############################################################

def myModel(size1D):

    model = Sequential()
    # model.add(LocallyConnected2D(1,  (1, 1), activation=tf.keras.activations.hard_sigmoid, input_shape=(size1D, size1D, 1)))
    # model.add(Convolution2D(1, (1, 1), activation=tf.keras.activations.hard_sigmoid, input_shape=(size1D, size1D, 1)))
    
    # model.add(BatchNormalization(center=False, scale=False, input_shape=(size1D, size1D, 1)))
    model.add(BatchNormalization(input_shape=(size1D, size1D, 1)))
    # model.add(layers.Activation(tf.keras.activations.sigmoid))
    model.add(layers.Activation(tf.keras.activations.hard_sigmoid))
    # model.add(layers.Activation(hardSigm2))
    
    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(doParam))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # model.add(BatchNormalization())
    
    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(doParam))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # model.add(DepthwiseConv2D(1, depth_multiplier=2, activation=tf.keras.activations.hard_sigmoid))    
        
    model.add(Flatten())
    
    # model.add(BatchNormalization(center=False, scale=False))
    
    model.add(BatchNormalization())
    model.add(layers.Activation(tf.keras.activations.sigmoid))
    
    # model.add(layers.Activation(tf.keras.activations.hard_sigmoid))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(doParam))
    # model.add(Dense(128, activation='sigmoid'))
    
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    # theWeights  = model.get_weights()
    # m           = theWeights[0]
    # m[0,0,0,0]  = 1;
    
    # m2          = theWeights[1]
    # m2[0]       = -0.5;
     
    return model

###############################################################

def myModel2(size1D):

    model = Sequential()
    # model.add(LocallyConnected2D(1,  (1, 1), activation=staircaseSemiLin, input_shape=(size1D, size1D, 1)))
    # model.add(Convolution2D(1, (1, 1), activation=staircaseSemiLin, input_shape=(size1D, size1D, 1)))
    
    # model.add(BatchNormalization(center=False, scale=False, input_shape=(size1D, size1D, 1)))
    model.add(BatchNormalization(input_shape=(size1D, size1D, 1)))
    # model.add(layers.Activation(staircaseUnbound))
    model.add(layers.Activation(staircaseSemiLin))
    # model.add(layers.Activation(staircaseSemiLin2))
    
    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(doParam))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(doParam))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # model.add(DepthwiseConv2D(1, depth_multiplier=2, activation=staircaseSemiLin))
        
    model.add(Flatten())
    
    # model.add(BatchNormalization(center=False, scale=False))
    
    model.add(BatchNormalization())
    # model.add(layers.Activation(staircaseUnbound))
    
    model.add(layers.Activation(staircaseSemiLin))
    
    # model.add(Convolution1D(3, 1, activation=staircaseSemiLin))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(doParam))
    # model.add(Dense(128, activation='sigmoid'))
    
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model

###############################################################
###############################################################

print("Loading data...")
    
train   = np.loadtxt("trainMnist784WC")

# X_train = train.reshape(train.shape[0], 1, size1D, size1D)
X_train = train.reshape(train.shape[0], size1D, size1D, 1)
X_train = X_train.astype('float32')
print(X_train.shape[0])

test   = np.loadtxt("testMnist2")

# X_test = test.reshape(test.shape[0], 1, size1D, size1D)
X_test = test.reshape(test.shape[0], size1D, size1D, 1)
X_test = X_test.astype('float32')
print(X_test.shape[0])

Y_train = np.loadtxt("mnistTrainClass")
Y_train = Y_train.astype('int32')

Y_test  = np.loadtxt("mnistTestClass")
Y_test  = Y_test.astype('int32')

##############################################################################

x_train = X_train[0:50000]
x_val   = X_train[50000:]
y_train = Y_train[0:50000]
y_val   = Y_train[50000:]

##############################################################################

model  = myModel(size1D)
model2 = myModel2(size1D)

for i in range(nbIt):
    
    checkpointer = ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True)
    
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[checkpointer], verbose=2)
    # model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_val, y_val), verbose=1)
    # model.fit(X_train, Y_train, batch_size=32, epochs=3, validation_data=(X_test, Y_test), verbose=1)
   
    score = model.evaluate(X_train, Y_train)
    print(score)
   
    score2 = model.evaluate(X_test, Y_test)
    print(score2)
    
    theWeights  = model.get_weights()
    # model2.set_weights(theWeights)

    
    # score = model2.evaluate(X_train, Y_train)
    # print(score)
   
    # score2 = model2.evaluate(X_test, Y_test)
    # print(score2)
   
    # print(i+1)
   
##############################################################################

# score = model.evaluate(X_train, Y_train)
# print(score)

# score = model.evaluate(X_test, Y_test)
# print(score)


# score = model2.evaluate(X_train, Y_train)
# print(score)

# score = model2.evaluate(X_test, Y_test)
# print(score)

##############################################################################

modelBest = load_model('weights.hdf5')

score = modelBest.evaluate(X_train, Y_train)
print(score)

score = modelBest.evaluate(X_test, Y_test)
print(score)


# print('\n')

# model3 = myModel2(size1D)

# theWeights  = modelBest.get_weights()
# model3.set_weights(theWeights)

    
# score = model3.evaluate(X_train, Y_train)
# print(score)
   
# score2 = model3.evaluate(X_test, Y_test)
# print(score2)

