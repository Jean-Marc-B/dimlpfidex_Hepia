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
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

np.random.seed(seed=None)

from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Activation, Flatten, Input
from keras.layers     import Convolution1D, Convolution2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization
from keras.models     import Model, load_model

from keras import layers
from keras import activations

from keras.callbacks  import ModelCheckpoint

from keras import backend as K

from stairObj import StairObj

###############################################################

start_time = time.time()

doParam            = 0.2

nbIt               = 5
nbStairsPerUnit    = 30
#size1D             = 28    # for MNIST images
size1D             = 32    # for Cifar images
nbChannels         = 3     # for Cifar images
#nbChannels         = 1     # for MNIST images

nbStairsPerUnitInv = 1.0/nbStairsPerUnit

nb_classes = 10 # for MNIST or Cifar10 images
hiknot = 5
nbQuantLevels = 100
K_val = 1.0

base_folder = "Cifar/"
train_data_file = base_folder + "trainData.txt"
train_class_file = base_folder + "trainClass.txt"
test_data_file = base_folder + "testData.txt"
test_class_file = base_folder + "testClass.txt"
base_model = base_folder + "baseModel.keras"
staircase_model = base_folder + "StairCaseModel.keras"
weights_first_layer = base_folder + "weights_first_layer.wts"
weights_deep_fidex_outfile = base_folder + "weights_deep_fidex.wts"
deep_fidex_train_inputs = base_folder + "deep_fidex_train_inputs.txt"
deep_fidex_test_inputs = base_folder + "deep_fidex_test_inputs.txt"
train_pred_file = base_folder + "train_pred.out"
test_pred_file = base_folder + "test_pred.out"

###############################################################
###############################################################

def output_data(data, data_file):
    try:
        with open(data_file, "w") as file:
            for var in data:
                for val in var:
                    file.write(str(val) + " ")
                file.write("\n")
            file.close()

    except (FileNotFoundError):
        raise ValueError(f"Error : File {data_file} not found.")
    except (IOError):
        raise ValueError(f"Error : Couldn't open file {data_file}.")

def staircaseUnbound(x):

   return (K.sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin(x):
#    h -> nStairsPerUnit
#    hard_sigmoid(x) =  max(0, min(1, x/6+0.5))
#    staircaseSemiLin(x, h) = hard_sigmoid(ceil(x*h)*(1/h))

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

def compute_first_hidden_layer(step, input_data, k, nb_stairs, hiknot, weights_outfile=None, mu=None, sigma=None):
    input_data = np.array(input_data)
    if step == "train": # Train datas
        if weights_outfile is None:
            raise ValueError("Error : weights file is None during computation of first hidden layer with train data.")
        mu = np.mean(input_data, axis=0) if mu is None else mu  # mean over variables
        sigma = np.std(input_data, axis=0) if sigma is None else sigma # std over variables
        sigma[sigma == 0] = 0.001  # Prevent division by zero

        weights = k/sigma
        biais = -k*mu/sigma

        # Save weights and bias
        if weights_outfile is not None:
         try:
               with open(weights_outfile, "w") as my_file:
                  for b in biais:
                     my_file.write(str(b))
                     my_file.write(" ")
                  my_file.write("\n")
                  for w in weights:
                     my_file.write(str(w))
                     my_file.write(" ")
                  my_file.close()
         except (FileNotFoundError):
               raise ValueError(f"Error : File {weights_outfile} not found.")
         except (IOError):
               raise ValueError(f"Error : Couldn't open file {weights_outfile}.")

    # Compute new data after first hidden layer
    h = k*(input_data-mu)/sigma # With indices : hij=K*(xij-muj)/sigmaj
    stair = StairObj(nb_stairs, hiknot)
    out_data = np.vectorize(stair.funct)(h) # Apply staircase activation function

    return (out_data, mu, sigma) if step == "train" else out_data

###############################################################

def create_model(use_staircase=False):

    model = Sequential()
    model.add(Input(shape=(size1D, size1D, nbChannels)))  # Explicit input shape definition
    # model.add(LocallyConnected2D(1,  (1, 1), activation=tf.keras.activations.hard_sigmoid, input_shape=(size1D, size1D, nbChannels)))
    # model.add(Convolution2D(1, (1, 1), activation=tf.keras.activations.hard_sigmoid, input_shape=(size1D, size1D, nbChannels)))

    # model.add(BatchNormalization(center=False, scale=False, input_shape=(size1D, size1D, nbChannels)))
    # model.add(BatchNormalization(input_shape=(size1D, size1D, nbChannels)))
    # model.add(layers.Activation(tf.keras.activations.sigmoid))
   #  if use_staircase:
   #    model.add(layers.Activation(staircaseSemiLin))
   #  else:
   #    model.add(layers.Activation(tf.keras.activations.hard_sigmoid))
    # model.add(layers.Activation(hardSigm2))
    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(doParam))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(BatchNormalization())

    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(doParam))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(DepthwiseConv2D(1, depth_multiplier=2, activation=tf.keras.activations.hard_sigmoid))

    model.add(Flatten(name="flatten_layer"))

    # model.add(BatchNormalization(center=False, scale=False, name="batchnorm_layer"))

    model.add(BatchNormalization(name="batchnorm_layer"))
    if use_staircase:
      model.add(layers.Activation(staircaseSemiLin))
    else:
      model.add(layers.Activation(tf.keras.activations.hard_sigmoid))

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
###############################################################

print("Loading data...")

train   = np.loadtxt(train_data_file)

# X_train = train.reshape(train.shape[0], nbChannels, size1D, size1D)
X_train = train.reshape(train.shape[0], size1D, size1D, nbChannels)
X_train = X_train.astype('float32')
print(X_train.shape)

test   = np.loadtxt(test_data_file)

# X_test = test.reshape(test.shape[0], nbChannels, size1D, size1D)
X_test = test.reshape(test.shape[0], size1D, size1D, nbChannels)
X_test = X_test.astype('float32')
print(X_test.shape)

Y_train = np.loadtxt(train_class_file)
Y_train = Y_train.astype('int32')

Y_test  = np.loadtxt(test_class_file)
Y_test  = Y_test.astype('int32')

##############################################################################
split_index = int(0.8 * len(X_train))
x_train = X_train[0:split_index]
x_val   = X_train[split_index:]
y_train = Y_train[0:split_index]
y_val   = Y_train[split_index:]

##############################################################################

# Compute first hidden layer

x_train = x_train.reshape(x_train.shape[0], size1D*size1D*nbChannels)
X_test = X_test.reshape(X_test.shape[0], size1D*size1D*nbChannels)
x_val = x_val.reshape(x_val.shape[0], size1D*size1D*nbChannels)

x_train_h1, mu, sigma = compute_first_hidden_layer("train", x_train, K_val, nbQuantLevels, hiknot, weights_first_layer)
X_test_h1 = compute_first_hidden_layer("test", X_test, K_val, nbQuantLevels, hiknot, mu=mu, sigma=sigma)
x_val_h1 = compute_first_hidden_layer("test", x_val, K_val, nbQuantLevels, hiknot, mu=mu, sigma=sigma)

x_train_h1 = x_train_h1.reshape(x_train_h1.shape[0], size1D, size1D, nbChannels)
X_test_h1 = X_test_h1.reshape(X_test_h1.shape[0], size1D, size1D, nbChannels)
x_val_h1 = x_val_h1.reshape(x_val_h1.shape[0], size1D, size1D, nbChannels)

print(f"Training set: {x_train_h1.shape}, {y_train.shape}")
print(f"Validation set: {x_val_h1.shape}, {y_val.shape}")
print(f"Test set: {X_test_h1.shape}, {Y_test.shape}")

##############################################################################

model  = create_model(use_staircase=False)
model2 = create_model(use_staircase=True)

bestScore = float('inf')


for epoch in range(nbIt):
    print(f"Epoch {epoch+1}")

    # Train the model for 1 epoch
    model.fit(x_train_h1, y_train, batch_size=32, epochs=1, validation_data=(x_val_h1, y_val), verbose=2)

    # Transfer the weights to model2
    model2.set_weights(model.get_weights())

    # Evalueate model2 on validation
    val_score = model2.evaluate(x_val_h1, y_val, verbose=0)
    print(f"Validation score with staircaseSemiLin: {val_score}")

    # Save weights if the model scores better
    if val_score[0] < bestScore:
        bestScore = val_score[0]
        model.save(base_model)
        print(f"*** New best validation score. Model saved at epoch {epoch+1}.")

    # Evaluate model2 on test set
    test_score = model2.evaluate(X_test_h1, Y_test, verbose=0)
    print(f"Test score with staircaseSemiLin: {test_score}\n")


modelBest  = load_model(base_model)
modelBest2 = create_model(use_staircase=True)
modelBest2.set_weights(modelBest.get_weights())
modelBest2.save(staircase_model)

print("model trained")

# Get all data outputs after the flatten layer
flatten_layer = modelBest2.get_layer('flatten_layer')  # Access the layer directly from the model
flatten_output_model = Model(inputs=modelBest2.inputs, outputs=flatten_layer.output)

flatten_output_train = flatten_output_model.predict(x_train_h1)
flatten_output_val = flatten_output_model.predict(x_val_h1)
flatten_output_test = flatten_output_model.predict(X_test_h1)
flatten_output_train_val = np.concatenate((flatten_output_train,flatten_output_val))

# Output deep Fidex input values
output_data(flatten_output_train_val, deep_fidex_train_inputs)
output_data(flatten_output_test, deep_fidex_test_inputs)


# Get batch_norm statistics to create hyperlocus
# BatchNorm : gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta
# We put :
# w = gamma / sqrt(self.moving_var + epsilon)
# b = (-self.moving_mean * gamma / sqrt(self.moving_var + epsilon)) + beta
bn_layer = modelBest2.get_layer('batchnorm_layer')
mean = bn_layer.moving_mean.numpy()  # Mean
variance = bn_layer.moving_variance.numpy()  # Variance
gamma = bn_layer.gamma.numpy()  # Scale
beta = bn_layer.beta.numpy()  # Center
epsilon = bn_layer.epsilon

weights = gamma/np.sqrt(variance+epsilon)
biais = -(gamma*mean)/np.sqrt(variance+epsilon) + beta

# Output weights
try:
    with open(weights_deep_fidex_outfile, "w") as my_file:
        for b in biais:
            my_file.write(str(b))
            my_file.write(" ")
        my_file.write("\n")
        for w in weights:
            my_file.write(str(w))
            my_file.write(" ")
        my_file.close()
except (FileNotFoundError):
    raise ValueError(f"Error : File {weights_deep_fidex_outfile} not found.")
except (IOError):
    raise ValueError(f"Error : Couldn't open file {weights_deep_fidex_outfile}.")

# Verify each of these and make sure it's computed on each train data...

print("Mean:", mean.shape)
print("Variance:", variance.shape)
print("Scale (gamma):", gamma.shape)
print("Center (beta):", beta.shape)
print("Epsilon :", epsilon)
print("Weights:", weights.shape)
print("Biais:", biais.shape)


train_pred = modelBest2.predict(x_train_h1)    # Predict the response for train dataset
test_pred = modelBest2.predict(X_test_h1)    # Predict the response for test dataset
valid_pred = modelBest2.predict(x_val_h1)   # Predict the response for validation dataset
train_valid_pred = np.concatenate((train_pred,valid_pred)) # We output predictions of both validation and training sets

output_data(train_valid_pred, train_pred_file)
output_data(test_pred, test_pred_file)

##############################################################################

print("\nBest result :")

score = modelBest2.evaluate(x_train_h1, y_train)
print("Train score : ", score)

score = modelBest2.evaluate(X_test_h1, Y_test)
print("Test score : ", score)

acc_train = modelBest2.evaluate(x_train_h1, y_train, verbose=0)[1]
acc_test = modelBest2.evaluate(X_test_h1, Y_test, verbose=0)[1]

formatted_acc_train = "{:.6f}".format(acc_train*100)
formatted_acc_test = "{:.6f}".format(acc_test*100)

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
