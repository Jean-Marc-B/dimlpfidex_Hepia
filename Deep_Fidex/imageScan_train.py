###############################################################

import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

np.random.seed(seed=None)

from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Flatten, Input, Convolution2D, DepthwiseConv2D, MaxPooling2D
from keras.layers     import BatchNormalization
from keras.applications     import ResNet50
from keras.optimizers import Adam

from keras.callbacks  import ModelCheckpoint

from utils import *

###############################################################

start_time = time.time()

nbIt               = 20

#dataset = "MNIST"
dataset = "CIFAR"

if dataset == "MNIST":     # for MNIST images
    size1D             = 28
    nbChannels         = 1
    nb_classes = 10
    base_folder = "Mnist/"

elif dataset == "CIFAR":     # for Cifar images
    size1D             = 32
    nbChannels         = 3
    nb_classes = 10
    base_folder = "Cifar/"

hiknot = 5
nbQuantLevels = 100
K_val = 1.0
resnet=False

train_data_file = base_folder + "trainData.txt"
train_class_file = base_folder + "trainClass.txt"
test_data_file = base_folder + "testData.txt"
test_class_file = base_folder + "testClass.txt"
model_file = base_folder + "Scan/scanModel.keras"
train_pred_file = base_folder + "Scan/train_pred.out"
test_pred_file = base_folder + "Scan/test_pred.out"
model_stats = base_folder + "Scan/stats_model.txt"
model_checkpoint_weights = base_folder + "Scan/weightsModel.weights.h5"

##############################################################################

print("Loading data...")

train   = np.loadtxt(train_data_file)
X_train = train.reshape(train.shape[0], size1D, size1D, nbChannels)
X_train = X_train.astype('float32')
print(X_train.shape)

test   = np.loadtxt(test_data_file)
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

print(f"Training set: {x_train.shape}, {y_train.shape}")
print(f"Validation set: {x_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {Y_test.shape}")

if (nbChannels == 1 and resnet):
    # B&W to RGB
    x_train = np.repeat(x_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)
    x_val = np.repeat(x_val, 3, axis=-1)
    nb_channels = 3

##############################################################################
if resnet:
    input_tensor = Input(shape=(size1D, size1D, 3))
    model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)
    model = Sequential()
    model.add(model_base)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(nb_classes, activation='softmax'))

    model.build((None, size1D, size1D, 3))  # Build the model with the input shape

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

else:
    model = Sequential()

    model.add(Input(shape=(size1D, size1D, nbChannels)))

    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(DepthwiseConv2D((5, 5), activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.3))

    model.add(Dense(nb_classes, activation='sigmoid'))

    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_checkpoint_weights, verbose=1, save_best_only=True, save_weights_only=True)
model.fit(x_train, y_train, batch_size=32, epochs=nbIt, validation_data=(x_val, y_val), callbacks=[checkpointer], verbose=2)

print("model trained")

##############################################################################

model.load_weights(model_checkpoint_weights)
model.save(model_file)

train_pred = model.predict(x_train)    # Predict the response for train dataset
test_pred = model.predict(X_test)    # Predict the response for test dataset
valid_pred = model.predict(x_val)   # Predict the response for validation dataset
train_valid_pred = np.concatenate((train_pred,valid_pred)) # We output predictions of both validation and training sets

# Output predictions
output_data(train_valid_pred, train_pred_file)
output_data(test_pred, test_pred_file)

##############################################################################

print("\nResult :")

with open(model_stats, "w") as myFile:
    score = model.evaluate(x_train, y_train)
    print("Train score : ", score)
    myFile.write(f"Train score : {score[1]}\n")

    score = model.evaluate(X_test, Y_test)
    print("Test score : ", score)
    myFile.write(f"Test score : {score[1]}\n")

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
