# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:24:08 2024

@author: jean-marc.boutay
"""
import os
import sys
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
import numpy as np
from utils import trainCNN, getHistogram, output_data

np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
from trainings import randForestsTrn

###############################################################

start_time = time.time()

with_train_cnn = False
train_cnn_only = False
with_hist_computation = False
with_train_second_model = False
with_global_rules = False

dataset = "MNIST"
#dataset = "CIFAR"

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

# Files
train_data_file = base_folder + "trainData.txt"
train_class_file = base_folder + "trainClass.txt"
test_data_file = base_folder + "testData.txt"
test_class_file = base_folder + "testClass.txt"
model_file = base_folder + "Scan/scanModel.keras"
train_pred_file = base_folder + "Scan/train_pred.out"
test_pred_file = base_folder + "Scan/test_pred.out"
train_histogram_file = base_folder + "Scan/train_hist.txt"
test_histogram_file = base_folder + "Scan/test_hist.txt"

second_model_stats = base_folder + "Scan/second_model_stats.txt"
second_model_train_pred = base_folder + "Scan/second_model_train_pred.txt"
second_model_test_pred = base_folder + "Scan/second_model_test_pred.txt"
second_model_output_rules = base_folder + "Scan/second_model_rules.rls"

global_rules_file = base_folder + "Scan/globalRules.rls"
attributes_file = base_folder + "Scan/attributes.txt"

# If we train :
model_checkpoint_weights = base_folder + "Scan/weightsModel.weights.h5"
model_stats = base_folder + "Scan/stats_model.txt"
resnet = False
nbIt = 4

# For histogram computation
filter_size = [7,7] # Size of filter applied to the image
stride = 1 # shift between each filter
nb_prob = 9 # Number of probas wanted (ex: NProb>=0.1, NProb>=0.2, etc.)
nb_histogram_attributes = nb_classes*nb_prob

# For Fidex
hiknot = 5
nbQuantLevels = 100
K_val = 1.0
dropout_hyp = 0.9
dropout_dim = 0.9




##############################################################################

print("\nLoading data...")

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

print("Data loaded.\n")

##############################################################################

# Get data and save in dataSet class

# Train CNN model
if with_train_cnn:
    trainCNN(size1D, nbChannels, nb_classes, resnet, nbIt, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats)
    if train_cnn_only:
        end_time = time.time()
        full_time = end_time - start_time
        full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

        print(f"\nFull execution time = {full_time} sec")
        sys.exit()


CNNModel = keras.saving.load_model(model_file)

if with_hist_computation:

    print("\nComputing train histograms...")

    # Get histograms for each train sample
    nb_train_samples = Y_train.shape[0]
    train_histograms = []
    for train_sample_id in range(100):
        image = X_train[train_sample_id]
        image = image.reshape(size1D, size1D, nbChannels)
        histogram = getHistogram(CNNModel, image, nb_classes, filter_size, stride, nb_prob)
        train_histograms.append(histogram)
    train_histograms = np.array(train_histograms)

    print("Train histograms computed.\n")

    print("Computing test histograms...")

    # Get histograms for each test sample
    nb_test_samples = Y_test.shape[0]
    test_histograms = []
    for test_sample_id in range(100):
        image = X_test[test_sample_id]
        image = image.reshape(size1D, size1D, nbChannels)
        histogram = getHistogram(CNNModel, image, nb_classes, filter_size, stride, nb_prob)
        test_histograms.append(histogram)

    test_histograms = np.array(test_histograms)
    print("Test histograms computed.")
    # Save in histograms in .npy file
    print("\nSaving histograms...")
    train_histograms = train_histograms.reshape(100, nb_histogram_attributes)
    test_histograms = test_histograms.reshape(100, nb_histogram_attributes)
    output_data(train_histograms, train_histogram_file)
    output_data(test_histograms, test_histogram_file)
    print("Histograms saved.")


train_class_file_temp = base_folder + "trainClass_temp.txt"
test_class_file_temp = base_folder + "testClass_temp.txt"

if with_train_second_model:

    # Train model
    command = (
        f'--train_data_file {train_histogram_file} '
        f'--train_class_file {train_class_file_temp} '
        f'--test_data_file {test_histogram_file} '
        f'--test_class_file {test_class_file_temp} '
        f'--stats_file {second_model_stats} '
        f'--train_pred_outfile {second_model_train_pred} '
        f'--test_pred_outfile {second_model_test_pred} '
        f'--rules_outfile {second_model_output_rules} '
        f'--nb_attributes {nb_histogram_attributes} '
        f'--nb_classes {nb_classes} '
        )

    print("\nTraining second model...\n")

    randForestsTrn(command)

    print("\nSecond model trained.")

# Define attributes file
probability_thresholds = [(1/(nb_prob+1))*i for i in range(1,nb_prob+1)]
with open(attributes_file, "w") as myFile:
    for i in range(nb_classes):
        for j in probability_thresholds:
            myFile.write(f"P>={j}_{i}\n")

if with_global_rules:
    command = (
        f'--train_data_file {train_histogram_file} '
        f'--train_pred_file {second_model_train_pred} '
        f'--train_class_file {train_class_file_temp} '
        f'--nb_attributes {nb_histogram_attributes} '
        f'--nb_classes {nb_classes} '
        f'--rules_file {second_model_output_rules} '
        f'--global_rules_outfile {global_rules_file} '
        f'--attributes_file {attributes_file} '
        f'--heuristic 1 '
        f'--nb_threads 4 '
    )

    print("\nComputing global rules...\n")
    fidex.fidexGloRules(command)

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
