# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:24:08 2024

@author: jean-marc.boutay

This script performs the training and evaluation of a Convolutional Neural Network (CNN) on a specified dataset
(either MNIST, CIFAR or HAPPY). It includes options for training the CNN, computing histograms from CNN predictions on filtered areas,
training a secondary model (e.g., random forests) based on these histograms and extracting global classification rules.
The script also generates images highlighting important areas of the input images based on the learned rules,
saving these images and information for further analysis. Various hyperparameters and file paths are specified to manage
the workflow, allowing for flexibility and customization of the experiment.

The histograms are the probabilities of each class on all the different filtered areas, Fidex is used on it to get rules,
we go back to the filtered images to see where the rule is applied on the image to see the important areas of the image with
respect to this rule.
"""
import os
import sys
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras
import numpy as np
import shutil
import re
import copy
from tensorflow.keras import Model
import tensorflow as tf
from constants import HISTOGRAM_ANTECEDENT_PATTERN
from utils import *
np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
from dimlpfidex import dimlp
from trainings import randForestsTrn, gradBoostTrn
from trainings.trnFun import get_attribute_file

###############################################################

start_time = time.time()


# What to launch
test_version = True # Whether to launch with minimal data



# Training CNN:
with_train_cnn = True

# Stats computation and second model training:
histogram_stats = False
activation_layer_stats = False
probability_stats = True
if probability_stats:
    use_multi_networks_stats = True

if histogram_stats + activation_layer_stats + probability_stats!= 1:
    raise ValueError("Error, you need to specify one of histogram_stats, activation_layer_stats, probability_stats.")

# Computation of statistics
with_stats_computation = True
# Train second model (with statistics data)
with_train_second_model = True

# Rule computation:
with_global_rules = True

# Image generation:
get_images = True # With histograms
simple_heat_map = False # Only evaluation on patches


##############################################################################

# Which dataset to launch
dataset = "MNIST"
#dataset = "CIFAR"
#dataset = "HAPPY"
#dataset = "testDataset"

if dataset == "MNIST":     # for MNIST images
    size1D             = 28
    nb_channels         = 1
    base_folder = "../../data/Mnist/"
    data_type = "integer"
    classes = {
        0:"0",
        1:"1",
        2:"2",
        3:"3",
        4:"4",
        5:"5",
        6:"6",
        7:"7",
        8:"8",
        9:"9",
    }

elif dataset == "CIFAR":     # for Cifar images
    size1D             = 32
    nb_channels         = 3
    base_folder = "../../data/Cifar/"
    data_type = "integer"
    classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

elif dataset == "HAPPY":     # for Happy images
    size1D             = 48
    nb_channels         = 1
    base_folder = "../../data/Happy/"
    data_type = "float"
    classes = {
        0: "happy",
        1: "not happy",
    }

elif dataset == "testDataset":
    size1D = 20
    nb_channels = 1
    base_folder = "Test/"
    data_type = "integer"
    classes = {
        0: "cl0",
        1: "cl1",
    }

nb_classes = len(classes)


##############################################################################

# Parameters

#----------------------------
# Folders
scan_folder = "ScanFull/"
if test_version:
    scan_folder = "Scan/"
if histogram_stats:
    scan_folder += "Histograms/"
elif activation_layer_stats:
    scan_folder += "Activations_Sum/"
elif probability_stats:
    if use_multi_networks_stats:
        scan_folder += "Probability_Multi_Nets_Images/"
    else:
        scan_folder += "Probability_Images/"


#----------------------------
# Files
test_particle = ""
if test_version:
    test_particle = "_test_version"
train_data_file = base_folder + "trainData" + test_particle + ".txt"
train_class_file = base_folder + "trainClass" + test_particle + ".txt"
test_data_file = base_folder + "testData" + test_particle + ".txt"
test_class_file = base_folder + "testClass" + test_particle + ".txt"
if test_version:
    train_data_file = base_folder + "trainData_test_version.txt"
    train_class_file = base_folder + "trainClass_test_version.txt"
    test_data_file = base_folder + "testData_test_version.txt"
    test_class_file = base_folder + "testClass_test_version.txt"
model_file = base_folder + scan_folder + "scanModel.keras"
train_pred_file = base_folder + scan_folder + "train_pred.out"
test_pred_file = base_folder + scan_folder + "test_pred.out"

attributes_file = base_folder + scan_folder + "attributes.txt"

#----------------------------
# If we train :
model_checkpoint_weights = base_folder + scan_folder + "weightsModel.weights.h5"
model_stats = base_folder + scan_folder + "stats_model.txt"
if test_version:
    model="small"
    nbIt = 4
    batch_size = 32
    batch_size_second_model = 32
else:
    model = "VGG"
    nbIt = 80
    batch_size = 64 # To avoid memory problems on GPU
    batch_size_second_model = 64

if activation_layer_stats:
    with_leaky_relu = True
else:
    with_leaky_relu = False

#----------------------------
# For stats computation

if histogram_stats:
    train_stats_file = base_folder + scan_folder + "train_hist.txt"
    test_stats_file = base_folder + scan_folder + "test_hist.txt"
elif activation_layer_stats:
    train_stats_file = base_folder + scan_folder + "train_activation_sum.txt"
    test_stats_file = base_folder + scan_folder + "test_activation_sum.txt"
elif probability_stats:
    train_stats_file = base_folder + scan_folder + "train_probability_images.txt"
    test_stats_file = base_folder + scan_folder + "test_probability_images.txt"
    train_stats_file_with_image = base_folder + scan_folder + "train_probability_images_with_original_img.txt"
    test_stats_file_with_image = base_folder + scan_folder + "test_probability_images_with_original_img.txt"

filter_size = [[7,7]] # Size of filter(s) applied to the image
if np.asarray(filter_size).ndim == 1:
    filter_size = [filter_size]
# Exemples : 7x7 : [7,7] 5x5 and 7x7 : [[5,5],[7,7]]
stride = [[1,1]] # shift between each filter (need to specify one per filter size)
if np.asarray(stride).ndim == 1:
    stride = [stride]
if len(stride) != len(filter_size):
    raise ValueError("Error : There is not the same amout of strides and filter sizes.")

nb_bins = 9 # Number of bins wanted (ex: NProb>=0.1, NProb>=0.2, etc.)
probability_thresholds = getProbabilityThresholds(nb_bins)
if histogram_stats:
    nb_stats_attributes = nb_classes*nb_bins

#----------------------------
# For second model training

if probability_stats:
    second_model = "cnn"
    #second_model = "randomForests"
    if use_multi_networks_stats:
        second_model = "cnn"
        with_hsl = False # Only if we have 3 chanels
        with_rg = True
else:
    # second_model = "randomForests"
    second_model = "gradientBoosting"
    # second_model = "dimlpTrn"
    # second_model = "dimlpBT"

if second_model in {"randomForests", "gradientBoosting"}:
    using_decision_tree_model = True
else:
    using_decision_tree_model = False

second_model_stats = base_folder + scan_folder + "second_model_stats.txt"
second_model_train_pred = base_folder + scan_folder + "second_model_train_pred.txt"
second_model_test_pred = base_folder + scan_folder + "second_model_test_pred.txt"
if using_decision_tree_model:
    second_model_output_rules = base_folder + scan_folder + "second_model_rules.rls"
else:
    second_model_output_rules = base_folder + scan_folder + "second_model_weights.wts"

#----------------------------
# For Fidex
global_rules_file = base_folder + scan_folder + "globalRules.json"
hiknot = 5
nbQuantLevels = 100
K_val = 1.0
dropout_hyp = 0.9
dropout_dim = 0.9
global_rules_with_test_stats = base_folder + scan_folder + "globalRulesWithStats.json"
global_rules_stats = base_folder + scan_folder + "global_rules_stats.txt"

if probability_stats:
    size_Height_proba_stat = size1D - filter_size[0][0] + 1 # Size of new image with probabilities from original image
    size_Width_proba_stat = size1D - filter_size[0][1] + 1
    output_size = (size_Height_proba_stat, size_Width_proba_stat, nb_classes + nb_channels) # Add nb_channels if adding the image for second cnn training
    nb_stats_attributes = size_Height_proba_stat*size_Width_proba_stat*(nb_classes + nb_channels) # Add nb_channels if adding the image for second cnn training
#----------------------------
# Folder for output images
rules_folder = base_folder + scan_folder + "Rules"

# Folder for heat maps
heat_maps_folder = base_folder + scan_folder + "Heat_maps"
##############################################################################

##############################################################################

# Display parameters
print("\n--------------------------------------------------------------------------")
print("Parameters :")
print("--------------------------------------------------------------------------\n")
print(f"Dataset : {dataset}")
print(f"Size : {size1D}x{size1D}x{nb_channels}")
print(f"Data type : {data_type}")
print(f"Number of attributes : ", {nb_stats_attributes})

print("Statistic :")
if histogram_stats:
    print("Histogram")
elif activation_layer_stats:
    print("Activation layer")
elif probability_stats:
    if use_multi_networks_stats:
        print("Probability with multiple networks")
    else:
        print("Probability")
else:
    print("UNKNOWN")

print("\n-------------")
print("Files :")
print("-------------")
print(f"Train data file : {train_data_file}")
print(f"Train class file : {train_class_file}")
print(f"Train prediction file : {train_pred_file}")
print(f"Test data file : {train_data_file}")
print(f"Test class file : {train_class_file}")
print(f"Test prediction file : {train_pred_file}")
print(f"Model file : {model_file}")

if with_train_cnn:
    print("\n-------------")
    print("Training :")
    print("-------------")
    print(f"Model checkpoint weights : {model_checkpoint_weights}")
    print(f"Model stats file : {model_stats}")
    print(f"Model : {model}")
    print(f"Number of iterations : {nbIt}")
    print(f"Batch size : {batch_size}")
    if activation_layer_stats:
        if with_leaky_relu:
            print("With Leaky Relu")
        else:
            print("Without Leaky Relu")

if get_images or simple_heat_map or with_stats_computation:
    print("\n-------------")
    print("Statistics :")
    print("-------------")
    print(f"Filter size : {filter_size}",)
    print(f"Stride : {stride}",)
if with_stats_computation:
    print(f"Train statistics file : {train_stats_file}")
    print(f"Test statistics file : {test_stats_file}")
    if probability_stats:
        print(f"Train statistics file with image : {train_stats_file_with_image}")
        print(f"Test statistics file with image: {test_stats_file_with_image}")
    elif histogram_stats:
        print(f"Number of bins : {nb_bins}")
        print(f"Probability threshold : {probability_thresholds}")
if (simple_heat_map and (not (with_stats_computation and histogram_stats))):
    print(f"Probability threshold : {probability_thresholds}")

if with_train_second_model:
    print("\n-------------")
    print("Second training :")
    print("-------------")
    print(f"Second model : {second_model}")
    if probability_stats and use_multi_networks_stats:
        if with_hsl:
            print("Using HSL")
        else:
            print("Not using HSL")
    print(f"Batch size second model: {batch_size_second_model}")
    print(f"Second model statistics file : {second_model_stats}")
    print(f"Second model train predictions file : {second_model_train_pred}")
    print(f"Second model output rules file : {second_model_output_rules}")

if with_global_rules:
    print("\n-------------")
    print("Fidex rules generation :")
    print("-------------")
    print(f"Global rules file : {global_rules_file}")
    print(f"Hiknot : {hiknot}")
    print(f"Number of quantization levels : {nbQuantLevels}")
    print(f"K : {K_val}")
    print(f"Dropout hyperplans : {dropout_hyp}")
    print(f"Dropout dimensions : {dropout_dim}")
    print(f"Global rules file with test statistics : {global_rules_with_test_stats}")
    print(f"Global rules statistics : {global_rules_stats}")

print("\n--------------------------------------------------------------------------")

##############################################################################

# Get data

print("\nLoading data...")

train   = np.loadtxt(train_data_file)
X_train = train.reshape(train.shape[0], size1D, size1D, nb_channels)
if data_type == "integer":
    X_train = X_train.astype('int32')
else:
    X_train = X_train.astype('float32')
print(X_train.shape)

test   = np.loadtxt(test_data_file)
X_test = test.reshape(test.shape[0], size1D, size1D, nb_channels)
if data_type == "integer":
    X_test = X_test.astype('int32')
else:
    X_test = X_test.astype('float32')
print(X_test.shape)

Y_train = np.loadtxt(train_class_file)
Y_train = Y_train.astype('int32')

Y_test  = np.loadtxt(test_class_file)
Y_test  = Y_test.astype('int32')

print("Data loaded.\n")


# Normalize if necessary
if data_type != "integer":
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
##############################################################################

# Train CNN model
if with_train_cnn:
    start_time_train_cnn = time.time()
    trainCNN(size1D, size1D, nb_channels, nb_classes, model, nbIt, batch_size, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats, with_leaky_relu)
    end_time_train_cnn = time.time()
    full_time_train_cnn = end_time_train_cnn - start_time_train_cnn
    full_time_train_cnn = "{:.6f}".format(full_time_train_cnn).rstrip("0").rstrip(".")
    print(f"\nTrain first CNN time = {full_time_train_cnn} sec")

print("Loading model...")
CNNModel = keras.saving.load_model(model_file)
print("Model loaded.")
if activation_layer_stats: # Get intermediate model
    input_channels = CNNModel.input_shape[-1]
    dummy_input = np.zeros((1, size1D, size1D, input_channels))
    _ = CNNModel(dummy_input)
    flatten_layer_output = CNNModel.get_layer("flatten").output
    intermediate_model = Model(inputs=CNNModel.inputs, outputs=flatten_layer_output)
    nb_stats_attributes = intermediate_model.output_shape[1]


##############################################################################
if test_version:
    nb_train_samples = 100
    nb_test_samples = 100
else:
    nb_train_samples = Y_train.shape[0]
    nb_test_samples = Y_test.shape[0]

# Compute histograms
if with_stats_computation:
    start_time_stats_computation = time.time()
    if histogram_stats:
        print("\nComputing train histograms...")

        # Get histograms for each train sample

        train_histograms = compute_histograms(nb_train_samples, X_train, size1D, nb_channels, CNNModel, nb_classes, filter_size, stride, nb_bins)
        print("\nTrain histograms computed.\n")

        print("Computing test histograms...")

        # Get histograms for each test sample
        test_histograms = compute_histograms(nb_test_samples, X_test, size1D, nb_channels, CNNModel, nb_classes, filter_size, stride, nb_bins)

        print("\nTest histograms computed.")
        # Save in histograms in .npy file
        print("\nSaving histograms...")
        train_histograms = train_histograms.reshape(nb_train_samples, nb_stats_attributes)
        test_histograms = test_histograms.reshape(nb_test_samples, nb_stats_attributes)
        print(train_histograms.shape)
        output_data(train_histograms, train_stats_file)
        output_data(test_histograms, test_stats_file)
        print("Histograms saved.")



    elif activation_layer_stats:

        print("\nComputing train sums of activation layer patches...")
        # Get sums for each train sample

        train_sums = compute_activation_sums(nb_train_samples, X_train, size1D, nb_channels, CNNModel, intermediate_model, nb_stats_attributes, filter_size, stride)
        # Normalization
        mean = np.mean(train_sums, axis=0)
        std = np.std(train_sums, axis=0)
        train_sums = (train_sums - mean) / std
        print("\nTrain sum of activation layer patches computed.\n")

        print("\nComputing test sums of activation layer patches...")
        # Get sums for each test sample

        test_sums = compute_activation_sums(nb_test_samples, X_test, size1D, nb_channels, CNNModel, intermediate_model, nb_stats_attributes, filter_size, stride)
        # Normalization
        test_sums = (test_sums - mean) / std
        print("\nTest sum of activation layer patches computed.\n")

        print("\nSaving sums...")
        output_data(train_sums, train_stats_file)
        output_data(test_sums, test_stats_file)
        print("Sums saved.")

    elif probability_stats: # We create an image out of the probabilities (for each class) of cropped areas of the original image
        print("\nComputing train probability images of patches...\n")
        # Get sums for each train sample

        train_probas = compute_proba_images(nb_train_samples, X_train, size1D, nb_channels, nb_classes, CNNModel, filter_size, stride)
        print(train_probas.shape)
        print("\nComputed train probability images of patches.")

        print("\nComputing test probability images of patches...\n")
        # Get sums for each train sample

        test_probas = compute_proba_images(nb_test_samples, X_test, size1D, nb_channels, nb_classes, CNNModel, filter_size, stride)
        print(test_probas.shape)
        print("\nComputed test probability images of patches.")

        print("\nSaving probability images...")

        output_data(train_probas, train_stats_file)
        output_data(test_probas, test_stats_file)
        print("Probability images saved.")

    end_time_stats_computation = time.time()
    full_time_stats_computation = end_time_stats_computation - start_time_stats_computation
    full_time_stats_computation = "{:.6f}".format(full_time_stats_computation).rstrip("0").rstrip(".")
    print(f"\nStats computation time = {full_time_stats_computation} sec")

##############################################################################
# Train second model with stats

if with_train_second_model:
    start_time_train_second_model = time.time()

    if probability_stats: # We create an image out of the probabilities (for each class) of cropped areas of the original image
        # Load probas of areas from file if necessary
        if not with_stats_computation:
            print("Loading probability stats...")
            train_probas = np.loadtxt(train_stats_file)
            train_probas = train_probas.astype('float32')
            test_probas = np.loadtxt(test_stats_file)
            test_probas = test_probas.astype('float32')
            print("Probability stats loaded.")
        #print(train_probas.shape) # (nb_train_samples, 4840) (22*22*10)
        #print(test_probas.shape) # (nb_test_samples, 4840)

        print("Adding original image...")
        train_probas = train_probas.reshape(nb_train_samples, size_Height_proba_stat, size_Width_proba_stat, nb_classes)
        X_train_reshaped = tf.image.resize(X_train, (size_Height_proba_stat, size_Width_proba_stat)) # Resize original image to the proba size
        train_probas = np.concatenate((train_probas, X_train_reshaped[:nb_train_samples]), axis=-1) # Concatenate the probas and the original image resized
        train_probas = train_probas.reshape(nb_train_samples, -1) # flatten for export

        test_probas = test_probas.reshape(nb_test_samples, size_Height_proba_stat, size_Width_proba_stat, nb_classes)
        X_test_reshaped = tf.image.resize(X_test, (size_Height_proba_stat, size_Width_proba_stat)) # Resize original image to the proba size
        test_probas = np.concatenate((test_probas, X_test_reshaped[:nb_test_samples]), axis=-1) # Concatenate the probas and the original image resized
        test_probas = test_probas.reshape(nb_test_samples, -1) # flatten for export

        # print(train_probas.shape) #(nb_train_samples, 5324) (22*22*11)
        # print(test_probas.shape)  #(nb_test_samples, 5324)

        # Save proba stats data with original image added
        output_data(train_probas, train_stats_file_with_image)
        output_data(test_probas, test_stats_file_with_image)

        train_stats_file = train_stats_file_with_image
        test_stats_file = test_stats_file_with_image

        print("original image added.")

        if second_model == "cnn":
            # Pass on the DIMLP layer
            train_probas_h1, mu, sigma = compute_first_hidden_layer("train", train_probas, K_val, nbQuantLevels, hiknot, second_model_output_rules)
            test_probas_h1 = compute_first_hidden_layer("test", test_probas, K_val, nbQuantLevels, hiknot, mu=mu, sigma=sigma)
            train_probas_h1 = train_probas_h1.reshape((nb_train_samples,)+output_size) #(100, 26, 26, 13)
            print("train_probas_h1 reshaped : ", train_probas_h1.shape)
            test_probas_h1 = test_probas_h1.reshape((nb_test_samples,)+output_size)
            #print(train_probas.shape)  # (nb_train_samples, 22, 22, 10)
            #print(test_probas.shape)  # (nb_train_samples, 22, 22, 10)
            second_model_file = base_folder + scan_folder + "scanSecondModel.keras"
            second_model_checkpoint_weights = base_folder + scan_folder + "weightsSecondModel.weights.h5"

            if not use_multi_networks_stats: # Train with a CNN now
                trainCNN(size_Height_proba_stat, size_Width_proba_stat, nb_classes+nb_channels, nb_classes, "small", 80, batch_size_second_model, second_model_file, second_model_checkpoint_weights, train_probas_h1, Y_train, test_probas_h1, Y_test, second_model_train_pred, second_model_test_pred, second_model_stats, False, True)

            else: # Create nb_classes networks and gather best probability among them. The images keep only the probabilities of areas for one class and add B&W image (or H and S of HSL)

                if test_version:
                    nbIt_current = 2
                else:
                    nbIt_current = 80
                models_folder = "Models/"
                # Create folder for all models
                if os.path.exists(base_folder + scan_folder + models_folder):
                    shutil.rmtree(base_folder + scan_folder + models_folder)
                os.makedirs(base_folder + scan_folder + models_folder)

                # Create each dataset
                for i in range(nb_classes):
                    print("Creating dataset n°",i,"...")

                    original_img_transformed_reshaped_train = X_train_reshaped # (100, 26, 26, 3)
                    original_img_transformed_reshaped_test = X_test_reshaped
                    if nb_channels == 3:
                        if with_hsl: # Transform in HSL(hsv in fact)
                            original_img_transformed_reshaped_train = tf.image.rgb_to_hsv(original_img_transformed_reshaped_train)
                            original_img_transformed_reshaped_test = tf.image.rgb_to_hsv(original_img_transformed_reshaped_test)
                        elif not with_rg: # Transform in black and white
                            original_img_transformed_reshaped_train = tf.image.rgb_to_grayscale(original_img_transformed_reshaped_train)
                            original_img_transformed_reshaped_test = tf.image.rgb_to_grayscale(original_img_transformed_reshaped_test)

                    # Create train data for each model
                    built_data_train = np.empty((nb_train_samples, size_Height_proba_stat, size_Width_proba_stat, 3))
                    # Add probas on first channel
                    built_data_train[:,:,:,0] = train_probas_h1[:,:,:,i]
                    # Add H and S on last 2 channels (or R and G)
                    if (with_hsl or with_rg) and nb_channels == 3:
                        built_data_train[:,:,:,1] = original_img_transformed_reshaped_train[..., 0]
                        built_data_train[:,:,:,2] = original_img_transformed_reshaped_train[..., 1]

                    else: # Add 1-probas and B&W on last 2 channels
                        built_data_train[:,:,:,1] = 1-train_probas_h1[:,:,:,i]
                        built_data_train[:,:,:,2] = original_img_transformed_reshaped_train[..., 0]
                    # built_data_train :  (100, 26, 26, 3)

                    # Create classes for these datas
                    built_Y_train = np.zeros((nb_train_samples, 2), dtype=int)
                    built_Y_train[Y_train[:, i] == 1, 0] = 1  # If condition is True, set [1, 0]
                    built_Y_train[Y_train[:, i] != 1, 1] = 1  # If condition is False, set [0, 1]
                    current_model_train_pred = base_folder + scan_folder + models_folder + "second_model_train_pred_" + str(i) + ".txt"
                    data_filename = "train_probability_images_with_original_img_" + str(i) + ".txt"
                    class_filename = "Y_train_probability_images_with_original_img_" + str(i) + ".txt"
                    built_data_train_flatten = built_data_train.reshape(nb_train_samples, size_Height_proba_stat*size_Width_proba_stat*3)

                    # output new train data
                    output_data(built_data_train_flatten, base_folder + scan_folder + models_folder + data_filename)
                    output_data(built_Y_train, base_folder + scan_folder + models_folder + class_filename)

                    # Create test data for each model
                    built_data_test = np.empty((nb_test_samples, size_Height_proba_stat, size_Width_proba_stat, 3))
                    # Add probas on first channel
                    built_data_test[:,:,:,0] = test_probas_h1[:,:,:,i]
                    # Add H and S on last 2 channels
                    if (with_hsl or with_rg) and nb_channels == 3:
                        built_data_test[:,:,:,1] = original_img_transformed_reshaped_test[..., 0]
                        built_data_test[:,:,:,2] = original_img_transformed_reshaped_test[..., 1]
                    else: # Add 1-probas and B&W on last 2 channels
                        built_data_test[:,:,:,1] = 1-test_probas_h1[:,:,:,i]
                        built_data_test[:,:,:,2] = original_img_transformed_reshaped_test[..., 0]

                    # Create classes for these datas
                    built_Y_test = np.zeros((nb_test_samples, 2), dtype=int)
                    built_Y_test[Y_test[:, i] == 1, 0] = 1  # If condition is True, set [1, 0]
                    built_Y_test[Y_test[:, i] != 1, 1] = 1  # If condition is False, set [0, 1]
                    current_model_test_pred = base_folder + scan_folder + models_folder + "second_model_test_pred_" + str(i) + ".txt"
                    data_filename = "test_probability_images_with_original_img_" + str(i) + ".txt"
                    class_filename = "Y_test_probability_images_with_original_img_" + str(i) + ".txt"
                    built_data_test_flatten = built_data_test.reshape(nb_test_samples, size_Height_proba_stat*size_Width_proba_stat*3)

                    # output new test data
                    output_data(built_data_test_flatten, base_folder + scan_folder + models_folder + data_filename)
                    output_data(built_Y_test, base_folder + scan_folder + models_folder + class_filename)

                    current_model_stats = base_folder + scan_folder + models_folder + "second_model_stats_" + str(i) +".txt"
                    current_model_file = base_folder + scan_folder + models_folder + "scanSecondModel_" + str(i) +".keras"
                    current_model_checkpoint_weights = base_folder + scan_folder + models_folder + "weightsSecondModel_" + str(i) +".weights.h5"

                    print("Dataset n°",i," created.")
                    # Train new model
                    trainCNN(size_Height_proba_stat, size_Width_proba_stat, 3, 2, "VGG", nbIt_current, batch_size_second_model, current_model_file, current_model_checkpoint_weights, built_data_train, built_Y_train, built_data_test, built_Y_test, current_model_train_pred, current_model_test_pred, current_model_stats, False, True)
                    print("Dataset n°",i," trained.")
                # Create test and train predictions

                train_pred_files = [f"{base_folder}{scan_folder}{models_folder}second_model_train_pred_{i}.txt" for i in range(nb_classes)]
                test_pred_files = [f"{base_folder}{scan_folder}{models_folder}second_model_test_pred_{i}.txt" for i in range(nb_classes)]

                # Gathering predictions for train and test
                print("Gathering train predictions...")
                gathering_predictions(train_pred_files, second_model_train_pred)
                print("Gathering test predictions...")
                gathering_predictions(test_pred_files, second_model_test_pred)

                # Compute and save predictions of the second (gathering of all models) model
                second_model_train_preds = np.argmax(np.loadtxt(second_model_train_pred), axis=1)
                second_model_test_preds = np.argmax(np.loadtxt(second_model_test_pred), axis=1)

                # Compute and save train and test accuracies of the second model
                train_accuracy = 0
                for i in range(nb_train_samples):
                    if np.argmax(Y_train[i]) == second_model_train_preds[i]:
                        train_accuracy += 1
                train_accuracy /= nb_train_samples

                test_accuracy = 0
                for i in range(nb_test_samples):
                    if np.argmax(Y_test[i]) == second_model_test_preds[i]:
                        test_accuracy += 1
                test_accuracy /= nb_test_samples

                with open(second_model_stats, "w") as myFile:
                    print("Train score : ", train_accuracy)
                    myFile.write(f"Train score : {train_accuracy}\n")

                    print("Test score : ", test_accuracy)
                    myFile.write(f"Test score : {test_accuracy}\n")

                print("Data sets created and all models trained.")

        else: # Using a Ranfom Forests to train the probas with images

            command = (
                f'--train_data_file {train_stats_file} '
                f'--train_class_file {train_class_file} '
                f'--test_data_file {test_stats_file} '
                f'--test_class_file {test_class_file} '
                f'--stats_file {second_model_stats} '
                f'--train_pred_outfile {second_model_train_pred} '
                f'--test_pred_outfile {second_model_test_pred} '
                f'--nb_attributes {nb_stats_attributes} '
                f'--nb_classes {nb_classes} '
                f'--root_folder . '
                )
            command += f'--rules_outfile {second_model_output_rules} '
            status = randForestsTrn(command)

    else: # (not with probabilities of areas)

        # Train model
        command = (
            f'--train_data_file {train_stats_file} '
            f'--train_class_file {train_class_file} '
            f'--test_data_file {test_stats_file} '
            f'--test_class_file {test_class_file} '
            f'--stats_file {second_model_stats} '
            f'--train_pred_outfile {second_model_train_pred} '
            f'--test_pred_outfile {second_model_test_pred} '
            f'--nb_attributes {nb_stats_attributes} '
            f'--nb_classes {nb_classes} '
            f'--root_folder . '
            )

        if using_decision_tree_model:
            command += f'--rules_outfile {second_model_output_rules} '
        else:
            command += f'--weights_outfile {second_model_output_rules} '

        print("\nTraining second model...\n")

        # match second_model:
        #     case "randomForests":
        #         status = randForestsTrn(command)
        #     case "gradientBoosting":
        #         status = gradBoostTrn(command)
        #     case "dimlpTrn":
        #         status = dimlp.dimlpTrn(command)
        #     case "dimlpBT":
        #         command += '--nb_dimlp_nets 15 '
        #         command += '--hidden_layers [25] '
        #         if test_version:
        #             command += '--nb_epochs 10 '
        #         status = dimlp.dimlpBT(command)

        if second_model == "randomForests":
            status = randForestsTrn(command)
        elif second_model == "gradientBoosting":
            status = gradBoostTrn(command)
        elif second_model == "dimlpTrn":
            status = dimlp.dimlpTrn(command)
        elif second_model == "dimlpBT":
            command += '--nb_dimlp_nets 15 '
            command += '--hidden_layers [25] '
            if test_version:
                command += '--nb_epochs 10 '
            status = dimlp.dimlpBT(command)

        if status != -1:
            print("\nSecond model trained.")

    end_time_train_second_model = time.time()
    full_time_train_second_model = end_time_train_second_model - start_time_train_second_model
    full_time_train_second_model = "{:.6f}".format(full_time_train_second_model).rstrip("0").rstrip(".")
    print(f"\nTrain second model time = {full_time_train_second_model} sec")

if histogram_stats:
    # Define attributes file for histograms
    with open(attributes_file, "w") as myFile:
        for i in range(nb_classes):
            for j in probability_thresholds:
                myFile.write(f"P_{i}>={j:.6g}\n")


if probability_stats:
    train_stats_file = train_stats_file_with_image
    test_stats_file = test_stats_file_with_image

##############################################################################
# Compute global rules

if with_global_rules:
    start_time_global_rules = time.time()
    command = (
        f'--train_data_file {train_stats_file} '
        f'--train_pred_file {second_model_train_pred} '
        f'--train_class_file {train_class_file} '
        f'--nb_classes {nb_classes} '
        f'--global_rules_outfile {global_rules_file} '
        f'--nb_attributes {nb_stats_attributes} '
        f'--heuristic 1 '
        f'--nb_threads 8 '
        f'--max_iterations 25 '
        f'--nb_quant_levels {nbQuantLevels} '
        f'--dropout_dim {dropout_dim} '
        f'--dropout_hyp {dropout_hyp} '
    )
    if histogram_stats:
        command += f'--attributes_file {attributes_file} '
    if using_decision_tree_model:
        command += f'--rules_file {second_model_output_rules} '
    else:
        command += f'--weights_file {second_model_output_rules} '

    print("\nComputing global rules...\n")
    status = fidex.fidexGloRules(command)
    if status != -1:
        print("\nGlobal rules computed.")

    command = (
        f'--test_data_file {test_stats_file} '
        f'--test_pred_file {second_model_test_pred} '
        f'--test_class_file {test_class_file} '
        f'--nb_classes {nb_classes} '
        f'--global_rules_file {global_rules_file} '
        f'--nb_attributes {nb_stats_attributes} '
        f'--global_rules_outfile {global_rules_with_test_stats} '
        f'--stats_file {global_rules_stats}'
    )

    print("\nComputing statistics on global rules...\n")
    status = fidex.fidexGloStats(command)
    if status != -1:
        print("\nStatistics computed.")

    end_time_global_rules = time.time()
    full_time_global_rules = end_time_global_rules - start_time_global_rules
    full_time_global_rules = "{:.6f}".format(full_time_global_rules).rstrip("0").rstrip(".")
    print(f"\nGlobal rules time = {full_time_global_rules} sec")

##############################################################################
# Get images explaining and illustrating the samples and rules

if get_images:
    print("Generation of images...")
    # Get rules and attributes
    global_rules = getRules(global_rules_file)
    if histogram_stats:
        attributes = get_attribute_file(attributes_file, nb_stats_attributes)[0]

    # Create folder for all rules
    if os.path.exists(rules_folder):
        shutil.rmtree(rules_folder)
    os.makedirs(rules_folder)

    # For each rule we get filter images for train samples covering the rule
    good_classes = [2,3,5]
    conteur = 0
    for id,rule in enumerate(global_rules[0:50]):

        # if conteur == 50:
        #     exit()
        # if rule.target_class not in good_classes:
        #     continue
        # else:
        #     conteur += 1

        if histogram_stats:
            rule.include_X = False
            for ant in rule.antecedents:
                ant.attribute = attributes[ant.attribute] # Get true name of attribute
        elif probability_stats:
            rule.include_X = False

        # Create folder for this rule
        rule_folder = f"{rules_folder}/rule_{id}_class_{classes[rule.target_class]}"
        if os.path.exists(rule_folder):
            shutil.rmtree(rule_folder)
        os.makedirs(rule_folder)

        # Add a readme containing the rule
        readme_file = rule_folder+'/Readme.md'
        rule_to_print = copy.deepcopy(rule)

        if histogram_stats:
            # Change antecedent with real class names
            for antecedent in rule_to_print.antecedents:
                match = re.match(HISTOGRAM_ANTECEDENT_PATTERN, antecedent.attribute)
                if match:
                    class_id = int(match.group(1))
                    pred_threshold = match.group(2)
                    class_name = classes[class_id]  # Get the class name
                    antecedent.attribute = f"P_{class_name}>={pred_threshold}"
                else:
                    raise ValueError("Wrong antecedent...")
        elif probability_stats:
            # attribut_de_test = 2024 # -> classe :  0, Height :  8, Width :  8

            # Change antecedent with area and class involved

            # Scales of changes of original image to reshaped image
            scale_h = size1D / size_Height_proba_stat
            scale_w = size1D / size_Width_proba_stat
            for antecedent in rule_to_print.antecedents: # TODO : handle stride, different filter sizes, etc
                # area_index (size_Height_proba_stat, size_Width_proba_stat) : 0 : (1,1), 1: (1,2), ...
                channel_id = antecedent.attribute % (nb_classes + nb_channels) # (probas of each class + image rgb concatenated)
                area_number = antecedent.attribute // (nb_classes + nb_channels)
                # channel_id = attribut_de_test % (nb_classes + nb_channels)
                # area_number = attribut_de_test // (nb_classes + nb_channels)
                area_Height = area_number // size_Width_proba_stat
                area_Width = area_number % size_Width_proba_stat
                # print("classe : ", channel_id)
                # print("Height : ", area_Height)
                # print("Width : ", area_Width)
                # exit()
                if channel_id < nb_classes: #Proba of area
                    class_name = classes[channel_id]
                    antecedent.attribute = f"P_class_{class_name}_area_[{area_Height}-{area_Height+filter_size[0][0]-1}]x[{area_Width}-{area_Width+filter_size[0][1]-1}]"
                else:
                    channel = channel_id - nb_classes #Pixel in concatenated original rgb image
                    # Conversion of resized coordinates into originals
                    height_original = round(area_Height * scale_h)
                    width_original = round(area_Width * scale_w)
                    antecedent.attribute = f"Pixel_{height_original}x{width_original}x{channel}"

        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as file:
            file.write(str(rule_to_print))

        # Create full image with all filters and save it
        for img_id in rule.covered_samples[0:10]:
            img = X_train[img_id]
            if histogram_stats:
                highlighted_image = highlight_area_histograms(CNNModel, img, data_type, filter_size, stride, rule, classes)
            elif activation_layer_stats:
                highlighted_image = highlight_area_activations_sum(CNNModel, intermediate_model, img, data_type, rule, filter_size, stride, classes)
            elif probability_stats:
                highlighted_image = highlight_area_probability_image(img, data_type, rule, size1D, size_Height_proba_stat, size_Width_proba_stat, filter_size, classes, nb_channels)
            highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png") # Save image


##############################################################################
# Get images explaining and illustrating the samples without rules or training twice

if simple_heat_map: # Only for one filter !

    # Create heat map folder
    if os.path.exists(heat_maps_folder):
        shutil.rmtree(heat_maps_folder)
    os.makedirs(heat_maps_folder)

    for id,img in enumerate(X_test[0:100]):
        heat_maps_img = get_heat_maps(CNNModel, img, data_type, filter_size, stride, probability_thresholds, classes)
        heat_maps_img.savefig(f"{heat_maps_folder}/sample_{id}.png")


end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
