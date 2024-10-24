# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:24:08 2024

@author: jean-marc.boutay

This script performs the training and evaluation of a Convolutional Neural Network (CNN) on a specified dataset
(either MNIST or CIFAR). It includes options for training the CNN, computing histograms from CNN predictions on filtered areas,
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
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
import numpy as np
import shutil
import re
import copy
from constants import HISTOGRAM_ANTECEDENT_PATTERN
from utils import trainCNN, compute_histograms, output_data, getRules, highlight_area, getProbabilityThresholds

np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
from trainings import randForestsTrn
from trainings.trnFun import get_attribute_file

###############################################################

start_time = time.time()


# What to launch
with_train_cnn = False
with_hist_computation = True
with_train_second_model = True
with_global_rules = True
get_images = True


##############################################################################

# Which dataset to launch
dataset = "MNIST"
#dataset = "CIFAR"

if dataset == "MNIST":     # for MNIST images
    size1D             = 28
    nb_channels         = 1
    nb_classes = 10
    base_folder = "Mnist/"
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
    nb_classes = 10
    base_folder = "Cifar/"
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


##############################################################################

# Parameters

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

global_rules_file = base_folder + "Scan/globalRules.json"
attributes_file = base_folder + "Scan/attributes.txt"

# If we train :
model_checkpoint_weights = base_folder + "Scan/weightsModel.weights.h5"
model_stats = base_folder + "Scan/stats_model.txt"
resnet = False
nbIt = 4

# For histogram computation
filter_size = [7,7] # Size of filter applied to the image
stride = 1 # shift between each filter
nb_bins = 9 # Number of bins wanted (ex: NProb>=0.1, NProb>=0.2, etc.)
nb_histogram_attributes = nb_classes*nb_bins

# For Fidex
hiknot = 5
nbQuantLevels = 100
K_val = 1.0
dropout_hyp = 0.9
dropout_dim = 0.9

# Folder for output images
rules_folder = base_folder + "Scan/Rules"


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

##############################################################################

# Train CNN model
if with_train_cnn:
    trainCNN(size1D, nb_channels, nb_classes, resnet, nbIt, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats)


CNNModel = keras.saving.load_model(model_file)


##############################################################################
# Compute histograms

if with_hist_computation:

    print("\nComputing train histograms...")

    # Get histograms for each train sample
    #nb_train_samples = Y_train.shape[0]
    nb_train_samples = 100

    train_histograms = compute_histograms(nb_train_samples, X_train, size1D, nb_channels, CNNModel, nb_classes, filter_size, stride, nb_bins)
    print("\nTrain histograms computed.\n")

    print("Computing test histograms...")

    # Get histograms for each test sample
    #nb_test_samples = Y_test.shape[0]
    nb_test_samples = 100
    test_histograms = compute_histograms(nb_test_samples, X_test, size1D, nb_channels, CNNModel, nb_classes, filter_size, stride, nb_bins)

    print("\nTest histograms computed.")
    # Save in histograms in .npy file
    print("\nSaving histograms...")
    train_histograms = train_histograms.reshape(nb_train_samples, nb_histogram_attributes)
    test_histograms = test_histograms.reshape(nb_test_samples, nb_histogram_attributes)
    output_data(train_histograms, train_histogram_file)
    output_data(test_histograms, test_histogram_file)
    print("Histograms saved.")

##############################################################################
# Train second model with histograms

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

# Define attributes file for histograms
probability_thresholds = getProbabilityThresholds(nb_bins)
with open(attributes_file, "w") as myFile:
    for i in range(nb_classes):
        for j in probability_thresholds:
            myFile.write(f"P_{i}>={j:.6g}\n")


##############################################################################
# Compute global rules

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


##############################################################################
# Get images explaining and illustrating the samples and rules

if get_images:
    # Get rules and attributes
    global_rules = getRules(global_rules_file)
    attributes = get_attribute_file(attributes_file, nb_histogram_attributes)[0]

    # Create folder for all rules
    if os.path.exists(rules_folder):
        shutil.rmtree(rules_folder)
    os.makedirs(rules_folder)

    # For each rule we get filter images for train samples covering the rule
    for id,rule in enumerate(global_rules[0:10]):
        rule.include_X = False
        for ant in rule.antecedents:
            ant.attribute = attributes[ant.attribute] # Get true name of attribute

        # Create folder for this rule
        rule_folder = f"{rules_folder}/rule_{id}_class_{classes[rule.target_class]}"
        if os.path.exists(rule_folder):
            shutil.rmtree(rule_folder)
        os.makedirs(rule_folder)

        # Add a readme containing the rule
        readme_file = rule_folder+'/Readme.md'
        rule_to_print = copy.deepcopy(rule)
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
        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as file:
            file.write(str(rule_to_print))

        # Create full image with all filters and save it
        for img_id in rule.covered_samples:
            img = X_train[img_id]
            highlighted_image = highlight_area(CNNModel, img, filter_size, stride, rule, classes)
            highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png") # Save image

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
