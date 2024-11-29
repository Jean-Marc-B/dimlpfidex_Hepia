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
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
import numpy as np
import shutil
import re
import copy
from tensorflow.keras import Model
from constants import HISTOGRAM_ANTECEDENT_PATTERN
from utils import trainCNN, compute_histograms, compute_activation_sums, compute_proba_images, output_data, getRules, highlight_area_histograms, highlight_area_activations_sum, highlight_area_probability_image, getProbabilityThresholds, get_heat_maps, compute_first_hidden_layer

np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
from dimlpfidex import dimlp
from trainings import randForestsTrn, gradBoostTrn
from trainings.trnFun import get_attribute_file

###############################################################

start_time = time.time()


# What to launch
test_version = False # Whether to launch with minimal data



# Training CNN:
with_train_cnn = True

# Stats computation and second model training:
histogram_stats = False
activation_layer_stats = False
probability_stats = True

if histogram_stats + activation_layer_stats + probability_stats != 1:
    raise ValueError("Error, you need to specify one of histogram_stats, activation_layer_stats and probability_stats.")


with_stats_computation = False
with_train_second_model = False

# Rule computation:
with_global_rules = False

# Image generation:
get_images = False # With histograms
simple_heat_map = False # Only evaluation on patches


##############################################################################

# Which dataset to launch
#dataset = "MNIST"
dataset = "CIFAR"

if dataset == "MNIST":     # for MNIST images
    size1D             = 28
    nb_channels         = 1
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
    base_folder = "CifarNewResnet/"
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
    scan_folder += "Probability_Images/"

#----------------------------
# Files
train_data_file = base_folder + "trainData.txt"
train_class_file = base_folder + "trainClass.txt"
test_data_file = base_folder + "testData.txt"
test_class_file = base_folder + "testClass.txt"
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
else:
    model = "resnet"
    nbIt = 80

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
second_model = "cnn"
if not probability_stats:
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

if probability_stats:
    size_Height_proba_stat = size1D - filter_size[0][0] + 1 # Size of new image qith probabilities from original image
    size_Width_proba_stat = size1D - filter_size[0][1] + 1
    nb_stats_attributes = size_Height_proba_stat*size_Width_proba_stat*nb_classes
    output_size = (size_Height_proba_stat, size_Width_proba_stat, nb_classes)
#----------------------------
# Folder for output images
rules_folder = base_folder + scan_folder + "Rules"

# Folder for heat maps
heat_maps_folder = base_folder + scan_folder + "Heat_maps"
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
    start_time_train_cnn = time.time()
    trainCNN(size1D, size1D, nb_channels, nb_classes, model, nbIt, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats, with_leaky_relu)
    end_time_train_cnn = time.time()
    full_time_train_cnn = end_time_train_cnn - start_time_train_cnn
    full_time_train_cnn = "{:.6f}".format(full_time_train_cnn).rstrip("0").rstrip(".")
    print(f"\nTrain first CNN time = {full_time_train_cnn} sec")

CNNModel = keras.saving.load_model(model_file)
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

    elif probability_stats:
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

if test_version:
    train_class_file = base_folder + "trainClass_temp.txt"
    test_class_file = base_folder + "testClass_temp.txt"

if with_train_second_model:
    start_time_train_second_model = time.time()

    if probability_stats:
        if not with_stats_computation:
            train_probas = np.loadtxt(train_stats_file)
            train_probas = train_probas.astype('float32')
            test_probas = np.loadtxt(test_stats_file)
            test_probas = test_probas.astype('float32')
        #print(train_probas.shape) # (nb_train_samples, 4840)
        #print(test_probas.shape) # (nb_test_samples, 4840)
        train_probas_h1, mu, sigma = compute_first_hidden_layer("train", train_probas, K_val, nbQuantLevels, hiknot, second_model_output_rules)
        test_probas_h1 = compute_first_hidden_layer("test", test_probas, K_val, nbQuantLevels, hiknot, mu=mu, sigma=sigma)
        train_probas_h1 = train_probas_h1.reshape((nb_train_samples,)+output_size)
        test_probas_h1 = test_probas_h1.reshape((nb_test_samples,)+output_size)
        #print(train_probas.shape)  # (nb_train_samples, 22, 22, 10)
        #print(test_probas.shape)  # (nb_train_samples, 22, 22, 10)
        second_model_file = base_folder + scan_folder + "scanSecondModel.keras"
        second_model_checkpoint_weights = base_folder + scan_folder + "weightsSecondModel.weights.h5"

        trainCNN(size_Height_proba_stat, size_Width_proba_stat, nb_classes, nb_classes, "small", 80, second_model_file, second_model_checkpoint_weights, train_probas_h1, Y_train[0:nb_train_samples], test_probas_h1, Y_test[0:nb_test_samples], second_model_train_pred, second_model_test_pred, second_model_stats, False, True)

    else:

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
        f'--nb_threads 4 '
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

    end_time_global_rules = time.time()
    full_time_global_rules = end_time_global_rules - start_time_global_rules
    full_time_global_rules = "{:.6f}".format(full_time_global_rules).rstrip("0").rstrip(".")
    print(f"\nGlobal rules time = {full_time_global_rules} sec")

##############################################################################
# Get images explaining and illustrating the samples and rules

if get_images:
    # Get rules and attributes
    global_rules = getRules(global_rules_file)
    if histogram_stats:
        attributes = get_attribute_file(attributes_file, nb_stats_attributes)[0]

    # Create folder for all rules
    if os.path.exists(rules_folder):
        shutil.rmtree(rules_folder)
    os.makedirs(rules_folder)

    # For each rule we get filter images for train samples covering the rule
    good_classes = [2,3,7]
    conteur = 0
    for id,rule in enumerate(global_rules):

        if conteur == 50:
            exit()
        if rule.target_class not in good_classes:
            continue
        else:
            conteur += 1

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
            # Change antecedent with area and class involved
            for antecedent in rule_to_print.antecedents: # TODO : handle stride, different filter sizes, etc
                # area_index (size_Height_proba_stat, size_Width_proba_stat) : 0 : (1,1), 1: (1,2), ...
                class_name = classes[antecedent.attribute % nb_classes]
                area_number = antecedent.attribute // nb_classes
                area_Height = area_number // size_Width_proba_stat
                area_Width = area_number % size_Width_proba_stat
                antecedent.attribute = f"P_class_{class_name}_area_[{area_Height}-{area_Height+filter_size[0][0]-1}]x[{area_Width}-{area_Width+filter_size[0][1]-1}]"

        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as file:
            file.write(str(rule_to_print))

        # Create full image with all filters and save it
        for img_id in rule.covered_samples[0:10]:
            img = X_train[img_id]
            if histogram_stats:
                highlighted_image = highlight_area_histograms(CNNModel, img, filter_size, stride, rule, classes)
            elif activation_layer_stats:
                highlighted_image = highlight_area_activations_sum(CNNModel, intermediate_model, img, rule, filter_size, stride, classes)
            elif probability_stats:
                highlighted_image = highlight_area_probability_image(img, rule, size_Width_proba_stat, filter_size, classes)
            highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png") # Save image


##############################################################################
# Get images explaining and illustrating the samples without rules or training twice

if simple_heat_map: # Only for one filter !

    # Create heat map folder
    if os.path.exists(heat_maps_folder):
        shutil.rmtree(heat_maps_folder)
    os.makedirs(heat_maps_folder)

    for id,img in enumerate(X_test[0:100]):
        heat_maps_img = get_heat_maps(CNNModel, img, filter_size, stride, probability_thresholds, classes)
        heat_maps_img.savefig(f"{heat_maps_folder}/sample_{id}.png")


end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
