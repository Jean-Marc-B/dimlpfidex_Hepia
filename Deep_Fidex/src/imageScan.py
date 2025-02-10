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
import argparse
import numpy as np
import keras
import tensorflow as tf

# Import intern modules
import utils.config as config
from utils.config import *
from utils.utils import *
from dimlpfidex import fidex, dimlp
from trainings import randForestsTrn, gradBoostTrn

# Import step files
from train import train_cnn
from stats import compute_stats
from second_train import train_second_model
from generate_rules import generate_rules
from images import generate_explaining_images
from heatmap import generate_heatmaps

# GPU arguments
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(seed=None)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

###############################################################

start_time = time.time()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image classification with explanation.")

    # Mandatory arguments
    parser.add_argument(
        "--dataset", type=str, required=True, choices=config.AVAILABLE_DATASETS,
        help=f"Choose a dataset in : {', '.join(config.AVAILABLE_DATASETS)}"
    )
    parser.add_argument(
        "--statistic", type=str, required=True, choices=config.AVAILABLE_STATISTICS,
        help=f"Choose a statistic in : {', '.join(config.AVAILABLE_STATISTICS)}"
    )

    # Optionals arguments
    parser.add_argument("--test", action="store_true", help="Test mode") # Launch with minimal version
    parser.add_argument("--train", action="store_true", help="Train a CNN")
    parser.add_argument("--stats", action="store_true", help="Compute statistics") # Stats computation
    parser.add_argument("--second_train", action="store_true", help="Train a second model")
    parser.add_argument("--rules", action="store_true", help="Compute global rules")
    parser.add_argument("--images", action="store_true", help="Generate explaining images")
    parser.add_argument("--heatmap", action="store_true", help="Generate a heatmap") # Only evaluation on patches

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()

    # Check on inline parameters

    # If generating rules, we force CPU only
    if args.rules:
        if any((args.train, args.stats, args.second_train, args.images)):
            raise ValueError("Global rules have to be computed alone because we don't want to use a GPU during global rules generation.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Get Parameter configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = config.load_config(args, script_dir)

    ##############################################################################

    # Get data

    print("\nLoading data...")

    train   = np.loadtxt(cfg["train_data_file"])
    X_train = train.reshape(train.shape[0], cfg["size1D"], cfg["size1D"], cfg["nb_channels"])
    if cfg["data_type"] == "integer":
        X_train = X_train.astype('int32')
    else:
        X_train = X_train.astype('float32')
    print(X_train.shape)

    test   = np.loadtxt(cfg["test_data_file"])
    X_test = test.reshape(test.shape[0], cfg["size1D"], cfg["size1D"], cfg["nb_channels"])
    if cfg["data_type"] == "integer":
        X_test = X_test.astype('int32')
    else:
        X_test = X_test.astype('float32')
    print(X_test.shape)

    Y_train = np.loadtxt(cfg["train_class_file"])
    Y_train = Y_train.astype('int32')

    Y_test  = np.loadtxt(cfg["test_class_file"])
    Y_test  = Y_test.astype('int32')

    print("Data loaded.\n")


    # Normalize if necessary
    if cfg["data_type"] != "integer":
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
    ##############################################################################

    ##############################################################################

    # TRAINING
    if args.train:
        train_cnn(cfg, X_train, Y_train, X_test, Y_test)

    print("Loading model...")
    CNNModel = keras.saving.load_model(cfg["model_file"])
    print("Model loaded.")

    # Get intermediate model if with activation_layer
    intermediate_model = None
    if args.statistic == "activation_layer": # Get intermediate model
        input_channels = CNNModel.input_shape[-1]
        dummy_input = np.zeros((1, cfg["size1D"], cfg["size1D"], input_channels))
        _ = CNNModel(dummy_input)
        flatten_layer_output = CNNModel.get_layer("flatten").output
        intermediate_model = Model(inputs=CNNModel.inputs, outputs=flatten_layer_output)
        cfg["nb_stats_attributes"] = intermediate_model.output_shape[1]

    # Define the number of train and test samples
    if args.test:
        nb_train_samples = 100
        nb_test_samples = 100
    else:
        nb_train_samples = Y_train.shape[0]
        nb_test_samples = Y_test.shape[0]

    # STATISTICS
    if args.stats:
        compute_stats(cfg, X_train, Y_train, X_test, Y_test, nb_train_samples, nb_test_samples, CNNModel, intermediate_model, args)

    # TRAIN SECOND MODEL
    if args.second_train:
        train_second_model(cfg, X_train, Y_train, X_test, Y_test, nb_train_samples, nb_test_samples, CNNModel, intermediate_model, args)

    # Define attributes file for histograms
    if args.statistic == "histogram":
        with open(cfg["attributes_file"], "w") as myFile:
            for i in range(cfg["nb_classes"]):
                for j in PROBABILITY_THRESHOLDS:
                    myFile.write(f"P_{i}>={j:.6g}\n")

    # Update stats file
    if args.statistic in ["probability", "probability_multi_nets"]:
        cfg["train_stats_file"] = cfg["train_stats_file_with_image"]
        cfg["test_stats_file"] = cfg["test_stats_file_with_image"]


    # GENERATE GLOBAL RULES
    if args.rules:
        generate_rules(cfg, args)

    # GENERATION OF EXPLAINING IMAGES ILLUSTRATING SAMPLES AND RULES
    if args.images:
        generate_explaining_images(cfg, X_train, Y_train, CNNModel, intermediate_model, args)

    # HEATMAP
    if args.heatmap:
        generate_heatmaps(cfg, X_test, CNNModel)

    end_time = time.time()
    full_time = end_time - start_time
    print(f"\nFull execution time = {full_time:.2f} sec")
