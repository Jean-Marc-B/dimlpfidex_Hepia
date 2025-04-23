# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:10 2025

@author: jean-marc.boutay
"""

import argparse
import time
import os

# Tensorflow modules
from tensorflow.keras.models import load_model, Model


# Import intern modules
import utils.config as config
from utils.config import *
from utils.utils import *
from train import train_model
from generate_rules import generate_rules
from images import generate_explaining_images

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Initialize random generator of numpy
np.random.seed(seed=None)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image classification with explanation based on first convolution.")

    # Mandatory arguments
    parser.add_argument(
        "--dataset", type=str, required=True, choices=config.AVAILABLE_DATASETS,
        help=f"Choose a dataset in : {', '.join(config.AVAILABLE_DATASETS)}"
    )

    # Optional arguments
    parser.add_argument("--test", action="store_true", help="Test mode") # Launch with minimal version
    parser.add_argument("--train", action="store_true", help="Train a CNN")
    parser.add_argument("--get_data", action="store_true", help="Get data after first convolution")
    parser.add_argument("--second_train", action="store_true", help="Train CNN with a DIMLP layer")
    parser.add_argument("--images", type=check_positive, metavar="N", help="Generate N explaining images", default=None)
    parser.add_argument("--rules", action="store_true", help="Compute global rules")
    parser.add_argument("--each_class", action="store_true", help="Generate N images per class")
    parser.add_argument("--folder_sufix", type=str, help="Add a sufix to the folder name")
    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()

    # Get user arguments
    args = parse_arguments()
    args.statistic = "convDimlpFilter"

    # If generating rules, we force CPU only
    if args.rules:
        if any((args.train, args.get_data, args.second_train, args.images is not None)):
            raise ValueError("Global rules have to be computed alone because we don't want to use a GPU during global rules generation.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Get Parameter configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(args, script_dir)

    # Load data
    X_train, Y_train, X_test, Y_test = load_data(cfg)

    # TRAINING
    if args.train:
        train_model(cfg, X_train, Y_train, X_test, Y_test, args, model = "big")

    if args.get_data or args.images:
        print("Loading first model...")
        model = load_model(cfg["model_file"])
        print("Model loaded.")

    # DATA THROUGH FIRST CONV LAYER
    if args.get_data:
        intermediate_model = Model(inputs=model.layers[0].input,
                                outputs=model.get_layer("first_conv_end").output)
        print("getting data after first convolution...")
        batch_size = 32
        X_train_conv = intermediate_model.predict(X_train, batch_size=batch_size)
        X_test_conv = intermediate_model.predict(X_test, batch_size=batch_size)
        print(f"Train feature map shape : {X_train_conv.shape}")
        print(f"Test feature map shape : {X_test_conv.shape}")
        # Output data after conv layer
        print("Saving feature map data...")
        np.save(cfg["train_feature_map_file_npy"], X_train_conv)
        np.save(cfg["test_feature_map_file_npy"], X_test_conv)
        output_data(X_train_conv.reshape(X_train_conv.shape[0], -1), cfg["train_feature_map_file"])
        output_data(X_test_conv.reshape(X_test_conv.shape[0], -1), cfg["test_feature_map_file"])

    # Load data if necessary
    if args.second_train or args.rules or args.images:
        if (not args.get_data):
            print("Loading feature map data...")
            X_train_conv = np.load(cfg["train_feature_map_file_npy"])
            if args.second_train:
                X_test_conv = np.load(cfg["test_feature_map_file_npy"])
        print(X_train_conv.shape)
        # if args.dataset == "Mnist_Guido":
        #     height = 12
        #     width = 12
        #     n_channels = 32
        # else:
        height = X_train_conv.shape[1]
        width = X_train_conv.shape[2]
        n_channels = X_train_conv.shape[3]
        nb_attr = height*width*n_channels

    #TRAINING WITH DIMLP
    if args.second_train:
        X_train_conv_h1, X_test_conv_h1 = apply_Dimlp(X_train_conv, X_test_conv, height, n_channels, K_VAL, NB_QUANT_LEVELS, HIKNOT, cfg["second_model_output_weights"], activation_fct_stairobj="identity")
        trainCNN(height, width, n_channels, cfg["nb_classes"], "big", 80, cfg["batch_size_second_model"], cfg["second_model_file"], cfg["second_model_checkpoint_weights"], X_train_conv_h1, Y_train, X_test_conv_h1, Y_test, cfg["second_model_train_pred"], cfg["second_model_test_pred"], cfg["second_model_stats"], remove_first_conv=True)

    # GENERATE GLOBAL RULES
    if args.rules:
        generate_rules(cfg, args, nb_attributes = nb_attr)

    # GENERATION OF EXPLAINING IMAGES ILLUSTRATING SAMPLES AND RULES
    if args.images is not None:
        generate_explaining_images(cfg, X_train, Y_train, model, None, args, height_feature_map=height, width_feature_map=width, nb_channels_feature_map=n_channels, data_in_rules=X_train_conv)

    end_time = time.time()
    print(f"\nFull execution time : {(end_time - start_time):.2f}sec")
