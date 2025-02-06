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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import numpy as np
import shutil
import re
import copy
import argparse
import utils.config as config
from utils.config import *
from tensorflow.keras import Model
import tensorflow as tf
from utils.constants import HISTOGRAM_ANTECEDENT_PATTERN
from utils.utils import *
np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
from dimlpfidex import dimlp
from trainings import randForestsTrn, gradBoostTrn
from trainings.trnFun import get_attribute_file

script_dir = os.path.dirname(os.path.abspath(__file__))

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
    parser.add_argument("--train_cnn", action="store_true", help="Train a CNN")
    parser.add_argument("--stats", action="store_true", help="Compute statistics") # Stats computation
    parser.add_argument("--train_second_model", action="store_true", help="Train a second model")
    parser.add_argument("--rules", action="store_true", help="Compute global rules")
    parser.add_argument("--images", action="store_true", help="Generate explaining images")
    parser.add_argument("--heatmap", action="store_true", help="Generate a heatmap") # Only evaluation on patches

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()

    # Check on inline parameters

    if args.rules:
        if any((args.train_cnn,
                args.stats,
                args.train_second_model,
                args.images)):
            raise ValueError("Global rules have to be computed alone because we don't want to use a GPU during global rules generation.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # No GPU when generating rules

    # Get Parameter configuration
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

    # Train CNN model
    if args.train_cnn:
        start_time_train_cnn = time.time()
        trainCNN(cfg["size1D"], cfg["size1D"], cfg["nb_channels"], cfg["nb_classes"], cfg["model"], cfg["nbIt"], cfg["batch_size"], cfg["model_file"], cfg["model_checkpoint_weights"], X_train, Y_train, X_test, Y_test, cfg["train_pred_file"], cfg["test_pred_file"], cfg["model_stats"], cfg["with_leaky_relu"])
        end_time_train_cnn = time.time()
        full_time_train_cnn = end_time_train_cnn - start_time_train_cnn
        full_time_train_cnn = "{:.6f}".format(full_time_train_cnn).rstrip("0").rstrip(".")
        print(f"\nTrain first CNN time = {full_time_train_cnn} sec")

    print("Loading model...")
    CNNModel = keras.saving.load_model(cfg["model_file"])
    print("Model loaded.")
    if args.statistic == "activation_layer": # Get intermediate model
        input_channels = CNNModel.input_shape[-1]
        dummy_input = np.zeros((1, cfg["size1D"], cfg["size1D"], input_channels))
        _ = CNNModel(dummy_input)
        flatten_layer_output = CNNModel.get_layer("flatten").output
        intermediate_model = Model(inputs=CNNModel.inputs, outputs=flatten_layer_output)
        cfg["nb_stats_attributes"] = intermediate_model.output_shape[1]


    ##############################################################################
    if args.test:
        nb_train_samples = 100
        nb_test_samples = 100
    else:
        nb_train_samples = Y_train.shape[0]
        nb_test_samples = Y_test.shape[0]

    # Compute histograms
    if args.stats:
        start_time_stats_computation = time.time()
        if args.statistic == "histogram":
            print("\nComputing train histograms...")

            # Get histograms for each train sample

            train_histograms = compute_histograms(nb_train_samples, X_train, cfg["size1D"], cfg["nb_channels"], CNNModel, cfg["nb_classes"], FILTER_SIZE, STRIDE, NB_BINS)
            print("\nTrain histograms computed.\n")

            print("Computing test histograms...")

            # Get histograms for each test sample
            test_histograms = compute_histograms(nb_test_samples, X_test, cfg["size1D"], cfg["nb_channels"], CNNModel, cfg["nb_classes"], FILTER_SIZE, STRIDE, NB_BINS)

            print("\nTest histograms computed.")
            # Save in histograms in .npy file
            print("\nSaving histograms...")
            train_histograms = train_histograms.reshape(nb_train_samples, cfg["nb_stats_attributes"])
            test_histograms = test_histograms.reshape(nb_test_samples, cfg["nb_stats_attributes"])
            print(train_histograms.shape)
            output_data(train_histograms, cfg["train_stats_file"])
            output_data(test_histograms, cfg["test_stats_file"])
            print("Histograms saved.")



        elif args.statistic == "activation_layer":

            print("\nComputing train sums of activation layer patches...")
            # Get sums for each train sample

            train_sums = compute_activation_sums(nb_train_samples, X_train, cfg["size1D"], cfg["nb_channels"], CNNModel, intermediate_model, cfg["nb_stats_attributes"], FILTER_SIZE, STRIDE)
            # Normalization
            mean = np.mean(train_sums, axis=0)
            std = np.std(train_sums, axis=0)
            train_sums = (train_sums - mean) / std
            print("\nTrain sum of activation layer patches computed.\n")

            print("\nComputing test sums of activation layer patches...")
            # Get sums for each test sample

            test_sums = compute_activation_sums(nb_test_samples, X_test, cfg["size1D"], cfg["nb_channels"], CNNModel, intermediate_model, cfg["nb_stats_attributes"], FILTER_SIZE, STRIDE)
            # Normalization
            test_sums = (test_sums - mean) / std
            print("\nTest sum of activation layer patches computed.\n")

            print("\nSaving sums...")
            output_data(train_sums, cfg["train_stats_file"])
            output_data(test_sums, cfg["test_stats_file"])
            print("Sums saved.")

        elif args.statistic == "probability" or args.statistic == "probability_multi_nets": # We create an image out of the probabilities (for each class) of cropped areas of the original image
            print("\nComputing train probability images of patches...\n")
            # Get sums for each train sample
            print(nb_train_samples, cfg["size1D"], cfg["nb_channels"], cfg["nb_classes"], FILTER_SIZE, STRIDE)
            train_probas = compute_proba_images(nb_train_samples, X_train, cfg["size1D"], cfg["nb_channels"], cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
            print(train_probas.shape)
            print("\nComputed train probability images of patches.")

            print("\nComputing test probability images of patches...\n")
            # Get sums for each train sample

            test_probas = compute_proba_images(nb_test_samples, X_test, cfg["size1D"], cfg["nb_channels"], cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
            print(test_probas.shape)
            print("\nComputed test probability images of patches.")

            print("\nSaving probability images...")

            output_data(train_probas, cfg["train_stats_file"])
            output_data(test_probas, cfg["test_stats_file"])
            print("Probability images saved.")

        end_time_stats_computation = time.time()
        full_time_stats_computation = end_time_stats_computation - start_time_stats_computation
        full_time_stats_computation = "{:.6f}".format(full_time_stats_computation).rstrip("0").rstrip(".")
        print(f"\nStats computation time = {full_time_stats_computation} sec")

    ##############################################################################
    # Train second model with stats

    if args.train_second_model:
        start_time_train_second_model = time.time()

        if args.statistic == "probability" or args.statistic == "probability_multi_nets": # We create an image out of the probabilities (for each class) of cropped areas of the original image
            # Load probas of areas from file if necessary
            if not args.stats:
                print("Loading probability stats...")
                train_probas = np.loadtxt(cfg["train_stats_file"])
                train_probas = train_probas.astype('float32')
                test_probas = np.loadtxt(cfg["test_stats_file"])
                test_probas = test_probas.astype('float32')
                print("Probability stats loaded.")
            #print(train_probas.shape) # (nb_train_samples, 4840) (22*22*10)
            #print(test_probas.shape) # (nb_test_samples, 4840)

            print("Adding original image...")
            train_probas = train_probas.reshape(nb_train_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"])
            X_train_reshaped = tf.image.resize(X_train, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"])) # Resize original image to the proba size
            train_probas = np.concatenate((train_probas, X_train_reshaped[:nb_train_samples]), axis=-1) # Concatenate the probas and the original image resized
            train_probas = train_probas.reshape(nb_train_samples, -1) # flatten for export

            test_probas = test_probas.reshape(nb_test_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"])
            X_test_reshaped = tf.image.resize(X_test, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"])) # Resize original image to the proba size
            test_probas = np.concatenate((test_probas, X_test_reshaped[:nb_test_samples]), axis=-1) # Concatenate the probas and the original image resized
            test_probas = test_probas.reshape(nb_test_samples, -1) # flatten for export

            # print(train_probas.shape) #(nb_train_samples, 5324) (22*22*11)
            # print(test_probas.shape)  #(nb_test_samples, 5324)

            # Save proba stats data with original image added
            output_data(train_probas, cfg["train_stats_file_with_image"])
            output_data(test_probas, cfg["test_stats_file_with_image"])

            cfg["train_stats_file"] = cfg["train_stats_file_with_image"]
            cfg["test_stats_file"] = cfg["test_stats_file_with_image"]

            print("original image added.")

            if cfg["second_model"] == "cnn":
                # Pass on the DIMLP layer
                train_probas_h1, mu, sigma = compute_first_hidden_layer("train", train_probas, K_VAL, NB_QUANT_LEVELS, HIKNOT, cfg["second_model_output_rules"])
                test_probas_h1 = compute_first_hidden_layer("test", test_probas, K_VAL, NB_QUANT_LEVELS, HIKNOT, mu=mu, sigma=sigma)
                train_probas_h1 = train_probas_h1.reshape((nb_train_samples,)+cfg["output_size"]) #(100, 26, 26, 13)
                print("train_probas_h1 reshaped : ", train_probas_h1.shape)
                test_probas_h1 = test_probas_h1.reshape((nb_test_samples,)+cfg["output_size"])
                #print(train_probas.shape)  # (nb_train_samples, 22, 22, 10)
                #print(test_probas.shape)  # (nb_train_samples, 22, 22, 10)
                second_model_file = os.path.join(cfg["files_folder"], "scanSecondModel.keras")
                second_model_checkpoint_weights = os.path.join(cfg["files_folder"], "weightsSecondModel.weights.h5")

                if args.statistic == "probability": # Train with a CNN now
                    trainCNN(cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"]+cfg["nb_channels"], cfg["nb_classes"], "small", 80, cfg["batch_size_second_model"], second_model_file, second_model_checkpoint_weights, train_probas_h1, Y_train, test_probas_h1, Y_test, cfg["second_model_train_pred"], cfg["second_model_test_pred"], cfg["second_model_stats"], False, True)

                else: # Create nb_classes networks and gather best probability among them. The images keep only the probabilities of areas for one class and add B&W image (or H and S of HSL)

                    if args.test:
                        nbIt_current = 2
                    else:
                        nbIt_current = 80

                    models_folder = os.path.join(cfg["files_folder"], "Models")
                    # Delete and recreate models folder
                    if os.path.exists(models_folder):
                        shutil.rmtree(models_folder)
                    os.makedirs(models_folder)

                    # Create each dataset
                    for i in range(cfg["nb_classes"]):
                        print("Creating dataset n°",i,"...")

                        original_img_transformed_reshaped_train = X_train_reshaped # (100, 26, 26, 3)
                        original_img_transformed_reshaped_test = X_test_reshaped
                        if cfg["nb_channels"] == 3:
                            if cfg["with_hsl"]: # Transform in HSL(hsv in fact)
                                original_img_transformed_reshaped_train = tf.image.rgb_to_hsv(original_img_transformed_reshaped_train)
                                original_img_transformed_reshaped_test = tf.image.rgb_to_hsv(original_img_transformed_reshaped_test)
                            elif not cfg["with_rg"]: # Transform in black and white
                                original_img_transformed_reshaped_train = tf.image.rgb_to_grayscale(original_img_transformed_reshaped_train)
                                original_img_transformed_reshaped_test = tf.image.rgb_to_grayscale(original_img_transformed_reshaped_test)

                        # Create train data for each model
                        built_data_train = np.empty((nb_train_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], 3))
                        # Add probas on first channel
                        built_data_train[:,:,:,0] = train_probas_h1[:,:,:,i]
                        # Add H and S on last 2 channels (or R and G)
                        if (cfg["with_hsl"] or cfg["with_rg"]) and cfg["nb_channels"] == 3:
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
                        current_model_train_pred = os.path.join(models_folder, f"second_model_train_pred_{i}.txt")
                        data_filename = f"train_probability_images_with_original_img_{i}.txt"
                        class_filename = f"Y_train_probability_images_with_original_img_{i}.txt"
                        built_data_train_flatten = built_data_train.reshape(nb_train_samples, cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*3)

                        # output new train data
                        output_data(built_data_train_flatten, os.path.join(models_folder, data_filename))
                        output_data(built_Y_train, os.path.join(models_folder, class_filename))

                        # Create test data for each model
                        built_data_test = np.empty((nb_test_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], 3))
                        # Add probas on first channel
                        built_data_test[:,:,:,0] = test_probas_h1[:,:,:,i]
                        # Add H and S on last 2 channels
                        if (cfg["with_hsl"] or cfg["with_rg"]) and cfg["nb_channels"] == 3:
                            built_data_test[:,:,:,1] = original_img_transformed_reshaped_test[..., 0]
                            built_data_test[:,:,:,2] = original_img_transformed_reshaped_test[..., 1]
                        else: # Add 1-probas and B&W on last 2 channels
                            built_data_test[:,:,:,1] = 1-test_probas_h1[:,:,:,i]
                            built_data_test[:,:,:,2] = original_img_transformed_reshaped_test[..., 0]

                        # Create classes for these datas
                        built_Y_test = np.zeros((nb_test_samples, 2), dtype=int)
                        built_Y_test[Y_test[:, i] == 1, 0] = 1  # If condition is True, set [1, 0]
                        built_Y_test[Y_test[:, i] != 1, 1] = 1  # If condition is False, set [0, 1]
                        current_model_test_pred = os.path.join(models_folder, f"second_model_test_pred_{i}.txt")
                        data_filename = f"test_probability_images_with_original_img_{i}.txt"
                        class_filename = f"Y_test_probability_images_with_original_img_{i}.txt"
                        built_data_test_flatten = built_data_test.reshape(nb_test_samples, cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*3)

                        # output new test data
                        output_data(built_data_test_flatten, os.path.join(models_folder, data_filename))
                        output_data(built_Y_test, os.path.join(models_folder, class_filename))

                        current_model_stats = os.path.join(models_folder, f"second_model_stats_{i}.txt")
                        current_model_file = os.path.join(models_folder, f"scanSecondModel_{i}.keras")
                        current_model_checkpoint_weights = os.path.join(models_folder, f"weightsSecondModel_{i}.weights.h5")

                        print("Dataset n°",i," created.")
                        # Train new model
                        trainCNN(cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], 3, 2, "VGG", nbIt_current, cfg["batch_size_second_model"], current_model_file, current_model_checkpoint_weights, built_data_train, built_Y_train, built_data_test, built_Y_test, current_model_train_pred, current_model_test_pred, current_model_stats, False, True)
                        print("Dataset n°",i," trained.")

                    # Create test and train predictions
                    train_pred_files = [
                        os.path.join(models_folder, f"second_model_train_pred_{i}.txt") for i in range(cfg["nb_classes"])
                    ]
                    test_pred_files = [
                        os.path.join(models_folder, f"second_model_test_pred_{i}.txt") for i in range(cfg["nb_classes"])
                    ]

                    # Gathering predictions for train and test
                    print("Gathering train predictions...")
                    gathering_predictions(train_pred_files, cfg["second_model_train_pred"])
                    print("Gathering test predictions...")
                    gathering_predictions(test_pred_files, cfg["second_model_test_pred"])

                    # Compute and save predictions of the second (gathering of all models) model
                    second_model_train_preds = np.argmax(np.loadtxt(cfg["second_model_train_pred"]), axis=1)
                    second_model_test_preds = np.argmax(np.loadtxt(cfg["second_model_test_pred"]), axis=1)

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

                    with open(cfg["second_model_stats"], "w") as myFile:
                        print("Train score : ", train_accuracy)
                        myFile.write(f"Train score : {train_accuracy}\n")

                        print("Test score : ", test_accuracy)
                        myFile.write(f"Test score : {test_accuracy}\n")

                    print("Data sets created and all models trained.")

            else: # Using a Ranfom Forests to train the probas with images

                command = (
                    f'--train_data_file {cfg["train_stats_file"]} '
                    f'--train_class_file {cfg["train_class_file"]} '
                    f'--test_data_file {cfg["test_stats_file"]} '
                    f'--test_class_file {cfg["test_class_file"]} '
                    f'--stats_file {cfg["second_model_stats"]} '
                    f'--train_pred_outfile {cfg["second_model_train_pred"]} '
                    f'--test_pred_outfile {cfg["second_model_test_pred"]} '
                    f'--nb_attributes {cfg["nb_stats_attributes"]} '
                    f'--nb_classes {cfg["nb_classes"]} '
                    f'--root_folder . '
                    )
                command += f'--rules_outfile {cfg["second_model_output_rules"]} '
                status = randForestsTrn(command)

        else: # (not with probabilities of areas)

            # Train model
            command = (
                f'--train_data_file {cfg["train_stats_file"]} '
                f'--train_class_file {cfg["train_class_file"]} '
                f'--test_data_file {cfg["test_stats_file"]} '
                f'--test_class_file {cfg["test_class_file"]} '
                f'--stats_file {cfg["second_model_stats"]} '
                f'--train_pred_outfile {cfg["second_model_train_pred"]} '
                f'--test_pred_outfile {cfg["second_model_test_pred"]} '
                f'--nb_attributes {cfg["nb_stats_attributes"]} '
                f'--nb_classes {cfg["nb_classes"]} '
                f'--root_folder . '
                )

            if cfg["using_decision_tree_model"]:
                command += f'--rules_outfile {cfg["second_model_output_rules"]} '
            else:
                command += f'--weights_outfile {cfg["second_model_output_rules"]} '

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
            #         if args.test:
            #             command += '--nb_epochs 10 '
            #         status = dimlp.dimlpBT(command)

            if cfg["second_model"] == "randomForests":
                status = randForestsTrn(command)
            elif cfg["second_model"] == "gradientBoosting":
                status = gradBoostTrn(command)
            elif cfg["second_model"] == "dimlpTrn":
                status = dimlp.dimlpTrn(command)
            elif cfg["second_model"] == "dimlpBT":
                command += '--nb_dimlp_nets 15 '
                command += '--hidden_layers [25] '
                if args.test:
                    command += '--nb_epochs 10 '
                status = dimlp.dimlpBT(command)

            if status != -1:
                print("\nSecond model trained.")

        end_time_train_second_model = time.time()
        full_time_train_second_model = end_time_train_second_model - start_time_train_second_model
        full_time_train_second_model = "{:.6f}".format(full_time_train_second_model).rstrip("0").rstrip(".")
        print(f"\nTrain second model time = {full_time_train_second_model} sec")

    if args.statistic == "histogram":
        # Define attributes file for histograms
        with open(cfg["attributes_file"], "w") as myFile:
            for i in range(cfg["nb_classes"]):
                for j in PROBABILITY_THRESHOLDS:
                    myFile.write(f"P_{i}>={j:.6g}\n")


    if args.statistic == "probability" or args.statistic == "probability_multi_nets":
        cfg["train_stats_file"] = cfg["train_stats_file_with_image"]
        cfg["test_stats_file"] = cfg["test_stats_file_with_image"]

    ##############################################################################
    # Compute global rules

    if args.rules:
        start_time_global_rules = time.time()
        command = (
            f'--train_data_file {cfg["train_stats_file"]} '
            f'--train_pred_file {cfg["second_model_train_pred"]} '
            f'--train_class_file {cfg["train_class_file"]} '
            f'--nb_classes {cfg["nb_classes"]} '
            f'--global_rules_outfile {cfg["global_rules_file"]} '
            f'--nb_attributes {cfg["nb_stats_attributes"]} '
            f'--heuristic 1 '
            f'--nb_threads 8 '
            f'--max_iterations 25 '
            f'--nb_quant_levels {NB_QUANT_LEVELS} '
            f'--dropout_dim {DROPOUT_DIM} '
            f'--dropout_hyp {DROPOUT_HYP} '
        )
        if args.statistic == "histogram":
            command += f'--attributes_file {cfg["attributes_file"]} '
        if cfg["using_decision_tree_model"]:
            command += f'--rules_file {cfg["second_model_output_rules"]} '
        else:
            command += f'--weights_file {cfg["second_model_output_rules"]} '

        print("\nComputing global rules...\n")
        status = fidex.fidexGloRules(command)
        if status != -1:
            print("\nGlobal rules computed.")

        command = (
            f'--test_data_file {cfg["test_stats_file"]} '
            f'--test_pred_file {cfg["second_model_test_pred"]} '
            f'--test_class_file {cfg["test_class_file"]} '
            f'--nb_classes {cfg["nb_classes"]} '
            f'--global_rules_file {cfg["global_rules_file"]} '
            f'--nb_attributes {cfg["nb_stats_attributes"]} '
            f'--global_rules_outfile {cfg["global_rules_with_test_stats"]} '
            f'--stats_file {cfg["global_rules_stats"]}'
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

    if args.images:
        print("Generation of images...")
        # Get rules and attributes
        global_rules = getRules(cfg["global_rules_file"])
        if args.statistic == "histogram":
            attributes = get_attribute_file(cfg["attributes_file"], cfg["nb_stats_attributes"])[0]

        # Create folder for all rules
        if os.path.exists(cfg["rules_folder"]):
            shutil.rmtree(cfg["rules_folder"])
        os.makedirs(cfg["rules_folder"])

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

            if args.statistic == "histogram":
                rule.include_X = False
                for ant in rule.antecedents:
                    ant.attribute = attributes[ant.attribute] # Get true name of attribute
            elif args.statistic == "probability" or args.statistic == "probability_multi_nets":
                rule.include_X = False

            # Create folder for this rule
            rule_folder = os.path.join(cfg["rules_folder"], f"rule_{id}_class_{cfg['classes'][rule.target_class]}")
            if os.path.exists(rule_folder):
                shutil.rmtree(rule_folder)
            os.makedirs(rule_folder)

            # Add a readme containing the rule
            readme_file = rule_folder+'/Readme.md'
            rule_to_print = copy.deepcopy(rule)

            if args.statistic == "histogram":
                # Change antecedent with real class names
                for antecedent in rule_to_print.antecedents:
                    match = re.match(HISTOGRAM_ANTECEDENT_PATTERN, antecedent.attribute)
                    if match:
                        class_id = int(match.group(1))
                        pred_threshold = match.group(2)
                        class_name = cfg["classes"][class_id]  # Get the class name
                        antecedent.attribute = f"P_{class_name}>={pred_threshold}"
                    else:
                        raise ValueError("Wrong antecedent...")
            elif args.statistic == "probability" or args.statistic == "probability_multi_nets":
                # attribut_de_test = 2024 # -> classe :  0, Height :  8, Width :  8

                # Change antecedent with area and class involved

                # Scales of changes of original image to reshaped image
                scale_h = cfg["size1D"] / cfg["size_Height_proba_stat"]
                scale_w = cfg["size1D"] / cfg["size_Width_proba_stat"]
                for antecedent in rule_to_print.antecedents: # TODO : handle stride, different filter sizes, etc
                    # area_index (size_Height_proba_stat, size_Width_proba_stat) : 0 : (1,1), 1: (1,2), ...
                    channel_id = antecedent.attribute % (cfg["nb_classes"] + cfg["nb_channels"]) # (probas of each class + image rgb concatenated)
                    area_number = antecedent.attribute // (cfg["nb_classes"] + cfg["nb_channels"])
                    # channel_id = attribut_de_test % (cfg["nb_classes"] + cfg["nb_channels"])
                    # area_number = attribut_de_test // (cfg["nb_classes"] + cfg["nb_channels"])
                    area_Height = area_number // cfg["size_Width_proba_stat"]
                    area_Width = area_number % cfg["size_Width_proba_stat"]
                    # print("classe : ", channel_id)
                    # print("Height : ", area_Height)
                    # print("Width : ", area_Width)
                    # exit()
                    if channel_id < cfg["nb_classes"]: #Proba of area
                        class_name = cfg["classes"][channel_id]
                        antecedent.attribute = f"P_class_{class_name}_area_[{area_Height}-{area_Height+FILTER_SIZE[0][0]-1}]x[{area_Width}-{area_Width+FILTER_SIZE[0][1]-1}]"
                    else:
                        channel = channel_id - cfg["nb_classes"] #Pixel in concatenated original rgb image
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
                if args.statistic == "histogram":
                    highlighted_image = highlight_area_histograms(CNNModel, img, FILTER_SIZE, STRIDE, rule, cfg["classes"])
                elif args.statistic == "activation_layer":
                    highlighted_image = highlight_area_activations_sum(CNNModel, intermediate_model, img, rule, FILTER_SIZE, STRIDE, cfg["classes"])
                elif args.statistic == "probability" or args.statistic == "probability_multi_nets":
                    highlighted_image = highlight_area_probability_image(img, rule, cfg["size1D"], cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], FILTER_SIZE, cfg["classes"], cfg["nb_channels"])
                highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png") # Save image


    ##############################################################################
    # Get images explaining and illustrating the samples without rules or training twice

    if args.heatmap: # Only for one filter !

        # Create heat map folder
        if os.path.exists(cfg["heat_maps_folder"]):
            shutil.rmtree(cfg["heat_maps_folder"])
        os.makedirs(cfg["heat_maps_folder"])

        for id,img in enumerate(X_test[0:100]):
            heat_maps_img = get_heat_maps(CNNModel, img, FILTER_SIZE, STRIDE, PROBABILITY_THRESHOLDS, cfg["classes"])
            heat_maps_img.savefig(f"{cfg['heat_maps_folder']}/sample_{id}.png")


    end_time = time.time()
    full_time = end_time - start_time
    full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

    print(f"\nFull execution time = {full_time} sec")
