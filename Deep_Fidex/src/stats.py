# stats.py
import time
import numpy as np
import tensorflow as tf

from utils.utils import (
    compute_histograms,
    compute_activation_sums,
    compute_proba_images,
    output_data,
    normalize_data
)
from tensorflow.keras import Model
from utils.config import *

def compute_stats(cfg, X_train, X_test, CNNModel, intermediate_model, args):
    """
    Compute statistics :
      - histograms
      - activations
      - probabilities
    with respect to the value of args.statistic.
    """
    start_time_stats_computation = time.time()
    nb_train_images = len(X_train)
    nb_test_images = len(X_test)
    train_data = X_train
    test_data = X_test
    if args.train_with_patches:
        train_pred = np.loadtxt(cfg["train_pred_file"])
        test_pred = np.loadtxt(cfg["test_pred_file"])
        nb_train_images = train_pred.shape[0] // (cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"])
        nb_test_images = test_pred.shape[0] // (cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"])
        train_data = train_pred
        test_data = test_pred

    if args.statistic == "histogram":
        print("\nComputing train histograms...")

        train_histograms = compute_histograms(nb_train_images, train_data, cfg["size1D"], cfg["nb_channels"], # Shape (nb_train_images(nb_images), nb_classes, nb_bins)
                                            CNNModel, cfg["nb_classes"], FILTER_SIZE, STRIDE, NB_BINS, cfg, args.train_with_patches)
        print("\nTrain histograms computed.\n")

        print("Computing test histograms...")

        test_histograms = compute_histograms(nb_test_images, test_data, cfg["size1D"], cfg["nb_channels"], # Shape (nb_test_images(nb_images), nb_classes, nb_bins)
                                            CNNModel, cfg["nb_classes"], FILTER_SIZE, STRIDE, NB_BINS, cfg, args.train_with_patches)
        print("\nTest histograms computed.")

        # Save in histograms in .npy file
        print("\nSaving histograms...")
        train_histograms = train_histograms.reshape(nb_train_images, cfg["nb_stats_attributes"])
        test_histograms = test_histograms.reshape(nb_test_images, cfg["nb_stats_attributes"])
        output_data(train_histograms, cfg["train_stats_file"])
        output_data(test_histograms, cfg["test_stats_file"])
        print("Histograms saved.")

    elif args.statistic == "activation_layer":
        print("\nComputing train sums of activation layer patches...")

        # Get sums for each train sample
        train_sums = compute_activation_sums(nb_train_images, X_train, cfg["size1D"], cfg["nb_channels"],
                                             CNNModel, intermediate_model, cfg["nb_stats_attributes"],
                                             FILTER_SIZE, STRIDE)
        # Normalization
        mean = np.mean(train_sums, axis=0)
        std = np.std(train_sums, axis=0)
        train_sums = (train_sums - mean) / std

        print("\nTrain sum of activation layer patches computed.\n")

        print("\nComputing test sums of activation layer patches...")
        # Get sums for each test sample
        test_sums = compute_activation_sums(nb_test_images, X_test, cfg["size1D"], cfg["nb_channels"],
                                            CNNModel, intermediate_model, cfg["nb_stats_attributes"],
                                            FILTER_SIZE, STRIDE)
        # Normalization
        test_sums = (test_sums - mean) / std
        print("\nTest sum of activation layer patches computed.\n")

        output_data(train_sums, cfg["train_stats_file"])
        output_data(test_sums, cfg["test_stats_file"])
        print("Sums saved.")

    elif args.statistic in ["probability", "probability_multi_nets"]: # We create an image out of the probabilities (for each class) of cropped areas of the original image
        if args.train_with_patches:
            print("\nGathering train probability of patches...")
            train_pred_data = train_pred # shape (nb_images_train*cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"], nb_classes)
            nb_images_train = train_pred_data.shape[0] // (cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"])
            train_probas = train_pred_data.reshape(nb_images_train, -1) # shape (nb_images_train, cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*nb_classes)

            print("\nGathering test probability of patches...")
            test_pred_data = test_pred # shape (nb_images_test*cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"], nb_classes)
            nb_images_test = test_pred_data.shape[0] // (cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"])
            test_probas = test_pred_data.reshape(nb_images_test, -1) # shape (nb_images_test, cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*nb_classes)
        else:
            print("\nComputing train probability images of patches...\n")
            # Get sums for each train sample
            train_probas = compute_proba_images(nb_train_images, X_train, cfg["size1D"], cfg["nb_channels"],
                                                cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
            # shape (nb_train_images(images), cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*nb_classes)
            print("\nComputed train probability images of patches.")

            print("\nComputing test probability images of patches...\n")
            # Get sums for each train sample
            test_probas = compute_proba_images(nb_test_images, X_test, cfg["size1D"], cfg["nb_channels"],
                                            cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
            print("\nComputed test probability images of patches.")

        output_data(train_probas, cfg["train_stats_file"])
        output_data(test_probas, cfg["test_stats_file"])
        print("Probability images saved.")

    end_time_stats_computation = time.time()
    full_time_stats_computation = end_time_stats_computation - start_time_stats_computation
    print(f"\nStats computation time = {full_time_stats_computation:.2f} sec")
