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

def compute_stats(cfg, X_train, Y_train, X_test, Y_test, CNNModel, intermediate_model, args):
    """
    Compute statistics :
      - histograms
      - activations
      - probabilities
    with respect to the value of args.statistic.
    """
    start_time_stats_computation = time.time()
    nb_train_samples = len(X_train)
    nb_test_samples = len(X_test)


    if args.statistic == "histogram":
        print("\nComputing train histograms...")
        train_histograms = compute_histograms(nb_train_samples, X_train, cfg["size1D"], cfg["nb_channels"],
                                              CNNModel, cfg["nb_classes"], FILTER_SIZE, STRIDE, NB_BINS)
        print("\nTrain histograms computed.\n")

        print("Computing test histograms...")
        test_histograms = compute_histograms(nb_test_samples, X_test, cfg["size1D"], cfg["nb_channels"],
                                             CNNModel, cfg["nb_classes"], FILTER_SIZE, STRIDE, NB_BINS)
        print("\nTest histograms computed.")

        # Save in histograms in .npy file
        print("\nSaving histograms...")
        train_histograms = train_histograms.reshape(nb_train_samples, cfg["nb_stats_attributes"])
        test_histograms = test_histograms.reshape(nb_test_samples, cfg["nb_stats_attributes"])
        output_data(train_histograms, cfg["train_stats_file"])
        output_data(test_histograms, cfg["test_stats_file"])
        print("Histograms saved.")

    elif args.statistic == "activation_layer":
        print("\nComputing train sums of activation layer patches...")

        # Get sums for each train sample
        train_sums = compute_activation_sums(nb_train_samples, X_train, cfg["size1D"], cfg["nb_channels"],
                                             CNNModel, intermediate_model, cfg["nb_stats_attributes"],
                                             FILTER_SIZE, STRIDE)
        # Normalization
        mean = np.mean(train_sums, axis=0)
        std = np.std(train_sums, axis=0)
        train_sums = (train_sums - mean) / std

        print("\nTrain sum of activation layer patches computed.\n")

        print("\nComputing test sums of activation layer patches...")
        # Get sums for each test sample
        test_sums = compute_activation_sums(nb_test_samples, X_test, cfg["size1D"], cfg["nb_channels"],
                                            CNNModel, intermediate_model, cfg["nb_stats_attributes"],
                                            FILTER_SIZE, STRIDE)
        # Normalization
        test_sums = (test_sums - mean) / std
        print("\nTest sum of activation layer patches computed.\n")

        output_data(train_sums, cfg["train_stats_file"])
        output_data(test_sums, cfg["test_stats_file"])
        print("Sums saved.")

    elif args.statistic in ["probability", "probability_multi_nets"]: # We create an image out of the probabilities (for each class) of cropped areas of the original image
        print("\nComputing train probability images of patches...\n")
        # Get sums for each train sample
        train_probas = compute_proba_images(nb_train_samples, X_train, cfg["size1D"], cfg["nb_channels"],
                                            cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
        print("\nComputed train probability images of patches.")

        print("\nComputing test probability images of patches...\n")
        # Get sums for each train sample
        test_probas = compute_proba_images(nb_test_samples, X_test, cfg["size1D"], cfg["nb_channels"],
                                           cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
        print("\nComputed test probability images of patches.")

        output_data(train_probas, cfg["train_stats_file"])
        output_data(test_probas, cfg["test_stats_file"])
        print("Probability images saved.")

    end_time_stats_computation = time.time()
    full_time_stats_computation = end_time_stats_computation - start_time_stats_computation
    print(f"\nStats computation time = {full_time_stats_computation:.2f} sec")
