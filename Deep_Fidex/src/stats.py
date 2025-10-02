# stats.py
import time
import numpy as np
import tensorflow as tf

from utils.utils import (
    getHistogram,
    generate_filtered_images_and_predictions,
    output_data
)
from tensorflow.keras import Model
from utils.config import *


def compute_histograms(nb_samples, data_or_predictions, size1D, nb_channels, CNNModel, nb_classes, filter_size, stride, nb_bins, cfg, train_with_patches=False):
    """
    Computes histograms for each sample in the dataset using the CNN model. It's the histogram of the probabilities of each class on the CNN
    evaluated on each area (or patches) added on the image (by a sliding filter). A patch is applied and outside of this area everything is 0.

    Parameters:
    - nb_samples: The number of samples in the dataset.
    - data_or_predictions: The dataset containing images to be processed or the predictions of each patch (if train_with_patches == True).
    - size1D: The size of one dimension of the input image (image is size1D x size1D).
    - nb_channels: The number of channels in the input images (1 for grayscale, 3 for RGB).
    - CNNModel: The CNN model used for making predictions.
    - nb_classes: The number of classes for classification.
    - filter_size: The size of the filter applied to the image (height, width), can be an array of tuples if we apply several filters.
    - stride: The stride value for moving the filter across the image (verticaly, horizontaly).
    - nb_bins: The number of bins used for computing the histogram.

    Returns:
    - histograms: A numpy array containing the computed histograms for each sample.
    """

    histograms = []
    nb_patches_per_image = cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]
    for sample_id in range(nb_samples):
        if train_with_patches:
            start_idx = sample_id * nb_patches_per_image
            end_idx = start_idx + nb_patches_per_image
            predictions = data_or_predictions[start_idx:end_idx, :]
        else:
            image = data_or_predictions[sample_id]
            image = image.reshape(size1D, size1D, nb_channels)
            predictions, _, _ = generate_filtered_images_and_predictions( # shape cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"], nb_classes
                cfg, CNNModel, image, filter_size, stride)

        histogram = getHistogram(CNNModel, predictions, nb_classes, filter_size, stride, nb_bins)
        histograms.append(histogram)
        if (sample_id+1) % 100 == 0 or sample_id+1 == nb_samples:
            progress = ((sample_id+1) / nb_samples) * 100
            progress_message = f"Progress : {progress:.2f}% ({sample_id+1}/{nb_samples})"
            print(f"\r{progress_message}", end='', flush=True)
    histograms = np.array(histograms)

    return histograms

###############################################################

def compute_activation_sums(cfg, nb_samples, data, size1D, nb_channels, CNNModel, intermediate_model, nb_stats_attributes, filter_size, stride):
    """
    Computes the sum of activations from an intermediate layer for each sample in the dataset using a sliding filter.

    This function applies a sliding filter across each image in the dataset and uses a CNN model to extract feature activations
    from an intermediate layer. The activations for each patch are then summed to produce a global activation sum for each sample.
    This method helps analyze the CNN model’s response over different areas of the image.

    Parameters:
    - nb_samples: The number of samples in the dataset.
    - data: The dataset containing images to be processed.
    - size1D: The size of one dimension of the input image (image is size1D x size1D).
    - nb_channels: The number of channels in the input images (1 for grayscale, 3 for RGB).
    - CNNModel: The full CNN model used for making predictions.
    - intermediate_model: A model stopping at the specific intermediate layer (e.g., Flatten layer) to capture activations.
    - nb_stats_attributes: The number of statistical attributes to store for each sample (dimensionality of the intermediate layer).
    - filter_size: The size of the filter applied to the image (height, width), can be an array of tuples if applying multiple filters.
    - stride: The stride value for moving the filter across the image (vertically, horizontally).

    Returns:
    - sums: A numpy array containing the activation sums for each sample, with shape (nb_samples, nb_stats_attributes).
    """
    sums = np.zeros((nb_samples, nb_stats_attributes))
    for sample_id in range(nb_samples):
        image = data[sample_id]
        image = image.reshape(size1D, size1D, nb_channels)
        activations, _ = generate_filtered_images_and_predictions(
        cfg, CNNModel, image, filter_size, stride, intermediate_model)
        sums[sample_id] = np.sum(activations, axis=0)
        if (sample_id+1) % 100 == 0 or sample_id+1 == nb_samples:
            progress = ((sample_id+1) / nb_samples) * 100
            progress_message = f"Progress : {progress:.2f}% ({sample_id+1}/{nb_samples})"
            print(f"\r{progress_message}", end='', flush=True)

    return sums

###############################################################

def compute_proba_images(cfg, nb_samples, data, size1D, nb_channels, nb_classes, CNNModel, filter_size, stride):
    my_filter_size = filter_size[0]
    output_size = ((size1D - my_filter_size[0] + 1)*(size1D - my_filter_size[1] + 1)*nb_classes)
    #print(output_size) # (4840)
    proba_images = np.zeros((nb_samples,output_size))

    for sample_id in range(nb_samples):
        image = data[sample_id]
        image = image.reshape(size1D, size1D, nb_channels)
        predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
            cfg, CNNModel, image, filter_size, stride)
        # print(predictions.shape)  # (484, 10)
        predictions = predictions.reshape(output_size)
        # print(predictions.shape) # (4840,)
        proba_images[sample_id] = predictions
    #print(proba_images.shape)  # (nb_samples, 4840)
    return proba_images

###############################################################

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
        train_sums = compute_activation_sums(cfg, nb_train_images, X_train, cfg["size1D"], cfg["nb_channels"],
                                             CNNModel, intermediate_model, cfg["nb_stats_attributes"],
                                             FILTER_SIZE, STRIDE)
        # Normalization
        mean = np.mean(train_sums, axis=0)
        std = np.std(train_sums, axis=0)
        train_sums = (train_sums - mean) / std

        print("\nTrain sum of activation layer patches computed.\n")

        print("\nComputing test sums of activation layer patches...")
        # Get sums for each test sample
        test_sums = compute_activation_sums(cfg, nb_test_images, X_test, cfg["size1D"], cfg["nb_channels"],
                                            CNNModel, intermediate_model, cfg["nb_stats_attributes"],
                                            FILTER_SIZE, STRIDE)
        # Normalization
        test_sums = (test_sums - mean) / std
        print("\nTest sum of activation layer patches computed.\n")

        output_data(train_sums, cfg["train_stats_file"])
        output_data(test_sums, cfg["test_stats_file"])
        print("Sums saved.")

    elif args.statistic in ["probability", "probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]: # We create an image out of the probabilities (for each class) of cropped areas of the original image
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
            train_probas = compute_proba_images(cfg, nb_train_images, X_train, cfg["size1D"], cfg["nb_channels"],
                                                cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
            # shape (nb_train_images(images), cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*nb_classes)
            print("\nComputed train probability images of patches.")

            print("\nComputing test probability images of patches...\n")
            # Get sums for each train sample
            test_probas = compute_proba_images(cfg, nb_test_images, X_test, cfg["size1D"], cfg["nb_channels"],
                                            cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE)
            print("\nComputed test probability images of patches.")

        output_data(train_probas, cfg["train_stats_file"])
        output_data(test_probas, cfg["test_stats_file"])
        print("Probability images saved.")

    end_time_stats_computation = time.time()
    full_time_stats_computation = end_time_stats_computation - start_time_stats_computation
    print(f"\nStats computation time = {full_time_stats_computation:.2f} sec")

def compute_HOG(cfg, X_train, X_test, nb_original_train_samples, nb_original_test_samples):
    """
    Compute HOG statistics for the training and testing datasets.

    This function computes the Histogram of Oriented Gradients (HOG) for each image in the training
    and testing datasets. The computed HOG features are then saved to the specified files.
    Images must be 8x8 pixels. We use 2x2 cells of 4x4 pixels, 8 orientations, and no block overlap.

    Parameters:
    - cfg: Configuration dictionary containing parameters such as file paths and HOG settings.
    - X_train: The training dataset containing images (each image must be 8x8).
    - X_test: The testing dataset containing images (each image must be 8x8).
    - nb_original_train_samples: The number of original training samples (images, not patches).
    - nb_original_test_samples: The number of original testing samples (images, not patches).

    Returns:
    None
    """
    from skimage.feature import hog
    import numpy as np

    # Note : The image should be float in [0,1] for skimage hog function

    # Convert images to grayscale if they are not already
    if cfg["nb_channels"] != 1:
        X_train = tf.image.rgb_to_grayscale(X_train).numpy()  # (62500, 8, 8, 1)
        X_test  = tf.image.rgb_to_grayscale(X_test).numpy()

    X_train = np.squeeze(X_train, axis=-1)  # becomes (N, 8, 8)
    X_test  = np.squeeze(X_test, axis=-1)

    if X_train.shape[1] != 8 or X_train.shape[2] != 8 or X_test.shape[1] != 8 or X_test.shape[2] != 8:
        raise ValueError("Images must be 8x8 pixels to compute HOG features with the current configuration.")
    print("Computing HOG features for training set...")
    # --- Compute HOG descriptors for training set ---
    hog_train = []
    for img in X_train:
        feat = hog(
            image=img,                 # single 8x8 grayscale image (values in [0,1])
            orientations=8,            # number of bins
            pixels_per_cell=(4, 4),    # cell size: 4x4 pixels -> 2x2 cells for an 8x8 image
            cells_per_block=(1, 1),    # no block overlap, block = single cell
            block_norm="L2-Hys",       # normalization method (still required for (1,1))
            transform_sqrt=False,      # optional: apply power law compression
            feature_vector=True,       # return 1D vector
            visualize=False            # no visualization
        )
        hog_train.append(feat)
    hog_train = np.array(hog_train)

    print("Computing HOG features for testing set...")
    # --- Compute HOG descriptors for testing set ---
    hog_test = []
    for img in X_test:
        feat = hog(
            image=img,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            block_norm="L2-Hys",
            transform_sqrt=False,
            feature_vector=True,
            visualize=False
        )
        hog_test.append(feat)
    hog_test = np.array(hog_test)

    #Modify to size nb_original_samples x (nb_patches*32) :
    hog_train = hog_train.reshape(nb_original_train_samples, -1)
    hog_test = hog_test.reshape(nb_original_test_samples, -1)

    # --- Save features to files defined in cfg ---
    np.savetxt(cfg["train_stats_file"], hog_train)
    np.savetxt(cfg["test_stats_file"], hog_test)

    print(f"HOG features for training set saved to {cfg['train_stats_file']} with shape {hog_train.shape}")
    print(f"HOG features for testing set saved to {cfg['test_stats_file']} with shape {hog_test.shape}")
