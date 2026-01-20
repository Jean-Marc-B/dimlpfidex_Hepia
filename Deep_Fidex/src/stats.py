# stats.py
import shap
import time
import os
import numpy as np
import tensorflow as tf

from utils.utils import (
    getHistogram,
    generate_filtered_images_and_predictions,
    compute_impact_patches,
    output_data
)
from utils.config import *
from skimage.feature import hog, local_binary_pattern
from scipy.fft import dctn


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

def compute_proba_images(cfg, nb_samples, data, size1D, nb_channels, nb_classes, CNNModel, filter_size, stride, args, baseline_preds=None):

    # Number of patches produced by the sliding window
    nb_patches = sum(
        ((size1D - f_sz[0]) // st[0] + 1) * ((size1D - f_sz[1]) // st[1] + 1)
        for f_sz, st in zip(filter_size, stride)
    )
    output_size = nb_patches * nb_classes
    #print(output_size) # (4840)
    proba_images = np.zeros((nb_samples,output_size))

    for sample_id in range(nb_samples):
        image = data[sample_id]
        image = image.reshape(size1D, size1D, nb_channels)
        baseline = None
        if baseline_preds is not None:
            baseline = baseline_preds[sample_id]
        if args.statistic in ["patch_impact_and_image", "patch_impact_and_stats"]:
            stats_batch_size = getattr(args, "stats_batch_size", 512)
            predictions, positions, nb_areas_per_filter = compute_impact_patches(
                cfg,
                CNNModel,
                image,
                filter_size,
                stride,
                baseline_pred=baseline,
                batch_size=stats_batch_size,
            )
        else:
            predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
                cfg, CNNModel, image, filter_size, stride)
        # print(predictions.shape)  # (484, 10)
        predictions = predictions.reshape(output_size)
        # print(predictions.shape) # (4840,)
        proba_images[sample_id] = predictions
    #print(proba_images.shape)  # (nb_samples, 4840)
    return proba_images

###############################################################

def compute_stats(cfg, X_train, X_test, CNNModel, intermediate_model, args, stats_file = None):
    """
    Compute statistics :
      - histograms
      - activations
      - probabilities
    with respect to the value of args.statistic.
    """

    if not stats_file:
        stats_file = [cfg["train_stats_file"], cfg["test_stats_file"]]

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
        output_data(train_histograms, stats_file[0])
        output_data(test_histograms, stats_file[1])
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

        output_data(train_sums, stats_file[0])
        output_data(test_sums, stats_file[1])
        print("Sums saved.")

    elif args.statistic in ["probability", "probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one", "patch_impact_and_image", "patch_impact_and_stats"]: # We create an image out of the probabilities (for each class) of cropped areas of the original image
        baseline_train = baseline_test = None
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
            if args.statistic in ["patch_impact_and_image", "patch_impact_and_stats"]:
                # use stored full-image predictions as baselines if available
                if os.path.exists(cfg["train_pred_file"]):
                    baseline_train = np.loadtxt(cfg["train_pred_file"])
                if os.path.exists(cfg["test_pred_file"]):
                    baseline_test = np.loadtxt(cfg["test_pred_file"])
            if args.statistic == ["patch_impact_and_image", "patch_impact_and_stats"]:
                print("\nComputing patch impacts on training set...\n")
            else:
                print("\nComputing train probability images of patches...\n")
            # Get sums for each train sample
            train_probas = compute_proba_images(cfg, nb_train_images, X_train, cfg["size1D"], cfg["nb_channels"],
                                                cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE, args, baseline_train)
            # shape (nb_train_images(images), cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*nb_classes)
            if args.statistic == ["patch_impact_and_image", "patch_impact_and_stats"]:
                print("\nComputed patch impacts on training set.")
            else:
                print("\nComputed train probability images of patches.")

            if args.statistic == ["patch_impact_and_image", "patch_impact_and_stats"]:
                print("\nComputing patch impacts on testing set...\n")
            else:
                print("\nComputing test probability images of patches...\n")
            # Get sums for each train sample
            test_probas = compute_proba_images(cfg, nb_test_images, X_test, cfg["size1D"], cfg["nb_channels"],
                                            cfg["nb_classes"], CNNModel, FILTER_SIZE, STRIDE, args, baseline_test)
            if args.statistic == ["patch_impact_and_image", "patch_impact_and_stats"]:
                print("\nComputed patch impacts on testing set.")
            else:
                print("\nComputed test probability images of patches.")

        output_data(train_probas, stats_file[0])
        output_data(test_probas, stats_file[1])
        if args.statistic == ["patch_impact_and_image", "patch_impact_and_stats"]:
            print("Patch impact images saved.")
        else:
            print("Probability images saved.")

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

    #Modify to size nb_original_samples x (nb_patches_per_image*32) :
    hog_train = hog_train.reshape(nb_original_train_samples, -1)
    hog_test = hog_test.reshape(nb_original_test_samples, -1)

    # --- Save features to files defined in cfg ---
    np.savetxt(cfg["train_stats_file"], hog_train)
    np.savetxt(cfg["test_stats_file"], hog_test)

    print(f"HOG features for training set saved to {cfg['train_stats_file']} with shape {hog_train.shape}")
    print(f"HOG features for testing set saved to {cfg['test_stats_file']} with shape {hog_test.shape}")


def compute_LBP_center_bits(cfg, X_train, X_test, nb_original_train_samples, nb_original_test_samples):
    """
    Compute per-patch LBP (method='default') bits at the *central pixel* only.
    R = floor((min(H,W)-1)/2), P = 8*R. Saves 0/1 bit vectors (length P).

    Parameters
    ----------
    cfg : dict
        - "nb_channels": 1 for grayscale, else converted to grayscale
        - "train_stats_file": path to save train features (txt)
        - "test_stats_file" : path to save test  features (txt)
    X_train, X_test : np.ndarray
        Arrays of shape (N, H, W, C) or (N, H, W, 1). Values can be any dtype.
    nb_original_train_samples, nb_original_test_samples : int
        For reshaping like in your HOG function (pass-through if already one
        LBP vector per “original sample”).

    Returns
    -------
    None (saves features to cfg["train_stats_file"], cfg["test_stats_file"])
    """


    # Convert images to grayscale if they are not already
    if cfg["nb_channels"] != 1:
        X_train = tf.image.rgb_to_grayscale(X_train).numpy()
        X_test  = tf.image.rgb_to_grayscale(X_test).numpy()

    if cfg["data_type"] == "float":
        X_train = (X_train * 255).astype(np.uint8)
        X_test  = (X_test  * 255).astype(np.uint8)
    X_train = np.squeeze(X_train, axis=-1)
    X_test  = np.squeeze(X_test, axis=-1)

    # ---- 2) Check sizes and derive R, P
    Ht, Wt = X_train.shape[1], X_train.shape[2]
    Hs, Ws = X_test.shape[1], X_test.shape[2]
    if (Ht, Wt) != (Hs, Ws):
        raise ValueError(f"Train and test images must share the same size. Got train {(Ht, Wt)} vs test {(Hs, Ws)}.")
    H, W = Ht, Wt

    R = int((min(H, W) - 1) // 2)
    if R < 1:
        raise ValueError(f"Patch too small for LBP (got H={H}, W={W} -> R={R}). Need min(H,W) >= 3.")
    P = int(8 * R)
    #P = 24

    # ---- 3) Central index (works for odd or even sizes)
    ci, cj = H // 2, W // 2  # floor(H/2), floor(W/2)

    # ---- 4) Helpers
    def lbp_center_bits(img):
        # skimage LBP expects float64 contiguous
        img_u8 = np.ascontiguousarray(img, dtype=np.uint8)
        lbp_map = local_binary_pattern(img_u8, P=P, R=R, method="default")
        code = int(lbp_map[ci, cj])
        # Decompose code into P bits (order matches scikit-image's sampling order)
        bits = np.fromiter(((code >> k) & 1 for k in range(P)), count=P, dtype=np.uint8)
        return bits

    # ---- 5) Compute features
    print(f"Computing LBP(center) with method='default', R={R}, P={P} for training set...")
    lbp_train = np.vstack([lbp_center_bits(im) for im in X_train]).astype(np.uint8)

    print(f"Computing LBP(center) with method='default', R={R}, P={P} for testing set...")
    lbp_test = np.vstack([lbp_center_bits(im) for im in X_test]).astype(np.uint8)

    # ---- 6) Reshape like your HOG code (one vector per original sample)
    lbp_train = lbp_train.reshape(nb_original_train_samples, -1)
    lbp_test  = lbp_test.reshape(nb_original_test_samples,  -1)

    # ---- 7) Save
    np.savetxt(cfg["train_stats_file"], lbp_train, fmt="%d")
    np.savetxt(cfg["test_stats_file"],  lbp_test,  fmt="%d")

    print(f"LBP(center) bits for training saved to {cfg['train_stats_file']} with shape {lbp_train.shape}")
    print(f"LBP(center) bits for testing  saved to {cfg['test_stats_file']} with shape {lbp_test.shape}")


def compute_little_patch_stats(cfg, X_train, X_test, nb_original_train_samples, nb_original_test_samples, stats_file = None):
    """
    Compute simple statistics (min, max, mean, std) for small patches in the training and testing datasets.
    This function computes basic statistics for each small patch in the training and testing datasets.
    The computed statistics are then saved to the specified files.
    Parameters:
    - cfg: Configuration dictionary containing parameters such as file paths.
    - X_train: The training dataset containing small patches.
    - X_test: The testing dataset containing small patches.
    - nb_original_train_samples: The number of original training samples (images, not patches).
    - nb_original_test_samples: The number of original testing samples (images, not patches).
    Returns:
    None
    """

    print("Computing simple statistics for training set...")
    train_stats = []
    for patch in X_train:
        vmin = patch.min(axis=(0, 1))
        vmax = patch.max(axis=(0, 1))
        mean = patch.mean(axis=(0, 1))
        std  = patch.std(axis=(0, 1))
        train_stats.append(np.concatenate([vmin, vmax, mean, std]))
    train_stats = np.array(train_stats)

    print("Computing simple statistics for testing set...")
    test_stats = []
    for patch in X_test:
        vmin = patch.min(axis=(0, 1))
        vmax = patch.max(axis=(0, 1))
        mean = patch.mean(axis=(0, 1))
        std  = patch.std(axis=(0, 1))
        test_stats.append(np.concatenate([vmin, vmax, mean, std]))
    test_stats = np.array(test_stats)

    # Modify to size nb_original_samples x (nb_patches_per_image*4) :
    train_stats = train_stats.reshape(nb_original_train_samples, -1)
    test_stats = test_stats.reshape(nb_original_test_samples, -1)

    # Save features to files defined in cfg
    if not stats_file:
        stats_file = [cfg["train_stats_file"], cfg["test_stats_file"]]
    np.savetxt(stats_file[0], train_stats)
    np.savetxt(stats_file[1], test_stats)

    print(f"Simple statistics for training set saved to {cfg['train_stats_file']} with shape {train_stats.shape}")
    print(f"Simple statistics for testing set saved to {cfg['test_stats_file']} with shape {test_stats.shape}")


def compute_Shap(cfg, X_train, X_test, model, nb_original_train_samples, nb_original_test_samples):
    """
    Compute SHAP values with DeepExplainer for the training and testing datasets.

    Keeps the original DeepExplainer workflow (for CNNs) to avoid breaking previous behaviour.
    """
    def unpack_data(data):
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        return data, None

    train_patches, _ = unpack_data(X_train)
    test_patches, _ = unpack_data(X_test)

    # Create a SHAP explainer with a random background subset (max 100 samples)
    nb_background = min(train_patches.shape[0], 100)
    if nb_background < train_patches.shape[0]:
        background_idx = np.random.choice(train_patches.shape[0], size=nb_background, replace=False)
        background_data = train_patches[background_idx].astype(np.float32)
    else:
        background_data = train_patches.astype(np.float32)
    explainer = shap.DeepExplainer(model, background_data)

    def _stack_deep_shap_outputs(shap_values):
        """
        Normalize SHAP outputs to a 5D array (N, H, W, C, K).
        Handles cases where DeepExplainer returns a singleton extra dimension.
        """
        if isinstance(shap_values, list):
            shap_patch = np.stack(shap_values, axis=-1)  # (N, H, W, C, K) or (N, H, W, C, 1, K)
        else:
            shap_patch = shap_values

        shap_patch = np.asarray(shap_patch)

        # Remove trailing singleton class dimension if present
        if shap_patch.ndim == 6 and shap_patch.shape[-1] == 1:
            shap_patch = np.squeeze(shap_patch, axis=-1)
        # Remove penultimate singleton if stacking produced (N,H,W,C,1,K)
        if shap_patch.ndim == 6 and shap_patch.shape[-2] == 1:
            shap_patch = np.squeeze(shap_patch, axis=-2)

        if shap_patch.ndim != 5:
            raise ValueError(f"Unexpected SHAP shape {shap_patch.shape}, expected 5D (N,H,W,C,K).")
        return shap_patch

    def patch_stats_from_shap(shap_values, nb_original_samples):
        """
        Compute min/max/mean/std per patch, per channel and per class.
        Final shape:
        - binary: (nb_original_samples, nb_patches_per_image * 4 * nb_channels)
        - otherwise: (nb_original_samples, nb_patches_per_image * nb_classes * 4 * nb_channels)
        """

        shap_patch = _stack_deep_shap_outputs(shap_values)
        print("SHAP patch shape (DeepExplainer):", shap_patch.shape)

        if shap_patch.ndim != 5:
            raise ValueError(f"Unexpected SHAP shape {shap_patch.shape}, expected 5D (N,H,W,C,K).")

        nb_classes_present = shap_patch.shape[-1]
        class_indices = list(range(nb_classes_present))

        per_class_stats = []
        for cls_idx in class_indices:
            sv = shap_patch[..., cls_idx]  # (N, H, W, C)
            vmin = sv.min(axis=(1, 2))
            vmax = sv.max(axis=(1, 2))
            mean = sv.mean(axis=(1, 2))
            std = sv.std(axis=(1, 2))
            per_class_stats.append(np.concatenate([vmin, vmax, mean, std], axis=1))

        patch_stats = np.concatenate(per_class_stats, axis=1)
        return patch_stats.reshape(nb_original_samples, -1)

    print("Computing SHAP values for training set (DeepExplainer)...")
    shap_values_train = explainer.shap_values(train_patches, check_additivity=False)
    shap_train_stats = patch_stats_from_shap(shap_values_train, nb_original_train_samples)

    print("Computing SHAP values for testing set (DeepExplainer)...")
    shap_values_test = explainer.shap_values(test_patches, check_additivity=False)
    shap_test_stats = patch_stats_from_shap(shap_values_test, nb_original_test_samples)

    # --- Save features to files defined in cfg ---
    np.savetxt(cfg["train_stats_file"], shap_train_stats)
    np.savetxt(cfg["test_stats_file"], shap_test_stats)

    print(f"SHAP patch stats for training set saved to {cfg['train_stats_file']} with shape {shap_train_stats.shape}")
    print(f"SHAP patch stats for testing set saved to {cfg['test_stats_file']} with shape {shap_test_stats.shape}")


# This function won't work with GPU if the model is too huge (out of memory). So if you have too many trees.
def compute_Shap_RF(cfg, X_train, X_test, model, nb_original_train_samples, nb_original_test_samples):
    """
    Compute SHAP values with TreeExplainer for datasets built from patches and optional positions.

    Parameters:
    - cfg: Configuration dictionary containing parameters such as file paths.
    - X_train, X_test: Either numpy arrays of patches (N, H, W, C) or tuples (patches, positions)
                       where positions is a list/array of shape (N, 2).
    - model: Tree-based model (e.g., RandomForest) used for making predictions.
    - nb_original_train_samples: Number of original training images.
    - nb_original_test_samples: Number of original testing images.
    """

    def unpack_data(data):
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        return data, None

    def to_feature_matrix(patches, positions=None):
        features = patches.reshape(patches.shape[0], -1).astype(np.float32)
        if positions is None:
            return features
        pos_array = np.asarray(positions, dtype=np.float32)
        if pos_array.shape[0] != features.shape[0]:
            raise ValueError(f"Positions length {pos_array.shape[0]} does not match patches {features.shape[0]}.")
        pos_array = pos_array.reshape(features.shape[0], -1)
        return np.concatenate([features, pos_array], axis=1)

    def stack_tree_shap(shap_values):
        if isinstance(shap_values, list):
            return np.stack(shap_values, axis=-1)  # (N, F, K)
        return shap_values[..., np.newaxis]       # (N, F, 1)

    def shap_values_with_progress(explainer, features, batch_size, label):
        """
        Compute SHAP values by batches to show progress and limit memory spikes.
        Returns the same type as TreeExplainer.shap_values (list or np.ndarray).
        """
        nb_samples = features.shape[0]
        shap_accum = None
        for start in range(0, nb_samples, batch_size):
            end = min(start + batch_size, nb_samples)
            print(f"{label} SHAP progress: {end}/{nb_samples} ({(end/nb_samples)*100:.1f}%)")
            shap_batch = explainer.shap_values(features[start:end], check_additivity=False)
            if isinstance(shap_batch, list):
                if shap_accum is None:
                    shap_accum = [arr for arr in shap_batch]
                else:
                    for idx, arr in enumerate(shap_batch):
                        shap_accum[idx] = np.concatenate([shap_accum[idx], arr], axis=0)
            else:
                if shap_accum is None:
                    shap_accum = shap_batch
                else:
                    shap_accum = np.concatenate([shap_accum, shap_batch], axis=0)
        return shap_accum

    def patch_stats_from_tree_shap(shap_values, patch_shape, nb_original_samples, feature_count_without_pos):
        shap_arr = stack_tree_shap(shap_values)  # (N, F, K)
        if shap_arr.shape[1] < feature_count_without_pos:
            raise ValueError(f"SHAP feature dimension {shap_arr.shape[1]} smaller than expected {feature_count_without_pos}.")

        shap_patch = shap_arr[:, :feature_count_without_pos, :]  # drop position SHAP values
        shap_patch = shap_patch.reshape(shap_arr.shape[0], patch_shape[0], patch_shape[1], patch_shape[2], shap_arr.shape[2])

        nb_classes_present = shap_patch.shape[-1]
        class_indices = [1] if nb_classes_present == 2 else list(range(nb_classes_present))

        per_class_stats = []
        for cls_idx in class_indices:
            sv = shap_patch[..., cls_idx]  # (N, H, W, C)
            vmin = sv.min(axis=(1, 2))
            vmax = sv.max(axis=(1, 2))
            mean = sv.mean(axis=(1, 2))
            std = sv.std(axis=(1, 2))
            per_class_stats.append(np.concatenate([vmin, vmax, mean, std], axis=1))

        patch_stats = np.concatenate(per_class_stats, axis=1)
        return patch_stats.reshape(nb_original_samples, -1)

    # Unpack data
    train_patches, train_positions = unpack_data(X_train)
    test_patches, test_positions = unpack_data(X_test)

    patch_shape = train_patches.shape[1:4]
    feature_count_without_pos = patch_shape[0] * patch_shape[1] * patch_shape[2]

    # Prepare feature matrices (flatten patches + positions if provided)
    train_features = to_feature_matrix(train_patches, train_positions)
    test_features = to_feature_matrix(test_patches, test_positions)

    # Path-dependent mode is much faster than the interventional default on many samples
    explainer = shap.GPUTreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw"  # path-dependent only supports raw output
    )

    batch_size = 2000

    print("Computing SHAP values for training set (TreeExplainer, path-dependent)...")
    shap_values_train = shap_values_with_progress(explainer, train_features, batch_size, "Train")
    shap_train_stats = patch_stats_from_tree_shap(shap_values_train, patch_shape, nb_original_train_samples, feature_count_without_pos)

    print("Computing SHAP values for testing set (TreeExplainer, path-dependent)...")
    shap_values_test = shap_values_with_progress(explainer, test_features, batch_size, "Test")
    shap_test_stats = patch_stats_from_tree_shap(shap_values_test, patch_shape, nb_original_test_samples, feature_count_without_pos)

    # --- Save features to files defined in cfg ---
    np.savetxt(cfg["train_stats_file"], shap_train_stats)
    np.savetxt(cfg["test_stats_file"], shap_test_stats)

    print(f"SHAP patch stats for training set saved to {cfg['train_stats_file']} with shape {shap_train_stats.shape}")
    print(f"SHAP patch stats for testing set saved to {cfg['test_stats_file']} with shape {shap_test_stats.shape}")


def compute_DCT(cfg, X_train, X_test, nb_original_train_samples, nb_original_test_samples, DCT_size):
    """
    Compute DCT coefficients for the training and testing datasets.

    This function computes the Discrete Cosine Transform (DCT) coefficients for each image in the training
    and testing datasets. The computed DCT features are then saved to the specified files.

    Parameters:
    - cfg: Configuration dictionary containing parameters such as file paths and HOG settings.
    - X_train: The training dataset containing images (each image must be 8x8).
    - X_test: The testing dataset containing images (each image must be 8x8).
    - nb_original_train_samples: The number of original training samples (images, not patches).
    - nb_original_test_samples: The number of original testing samples (images, not patches).
    - DCT_size: The number of DCT coefficients to retain for each patch (DCT_size x DCT_size).

    Returns:
    None
    """

    def compute_split(X, DCT_size):
            """
            X : (nb_samples, N, M, C)
            return : (nb_samples, C*DCT_size*DCT_size)
            """

            nb_samples, N, M, C = X.shape
            k = min(DCT_size, N, M) # In case DCT_size > N or M


            # 2D DCT
            Y = dctn(X, type=2, norm='ortho', axes=(1, 2))

            # keep only low frequencies
            Y = Y[:, :k, :k, :]   # (nb_samples, DCT_size, DCT_size, C)

            # Flatten : (nb_samples, C*DCT_size*DCT_size)
            return Y.reshape(nb_samples, -1)

    print("Computing DCT coefficients for training set...")
    dct_train = compute_split(X_train, DCT_size)

    print("Computing DCT coefficients for testing set...")
    dct_test = compute_split(X_test, DCT_size)

    #Modify to size nb_original_samples x (nb_patches_per_image*C*DCT_size*DCT_size) :
    dct_train = dct_train.reshape(nb_original_train_samples, -1)
    dct_test = dct_test.reshape(nb_original_test_samples, -1)

    # --- Save coefficients to files defined in cfg ---
    np.savetxt(cfg["train_stats_file"], dct_train)
    np.savetxt(cfg["test_stats_file"], dct_test)

    print(f"DCT coefficients for training set saved to {cfg['train_stats_file']} with shape {dct_train.shape}")
    print(f"DCT coefficients for testing set saved to {cfg['test_stats_file']} with shape {dct_test.shape}")
