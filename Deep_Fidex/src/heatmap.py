# heatmap.py
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from utils.config import *
from utils.utils import generate_filtered_images_and_predictions, image_to_rgb

# Only for one filter !
def get_heat_maps(CNNModel, image, filter_size, classes, predictions, positions, nb_areas_per_filter):
    """
    Generates heat maps for each class in an image by applying a sliding filter, computing predictions,
    and overlaying the class-specific heat maps on the original image.

    Parameters
    ----------
    CNNModel : keras.Model
        The CNN model used for predicting the class probabilities for each filtered area in the image.
    image : numpy.ndarray
        The input image to be processed. It can be a 2D array (grayscale) or a 3D array (RGB or grayscale with a third dimension).
    filter_size : tuple
        The size of the filter applied to the image (height, width) to extract regions for predictions.
    classes : list of str
        A list of class names corresponding to the output classes for each heat map.
    predictions : numpy.ndarray
        The predictions of each area in the image
    positions : tuple
        The positions of each area in the image
    nb_areas_per_filter : list of int
        The number of areas in the image for each filter size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the original image and heat maps for each class, organized in a grid layout.
    """

    real_filter_size = filter_size[0]

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)

    total_images = len(classes)+1
    # We set a maximum number of columns per row for better visualization
    max_columns = 4
    num_columns = min(max_columns, total_images)
    num_rows = (total_images + num_columns - 1) // num_columns  # Calculate the number of rows

    # Create the matplotlib figure with dynamic rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
    fig.suptitle("Original image with heat maps for each class", fontsize=16)

    # If axes is a single AxesSubplot, convert it to a list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Show the original image on top left
    axes[0].imshow(original_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for class_id in range(len(classes)):
        # For each pixel we compute the mean of the predictions touching this pixel and normalize it to get the intensity
        sum_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        count_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

        # For each area (it's prediction, position and filter size)
        for i, (prediction, position) in enumerate(zip(predictions, positions)):
            # Get location
            top_left = position
            bottom_right = (position[0] + real_filter_size[0], position[1] + real_filter_size[1]) # Goes one pixel too long but handled correctly then

            # Add the area prediction and increase the counter
            sum_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += prediction[class_id]
            count_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += 1

        # Mean of prediction for each pixel
        intensity_map = np.divide(sum_map, count_map, where=(count_map != 0))
        normalized_intensity = intensity_map * 255

        # Apply red filter
        heat_map_img = original_image_rgb.copy().astype(np.float32)
        #heat_map_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        heat_map_img[:, :, 0] = np.clip(heat_map_img[:, :, 0].astype(float) + normalized_intensity.astype(float), 0, 255)
        heat_map_img = heat_map_img.astype(np.uint8)


        axes[class_id+1].imshow(heat_map_img)
        axes[class_id+1].set_title(f"Heat map for class {classes[class_id]}")
        axes[class_id+1].axis('off')

    # Hide any remaining empty subplots if total_images < num_rows * num_columns
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() # Adjust spacing
    plt.subplots_adjust(top=0.85)  # Let space for the main title

    plt.close(fig)

    return fig

def generate_heatmaps(cfg, X_test, CNNModel, args, test_positions=None):
    """
    Generate heatmaps for X_test
    """
    if os.path.exists(cfg["heat_maps_folder"]):
        shutil.rmtree(cfg["heat_maps_folder"])
    os.makedirs(cfg["heat_maps_folder"])

    if args.train_with_patches:
        test_positions = np.array(test_positions)
        test_pred = np.loadtxt(cfg["test_pred_file"])
        nb_patches_per_image = cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]

    for idx, img in enumerate(X_test[:100]):

        if args.train_with_patches:
            nb_areas_per_filter = [nb_patches_per_image]
            start_idx = idx * nb_patches_per_image
            end_idx = start_idx + nb_patches_per_image
            predictions = test_pred[start_idx:end_idx, :]
            positions = test_positions[start_idx:end_idx, :]
        else:
            predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
            cfg, CNNModel, img, FILTER_SIZE, STRIDE)


        heat_maps_img = get_heat_maps(CNNModel, img, FILTER_SIZE, cfg["classes"], predictions, positions, nb_areas_per_filter)
        heat_maps_img.savefig(f"{cfg['heat_maps_folder']}/sample_{idx}.png")
        plt.close(heat_maps_img)
