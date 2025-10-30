# images.py
import os
import shutil
import copy
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
import textwrap

from utils.utils import (
    getRules,
    image_to_rgb,
    denormalize_to_255,
    get_top_ids,
    generate_filtered_images_and_predictions,
    getCoveredSamples,
    getCoveringRulesForSample
)
from trainings.trnFun import get_attribute_file
from utils.constants import HISTOGRAM_ANTECEDENT_PATTERN
from utils.config import *


def highlight_area_histograms(CNNModel, image, true_class, filter_size, rule, classes, predictions, positions, nb_areas_per_filter):
    """
    Highlights important areas in an image based on the rule's antecedents using the CNN model.

    Parameters:
    - CNNModel: The trained CNN model used to make predictions on filtered areas of the image.
    - image: The input image (2D grayscale or 3D RGB) to be processed and highlighted.
    - filter_size: A tuple or a list of tuples representing the size (height, width) of the filter applied on the image.
    - rule: An object containing antecedents which specify conditions (class ID and prediction threshold)
            for highlighting areas based on CNN predictions.
    - classes: A list or dictionary that maps class IDs to class names for better interpretability.
    - predictions : The predictions of each area in the image
    - positions : The positions of each area in the image
    - nb_areas_per_filter : The number of areas in the image for each filter size.

    Returns:
    - fig: The matplotlib figure containing subplots with:
        1. The original image.
        2. The image with all filters applied based on the rule's antecedents.
        3. Individual images showing the highlighted areas for each antecedent separately.
    """

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)
    original_image_rgb = denormalize_to_255(original_image_rgb)

    # Initialize the combined intensity maps for green and red
    combined_green_intensity = np.zeros((image.shape[0], image.shape[1]))
    combined_red_intensity = np.zeros((image.shape[0], image.shape[1]))

    individual_maps = []

    # Identify current filter size
    filter_index = 0
    filter_start_idx = 0
    current_filter_size = filter_size[filter_index]

    for antecedent in rule.antecedents: # For each antecedent
        # Get the class id and prediction threshold of this antecedent
        match = re.match(HISTOGRAM_ANTECEDENT_PATTERN, antecedent.attribute)
        if match:
            pred_threshold = float(match.group(2))
            class_id = int(match.group(1))
        else:
            raise ValueError(f"Wrong antecedant format : {antecedent.attribute}")

        individual_intensity_map = np.zeros((image.shape[0], image.shape[1]))

        # For each area (it's prediction, position and filter size)
        for i, (prediction, position) in enumerate(zip(predictions, positions)):
            # Update filter size if we exceed current limit
            if i >= filter_start_idx + nb_areas_per_filter[filter_index]:
                filter_start_idx += nb_areas_per_filter[filter_index]
                filter_index += 1
                current_filter_size = filter_size[filter_index]

            class_prob = prediction[class_id] # Prediction of the area

            # Check if the prediction with this area satisfies the antecedent
            if (class_prob >= pred_threshold):
                top_left = position
                bottom_right = (position[0] + current_filter_size[0], position[1] + current_filter_size[1]) # Goes one pixel too long but handled correctly then

                # Accumulate the intensity of the activation in the individual color map
                individual_intensity_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += 1

                # Accumulate the intensity in the combined map for each activations
                if antecedent.inequality:
                    combined_green_intensity[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += 1
                else:
                    combined_red_intensity[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += 1

        individual_maps.append((individual_intensity_map, antecedent))



    # Create the image combined with all filter of all antecedents

    max_green = np.max(combined_green_intensity) if np.max(combined_green_intensity) > 0 else 1
    max_red = np.max(combined_red_intensity) if np.max(combined_red_intensity) > 0 else 1

    # Normalise the intensity maps combined between 0 and 255
    green_overlay = (combined_green_intensity / max_green) * 255
    red_overlay = (combined_red_intensity / max_red) * 255

    combined_image = original_image_rgb.copy().astype(np.float32)

    # Apply the combined filters
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0].astype(float) + red_overlay.astype(float), 0, 255)  # Ajout du rouge
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1].astype(float) + green_overlay.astype(float), 0, 255)  # Ajout du vert

    combined_image = combined_image.astype(np.uint8)

    # Determine the number of rows and columns for the subplots
    num_antecedents = len(rule.antecedents)
    total_images = num_antecedents + 2  # Original, combined, and individual filters

    # We set a maximum number of columns per row for better visualization
    max_columns = 4
    num_columns = min(max_columns, total_images)
    num_rows = (total_images + num_columns - 1) // num_columns  # Calculate the number of rows

    # Create the matplotlib figure with dynamic rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
    fig.suptitle("Original, Combined, and Individual Highlighted Areas for each rule antecedent", fontsize=16)

    # If axes is a single AxesSubplot, convert it to a list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Show the original image on top left
    axes[0].imshow(original_image_rgb)
    axes[0].set_title(f"Original Image of class {true_class}")
    axes[0].axis('off')

    # Show combined image
    axes[1].imshow(combined_image)
    axes[1].set_title("Combined Filters")
    axes[1].axis('off')

    # Show each image with an individual filter (for each antecedent)
    for i, (intensity_map, antecedent) in enumerate(individual_maps, start=2):
        # Normlaise the intensity of the filter for the antecedent
        max_intensity = np.max(intensity_map) if np.max(intensity_map) > 0 else 1
        normalized_intensity = (intensity_map / max_intensity) * 255

        individual_image = original_image_rgb.copy().astype(np.float32)

        # Apply filter green for >= or red for <
        if antecedent.inequality:
            # Apply green filter
            individual_image[:, :, 1] = np.clip(individual_image[:, :, 1].astype(float) + normalized_intensity.astype(float), 0, 255)
        else:
            # Apply red filter
            individual_image[:, :, 0] = np.clip(individual_image[:, :, 0].astype(float) + normalized_intensity.astype(float), 0, 255)

        individual_image = individual_image.astype(np.uint8)

        # Show an image for each individual antecedant
        match = re.match(HISTOGRAM_ANTECEDENT_PATTERN, antecedent.attribute)
        if match:
            class_id = int(match.group(1))
            pred_threshold = match.group(2)
            class_name = classes[class_id]  # Get the class name
        ineq = ">=" if antecedent.inequality else "<"

        axes[i].imshow(individual_image)
        axes[i].set_title(f"Filter for P_{class_name}>={pred_threshold}{ineq}{antecedent.value}")
        axes[i].axis('off')

    # Hide any remaining empty subplots if total_images < num_rows * num_columns
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() # Adjust spacing
    plt.subplots_adjust(top=0.85)  # Let space for the main title

    plt.close(fig)

    return fig

###############################################################

def highlight_area_activations_sum(cfg, CNNModel, intermediate_model, image, true_class, rule, filter_size, stride, classes):

    nb_top_filters = 20 # Number of filters to show in an image

    activations, positions = generate_filtered_images_and_predictions( #nb_filters x nb_activations
            cfg, CNNModel, image, filter_size, stride, intermediate_model)

    filter_size = filter_size[0]

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)
    original_image_rgb = denormalize_to_255(original_image_rgb)

    # Initialize the combined intensity maps for green and red
    combined_green_intensity = np.zeros((image.shape[0], image.shape[1]))
    combined_red_intensity = np.zeros((image.shape[0], image.shape[1]))

    individual_maps = []

    for antecedent in rule.antecedents:
        # WARNING : Depending on the activation before the flatten layer, all smaller values are 0 which cannot be well interpreted
        # Get top X ids of patches on this activation
        if antecedent.inequality: #>=
            patches_idx = get_top_ids(activations[:,antecedent.attribute], nb_top_filters) # The attribute is the id of the activation
            #print("big", activations[patches_idx,antecedent.attribute])
        else: #<
            patches_idx = get_top_ids(activations[:,antecedent.attribute], nb_top_filters, False)
            #print("small", activations[patches_idx,antecedent.attribute])

        for i in patches_idx:
            position = positions[i]
            top_left = position
            bottom_right = (position[0] + filter_size[0], position[1] + filter_size[1]) # Goes one pixel too long but handled correctly then

            individual_intensity_map = np.zeros((image.shape[0], image.shape[1]))

            #Calculate intensity based on activation values
            activation_values = activations[patches_idx, antecedent.attribute]
            min_val, max_val = activation_values.min(), activation_values.max()
            normalized_intensities = (activation_values - min_val) / (max_val - min_val + 1e-5)

            # Apply intensity to patches
            for i, idx in enumerate(patches_idx):  # Utilisation directe de l'index dans patches_idx
                position = positions[idx]
                intensity = normalized_intensities[i]
                top_left = position
                bottom_right = (position[0] + filter_size[0], position[1] + filter_size[1]) # Goes one pixel too long but handled correctly then

                if antecedent.inequality:  # Apply green filter
                    combined_green_intensity[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += intensity
                    individual_intensity_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += intensity
                else:  # Apply red filter
                    combined_red_intensity[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += intensity
                    individual_intensity_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] += intensity

        individual_maps.append(individual_intensity_map)

    # Renormalization of intensities for each channel
    if np.max(combined_green_intensity) > 0:
        combined_green_intensity = np.clip(combined_green_intensity / np.max(combined_green_intensity), 0, 1)
    else:
        combined_green_intensity = np.zeros_like(combined_green_intensity)  # Laisser l'intensité à zéro si aucune valeur

    if np.max(combined_red_intensity) > 0:
        combined_red_intensity = np.clip(combined_red_intensity / np.max(combined_red_intensity), 0, 1)
    else:
        combined_red_intensity = np.zeros_like(combined_red_intensity)  # Laisser l'intensité à zéro si aucune valeur

    # Combine green and red intensities into a single image
    combined_image = original_image_rgb.copy()
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1].astype(float) + combined_green_intensity.astype(float) * 255, 0, 255)  # Green channel
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0].astype(float) + combined_red_intensity.astype(float) * 255, 0, 255)    # Red channel

    # Plotting

    # Determine the number of rows and columns for the subplots
    num_antecedents = len(rule.antecedents)
    total_images = num_antecedents + 2  # Original, combined, and individual filters

    # We set a maximum number of columns per row for better visualization
    max_columns = 4
    num_columns = min(max_columns, total_images)
    num_rows = (total_images + num_columns - 1) // num_columns  # Calculate the number of rows

    # Create the matplotlib figure with dynamic rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
    fig.suptitle("Original, Combined, and Individual Highlighted Areas for each rule antecedent", fontsize=16)

    # If axes is a single AxesSubplot, convert it to a list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Show the original image on top left
    axes[0].imshow(original_image_rgb)
    axes[0].set_title(f"Original Image of class {true_class}")
    axes[0].axis('off')

    # Show combined image
    axes[1].imshow(combined_image.astype(np.uint8))
    axes[1].set_title("Combined Filters")
    axes[1].axis('off')

    # Show each antecedent image
    for i, intensity_map in enumerate(individual_maps):
        antecedent = rule.antecedents[i]
        # Renormalize each intensity map
        max_intensity = np.max(intensity_map)
        if max_intensity > 0:  # Eviter la division par zéro
            intensity_map = intensity_map / max_intensity

        # Create filter image
        filtered_image = original_image_rgb.copy()
        if antecedent.inequality:
            filtered_image[:, :, 1] = np.clip(filtered_image[:, :, 1].astype(float) + intensity_map.astype(float) * 255, 0, 255)
        else:
            filtered_image[:, :, 0] = np.clip(filtered_image[:, :, 0].astype(float) + intensity_map.astype(float) * 255, 0, 255)

        filtered_image = filtered_image.astype(np.uint8)
        ineq = ">=" if antecedent.inequality else "<"
        axes[i+2].imshow(filtered_image)
        axes[i+2].set_title(f"Top {nb_top_filters} Filter for Sum(A{antecedent.attribute}){ineq}{antecedent.value}")
        axes[i+2].axis('off')

    # Hide any remaining empty subplots if total_images < num_rows * num_columns
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() # Adjust spacing
    plt.subplots_adjust(top=0.85)  # Let space for the main title

    plt.close(fig)

    return fig

###############################################################

def split_title(title, max_line_length=50):
    """Split title into two lines, regardless of length."""
    wrapped = textwrap.wrap(title, max_line_length)
    if len(wrapped) == 1:
        # For short titles, artificially split in the middle
        midpoint = len(title) // 2
        for i in range(midpoint, 0, -1):
            if title[i] == "_":  # Prefer splitting at underscores or spaces
                return title[:i] + "\n" + title[i+1:]
        return title[:midpoint] + "\n" + title[midpoint:]
    return "\n".join(wrapped[:2])

def highlight_area_probability_image(image, true_class, rule, size1D, size_Height_proba_stat, size_Width_proba_stat, filter_size, classes, nb_channels, statistic):

    if statistic in ["probability_and_image", "probability_and_HOG_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one", "HOG_and_image", "HOG"]:
        prob_and_img_in_one_matrix = False
    else:
        prob_and_img_in_one_matrix = True

    nb_classes = len(classes)

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)
    original_image_rgb = denormalize_to_255(original_image_rgb)

    filtered_images = []
    combined_image_intensity = np.zeros_like(original_image_rgb, dtype=float)
    mask_green_combined = np.zeros((size1D, size1D), dtype=bool)
    mask_red_combined = np.zeros((size1D, size1D), dtype=bool)

    scale_h = size1D / size_Height_proba_stat # For probabilities and image in one matrix
    scale_w = size1D / size_Width_proba_stat

    if statistic in ["HOG_and_image", "HOG"]:
        nb_chanels_stats = 32
    else:
        nb_chanels_stats = nb_classes
    split_id_image = size_Height_proba_stat * size_Width_proba_stat * nb_chanels_stats # For separated probability(or HOG) and image
    if statistic == "probability_and_HOG_and_image":
        split_id_image += size_Height_proba_stat * size_Width_proba_stat*32

    for antecedent in rule.antecedents:
        filtered_image_intensity = np.zeros_like(original_image_rgb, dtype=float)

        if prob_and_img_in_one_matrix:
            area_Height, area_Width, channel_id = np.unravel_index(antecedent.attribute, (size_Height_proba_stat, size_Width_proba_stat, nb_classes + nb_channels))
            is_proba_part = channel_id < nb_classes
        else:
            if statistic == "HOG":
                is_proba_part = True
            else:
                is_proba_part = antecedent.attribute < split_id_image

        if is_proba_part: # proba or HOG part
            if not prob_and_img_in_one_matrix:
                adapted_antecedent_attr = antecedent.attribute
                if statistic == "probability_and_HOG_and_image":
                    split_id_HOG = size_Height_proba_stat * size_Width_proba_stat*nb_classes
                    if antecedent.attribute >= split_id_HOG: # HOG part
                        adapted_antecedent_attr = antecedent.attribute - split_id_HOG
                        nb_chanels_stats = 32
                area_Height, area_Width, channel_id = np.unravel_index(adapted_antecedent_attr, (size_Height_proba_stat, size_Width_proba_stat, nb_chanels_stats))

            start_h = area_Height * STRIDE[0][0]
            start_w = area_Width  * STRIDE[0][1]
            end_h   = start_h + filter_size[0][0] - 1
            end_w   = start_w + filter_size[0][1] - 1

            if antecedent.inequality:  # >=
                filtered_image_intensity[start_h:end_h+1, start_w:end_w+1, 1] += 1
                combined_image_intensity[start_h:end_h+1, start_w:end_w+1, 1] += 1
            else:  # <
                filtered_image_intensity[start_h:end_h+1, start_w:end_w+1, 0] += 1
                combined_image_intensity[start_h:end_h+1, start_w:end_w+1, 0] += 1

        else: # image part
            if not prob_and_img_in_one_matrix:
                height, width, channel = np.unravel_index(antecedent.attribute - split_id_image, (size1D, size1D, nb_channels))
            else:
                height = round(area_Height * scale_h)
                width = round(area_Width * scale_w)

            if antecedent.inequality:  # >=
                filtered_image_intensity[height, width, :] = [0, 255, 0]  # Green
                combined_image_intensity[height, width, :] = [0, 255, 0]
            else:  # <
                filtered_image_intensity[height, width, :] = [255, 0, 0]  # Red
                combined_image_intensity[height, width, :] = [255, 0, 0]


        # We force pixel specifics to be green or red
        mask_green = (filtered_image_intensity[:, :, 1] >= 255).astype(bool)   # Pixels marked for green
        mask_red = (filtered_image_intensity[:, :, 0] >= 255).astype(bool)     # Pixels marked for red

        mask_green_combined |= mask_green
        mask_red_combined |= mask_red

        # Normalize to be between 0 and 255.
        max_val = np.max(filtered_image_intensity)
        if max_val > 0:
            filtered_image_intensity = np.clip(filtered_image_intensity / max_val * 255, 0, 255).astype(np.uint8)
        else:
            filtered_image_intensity = np.zeros_like(filtered_image_intensity, dtype=np.uint8)

        filtered_image = original_image_rgb.copy()
        filtered_image[:, :, 1] = np.clip(filtered_image[:, :, 1].astype(float) + filtered_image_intensity[:, :, 1].astype(float), 0, 255)  # Green channel type unint16 is mandatory otherwise addition will be cyclic (255+1=0)
        filtered_image[:, :, 0] = np.clip(filtered_image[:, :, 0].astype(float) + filtered_image_intensity[:, :, 0].astype(float), 0, 255)  # Red channel

        # Green Pixels
        filtered_image[mask_green] = [0, 255, 0]
        # Red Pixels
        filtered_image[mask_red] = [255, 0, 0]

        filtered_images.append(filtered_image)

    # Normalise combined image between 0 and 255

    combined_masked_intensity = combined_image_intensity.copy()
    combined_masked_intensity[mask_green_combined | mask_red_combined] = 0  # Ignore marked pixels for max calculation

    # Normalizazion without marked pixels
    combined_max_value = np.max(combined_masked_intensity)

    if combined_max_value > 0:
        combined_image_intensity = np.clip(combined_image_intensity / combined_max_value * 255, 0, 255) # Normalize to be between 0 and 255
    else:
        combined_image_intensity = np.zeros_like(combined_image_intensity)

    combined_image = original_image_rgb.copy()
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1].astype(float) + combined_image_intensity[:, :, 1].astype(float), 0, 255)  # Green channel
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0].astype(float) + combined_image_intensity[:, :, 0].astype(float), 0, 255)  # Red channel

    # Green Pixels
    combined_image[mask_green_combined] = [0, 255, 0]
    # Red Pixels
    combined_image[mask_red_combined] = [255, 0, 0]

    # Plotting

    # Determine the number of rows and columns for the subplots
    num_antecedents = len(rule.antecedents)
    total_images = num_antecedents + 2  # Original, combined, and individual filters

    # We set a maximum number of columns per row for better visualization
    max_columns = 2
    num_columns = min(max_columns, total_images)
    num_rows = (total_images + num_columns - 1) // num_columns  # Calculate the number of rows

    # Create the matplotlib figure with dynamic rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows), constrained_layout=True)
    fig.suptitle("Original, Combined, and Individual Highlighted Areas for each rule antecedent", fontsize=16)

    # If axes is a single AxesSubplot, convert it to a list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Show the original image on top left
    axes[0].imshow(original_image_rgb)
    axes[0].set_title(f"Original Image of class {true_class}", fontsize=20)
    #axes[0].set_title(f"Original Image of class {true_class}")
    axes[0].axis('off')
    # Show combined image
    axes[1].imshow(combined_image.astype(np.uint8))
    axes[1].set_title("Combined Filters", fontsize=20)
    #axes[1].set_title("Combined Filters")
    axes[1].axis('off')

    # Show each antecedent image
    for i,img in enumerate(filtered_images):
        antecedent = rule.antecedents[i]
        ineq = ">=" if antecedent.inequality else "<"
        axes[i+2].imshow(img.astype(np.uint8))

        if prob_and_img_in_one_matrix:
            area_Height, area_Width, channel_id = np.unravel_index(antecedent.attribute, (size_Height_proba_stat, size_Width_proba_stat, nb_chanels_stats + nb_channels))
            is_proba_part = channel_id < nb_classes
        else:
            if statistic == "HOG":
                is_proba_part = True
            else:
                is_proba_part = antecedent.attribute < split_id_image

        if is_proba_part: # proba part
            in_HOG = False
            if statistic in ["HOG_and_image", "HOG"]:
                in_HOG = True
            if not prob_and_img_in_one_matrix:
                adapted_antecedent_attr = antecedent.attribute
                if statistic == "probability_and_HOG_and_image":
                    split_id_HOG = size_Height_proba_stat * size_Width_proba_stat*nb_classes
                    if antecedent.attribute >= split_id_HOG: # HOG part
                        in_HOG = True
                        adapted_antecedent_attr = antecedent.attribute - split_id_HOG
                        nb_chanels_stats = 32
                area_Height, area_Width, channel_id = np.unravel_index(adapted_antecedent_attr, (size_Height_proba_stat, size_Width_proba_stat, nb_chanels_stats))

            start_h = area_Height * STRIDE[0][0]
            start_w = area_Width  * STRIDE[0][1]
            end_h   = start_h + filter_size[0][0] - 1
            end_w   = start_w + filter_size[0][1] - 1
            if in_HOG:
                statname = f"Descriptor_vector#{channel_id}"
            else:
                class_name = classes[channel_id]
                statname = f"P_class_{class_name}"
            axes[i+2].set_title(
                f"{statname}_area_[{start_h}-{end_h}]x[{start_w}-{end_w}]{ineq}{antecedent.value:.6f}\n"
                f"Covering size : {rule.coveringSizesWithNewAntecedent[i]}\n"
                f"Gain of fidelity : {rule.increasedFidelity[i]:.6f}\n"
                f"Change in accuracy : {rule.accuracyChanges[i]:.6f}"
                )


        else: # image part
            if not prob_and_img_in_one_matrix:
                height, width, channel = np.unravel_index(antecedent.attribute - split_id_image, (size1D, size1D, nb_channels))
            else:
                height = round(area_Height * scale_h)
                width = round(area_Width * scale_w)
                channel = antecedent.attribute % nb_channels
            axes[i+2].set_title(
                f"Pixel_{height}x{width}x{channel}{ineq}{antecedent.value:.6f}\n"
                f"Covering size : {rule.coveringSizesWithNewAntecedent[i]}\n"
                f"Gain of fidelity : {rule.increasedFidelity[i]:.6f}\n"
                f"Change in accuracy : {rule.accuracyChanges[i]:.6f}"
                )
        axes[i+2].axis('off')

    # Hide any remaining empty subplots if total_images < num_rows * num_columns
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    # plt.tight_layout() # Adjust spacing
    # plt.subplots_adjust(top=0.7)  # Let space for the main title

    plt.close(fig)

    return fig

###############################################################

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Resizing

def get_receptive_field_coordinates(model, layer_name, output_pixel):
    """
    Returns the coordinates of the 4 corners of the receptive field in the original input image
    that influence the specified output pixel in the given layer.

    Parameters:
      - model: the loaded Keras model.
      - layer_name: the name of the layer (up to which the calculation is performed).
      - output_pixel: a tuple (row, col) indicating the position of the pixel in the layer's output.

    Returns:
      A tuple of 4 points corresponding to the corners (top_left, top_right,
      bottom_left, bottom_right) with coordinates (row, col) in the original input image.

    We assume that the model contains at most one of each type of layer among:
      - Convolution2D (with kernel_size, strides, and padding)
      - Dropout (which does not affect the spatial dimensions)
      - MaxPooling2D (with pool_size, strides, and padding, here simplified with padding=0)
      - Resizing (which resizes the input image)
    """

    # Retrieve the list of layers from the input up to the target layer
    layer_configs = []
    # Get original input size from the model's input shape (ignoring batch dimension)
    orig_height = model.input_shape[1]
    orig_width = model.input_shape[2]
    for layer in model.layers:
        if isinstance(layer, Resizing):
            # Get target size from the Resizing layer
            new_height = layer.height
            new_width = layer.width
            # Compute scaling factors to map coordinates from the resized image back to the original image
            row_scale = orig_height / new_height
            col_scale = orig_width / new_width
            layer_configs.append({'type': 'resize', 'row_scale': row_scale, 'col_scale': col_scale})
        elif isinstance(layer, Conv2D):
            # Assume a square kernel and identical strides for height and width.
            kernel_size = layer.kernel_size[0]
            stride = layer.strides[0]
            # For simplicity, if padding is 'same', assume a padding of floor(kernel_size/2)
            padding = kernel_size // 2 if layer.padding == 'same' else 0
            layer_configs.append({'type': 'conv', 'kernel': kernel_size, 'stride': stride, 'padding': padding})
        elif isinstance(layer, MaxPooling2D):
            pool_size = layer.pool_size[0]
            stride = layer.strides[0]
            # For simplicity, assume no padding (padding='valid').
            # Use getattr in case the 'padding' attribute is not present.
            padding = pool_size // 2 if getattr(layer, 'padding', 'valid') == 'same' else 0
            layer_configs.append({'type': 'maxpool', 'kernel': pool_size, 'stride': stride, 'padding': padding})
        elif isinstance(layer, Dropout):
            layer_configs.append({'type': 'dropout'})

        if layer.name == layer_name:
            break

    # Initialize the receptive field at the output of the target layer
    row_start, row_end = output_pixel[0], output_pixel[0]
    col_start, col_end = output_pixel[1], output_pixel[1]

    # Iterate in reverse order (from the target layer back to the original input)
    for config in reversed(layer_configs):
        if config['type'] in ['conv', 'maxpool']:
            k = config['kernel']
            s = config['stride']
            p = config['padding']
            # Update for the vertical dimension
            row_start = row_start * s - p
            row_end   = row_end * s - p + k - 1
            # Update for the horizontal dimension
            col_start = col_start * s - p
            col_end   = col_end * s - p + k - 1
        elif config['type'] == 'resize':
            # Undo the resizing transformation by scaling coordinates back to the original image
            row_start = row_start * config['row_scale']
            row_end   = row_end * config['row_scale']
            col_start = col_start * config['col_scale']
            col_end   = col_end * config['col_scale']
        # Dropout does not affect spatial dimensions

    # Get integer indices
    row_start = math.floor(row_start)
    row_end = math.ceil(row_end)
    col_start = math.floor(col_start)
    col_end = math.ceil(col_end)

    # Stay inside the image
    row_start = max(0, min(orig_height - 1, row_start))
    row_end = max(0, min(orig_height - 1, row_end))
    col_start = max(0, min(orig_width - 1, col_start))
    col_end = max(0, min(orig_width - 1, col_end))

    # Return the 4 corners as tuples (row, col)
    top_left = (row_start, col_start)
    top_right = (row_start, col_end)
    bottom_left = (row_end, col_start)
    bottom_right = (row_end, col_end)
    return top_left, top_right, bottom_left, bottom_right


###############################################################

def highlight_area_first_conv(img, true_class, rule, model, height_feature_map, width_feature_map, nb_channels_feature_map):
    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(img)
    original_image_rgb = denormalize_to_255(original_image_rgb)

    filtered_images = []
    combined_image_intensity = np.zeros_like(original_image_rgb, dtype=float)

    feature_coords = []
    for antecedent in rule.antecedents:
        feature_coords.append(np.unravel_index(antecedent.attribute, (height_feature_map, width_feature_map, nb_channels_feature_map)))
        top_left, top_right, bottom_left, bottom_right = get_receptive_field_coordinates(model, "first_conv_end", feature_coords[-1])
        filtered_image_intensity = np.zeros_like(original_image_rgb, dtype=float)

        if antecedent.inequality:  # >=
            filtered_image_intensity[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1, 1] += 1
            combined_image_intensity[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1, 1] += 1
        else:  # <
            filtered_image_intensity[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1, 0] += 1
            combined_image_intensity[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1, 0] += 1
        if np.max(filtered_image_intensity) == 0:
            print(antecedent)
            print(top_left, top_right, bottom_left, bottom_right)
        filtered_image_intensity = np.clip(filtered_image_intensity / np.max(filtered_image_intensity) * 255, 0, 255).astype(np.uint8) # Normalize to be between 0 and 255.
        filtered_image = original_image_rgb.copy()
        filtered_image[:, :, 1] = np.clip(filtered_image[:, :, 1].astype(float) + filtered_image_intensity[:, :, 1].astype(float), 0, 255)  # Green channel type unint16 is mandatory otherwise addition will be cyclic (255+1=0)
        filtered_image[:, :, 0] = np.clip(filtered_image[:, :, 0].astype(float) + filtered_image_intensity[:, :, 0].astype(float), 0, 255)  # Red channel

        filtered_images.append(filtered_image)

    # Normalise combined image between 0 and 255
    combined_masked_intensity = combined_image_intensity.copy()
    combined_max_value = np.max(combined_masked_intensity)

    if combined_max_value > 0:
        combined_image_intensity = np.clip(combined_image_intensity / combined_max_value * 255, 0, 255) # Normalize to be between 0 and 255
    else:
        combined_image_intensity = np.zeros_like(combined_image_intensity)

    combined_image = original_image_rgb.copy()
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1].astype(float) + combined_image_intensity[:, :, 1].astype(float), 0, 255)  # Green channel
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0].astype(float) + combined_image_intensity[:, :, 0].astype(float), 0, 255)  # Red channel

    # Plotting

    # Determine the number of rows and columns for the subplots
    num_antecedents = len(rule.antecedents)
    total_images = num_antecedents + 2  # Original, combined, and individual filters

    # We set a maximum number of columns per row for better visualization
    max_columns = 4
    num_columns = min(max_columns, total_images)
    num_rows = (total_images + num_columns - 1) // num_columns  # Calculate the number of rows

    # Create the matplotlib figure with dynamic rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
    fig.suptitle("Original, Combined, and Individual Highlighted Areas for each rule antecedent", fontsize=16)

    # If axes is a single AxesSubplot, convert it to a list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Show the original image on top left
    axes[0].imshow(original_image_rgb)
    axes[0].set_title(f"Original Image of class {true_class}")
    axes[0].axis('off')

    # Show combined image
    axes[1].imshow(combined_image.astype(np.uint8))
    axes[1].set_title("Combined Filters")
    axes[1].axis('off')


    # Show each antecedent image
    for i,img in enumerate(filtered_images):
        antecedent = rule.antecedents[i]
        feature_height, feature_width, feature_channel = feature_coords[i]
        img = img.astype(np.uint8)
        ineq = ">=" if antecedent.inequality else "<"
        axes[i+2].imshow(img)
        axes[i+2].set_title(f"Feature_map_{feature_height}x{feature_width}x{feature_channel}{ineq}{antecedent.value:.6f}")
        axes[i+2].axis('off')
    # Hide any remaining empty subplots if total_images < num_rows * num_columns
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() # Adjust spacing
    plt.subplots_adjust(top=0.85)  # Let space for the main title

    plt.close(fig)

    return fig
###############################################################

def generate_explaining_images(cfg, X_train, Y_train, CNNModel, intermediate_model, args, train_positions=None, height_feature_map=-1, width_feature_map=-1, nb_channels_feature_map=-1, data_in_rules=None):
    """
    Generate explaining images.
    """
    print("Generation of images...")

    # Get train predictions
    if getattr(args, "train_with_patches", False):
        print("Loading train predictions...")
        train_positions = np.array(train_positions)
        train_pred = np.loadtxt(cfg["train_pred_file"])
        nb_patches_per_image = cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]
        print("Train predictions loaded.")

    # 1) Load rules
    global_rules = getRules(cfg["global_rules_file"])


    # 2) Load attributes
    if args.statistic == "histogram":
        attributes = get_attribute_file(cfg["attributes_file"], cfg["nb_stats_attributes"])[0]

    # 3) Create out folder
    if os.path.exists(cfg["rules_folder"]):
        shutil.rmtree(cfg["rules_folder"])
    os.makedirs(cfg["rules_folder"])


    # 4) For each rule we get filter images for train samples covering the rule (if image_version, it's the opposite, for each image we get filter images for rules covered by the train sample)
    nb_rules = args.images
    counters = [0] * cfg["nb_classes"]

    for rule_id, rule in enumerate(global_rules):
        if args.each_class:
            if all(count >= nb_rules for count in counters):
                break
        else:
            if sum(counters) >= nb_rules:
                break

        if args.each_class and counters[rule.target_class] >= nb_rules:
            continue
        else:
            counters[rule.target_class] += 1

        if args.statistic == "histogram":
            rule.include_X = False
            for ant in rule.antecedents:
                ant.attribute = attributes[ant.attribute] # Replace the attribute's index by its true name
        elif args.statistic in ["probability", "probability_and_image", "probability_and_HOG_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one", "convDimlpFilter", "HOG_and_image", "HOG"]:
                rule.include_X = False
        # Create folder for this rule
        rule_folder = os.path.join(cfg["rules_folder"], f"rule_{rule_id}_class_{cfg['classes'][rule.target_class]}")
        if os.path.exists(rule_folder):
            shutil.rmtree(rule_folder)
        os.makedirs(rule_folder)

        # Add a readme containing the rule
        readme_file = os.path.join(rule_folder, 'Readme.md')
        rule_to_print = copy.deepcopy(rule)
        rule_to_print.target_class = cfg['classes'][rule.target_class]
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
        elif args.statistic == "probability":
            # Change antecedent with area and class involved

            # Scales of changes of original image to reshaped image
            scale_h = cfg["size1D"] / cfg["size_Height_proba_stat"]
            scale_w = cfg["size1D"] / cfg["size_Width_proba_stat"]
            for antecedent in rule_to_print.antecedents: # TODO : handle stride, different filter sizes, etc
                # area_index (size_Height_proba_stat, size_Width_proba_stat) : 0 : (1,1), 1: (1,2), ...
                area_Height, area_Width, channel_id = np.unravel_index(antecedent.attribute, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"] + cfg["nb_channels"]))

                start_h = area_Height * STRIDE[0][0]
                start_w = area_Width  * STRIDE[0][1]
                end_h   = start_h + FILTER_SIZE[0][0] - 1
                end_w   = start_w + FILTER_SIZE[0][1] - 1

                if channel_id < cfg["nb_classes"]: #Proba of area
                    class_name = cfg["classes"][channel_id]
                    antecedent.attribute = f"P_class_{class_name}_area_[{start_h}-{end_h}]x[{start_w}-{end_w}]"
                else:
                    channel = channel_id - cfg["nb_classes"] #Pixel in concatenated original rgb image
                    # Conversion of resized coordinates into originals
                    height_original = round(area_Height * scale_h)
                    width_original = round(area_Width * scale_w)
                    antecedent.attribute = f"Pixel_{height_original}x{width_original}x{channel}"
        elif args.statistic in ["probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]:
            split_id = cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"] * cfg["nb_classes"]
            for antecedent in rule_to_print.antecedents:
                if antecedent.attribute < split_id: # proba part
                    area_Height, area_Width, channel_id = np.unravel_index(antecedent.attribute, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"]))

                    start_h = area_Height * STRIDE[0][0]
                    start_w = area_Width  * STRIDE[0][1]
                    end_h   = start_h + FILTER_SIZE[0][0] - 1
                    end_w   = start_w + FILTER_SIZE[0][1] - 1

                    class_name = cfg["classes"][channel_id]
                    antecedent.attribute = f"P_class_{class_name}_area_[{start_h}-{end_h}]x[{start_w}-{end_w}]"
                else: # image part
                    height, width, channel = np.unravel_index(antecedent.attribute - split_id, (cfg["size1D"], cfg["size1D"], cfg["nb_channels"]))
                    antecedent.attribute = f"Pixel_{height}x{width}x{channel}"
        elif args.statistic in ["HOG_and_image", "HOG"]:
            split_id = cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"] * 32
            for antecedent in rule_to_print.antecedents:
                if args.statistic == "HOG" or antecedent.attribute < split_id: # HOG part
                    area_Height, area_Width, channel_id = np.unravel_index(antecedent.attribute, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], 32))

                    start_h = area_Height * STRIDE[0][0]
                    start_w = area_Width  * STRIDE[0][1]
                    end_h   = start_h + FILTER_SIZE[0][0] - 1
                    end_w   = start_w + FILTER_SIZE[0][1] - 1

                    antecedent.attribute = f"Descriptor_vector#{channel_id}_area_[{start_h}-{end_h}]x[{start_w}-{end_w}]"
                else: # image part
                    height, width, channel = np.unravel_index(antecedent.attribute - split_id, (cfg["size1D"], cfg["size1D"], cfg["nb_channels"]))
                    antecedent.attribute = f"Pixel_{height}x{width}x{channel}"
        elif args.statistic == "probability_and_HOG_and_image":
            split_id_proba = cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"] * cfg["nb_classes"]
            split_id_HOG = split_id_proba + cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"] * 32
            for antecedent in rule_to_print.antecedents:
                if antecedent.attribute < split_id_proba: # proba part
                    area_Height, area_Width, channel_id = np.unravel_index(antecedent.attribute, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"]))

                    start_h = area_Height * STRIDE[0][0]
                    start_w = area_Width  * STRIDE[0][1]
                    end_h   = start_h + FILTER_SIZE[0][0] - 1
                    end_w   = start_w + FILTER_SIZE[0][1] - 1

                    class_name = cfg["classes"][channel_id]
                    antecedent.attribute = f"P_class_{class_name}_area_[{start_h}-{end_h}]x[{start_w}-{end_w}]"
                elif antecedent.attribute < split_id_HOG: # HOG part
                    adjusted_attribute = antecedent.attribute - split_id_proba
                    area_Height, area_Width, channel_id = np.unravel_index(adjusted_attribute, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], 32))

                    start_h = area_Height * STRIDE[0][0]
                    start_w = area_Width  * STRIDE[0][1]
                    end_h   = start_h + FILTER_SIZE[0][0] - 1
                    end_w   = start_w + FILTER_SIZE[0][1] - 1

                    antecedent.attribute = f"Descriptor_vector#{channel_id}_area_[{start_h}-{end_h}]x[{start_w}-{end_w}]"
                else: # image part
                    adjusted_attribute = antecedent.attribute - split_id_HOG
                    height, width, channel = np.unravel_index(adjusted_attribute, (cfg["size1D"], cfg["size1D"], cfg["nb_channels"]))
                    antecedent.attribute = f"Pixel_{height}x{width}x{channel}"
        elif args.statistic == "convDimlpFilter":
            if -1 in [height_feature_map, width_feature_map, nb_channels_feature_map]:
                raise ValueError("Missing height, width or nb_channels of feature map")
            for antecedent in rule_to_print.antecedents:
                feature_height, feature_width, feature_channel = np.unravel_index(antecedent.attribute, (height_feature_map, width_feature_map, nb_channels_feature_map))
                antecedent.attribute = f"Feature_map_{feature_height}x{feature_width}x{feature_channel}"

        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as file:
            file.write(str(rule_to_print))

        # We create and save an image for each covered sample
        if not cfg["global_rules_file"].endswith(".json"):
            if data_in_rules is None:
                data_in_rules = X_train
            rule.covered_samples = getCoveredSamples(rule, data_in_rules)[1]
        for img_id in rule.covered_samples[0:10]:
            img = X_train[img_id]
            true_class = cfg['classes'][np.argmax(Y_train[img_id])]
            if args.statistic == "histogram":
                if getattr(args, "train_with_patches", False):
                    nb_areas_per_filter = [nb_patches_per_image]
                    start_idx = img_id * nb_patches_per_image
                    end_idx = start_idx + nb_patches_per_image
                    predictions = train_pred[start_idx:end_idx, :]
                    positions = train_positions[start_idx:end_idx, :]
                else:
                    predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
                    cfg, CNNModel, img, FILTER_SIZE, STRIDE)
                highlighted_image = highlight_area_histograms(CNNModel, img, true_class, FILTER_SIZE, rule, cfg["classes"], predictions, positions, nb_areas_per_filter)
            elif args.statistic == "activation_layer":
                highlighted_image = highlight_area_activations_sum(cfg, CNNModel, intermediate_model, img, true_class, rule, FILTER_SIZE, STRIDE, cfg["classes"])
            elif args.statistic in ["probability", "probability_and_image","probability_multi_nets", "probability_and_HOG_and_image", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one", "HOG_and_image", "HOG"]:
                highlighted_image = highlight_area_probability_image(img, true_class, rule, cfg["size1D"], cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], FILTER_SIZE, cfg["classes"], cfg["nb_channels"], args.statistic)
            elif args.statistic == "convDimlpFilter":
                highlighted_image = highlight_area_first_conv(img, true_class, rule, CNNModel, height_feature_map, width_feature_map, nb_channels_feature_map)

            highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png")
            plt.close(highlighted_image)









def generate_explaining_images_img_version(cfg, X_train, Y_train, CNNModel, intermediate_model, args, train_positions=None, height_feature_map=-1, width_feature_map=-1, nb_channels_feature_map=-1, data_in_rules=None):
    """
    Generate explaining images.
    """
    print("Generation of images...")

    # Get train predictions
    if getattr(args, "train_with_patches", False):
        print("Loading train predictions...")
        train_positions = np.array(train_positions)
        train_pred = np.loadtxt(cfg["train_pred_file"])
        nb_patches_per_image = cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]
        print("Train predictions loaded.")

    # 1) Load rules
    global_rules = getRules(cfg["global_rules_file"])

    # 2) Load attributes
    if args.statistic == "histogram":
        attributes = get_attribute_file(cfg["attributes_file"], cfg["nb_stats_attributes"])[0]

    # 3) Create out folder
    if os.path.exists(cfg["rules_folder"]):
        shutil.rmtree(cfg["rules_folder"])
    os.makedirs(cfg["rules_folder"])


    # 4) For each image we get filter images for rules covered by the train sample
    nb_images = args.images
    counters = [0] * cfg["nb_classes"]

    img_id_shuffled = list(range(len(X_train)))
    random.shuffle(img_id_shuffled)
    for img_id in img_id_shuffled:
        img = X_train[img_id]
        img_class_id = np.argmax(Y_train[img_id])
        true_class = cfg['classes'][img_class_id]
        if args.each_class:
            if all(count >= nb_images for count in counters):
                break
        else:
            if sum(counters) >= nb_images:
                break

        if args.each_class and counters[img_class_id] >= nb_images:
            continue
        else:
            counters[img_class_id] += 1




        # We create and save an image for each rules covering the sample
        if data_in_rules is None:
            data_in_rules = X_train
        covering_rules_list, covering_rules_ids = getCoveringRulesForSample(data_in_rules[img_id], global_rules)
        if len(covering_rules_ids) <= 1 :
            counters[img_class_id] -= 1
            continue

        # Create folder for this image
        rule_folder = os.path.join(cfg["rules_folder"], f"image_{img_id}_class_{true_class}")
        if os.path.exists(rule_folder):
            shutil.rmtree(rule_folder)
        os.makedirs(rule_folder)

        for rule, id_rule in zip(covering_rules_list, covering_rules_ids):
            if args.statistic == "histogram":
                rule.include_X = False
                for ant in rule.antecedents:
                    ant.attribute = attributes[ant.attribute] # Replace the attribute's index by its true name
            elif args.statistic in ["probability", "probability_and_image", "probability_and_HOG_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one", "convDimlpFilter", "HOG_and_image", "HOG"]:
                    rule.include_X = False

            if args.statistic == "histogram":
                if getattr(args, "train_with_patches", False):
                    nb_areas_per_filter = [nb_patches_per_image]
                    start_idx = img_id * nb_patches_per_image
                    end_idx = start_idx + nb_patches_per_image
                    predictions = train_pred[start_idx:end_idx, :]
                    positions = train_positions[start_idx:end_idx, :]
                else:
                    predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
                    cfg, CNNModel, img, FILTER_SIZE, STRIDE)
                highlighted_image = highlight_area_histograms(CNNModel, img, true_class, FILTER_SIZE, rule, cfg["classes"], predictions, positions, nb_areas_per_filter)
            elif args.statistic == "activation_layer":
                highlighted_image = highlight_area_activations_sum(cfg, CNNModel, intermediate_model, img, true_class, rule, FILTER_SIZE, STRIDE, cfg["classes"])
            elif args.statistic in ["probability", "probability_and_image", "probability_and_HOG_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one", "HOG_and_image", "HOG"]:
                highlighted_image = highlight_area_probability_image(img, true_class, rule, cfg["size1D"], cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], FILTER_SIZE, cfg["classes"], cfg["nb_channels"], args.statistic)
            elif args.statistic == "convDimlpFilter":
                highlighted_image = highlight_area_first_conv(img, true_class, rule, CNNModel, height_feature_map, width_feature_map, nb_channels_feature_map)

            highlighted_image.savefig(f"{rule_folder}/rule_{id_rule}.png")
            plt.close(highlighted_image)
