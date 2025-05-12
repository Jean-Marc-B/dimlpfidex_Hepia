
import numpy as np
import tensorflow as tf
from .stairObj import StairObj
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Input, Conv2D,
                                     DepthwiseConv2D, MaxPooling2D, LeakyReLU, Resizing,
                                     BatchNormalization, GlobalAveragePooling2D, Concatenate)
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session
from .rule import Rule
from .antecedent import Antecedent
import json
import math
import argparse
from PIL import Image
import os
import time
import re
import matplotlib.pyplot as plt
import gc
from .constants import HISTOGRAM_ANTECEDENT_PATTERN

nbStairsPerUnit    = 30
nbStairsPerUnitInv = 1.0/nbStairsPerUnit


def load_data(cfg):
    print("\nLoading data...")

    # Load train data
    train = np.loadtxt(cfg["train_data_file"])
    print("train data shape : ", train.shape)
    X_train = train.reshape(train.shape[0], cfg["size1D"], cfg["size1D"], cfg["nb_channels"])
    X_train = X_train.astype('int32' if cfg["data_type"] == "integer" else 'float32')

    # Load test data
    test = np.loadtxt(cfg["test_data_file"])
    print("test data shape : ", test.shape)
    X_test = test.reshape(test.shape[0], cfg["size1D"], cfg["size1D"], cfg["nb_channels"])
    X_test = X_test.astype('int32' if cfg["data_type"] == "integer" else 'float32')

    # Load labels
    Y_train = np.loadtxt(cfg["train_class_file"]).astype('int32')
    Y_test = np.loadtxt(cfg["test_class_file"]).astype('int32')

    # Normalize if necessary
    if cfg["data_type"] == "integer":
        X_train = normalize_data(X_train, 0, 255)
        X_test = normalize_data(X_test, 0, 255)

    print("Data loaded.\n")

    return X_train, Y_train, X_test, Y_test

def output_data(data, data_file):
    """
    Outputs a given dataset to a specified file.

    Parameters:
    data (np.ndarray): The dataset to be written, structured as a 2D array.
    data_file (str): Path to the output file where the data will be saved.

    Raises:
    ValueError: If the file is not found or cannot be opened.
    """
    try:
        with open(data_file, "w") as file:
            for var in data:
                for val in var:
                    file.write(str(val) + " ")
                file.write("\n")
            file.close()

    except (FileNotFoundError):
        raise ValueError(f"Error : File {data_file} not found.")
    except (IOError):
        raise ValueError(f"Error : Couldn't open file {data_file}.")

def staircaseUnbound(x):
    """
    Applies an unbounded staircase activation function to the input.

    Parameters:
    x (tf.Tensor): Input tensor to be transformed using the staircase function.

    Returns:
    tf.Tensor: Transformed tensor using the staircase activation.
    """

    return (K.sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin(x):
    """
    Applies a semi-linear staircase activation function to the input.

    Parameters:
    x (tf.Tensor): Input tensor to be transformed using the semi-linear staircase function.

    Returns:
    tf.Tensor: Transformed tensor using the semi-linear staircase activation.
    """
#   h -> nStairsPerUnit
#   hard_sigmoid(x) =  max(0, min(1, x/6+0.5))
#   staircaseSemiLin(x, h) = hard_sigmoid(ceil(x*h)*(1/h))

    return (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin2(x):
    """
    Applies a modified version of the semi-linear staircase activation function to the input.

    Parameters:
    x (tf.Tensor): Input tensor to be transformed.

    Returns:
    tf.Tensor: Transformed tensor adjusted to range between -3 and 3 based on the staircase activation.
    """
    a = (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))
    a = (a - 0.5)*6.0
    return a

###############################################################

def hardSigm2(x):
    """
    Applies a modified version of the hard sigmoid activation function to the input.

    Parameters:
    x (tf.Tensor): Input tensor to be transformed.

    Returns:
    tf.Tensor: Transformed tensor adjusted to range between -3 and 3.
    """
    a = tf.keras.activations.hard_sigmoid(x)
    a = (a - 0.5)*6.0
    return a

###############################################################

def staircaseBound(x):
    """
    Applies a bounded staircase activation function to the input.

    Parameters:
    x (tf.Tensor): Input tensor to be transformed using the bounded staircase function.

    Returns:
    tf.Tensor: Transformed tensor using the bounded staircase activation.
    """
    a = tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit)*0.5 * nbStairsPerUnitInv)
    a = (a - 0.5)*10.0
    return(K.sigmoid(a))

###############################################################

def zeroThreshold(x):
    """
    Applies a zero-threshold activation, setting values greater than or equal to zero to 1, and others to 0.

    Parameters:
    x (tf.Tensor): Input tensor.

    Returns:
    tf.Tensor: Tensor with values set to 1 or 0 based on the threshold.
    """
    return tf.cast(tf.greater_equal(x, 0), tf.float32)

###############################################################

def compute_first_hidden_layer(step, input_data, k, nb_stairs, hiknot, weights_outfile=None, mu=None, sigma=None, activation_fct_stairobj = "sigmoid"):
    """
    Computes the first hidden layer of the model using a staircase activation function.

    Parameters:
    step (str): Specifies whether to compute using "train" or "test" data.
    input_data (np.ndarray): Input data to be transformed.
    k (float): Scaling factor for normalization.
    nb_stairs (int): Number of staircase steps for quantization.
    hiknot (int): Parameter for the staircase function.
    weights_outfile (str, optional): File path to save weights if using training data.
    mu (np.ndarray, optional): Mean values for normalization (training step).
    sigma (np.ndarray, optional): Standard deviation values for normalization (training step).

    Returns:
    np.ndarray: Transformed data after applying the first hidden layer.
    """
    input_data = np.array(input_data)
    if step == "train": # Train datas
        if weights_outfile is None:
            raise ValueError("Error : weights file is None during computation of first hidden layer with train data.")
        mu = np.mean(input_data, axis=0) if mu is None else mu  # mean over variables
        sigma = np.std(input_data, axis=0) if sigma is None else sigma # std over variables
        sigma[sigma == 0] = 0.001  # Prevent division by zero

        weights = k/sigma
        biais = -k*mu/sigma

        # Save weights and bias
        if weights_outfile is not None:
            try:
                print("Saving weights...")
                with open(weights_outfile, "w") as my_file:
                    for b in biais:
                        my_file.write(str(b))
                        my_file.write(" ")
                    my_file.write("\n")
                    for w in weights:
                        my_file.write(str(w))
                        my_file.write(" ")
                    my_file.close()
            except (FileNotFoundError):
                raise ValueError(f"Error : File {weights_outfile} not found.")
            except (IOError):
                raise ValueError(f"Error : Couldn't open file {weights_outfile}.")

    # Compute new data after first hidden layer
    h = k*(input_data-mu)/sigma # With indices : hij=K*(xij-muj)/sigmaj
    stair = StairObj(nb_stairs, hiknot, activation_fct_stairobj)
    out_data = np.vectorize(stair.funct)(h) # Apply staircase activation function

    return (out_data, mu, sigma) if step == "train" else out_data

###############################################################

def apply_Dimlp(x_train, x_test, size1D, nb_channels, K_val, nb_quant_levels, hiknot, output_weights, activation_fct_stairobj="sigmoid"):
    print("Applying DIMLP layer...")
    x_train = x_train.reshape(x_train.shape[0], size1D*size1D*nb_channels)
    x_test = x_test.reshape(x_test.shape[0], size1D*size1D*nb_channels)

    # print("avant")
    # for i in range(0,50):
    #     #if (x_train[i][70]) != 0:
    #     print(x_train[i][70])

    x_train_h1, mu, sigma = compute_first_hidden_layer("train", x_train, K_val, nb_quant_levels, hiknot, output_weights, activation_fct_stairobj=activation_fct_stairobj)
    x_test_h1 = compute_first_hidden_layer("test", x_test, K_val, nb_quant_levels, hiknot, mu=mu, sigma=sigma, activation_fct_stairobj=activation_fct_stairobj)

    # print("apres")
    # for i in range(0,50):
    # #     if (x_train_h1[i][70]) != 0.4750208125210601:
    #     print(x_train_h1[i][70])




    x_train_h1 = x_train_h1.reshape(x_train_h1.shape[0], size1D, size1D, nb_channels)
    x_test_h1 = x_test_h1.reshape(x_test_h1.shape[0], size1D, size1D, nb_channels)

    print(f"Training set: {x_train_h1.shape}")
    print(f"Testing set: {x_test_h1.shape}")

    return x_train_h1, x_test_h1

###############################################################

def check_minimal_rule(rule):
    """
    Checks and returns the minimal version of a given rule by reducing redundant antecedents.

    Parameters:
    rule (Rule): The rule object containing antecedents.

    Returns:
    Rule: The minimal version of the rule with reduced antecedents.
    """
    antecedent_dict = {}

    for antecedent in rule.antecedents:
        attribute = antecedent.attribute

        if attribute not in antecedent_dict:
            antecedent_dict[attribute] = {'ge': None, 'lt': None}

        if antecedent.inequality:  # >=
            if antecedent_dict[attribute]['ge'] is None or antecedent.value >= antecedent_dict[attribute]['ge'].value:
                antecedent_dict[attribute]['ge'] = antecedent
        else:  # Cas <
            if antecedent_dict[attribute]['lt'] is None or antecedent.value < antecedent_dict[attribute]['lt'].value:
                antecedent_dict[attribute]['lt'] = antecedent

    minimal_antecedents = []
    for attribute, antecedents in antecedent_dict.items():
        if antecedents['ge']:
            minimal_antecedents.append(antecedents['ge'])
        if antecedents['lt']:
            minimal_antecedents.append(antecedents['lt'])

    minimal_rule = Rule(minimal_antecedents, rule.target_class, rule.covering_size, rule.fidelity, rule.accuracy, rule.confidence)

    return minimal_rule


###############################################################

def ruleToIMLP(current_rule, nb_attributes):
    """
    Converts a rule into an Interpretable Multi Layer Perceptron (IMLP) model.

    Parameters:
    current_rule (Rule): The rule object containing antecedents.
    nb_attributes (int): The number of attributes in the dataset.

    Returns:
    keras.Sequential: A Sequential model implementing the rule as an IMLP.
    """

    # Take ancient model's first layers, but clone them to avoid shared state
    IMLP = Sequential()

    # Construct first layer
    w_1_layer = np.zeros((nb_attributes, 2*nb_attributes))
    b_1_layer = np.zeros(2*nb_attributes)

    for ant in current_rule.antecedents:
        ant_attribute = ant.attribute
        ant_value = ant.value
        if ant.inequality: # >=
            w_1_layer[ant_attribute, ant_attribute] = 1
            b_1_layer[ant_attribute] = -ant_value
        else: # <
            w_1_layer[ant_attribute, nb_attributes + ant_attribute] = -1
            b_1_layer[nb_attributes + ant_attribute] = ant_value

    IMLP_layer_1 = Dense(2*nb_attributes, activation=zeroThreshold, use_bias=True)
    IMLP_layer_1.build((None, nb_attributes))
    IMLP_layer_1.set_weights([w_1_layer, b_1_layer])

    # Construct second layer
    w_2_layer = np.zeros((2*nb_attributes, 2))
    b_2_layer = np.array([-len(current_rule.antecedents) + 0.5, len(current_rule.antecedents) - 0.5])
    for ant in current_rule.antecedents:
        if ant.inequality:  # >=
            w_2_layer[ant.attribute, 0] = 1 #COVERED
            w_2_layer[ant.attribute, 1] = -1 #NOT COVERED
        else:  # <
            w_2_layer[nb_attributes + ant.attribute, 0] = 1
            w_2_layer[nb_attributes + ant.attribute, 1] = -1

    IMLP_layer_2 = Dense(2, activation=zeroThreshold, use_bias=True)
    IMLP_layer_2.build((None, 2*nb_attributes))
    IMLP_layer_2.set_weights([w_2_layer, b_2_layer])

    # Add the 2 layers to the IMLP model
    IMLP.add(IMLP_layer_1)
    IMLP.add(IMLP_layer_2)

    return IMLP


###############################################################

def getRules(rules_file, with_covered_samples=False, data=None):
    """
    Loads and parses rules from a specified file.

    Parameters:
    rules_file (str): The path to the file containing rules (in JSON or text format).

    Returns:
    list: A list of Rule objects parsed from the file.
    """
    print("Getting rules...")
    rules = []
    if rules_file.endswith(".json"):
        with open(rules_file, "r") as myFile:
            data = json.load(myFile)
            for rule_data in data['rules']:
                antecedents = []

                # Extract antecedents
                for antecedent_data in rule_data['antecedents']:
                    attribute = antecedent_data['attribute']
                    inequality = antecedent_data['inequality']
                    value = antecedent_data['value']
                    antecedents.append(Antecedent(attribute, inequality, value))

                # Create a Rule object and append it to the list
                rules.append(Rule(antecedents, rule_data['outputClass'], rule_data['coveringSize'], rule_data['fidelity'], rule_data['accuracy'], rule_data['confidence'], rule_data['coveredSamples']))

    else:
        with open(rules_file, "r") as myFile:
            line = myFile.readline()
            while line:
                if line.startswith("Rule "):
                    rule_line = line.strip().split(": ")[1]
                    [antecedents_str, class_str] = rule_line.split("->")
                    rule_class = int(class_str.split()[-1])
                    antecedents_str = antecedents_str.split()
                    antecedents = []
                    for antecedent in antecedents_str:
                        if ">=" in antecedent:
                            inequality=True
                            [attribute, value] = antecedent.split(">=")
                            attribute = int(attribute[1:])
                            value = float(value)
                        else:
                            inequality=False
                            [attribute, value] = antecedent.split("<")
                            attribute = int(attribute[1:])
                            value = float(value)
                        antecedents.append(Antecedent(attribute, inequality, value))
                    line = myFile.readline()
                    cov_size = int(line.split(" : ")[1])
                    line = myFile.readline()
                    fidelity = float(line.split(" : ")[1])
                    line = myFile.readline()
                    accuracy = float(line.split(" : ")[1])
                    line = myFile.readline()
                    confidence = float(line.split(" : ")[1])
                    new_rule = Rule(antecedents, rule_class, cov_size, fidelity, accuracy, confidence)
                    if with_covered_samples:
                        new_rule.covered_samples = getCoveredSamples(new_rule, data)[1]

                    rules.append(new_rule)
                line = myFile.readline()
    print("Rules obtained.")
    return rules

###############################################################

def getCoveredSamples(rule, samples):
    """
    Identifies the samples that a given rule covers.

    Parameters:
    rule (Rule): The rule object.
    samples (np.ndarray): Dataset samples to be checked against the rule.

    Returns:
    tuple: A tuple containing the list of covered samples and their respective indices.
    """
    covered_samples = [
        (sample, n)
        for n, sample in enumerate(samples)
        if all(
            (sample[antecedant.attribute] >= antecedant.value if antecedant.inequality
             else sample[antecedant.attribute] < antecedant.value)
            for antecedant in rule.antecedents
        )
    ]

    if covered_samples:
        covered_samples_list, covered_samples_ids = zip(*covered_samples)
        return list(covered_samples_list), list(covered_samples_ids)
    else:
        return [], []


###############################################################

def getCoveringRulesForSample(sample, rules):
    """
    Identifies the rules that a given sample covers.

    Parameters:
    sample (np.ndarray): Sample to analyse.
    rules (list[Rule]): The rules to be checked against the sample

    Returns:
    tuple: A tuple containing the list of rules and their respective indices.
    """

    covering_rules = [
        (rule, n)
        for n, rule in enumerate(rules)
        if all(
            (sample[antecedant.attribute] >= antecedant.value if antecedant.inequality
             else sample[antecedant.attribute] < antecedant.value)
            for antecedant in rule.antecedents
        )
    ]

    if covering_rules:
        covering_rules_list, covering_rules_ids = zip(*covering_rules)
        return list(covering_rules_list), list(covering_rules_ids)
    else:
        return [], []

###############################################################
# For 1D images
def reshape_and_pad(image, nb_attributes):
    """
    Reshapes and pads a 1D image to a 2D format based on the number of attributes.

    Parameters:
    image (np.ndarray): The 1D image array.
    nb_attributes (int): The number of attributes in the image.

    Returns:
    tuple: A tuple containing the reshaped image, its height, and width.
    """
    side_length = math.ceil(math.sqrt(nb_attributes))
    height = side_length
    while side_length * (height - 1) >= nb_attributes:
        height -= 1
    # Add 0 padding
    total_size = height * side_length
    padding_needed = total_size - nb_attributes
    image = np.pad(image, ((0, padding_needed)), mode='constant')
    return image, height, side_length

###############################################################
# Images need to have 1 or 3 channels. If 3 channels, it's flatten. Padding used for 1D images only
def process_rules(rules, X_test, X_train, image_save_folder, nb_channels, classes, with_pad = False, size1D=None, normalize=False, normalized01=False, show_images=False):
    """
    Processes rules and generates images highlighting areas activated by each rule.

    Parameters:
    rules (list): List of Rule objects to be processed.
    X_test (np.ndarray): Test dataset.
    X_train (np.ndarray): Training dataset.
    image_save_folder (str): Folder path to save the generated images.
    nb_channels (int): Number of channels in the images.
    classes (dict): Mapping of class IDs to class names.
    with_pad (bool, optional): If True, applies padding to images.
    size1D (int, optional): Size of the images if they are square.
    normalize (bool, optional): If True, normalizes pixel values between 0 and 255.
    normalized01 (bool, optional): If True, normalizes pixel values between 0 and 1.
    show_images (bool, optional): If True, displays the images.
    """
    if size1D:
        nb_rows = size1D
        nb_cols = size1D
    for id_sample, rule in enumerate(rules):
        print(f"Processing sample {id_sample}")

        # Create a new folder for the current sample
        current_dir = os.path.join(image_save_folder, f"sample_{id_sample}_class_{classes[rule.target_class]}")
        os.makedirs(current_dir)

        # Get the covered samples
        covered_samples, covered_samples_ids = getCoveredSamples(rule, X_train)
        # Process the test image
        baseimage = X_test[id_sample]
        if with_pad:
            nb_attributes = X_test.shape[1]
            baseimage, nb_rows, nb_cols = reshape_and_pad(baseimage, nb_attributes)
        image_path = os.path.join(current_dir, f"_test_img_{id_sample}_out.png")
        get_image(rule, baseimage, image_path, nb_rows, nb_cols, nb_channels, normalize, normalized01=normalized01, show_images=show_images)
        for id in covered_samples_ids:
            baseimage = X_train[id]
            if with_pad:
                baseimage, nb_rows, nb_cols = reshape_and_pad(baseimage, nb_attributes)
            image_path = current_dir + '/_train_img_'+ str(id) + '_out.png'
            get_image(rule, baseimage, image_path, nb_rows, nb_cols, nb_channels, normalize, show_images=show_images)

###############################################################

# Accepts only 1 or 3 channel images
def get_image(rule, baseimage, image_path, nb_rows, nb_cols, nb_channels, normalize=False, normalized01=False, show_images=False):
    """
    Generates and saves an image highlighting the areas activated by a rule.

    Parameters:
    rule (Rule): The rule object containing antecedents.
    baseimage (np.ndarray): The image to process.
    image_path (str): The path where the image will be saved.
    nb_rows (int): The number of rows in the image.
    nb_cols (int): The number of columns in the image.
    nb_channels (int): The number of channels in the image (1 or 3).
    normalize (bool, optional): If True, normalizes the image pixel values between 0 and 255.
    normalized01 (bool, optional): If True, normalizes pixel values between 0 and 1.
    show_images (bool, optional): If True, displays the processed image.
    """

    #normalize values between 0 and 255
    if normalize:
        max_val = np.max(baseimage)
        min_val = np.min(baseimage)
        baseimage = (255 * (baseimage - min_val) / (max_val - min_val)).astype(np.uint8)
    elif normalized01:
        baseimage = (baseimage * 255).astype(np.uint8)

    if nb_channels == 1:
        colorimage = np.stack([baseimage] * 3, axis=-1)
    else:
        colorimage = baseimage

    # Change pixel color when appearing in rule
    for antecedent in rule.antecedents:
        if antecedent.inequality == False:
            if nb_channels == 1:
                colorimage[antecedent.attribute]=[255,0,0]
            else: # In this case, data was flatten at start so there were 3x more attributs
                colorimage[antecedent.attribute - (antecedent.attribute % 3)]=255
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+1]=0
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+2]=0
        else:
            if nb_channels == 1:
                colorimage[antecedent.attribute]=[0,255,0]
            else:
                colorimage[antecedent.attribute - (antecedent.attribute % 3)]=0
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+1]=255
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+2]=0

    # Change image dimension
    colorimage_array = np.array(colorimage).reshape(nb_rows, nb_cols, 3) # Reshape to (size1D, size1D, 3)

    colorimage = Image.fromarray(colorimage_array.astype('uint8'))
    colorimage.save(image_path)

    if show_images:
        colorimage.show()


###############################################################
# Replace relu by leaky-relu in resnet

# def replace_ReLU_with_LeakyReLU(layer):
#     # Check if the layer is an instance of ReLU
#     if isinstance(layer, tf.keras.layers.ReLU):
#         # Create a new LeakyReLU layer with the desired alpha
#         return tf.keras.layers.LeakyReLU(alpha=0.3)
#     # If it's not a ReLU layer, return it as is
#     return layer


# Clone the model and replace ReLU layers with LeakyReLU layers
def clone_and_replace(model):
    # Serialize the model to get its configuration
    config = model.get_config()

    # Recursively replace ReLU activations with LeakyReLU in the config
    def replace_activation(layer_config):
        # Check if the layer is an Activation layer with relu activation
        if layer_config.get('class_name') == 'Activation':
            if layer_config.get('config', {}).get('activation') == 'relu':
                print(f"CHANGEMENT Activation Layer: {layer_config.get('name')}")
                # Replace Activation layer with LeakyReLU layer
                layer_config['class_name'] = 'LeakyReLU'
                # Remove 'activation' from config
                layer_config['config'].pop('activation', None)
                # Add 'alpha' parameter for LeakyReLU
                layer_config['config']['alpha'] = 0.3

        # Check if the layer is a ReLU layer
        elif layer_config.get('class_name') == 'ReLU':
            print(f"CHANGEMENT ReLU Layer: {layer_config.get('name')}")
            layer_config['class_name'] = 'LeakyReLU'
            # Update 'alpha' parameter for LeakyReLU
            layer_config['config']['alpha'] = 0.3

        # Check if the layer has 'activation' parameter set to 'relu'
        elif 'activation' in layer_config.get('config', {}) and layer_config['config']['activation'] == 'relu':
            print(f"CHANGEMENT activation in Layer: {layer_config.get('name')}")
            # Set activation to None
            layer_config['config']['activation'] = None
            # We will need to handle adding a LeakyReLU layer after this layer separately
            # For now, you can leave it as is or modify the model architecture accordingly

        # Recursively process nested configurations
        if 'config' in layer_config:
            for key, value in layer_config['config'].items():
                if isinstance(value, dict):
                    replace_activation(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            replace_activation(item)
        return layer_config

    new_config = replace_activation(config)
    # Reconstruct the model from the modified config
    new_model = tf.keras.Model.from_config(new_config)
    # Load weights from the original model
    new_model.set_weights(model.get_weights())
    return new_model

###############################################################
# Convert an image to RGB if necessary
def image_to_rgb(image):
    if len(image.shape) == 2:
        original_image_rgb = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        original_image_rgb = np.squeeze(image, axis=2)  # Remove the supplementary dimension
        original_image_rgb = np.stack((original_image_rgb,) * 3, axis=-1)
    else:
        original_image_rgb = image.copy()

    return original_image_rgb

#Normalize data to range [0,1]
def normalize_data(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)

    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(data, dtype=np.float32)

    # Normalize between 0 and 1
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.astype(np.float32)

def denormalize_to_255(data, min_val=0, max_val=255):
    """
    Converts normalized data [0,1] back to values in the range [0,255].

    Arguments:
    - data: numpy array containing normalized values (float32 between 0 and 1).
    - min_val: original minimum value (default is 0).
    - max_val: original maximum value (default is 255).

    Returns:
    - Integer (uint8) array with values in the range [0,255].
    """
    # Restore the original scale
    denormalized_data = data * (max_val - min_val) + min_val

    # Clip values and convert to uint8
    return np.clip(denormalized_data, min_val, max_val).astype(np.uint8)

# Convert image to black and white
def image_to_black_and_white(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def gathering_predictions(file_list, output_file):
    """
    Reads a list of class predictions files, extracts the first value(prediction of the sample for this class) of each sample,
    concatenates these values across all files, and saves them to a final output file.
    """
    concatenated_data = []

    # Loop over the files
    for class_id, file_name in enumerate(file_list):
        print(f"Processing {file_name} for class {class_id}...")

        with open(file_name, 'r') as file:
            current_data = np.loadtxt(file, usecols=0)

        # Add to the concatenated list
        concatenated_data.append(current_data)
        del current_data

    # Stack all probs values horizontally
    final_data = np.column_stack(concatenated_data)

    # Normalize each row so that the sum of values across columns equals 1
    row_sums = final_data.sum(axis=1, keepdims=True)  # Compute the sum of each row
    final_data = final_data / row_sums  # Element-wise division for normalization

    # Save to output file
    with open(output_file, 'w') as output:
        np.savetxt(output, final_data, fmt='%.6f')
        print(f"{output_file} written.")

###############################################################
# Train a CNN with a Resnet or with a small model

def trainCNN(height, width, nbChannels, nb_classes, model, nbIt, batch_size, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats, with_leaky_relu=False, remove_first_conv=False):
    """
    Trains a Convolutional Neural Network (CNN) using either a ResNet architecture or a small custom model.

    Parameters:
    - height: The size of the first dimension of the input image (image is height x width).
    - width: The size of the second dimension of the input image (image is height x width).
    - nbChannels: The number of channels in the input images (1 for grayscale, 3 for RGB).
    - nb_classes: The number of classes for classification.
    - model: Can be resnet, VGG, VGG_metadatas, VGG_and_big, big, small, MLP or MLP_Patch indicating the model architecture to use (small is a smaller custom model).
    - nbIt: The number of epochs to train the model.
    - model_file: File path to save the trained model.
    - model_checkpoint_weights: File path for saving the best model weights during training.
    - X_train, Y_train: Training dataset (images and labels).
    - X_test, Y_test: Test dataset (images and labels).
    - train_pred_file: File path to save the training set predictions.
    - test_pred_file: File path to save the test set predictions.
    - model_stats: File path to save the model's performance statistics.

    Returns:
    - None. The trained model is saved to the specified file, and predictions and performance metrics are saved.
    """

    start_time = time.time()

    print("Training CNN...\n")

    # To avoid memory problems on GPU we clear GPU memory before training
    clear_session()
    gc.collect()
    tf.keras.backend.clear_session()

    if model not in ["resnet", "VGG", "VGG_metadatas", "VGG_and_big", "small", "big", "MLP", "MLP_Patch"]:
        raise ValueError("The model needs to be one of resnet, VGG, VGG_metadatas, VGG_and_big, small, big, MLP or MLP_Patch")

    if model == "MLP_Patch":
        if len(X_train) != 2 or len(X_test) != 2 or len(X_train[1][0]) != 2 or len(X_test[1][0]) != 2:
            raise ValueError("Wrong shape of data when training with patches.")
        if not isinstance(X_train, tuple) or not isinstance(X_test, tuple):
            raise ValueError("X_train and X_test must be tuples (image patches, positions).")

    if model == "VGG_metadatas":
        if len(X_train) != 2 or len(X_test) != 2:
            raise ValueError("Wrong shape of data when training VGG with metadatas.")
        if not isinstance(X_train, tuple) or not isinstance(X_test, tuple):
            raise ValueError("X_train and X_test must be tuples (images, metadatas).")

    if model == "VGG_and_big":
        if len(X_train) != 2 or len(X_test) != 2 or len(height) != 2 or len(width) != 2 or len(nbChannels) != 2:
            raise ValueError("Wrong shape of data when training VGG with metadatas.")
        if not isinstance(X_train, tuple) or not isinstance(X_test, tuple) or not isinstance(nbChannels, tuple):
            raise ValueError("X_train, X_test, height, width and nb_channels must be tuples (image, probas).")

    if model in ["MLP_Patch", "VGG_metadatas", "VGG_and_big"]: # meta = probas for VGG_and_big
        X_train, meta_train = X_train
        X_test, meta_test = X_test
        meta_train = np.array(meta_train)
        meta_test = np.array(meta_test)
        X_test = np.array(X_test)

    split_index = int(0.8 * len(X_train))
    x_train = X_train[0:split_index]
    x_val   = X_train[split_index:]
    y_train = Y_train[0:split_index]
    y_val   = Y_train[split_index:]

    if model in ["MLP_Patch", "VGG_metadatas", "VGG_and_big"]:
        meta_train, meta_val = meta_train[:split_index], meta_train[split_index:]

    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {Y_test.shape}")

    if model in ["VGG_metadatas", "VGG_and_big"]:
        print(f"Training set meta: {meta_train.shape}")
        print(f"Validation set meta: {meta_val.shape}")
        print(f"Test set meta: {meta_test.shape}")
    if model == "VGG_and_big":
        nbChannel_img = nbChannels[0]
    else:
        nbChannel_img = nbChannels
    if (nbChannel_img == 1 and model in ["resnet", "VGG", "VGG_and_big", "VGG_metadatas"]):
        # B&W to RGB
        x_train = np.repeat(x_train, 3, axis=-1)
        X_test = np.repeat(X_test, 3, axis=-1)
        x_val = np.repeat(x_val, 3, axis=-1)
        if model == "VGG_and_big":
            nbChannels = (3, nbChannels[1])
        else:
            nbChannels=3

    if model in ["MLP_Patch", "VGG_metadatas", "VGG_and_big"]:
        x_train = [x_train, meta_train]
        x_val = [x_val,meta_val]
        X_test = [X_test,meta_test]

    ##############################################################################
    if model == "resnet":

        # Load the ResNet50 model with pretrained weights
        input_tensor = Input(shape=(height, width, 3))
        resized_input = Resizing(224, 224, name='resizing_layer')(input_tensor)
        model_base = ResNet50(include_top=False, weights='imagenet', input_tensor=resized_input)
        # Freeze layers of ResNet
        # for layer in model_base.layers:
        #     layer.trainable = False

        if with_leaky_relu:
            # Name of the last ReLU activation
            last_activation_layer_name = 'conv5_block3_out'

            # Find index of this layer
            last_activation_layer_index = None
            for idx, layer in enumerate(model_base.layers):
                if layer.name == last_activation_layer_name:
                    last_activation_layer_index = idx
                    break

            if last_activation_layer_index is None:
                print(f"Layer {last_activation_layer_name} not found in the model, cannot use LeakyRelu.")
            else:
                # Get output of previous layer
                last_conv_layer_output = model_base.layers[last_activation_layer_index - 1].output

                # Apply LeakyReLU instead of ReLU
                x = LeakyReLU(alpha=0.3, name='conv5_block3_out')(last_conv_layer_output)
        else:
            # If not using LeakyReLU, use the default output of the base model
            x = model_base.output


        # x = Flatten()(x)
        # x = Dropout(0.5)(x)
        # x = BatchNormalization()(x)

        x = GlobalAveragePooling2D(name="flatten")(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        outputs = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=model_base.input, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.00001),
                    loss='categorical_crossentropy',
                    metrics=['acc'])

        model.summary()

    # if resnet:
    #     input_tensor = Input(shape=(height, width, 3))
    #     model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)
    #     model = Sequential()
    #     model.add(model_base)
    #     model.add(Flatten())
    #     model.add(Dropout(0.5))
    #     model.add(BatchNormalization())
    #     model.add(Dense(nb_classes, activation='softmax'))

    #     model.build((None, height, width, 3))  # Build the model with the input shape

    #     model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['acc'])
    #     model.summary()

    elif model == "VGG":
        if with_leaky_relu:
            raise ValueError("VGG with leakyRelu is not yet implemented.")

        # if height < 32 or width < 32:
        #     input_tensor = Input(shape=(32, 32, 3))
        # else:
        input_tensor = Input(shape=(height, width, 3))

        resized_input = Resizing(224, 224, name='resizing_layer')(input_tensor)

        # charge pre-trained model vgg with imageNet weights
        model_base = VGG16(include_top=False, weights="imagenet", input_tensor=resized_input)

        # Freeze layers of VGG
        # for layer in model_base.layers:
        #     layer.trainable = False

        model = Sequential()
        #if height < 32 or width < 32:
        #    model.add(Resizing(32, 32))
        model.add(model_base)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(nb_classes, activation='softmax'))

        model.build((None, height, width, 3))  # Build the model with the input shape

        model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

    elif model == "VGG_metadatas":
        if with_leaky_relu:
            raise ValueError("VGG with leakyRelu is not yet implemented.")

        # IMAGE BRANCH
        image_input = Input(shape=(height, width, 3))

        resized_image_input = Resizing(224, 224, name='resizing_layer')(image_input)

        # charge pre-trained model vgg with imageNet weights
        model_base = VGG16(include_top=False, weights="imagenet", input_tensor=resized_image_input)

        x = model_base.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        # METADATA BRANCH
        metadatas_input = Input(shape=(meta_train.shape[1],))
        y = Dense(64, activation='relu')(metadatas_input)
        y = BatchNormalization()(y)
        y = Dense(64, activation='relu')(y)

        # MERGING BRANCHES
        merged = Concatenate()([x, y])
        merged = Dense(256, activation='relu')(merged)
        merged = Dense(128, activation='relu')(merged)
        merged = Dense(64, activation='relu')(merged)
        output = Dense(nb_classes, activation='softmax')(merged)

        # FINAL MODEL
        model = Model(inputs=[image_input, metadatas_input], outputs=output)

        # COMPILATION
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
        model.summary()

    elif model == "VGG_and_big": # A VGG for images and a big for probabilities, reuniting at the end
        if with_leaky_relu:
            raise ValueError("VGG with leakyRelu is not yet implemented.")

        # IMAGE BRANCH
        image_input = Input(shape=(height[0], width[0], nbChannels[0]))

        resized_image_input = Resizing(224, 224, name='resizing_layer')(image_input)

        # charge pre-trained model vgg with imageNet weights
        model_base = VGG16(include_top=False, weights="imagenet", input_tensor=resized_image_input)

        x = model_base.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        # PROBABILITY BRANCH

        probability_input = Input(shape=(height[1], width[1], nbChannels[1]))
        y = Resizing(2*height[1], 2*width[1])(probability_input)

        # Première couche de convolution
        y = Conv2D(64, (3, 3), activation=None, padding='same', strides=2, kernel_regularizer=l2(0.0005))(y)
        y = BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
        y = MaxPooling2D(pool_size=(2, 2), name='first_conv_end')(y)

        # Deuxième couche de convolution
        y = Conv2D(128, (3, 3), activation=None, padding='same', kernel_regularizer=l2(0.0005))(y)
        y = BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
        y = MaxPooling2D(pool_size=(2, 2))(y)

        # Troisième couche de convolution
        y = Conv2D(256, (3, 3), activation=None, padding='same', kernel_regularizer=l2(0.0005))(y)
        y = BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
        y = MaxPooling2D(pool_size=(2, 2))(y)

        # Passage à une couche dense
        y = Flatten()(y)
        y = Dense(256, activation=None)(y)
        y = BatchNormalization()(y)
        y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.4)(y)

        # MERGING BRANCHES
        merged = Concatenate()([x, y])
        merged = Dense(256, activation='relu')(merged)
        merged = Dense(128, activation='relu')(merged)
        merged = Dense(64, activation='relu')(merged)
        output = Dense(nb_classes, activation='softmax')(merged)

        # FINAL MODEL
        model = Model(inputs=[image_input, probability_input], outputs=output)

        # COMPILATION
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
        model.summary()

    elif model == "small":
        model = Sequential()
        model.add(Input(shape=(height, width, nbChannels)))

        if not remove_first_conv:
            model.add(Conv2D(32, (5, 5), activation='relu'))
            model.add(Dropout(0.3))
            model.add(MaxPooling2D(pool_size=(2, 2), name='first_conv_end'))

        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # if with_leaky_relu:
        #     model.add(DepthwiseConv2D((5, 5), activation='leaky_relu'))
        # else:
        #     model.add(DepthwiseConv2D((5, 5), activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(nb_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif model == "big":
        model = Sequential()
        model.add(Input(shape=(height, width, nbChannels)))

        if not remove_first_conv:
            model.add(Resizing(2*height, 2*width))

            # Première couche de convolution
            model.add(Conv2D(64, (3, 3), activation=None, padding='same', strides=2, kernel_regularizer=l2(0.0005)))
            model.add(BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), name='first_conv_end'))

        # Deuxième couche de convolution
        model.add(Conv2D(128, (3, 3), activation=None, padding='same', kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Troisième couche de convolution
        model.add(Conv2D(256, (3, 3), activation=None, padding='same', kernel_regularizer=l2(0.0005)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Passage à une couche dense
        model.add(Flatten())
        model.add(Dense(512, activation=None))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(Dropout(0.4))

        # Couche de sortie
        model.add(Dense(nb_classes, activation='softmax'))

        model.summary()

        # Compilation du modèle
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


    elif model == "MLP":
        model = Sequential()
        model.add(Input(shape=(height, width, nbChannels)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

    elif model == "MLP_Patch":
        patch_input = Input(shape=(height, width, nbChannels))

        x = Flatten()(patch_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        coord_input = Input(shape=(2,))
        y = Dense(8, activation='relu')(coord_input)

        merged = Concatenate()([x, y])
        merged = Dense(64, activation='relu')(merged)
        output = Dense(nb_classes, activation='softmax')(merged)

        model = Model(inputs=[patch_input, coord_input], outputs=output)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

    checkpointer = ModelCheckpoint(filepath=model_checkpoint_weights, verbose=1, save_best_only=True, save_weights_only=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nbIt, validation_data=(x_val, y_val), callbacks=[checkpointer], shuffle=True, verbose=2)

    print("\nCNN trained\n")

    end_time = time.time()
    full_time = end_time - start_time
    full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

    print(f"Training time = {full_time} sec\n")

    ##############################################################################

    model.load_weights(model_checkpoint_weights)
    model.save(model_file)

    train_pred = model.predict(x_train)    # Predict the response for train dataset
    test_pred = model.predict(X_test)    # Predict the response for test dataset
    valid_pred = model.predict(x_val)   # Predict the response for validation dataset
    train_valid_pred = np.concatenate((train_pred,valid_pred)) # We output predictions of both validation and training sets

    # Output predictions
    output_data(train_valid_pred, train_pred_file)
    output_data(test_pred, test_pred_file)

    ##############################################################################

    print("\nResult :")

    with open(model_stats, "w") as myFile:
        score = model.evaluate(x_train, y_train)
        print("Train score : ", score)
        myFile.write(f"Train score : {score[1]}\n")

        score = model.evaluate(X_test, Y_test)
        print("Test score : ", score)
        myFile.write(f"Test score : {score[1]}\n")

###############################################################

def generate_filtered_images_and_predictions(cfg, CNNModel, image, filter_size, stride, intermediate_model=None):

    """
    Generates filtered versions of an image by applying a sliding filter and predicts each filtered image using the CNN model.

    Parameters:
    - CNNModel: The CNN model used for predicting the filtered images.
    - image: The input image to be processed (2D grayscale or 3D RGB).
    - filter_size: The size of the filter applied to the image (height, width), can be an array of tuples if we apply several filters.
    - stride: The stride value for moving the filter across the image (verticaly, horizontaly).

    Returns:
    - predictions: Predictions from the CNN model for each filtered image.
    - positions: A list of (row, column) tuples indicating the top-left position of each filter applied.
    """

    if (cfg["model"] != "RF" and image.shape[2] == 1 and CNNModel.input_shape[-1] == 3):
        # B&W to RGB
        image = np.repeat(image, 3, axis=-1)

    image_size = [image.shape[0], image.shape[1]]
    filtered_images = []
    positions = []  # To keep the position of each filter
    nb_areas_per_filter = [] # Keep track of the number of areas for each filter

    for filter_sz, strd in zip(filter_size, stride):
        nb_areas = 0
        current_pixel = [0, 0]
        while current_pixel[0] + filter_sz[0] <= image_size[0]:
            while current_pixel[1] + filter_sz[1] <= image_size[1]:
                # Create filtered image
                filtered_img = np.zeros_like(image)
                filtered_img[current_pixel[0]:current_pixel[0]+filter_sz[0],
                            current_pixel[1]:current_pixel[1]+filter_sz[1]] = \
                    image[current_pixel[0]:current_pixel[0]+filter_sz[0],
                        current_pixel[1]:current_pixel[1]+filter_sz[1]]
                filtered_images.append(filtered_img)
                positions.append((current_pixel[0], current_pixel[1]))
                nb_areas += 1

                current_pixel[1] += strd[1]  # Move horizontally

            current_pixel[1] = 0  # reset the column
            current_pixel[0] += strd[0]  # Move vertically
        nb_areas_per_filter.append(nb_areas)

    filtered_images = np.array(filtered_images)

    # Get predictions
    if intermediate_model is not None:
        predictions = intermediate_model.predict(filtered_images, verbose=0)
        return predictions, positions

    else:
        if cfg["model"] == "RF":
            filtered_images_flattened = filtered_images.reshape(filtered_images.shape[0], -1)
            predictions = CNNModel.predict_proba(filtered_images_flattened)
        else:
            predictions = CNNModel.predict(filtered_images, verbose=0)
        # if sample_id == 56:
        #     index = positions.index((8,8))
        #     prediction_at_target = predictions[index]
        #     print(f"La prédiction pour le sample_id {sample_id} à la position Height=8 et Width=8 et Classe=0 est : {prediction_at_target[0]}")
        return predictions, positions, nb_areas_per_filter

###############################################################
# Generate patches for the dataset
def create_patches(X_train, Y_train, X_test, Y_test, filter_size, stride):
    """
    Generates patches for the dataset by applying a sliding window approach.

    Parameters:
    - X_train, Y_train: Training images and their labels.
    - X_test, Y_test: Testing images and their labels.
    - filter_size: Tuple (height, width) defining the size of the patches.
    - stride: Tuple (vertical_step, horizontal_step) defining the step size.

    Returns:
    - X_train_patches: List of training image patches.
    - Y_train_patches: List of corresponding training labels.
    - X_test_patches: List of testing image patches.
    - Y_test_patches: List of corresponding testing labels.
    - nb_areas: Number of patches per original image.
    """
    def extract_patches(X, Y, filter_size, stride):
        start_time = time.time()
        patches = []
        labels = []
        positions = []
        img_height, img_width = X.shape[1:3]  # For shape (num_samples, height, width, channels)

        # Compute number of patches per image
        nb_areas = ((img_height - filter_size[0]) // stride[0] + 1) * ((img_width - filter_size[1]) // stride[1] + 1)

        for img, label in zip(X, Y):
            current_pixel = [0, 0]
            while current_pixel[0] + filter_size[0] <= img_height:
                while current_pixel[1] + filter_size[1] <= img_width:
                    # Extract patch
                    patch = img[current_pixel[0]:current_pixel[0] + filter_size[0],
                                current_pixel[1]:current_pixel[1] + filter_size[1]]
                    patches.append(patch)
                    labels.append(label)  # Assign the same label as the original image
                    positions.append((current_pixel[0], current_pixel[1]))

                    current_pixel[1] += stride[1]  # Move horizontally

                current_pixel[1] = 0  # Reset column position
                current_pixel[0] += stride[0]  # Move vertically

        end_time = time.time()
        print(f"Extracting executed in {end_time - start_time:.2f} seconds")
        return np.array(patches), np.array(labels), positions, nb_areas

    print("Extracting train patches...")
    X_train_patches, Y_train_patches, train_positions, nb_areas = extract_patches(X_train, Y_train, filter_size, stride)
    print("Extracting test patches...")
    X_test_patches, Y_test_patches, test_positions, _ = extract_patches(X_test, Y_test, filter_size, stride)  # nb_areas is the same for all images

    return X_train_patches, Y_train_patches, train_positions, X_test_patches, Y_test_patches, test_positions, nb_areas

###############################################################
# get probability thresholds
def getProbabilityThresholds(nb_bins):
    """
    Computes the probability thresholds for binning probabilities into a histogram.

    Parameters:
    - nb_bins: The number of bins used for the histogram.

    Returns:
    - A list of probability thresholds, evenly spaced between 0 and 1.
    """

    return [(1 / (nb_bins + 1)) * i for i in range(1, nb_bins + 1)]


###############################################################

# get histogram for an image on a CNN
def getHistogram(CNNModel, predictions, nb_classes, filter_size, stride, nb_bins):
    """
    Computes a histogram for a given image based on CNN model predictions for different areas of the image.

    Parameters:
    - CNNModel: The CNN model used for predicting filtered images.
    - image: The input image to be processed.
    - nb_classes: The number of classes for classification.
    - filter_size: The size of the filter applied to the image (height, width), can be an array of tuples if we apply several filters.
    - stride: The stride value for moving the filter across the image (verticaly, horizontaly).
    - nb_bins: The number of bins used for computing the histogram.

    Returns:
    - histogram_scores: A list of numpy arrays where each array corresponds to a class, containing counts of probabilities
      falling into each bin.
    """

    probability_thresholds = getProbabilityThresholds(nb_bins)
    histogram_scores = [np.zeros(nb_bins, dtype=int) for _ in range(nb_classes)]

    # Evaluate images
    for prediction in predictions:
        # Loop on classes
        for class_idx in range(nb_classes):
            class_prob = prediction[class_idx]  # Probability of class
            for prob_idx, threshold in enumerate(probability_thresholds):
                if class_prob >= threshold:
                    histogram_scores[class_idx][prob_idx] += 1 # +1 if probability score is high enough


    return histogram_scores


###############################################################

def get_top_ids(array, X=10, largest=True):
    """
    Returns the indices of the X largest or smallest values in an array (not sorted).

    Parameters:
    - array (np.ndarray): One-dimensional array of values.
    - X (int): Number of indices to return.
    - largest (bool): If True, returns the indices of the largest values; otherwise, the smallest.

    Returns:
    - np.ndarray: Indices of the X largest or smallest values.
    """

    if largest:
        return np.argpartition(array, -X)[-X:] # Get X greater indices (not sorted)
    else:
        return np.argpartition(array, X)[:X] # Get X smaller indices (not sorted)

###############################################################

def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number. Enter an positive integer (≥ 0).")
    return ivalue

###############################################################
###############################################################
