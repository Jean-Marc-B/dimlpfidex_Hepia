
import numpy as np
import tensorflow as tf
from stairObj import StairObj
from keras import backend as K
from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Flatten, Input, Convolution2D, DepthwiseConv2D, MaxPooling2D, LeakyReLU
from keras.layers     import BatchNormalization
from keras.applications     import ResNet50
from keras.optimizers import Adam
from tensorflow.keras.models import Model

from keras.callbacks  import ModelCheckpoint
from rule import Rule
from antecedent import Antecedent
import json
import math
from PIL import Image
import os
import time
import re
import matplotlib.pyplot as plt
from constants import HISTOGRAM_ANTECEDENT_PATTERN

nbStairsPerUnit    = 30
nbStairsPerUnitInv = 1.0/nbStairsPerUnit


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

def compute_first_hidden_layer(step, input_data, k, nb_stairs, hiknot, weights_outfile=None, mu=None, sigma=None):
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
    stair = StairObj(nb_stairs, hiknot)
    out_data = np.vectorize(stair.funct)(h) # Apply staircase activation function

    return (out_data, mu, sigma) if step == "train" else out_data

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

def getRules(rules_file):
    """
    Loads and parses rules from a specified file.

    Parameters:
    rules_file (str): The path to the file containing rules (in JSON or text format).

    Returns:
    list: A list of Rule objects parsed from the file.
    """
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
                if line.startswith("Rule for sample"):

                    myFile.readline()
                    rule_line = myFile.readline().strip()
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
                    rules.append(Rule(antecedents, rule_class, cov_size, fidelity, accuracy, confidence))
                line = myFile.readline()
    return rules

###############################################################

def getCovering(rule, samples):
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
        covered_samples, covered_samples_ids = getCovering(rule, X_train)
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

###############################################################
# Train a CNN with a Resnet or with a small model

def trainCNN(sizeX, sizeY, nbChannels, nb_classes, resnet, nbIt, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats, with_leaky_relu, with_probability=False):
    """
    Trains a Convolutional Neural Network (CNN) using either a ResNet architecture or a small custom model.

    Parameters:
    - sizeX: The size of the first dimension of the input image (image is sizeX x sizeY).
    - sizeY: The size of the second dimension of the input image (image is sizeX x sizeY).
    - nbChannels: The number of channels in the input images (1 for grayscale, 3 for RGB).
    - nb_classes: The number of classes for classification.
    - resnet: Boolean indicating whether to use a ResNet architecture (True) or a smaller custom model (False).
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

    split_index = int(0.8 * len(X_train))
    x_train = X_train[0:split_index]
    x_val   = X_train[split_index:]
    y_train = Y_train[0:split_index]
    y_val   = Y_train[split_index:]

    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {Y_test.shape}")

    if (nbChannels == 1 and resnet and not with_probability):
        # B&W to RGB
        x_train = np.repeat(x_train, 3, axis=-1)
        X_test = np.repeat(X_test, 3, axis=-1)
        x_val = np.repeat(x_val, 3, axis=-1)
        nbChannels = 3

    ##############################################################################
    if resnet:

        # Load the ResNet50 model with pretrained weights
        input_tensor = Input(shape=(sizeX, sizeY, 3))
        model_base = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

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

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        outputs = Dense(nb_classes, activation='softmax')(x)

        model = Model(inputs=model_base.input, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.00001),
                    loss='categorical_crossentropy',
                    metrics=['acc'])

        model.summary()

    # if resnet:
    #     input_tensor = Input(shape=(sizeX, sizeY, 3))
    #     model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)
    #     model = Sequential()
    #     model.add(model_base)
    #     model.add(Flatten())
    #     model.add(Dropout(0.5))
    #     model.add(BatchNormalization())
    #     model.add(Dense(nb_classes, activation='softmax'))

    #     model.build((None, sizeX, sizeY, 3))  # Build the model with the input shape

    #     model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['acc'])
    #     model.summary()

    else:
        model = Sequential()

        model.add(Input(shape=(sizeX, sizeY, nbChannels)))

        model.add(Convolution2D(32, (5, 5), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if with_leaky_relu:
            model.add(DepthwiseConv2D((5, 5), activation='leaky_relu'))
        else:
            model.add(DepthwiseConv2D((5, 5), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(nb_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=model_checkpoint_weights, verbose=1, save_best_only=True, save_weights_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=nbIt, validation_data=(x_val, y_val), callbacks=[checkpointer], verbose=2)

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

def compute_histograms(nb_samples, data, size1D, nb_channels, CNNModel, nb_classes, filter_size, stride, nb_bins):
    """
    Computes histograms for each sample in the dataset using the CNN model. It's the histogram of the probabilities of each class on the CNN
    evaluated on each area (or patches) added on the image (by a sliding filter). A patch is applied and outside of this area everything is 0.

    Parameters:
    - nb_samples: The number of samples in the dataset.
    - data: The dataset containing images to be processed.
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
    for sample_id in range(nb_samples):
        image = data[sample_id]
        image = image.reshape(size1D, size1D, nb_channels)
        histogram = getHistogram(CNNModel, image, nb_classes, filter_size, stride, nb_bins)
        histograms.append(histogram)
        if (sample_id+1) % 100 == 0 or sample_id+1 == nb_samples:
            progress = ((sample_id+1) / nb_samples) * 100
            progress_message = f"Progress : {progress:.2f}% ({sample_id+1}/{nb_samples})"
            print(f"\r{progress_message}", end='', flush=True)
    histograms = np.array(histograms)

    return histograms

###############################################################

def compute_activation_sums(nb_samples, data, size1D, nb_channels, CNNModel, intermediate_model, nb_stats_attributes, filter_size, stride):
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
        CNNModel, image, filter_size, stride, intermediate_model)
        sums[sample_id] = np.sum(activations, axis=0)
        if (sample_id+1) % 100 == 0 or sample_id+1 == nb_samples:
            progress = ((sample_id+1) / nb_samples) * 100
            progress_message = f"Progress : {progress:.2f}% ({sample_id+1}/{nb_samples})"
            print(f"\r{progress_message}", end='', flush=True)

    return sums

def compute_proba_images(nb_samples, data, size1D, nb_channels, nb_classes, CNNModel, filter_size, stride):
    my_filter_size = filter_size[0]
    output_size = ((size1D - my_filter_size[0] + 1)*(size1D - my_filter_size[1] + 1)*nb_classes)
    #print(output_size) # (4840)
    proba_images = np.zeros((nb_samples,output_size))

    for sample_id in range(nb_samples):
        image = data[sample_id]
        image = image.reshape(size1D, size1D, nb_channels)
        predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
            CNNModel, image, filter_size, stride)
        #print(predictions.shape)  # (484, 10)
        predictions = predictions.reshape(output_size)
        # print(predictions.shape) # (4840,)
        proba_images[sample_id] = predictions
    #print(proba_images.shape)  # (nb_samples, 4840)
    return proba_images

def generate_filtered_images_and_predictions(CNNModel, image, filter_size, stride, intermediate_model=None):

    """
    Generates filtered versions of an image by applying a sliding filter and predicts each filtered image using the CNN model.

    Parameters:
    - CNNModel: The CNN model used for predicting the filtered images.
    - image: The input image to be processed (2D grayscale or 3D RGB).
    - filter_size: The size of the filter applied to the image (height, width), can be an array of tuples if we apply several filters.
    - stride: The stride value for moving the filter across the image (verticaly, horizontaly).

    Returns:
    - filtered_images: An array of filtered images created by applying the filter at different positions.
    - predictions: Predictions from the CNN model for each filtered image.
    - positions: A list of (row, column) tuples indicating the top-left position of each filter applied.
    """
    if (image.shape[2] == 1 and CNNModel.input_shape[-1] == 3):
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
        predictions = CNNModel.predict(filtered_images, verbose=0)
        return predictions, positions, nb_areas_per_filter

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
def getHistogram(CNNModel, image, nb_classes, filter_size, stride, nb_bins):
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

    predictions, _, _ = generate_filtered_images_and_predictions(
        CNNModel, image, filter_size, stride)

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

def highlight_area_histograms(CNNModel, image, filter_size, stride, rule, classes):
    """
    Highlights important areas in an image based on the rule's antecedents using the CNN model.

    Parameters:
    - CNNModel: The trained CNN model used to make predictions on filtered areas of the image.
    - image: The input image (2D grayscale or 3D RGB) to be processed and highlighted.
    - filter_size: A tuple or a list of tuples representing the size (height, width) of the filter applied on the image.
    - stride: The stride value used to slide the filter across the image (verticaly, horizontaly).
    - rule: An object containing antecedents which specify conditions (class ID and prediction threshold)
            for highlighting areas based on CNN predictions.
    - classes: A list or dictionary that maps class IDs to class names for better interpretability.

    Returns:
    - fig: The matplotlib figure containing subplots with:
        1. The original image.
        2. The image with all filters applied based on the rule's antecedents.
        3. Individual images showing the highlighted areas for each antecedent separately.
    """
    predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
    CNNModel, image, filter_size, stride)

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)

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
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0] + red_overlay, 0, 255)  # Ajout du rouge
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1] + green_overlay, 0, 255)  # Ajout du vert

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
    axes[0].set_title("Original Image")
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
            individual_image[:, :, 1] = np.clip(individual_image[:, :, 1] + normalized_intensity, 0, 255)
        else:
            # Apply red filter
            individual_image[:, :, 0] = np.clip(individual_image[:, :, 0] + normalized_intensity, 0, 255)

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

def highlight_area_activations_sum(CNNModel, intermediate_model, image, rule, filter_size, stride, classes):

    nb_top_filters = 20 # Number of filters to show in an image

    activations, positions = generate_filtered_images_and_predictions( #nb_filters x nb_activations
            CNNModel, image, filter_size, stride, intermediate_model)

    filter_size = filter_size[0]

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)

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
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1] + combined_green_intensity * 255, 0, 255)  # Green channel
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0] + combined_red_intensity * 255, 0, 255)    # Red channel

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
    axes[0].set_title("Original Image")
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
            filtered_image[:, :, 1] = np.clip(filtered_image[:, :, 1] + intensity_map * 255, 0, 255)
        else:
            filtered_image[:, :, 0] = np.clip(filtered_image[:, :, 0] + intensity_map * 255, 0, 255)

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

def highlight_area_probability_image(image, rule, sizeY_proba_stat, filter_size, classes):

    nb_classes = len(classes)

    # Convert to RGB if necessary
    original_image_rgb = image_to_rgb(image)

    filtered_images = []
    combined_image_intensity = np.zeros_like(original_image_rgb, dtype=float)

    for antecedent in rule.antecedents:
        # Get attribute information
        area_number = antecedent.attribute // nb_classes
        area_X = area_number // sizeY_proba_stat
        area_X_end = area_X+filter_size[0][0]-1
        area_Y = area_number % sizeY_proba_stat
        area_Y_end = area_Y+filter_size[0][1]-1
        # print(area_X, "-", area_X_end)
        # print(area_Y, "-", area_Y_end)

        filtered_image_intensity = np.zeros_like(original_image_rgb, dtype=float)
        if antecedent.inequality:  # >=
            filtered_image_intensity[area_X:area_X_end+1, area_Y:area_Y_end+1, 1] += 1
            combined_image_intensity[area_X:area_X_end+1, area_Y:area_Y_end+1, 1] += 1
        else:  # <
            filtered_image_intensity[area_X:area_X_end+1, area_Y:area_Y_end+1, 0] += 1
            combined_image_intensity[area_X:area_X_end+1, area_Y:area_Y_end+1, 0] += 1

        filtered_image_intensity = np.clip(filtered_image_intensity / np.max(filtered_image_intensity) * 255, 0, 255).astype(np.uint8)

        filtered_image = original_image_rgb.copy()
        filtered_image[:, :, 1] = np.clip(filtered_image[:, :, 1] + filtered_image_intensity[:, :, 1], 0, 255)  # Green channel
        filtered_image[:, :, 0] = np.clip(filtered_image[:, :, 0] + filtered_image_intensity[:, :, 0], 0, 255)    # Red channel

        filtered_images.append(filtered_image)
    # Normalise combined image between 0 and 255
    combined_image_intensity = np.clip(combined_image_intensity / np.max(combined_image_intensity) * 255, 0, 255).astype(np.uint8)

    combined_image = original_image_rgb.copy()
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1] + combined_image_intensity[:, :, 1], 0, 255)  # Green channel
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0] + combined_image_intensity[:, :, 0], 0, 255)    # Red channel


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
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Show combined image
    axes[1].imshow(combined_image.astype(np.uint8))
    axes[1].set_title("Combined Filters")
    axes[1].axis('off')

    # Show each antecedent image
    for i,img in enumerate(filtered_images):
        antecedent = rule.antecedents[i]
        area_number = antecedent.attribute // nb_classes
        area_X = area_number // sizeY_proba_stat
        area_X_end = area_X+filter_size[0][0]-1
        area_Y = area_number % sizeY_proba_stat
        area_Y_end = area_Y+filter_size[0][1]-1
        class_name = classes[antecedent.attribute % nb_classes]
        img = img.astype(np.uint8)
        ineq = ">=" if antecedent.inequality else "<"
        axes[i+2].imshow(img)
        axes[i+2].set_title(f"P_class_{class_name}_area_[{area_X}-{area_X_end}]x[{area_Y}-{area_Y_end}]{ineq}{antecedent.value:.6f}")
        axes[i+2].axis('off')
    # Hide any remaining empty subplots if total_images < num_rows * num_columns
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() # Adjust spacing
    plt.subplots_adjust(top=0.85)  # Let space for the main title

    plt.close(fig)

    return fig

###############################################################

# Only for one filter !
def get_heat_maps(CNNModel, image, filter_size, stride, probability_thresholds, classes):
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
    stride : tuple
        The stride (vertical, horizontal) defining the step size for moving the filter across the image.
    probability_thresholds : list of float
        Thresholds for the probability predictions for each class, used to highlight areas of interest.
    classes : list of str
        A list of class names corresponding to the output classes for each heat map.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure containing the original image and heat maps for each class, organized in a grid layout.
    """
    predictions, positions, nb_areas_per_filter = generate_filtered_images_and_predictions(
    CNNModel, image, filter_size, stride)

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
        heat_map_img[:, :, 0] = np.clip(heat_map_img[:, :, 0] + normalized_intensity, 0, 255)
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

###############################################################
###############################################################

"""
# Toy test example
ant1 = Antecedent(0, True, 0.5)   # X0 >= 0.5
ant2 = Antecedent(1, False, 0.3)  # X1 < 0.3
ant3 = Antecedent(2, True, 0.8)   # X2 >= 0.8
ant4 = Antecedent(3, False, 0.1)  # X3 < 0.1
ant5 = Antecedent(4, True, 0.9)   # X4 >= 0.9
ant6 = Antecedent(4, False, 1)   # X4 >= 0.9
current_rule = Rule([ant1, ant2, ant3, ant4, ant5, ant6], 1)
nb_attributes = 5

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


X_0_fail_ge = np.array([[0.4, 0.2, 0.85, 0.05, 0.95]])  # Echoue sur X_0 >= 0.5
X_1_fail_lt = np.array([[0.6, 0.4, 0.85, 0.05, 0.95]])  # Echoue sur X_1 < 0.3
X_2_pass = np.array([[0.6, 0.2, 0.85, 0.05, 0.95]])     # Respecte toutes les règles
X_3_fail = np.array([[0.6, 0.2, 0.85, 0.05, 1.1]])     # Respecte pas toutes les règles

# Prédictions
prediction_0_fail_ge = IMLP.predict(X_0_fail_ge)
prediction_1_fail_lt = IMLP.predict(X_1_fail_lt)
prediction_2_pass = IMLP.predict(X_2_pass)
prediction_3_fail = IMLP.predict(X_3_fail)

print(f"Prediction for X_0_fail_ge: {prediction_0_fail_ge}")  # Devrait donner [0,1]
print(f"Prediction for X_1_fail_lt: {prediction_1_fail_lt}")  # Devrait donner [0,1]
print(f"Prediction for X_2_pass: {prediction_2_pass}")        # Devrait donner [1,0]
print(f"Prediction for X_2_pass: {prediction_3_fail}")        # Devrait donner [0,1]

"""
