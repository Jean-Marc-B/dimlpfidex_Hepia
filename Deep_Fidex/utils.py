
import numpy as np
import tensorflow as tf
from stairObj import StairObj
from keras import backend as K
from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Flatten, Input, Convolution2D, DepthwiseConv2D, MaxPooling2D
from keras.layers     import BatchNormalization
from keras.applications     import ResNet50
from keras.optimizers import Adam

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

   return (K.sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin(x):
#    h -> nStairsPerUnit
#    hard_sigmoid(x) =  max(0, min(1, x/6+0.5))
#    staircaseSemiLin(x, h) = hard_sigmoid(ceil(x*h)*(1/h))

   return (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin2(x):

   a = (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))
   a = (a - 0.5)*6.0
   return a

###############################################################

def hardSigm2(x):

   a = tf.keras.activations.hard_sigmoid(x)
   a = (a - 0.5)*6.0
   return a

###############################################################

def staircaseBound(x):

   a = tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit)*0.5 * nbStairsPerUnitInv)
   a = (a - 0.5)*10.0
   return(K.sigmoid(a))

###############################################################

def zeroThreshold(x):
    return tf.cast(tf.greater_equal(x, 0), tf.float32)

###############################################################

def compute_first_hidden_layer(step, input_data, k, nb_stairs, hiknot, weights_outfile=None, mu=None, sigma=None):
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
# Train a CNN with a Resnet or with a small model

def trainCNN(size1D, nbChannels, nb_classes, resnet, nbIt, model_file, model_checkpoint_weights, X_train, Y_train, X_test, Y_test, train_pred_file, test_pred_file, model_stats):
    """
    Trains a Convolutional Neural Network (CNN) using either a ResNet architecture or a small custom model.

    Parameters:
    - size1D: The size of one dimension of the input image (image is size1D x size1D).
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

    if (nbChannels == 1 and resnet):
        # B&W to RGB
        x_train = np.repeat(x_train, 3, axis=-1)
        X_test = np.repeat(X_test, 3, axis=-1)
        x_val = np.repeat(x_val, 3, axis=-1)
        nbChannels = 3

    ##############################################################################
    if resnet:
        input_tensor = Input(shape=(size1D, size1D, 3))
        model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)
        model = Sequential()
        model.add(model_base)
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(nb_classes, activation='softmax'))

        model.build((None, size1D, size1D, 3))  # Build the model with the input shape

        model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

    else:
        model = Sequential()

        model.add(Input(shape=(size1D, size1D, nbChannels)))

        model.add(Convolution2D(32, (5, 5), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(DepthwiseConv2D((5, 5), activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation='sigmoid'))
        model.add(Dropout(0.3))

        model.add(Dense(nb_classes, activation='sigmoid'))

        model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

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
    - filter_size: The size of the filter applied to the image.
    - stride: The stride value for moving the filter across the image.
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

def generate_filtered_images_and_predictions(CNNModel, image, filter_size, stride):

    """
    Generates filtered versions of an image by applying a sliding filter and predicts each filtered image using the CNN model.

    Parameters:
    - CNNModel: The CNN model used for predicting the filtered images.
    - image: The input image to be processed (2D grayscale or 3D RGB).
    - filter_size: The size of the filter applied to the image (height, width).
    - stride: The stride value for moving the filter across the image.

    Returns:
    - filtered_images: An array of filtered images created by applying the filter at different positions.
    - predictions: Predictions from the CNN model for each filtered image.
    - positions: A list of (row, column) tuples indicating the top-left position of each filter applied.
    """

    image_size = [image.shape[0], image.shape[1]]
    filtered_images = []
    positions = []  # To keep the position of each filter

    current_pixel = [0, 0]
    while current_pixel[0] + filter_size[0] <= image_size[0]:
        while current_pixel[1] + filter_size[1] <= image_size[1]:
            # Create filtered image
            filtered_img = np.zeros_like(image)
            filtered_img[current_pixel[0]:current_pixel[0]+filter_size[0],
                         current_pixel[1]:current_pixel[1]+filter_size[1]] = \
                image[current_pixel[0]:current_pixel[0]+filter_size[0],
                      current_pixel[1]:current_pixel[1]+filter_size[1]]
            filtered_images.append(filtered_img)
            positions.append((current_pixel[0], current_pixel[1]))

            current_pixel[1] += stride  # Move horizontally

        current_pixel[1] = 0  # reset the column
        current_pixel[0] += stride  # Move vertically

    filtered_images = np.array(filtered_images)

    # Get predictions
    predictions = CNNModel.predict(filtered_images, verbose=0)

    return filtered_images, predictions, positions

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
    - filter_size: The size of the filter applied to the image (height, width).
    - stride: The stride value for moving the filter across the image.
    - nb_bins: The number of bins used for computing the histogram.

    Returns:
    - histogram_scores: A list of numpy arrays where each array corresponds to a class, containing counts of probabilities
      falling into each bin.
    """

    filtered_images, predictions, _ = generate_filtered_images_and_predictions(
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

def highlight_area(CNNModel, image, filter_size, stride, rule, classes):
    """
    Highlights important areas in an image based on the rule's antecedents using the CNN model.

    Parameters:
    - CNNModel: The trained CNN model used to make predictions on filtered areas of the image.
    - image: The input image (2D grayscale or 3D RGB) to be processed and highlighted.
    - filter_size: A tuple representing the size (height, width) of the filter applied on the image.
    - stride: The stride value used to slide the filter across the image.
    - rule: An object containing antecedents which specify conditions (class ID and prediction threshold)
            for highlighting areas based on CNN predictions.
    - classes: A list or dictionary that maps class IDs to class names for better interpretability.

    Returns:
    - fig: The matplotlib figure containing subplots with:
        1. The original image.
        2. The image with all filters applied based on the rule's antecedents.
        3. Individual images showing the highlighted areas for each antecedent separately.
    """
    filtered_images, predictions, positions = generate_filtered_images_and_predictions(
    CNNModel, image, filter_size, stride)

    # Convert to RGB if necessary
    if len(image.shape) == 2:
        original_image_rgb = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        original_image_rgb = np.squeeze(image, axis=2)  # Remove the supplementary dimension
        original_image_rgb = np.stack((original_image_rgb,) * 3, axis=-1)
    else:
        original_image_rgb = image.copy()

    # Initialize the combined intensity maps for green and red
    combined_green_intensity = np.zeros((image.shape[0], image.shape[1]))
    combined_red_intensity = np.zeros((image.shape[0], image.shape[1]))

    individual_maps = []

    for antecedent in rule.antecedents: # For each antecedent
        # Get the class id and prediction threshold of this antecedent
        match = re.match(HISTOGRAM_ANTECEDENT_PATTERN, antecedent.attribute)
        if match:
            pred_threshold = float(match.group(2))
            class_id = int(match.group(1))
        else:
            raise ValueError(f"Wrong antecedant format : {antecedent.attribute}")

        individual_intensity_map = np.zeros((image.shape[0], image.shape[1]))

        # For each area
        for (prediction, position) in zip(predictions, positions):
            class_prob = prediction[class_id] # Prediction of the area

            # Check if the prediction with this area satisfies the antecedent
            if (class_prob >= pred_threshold):
                top_left = position
                bottom_right = (position[0] + filter_size[0], position[1] + filter_size[1])

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
        if antecedent.inequality:
            ineq = ">="
        else:
            ineq = "<"

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
