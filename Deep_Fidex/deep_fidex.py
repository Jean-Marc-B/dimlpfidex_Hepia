# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:20:16 2024

@author: jean-marc.boutay
"""

import os
import sys
import time
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from antecedent import Antecedent
from rule import Rule
from keras.models import load_model
from utils import *

np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
fidex.fidex()

###############################################################

start_time = time.time()

nbStairsPerUnit    = 30
size1D             = 28    # for MNIST images
#size1D             = 32    # for Cifar images
#nbChannels         = 3     # for Cifar images
nbChannels         = 1     # for MNIST images

nb_classes = 10 # for MNIST and Cifar images
hiknot = 5
hiknot_deep = 3
nbQuantLevels = 100
K_val = 1.0
dropout_hyp = 0.9
dropout_dim = 0.9

#base_folder = "Cifar/" # for Cifar images
base_folder = "Mnist/" # for Mnist images
train_data_file = "trainData.txt"
train_class_file = "trainClass.txt"
test_data_file = "testData.txt"
test_class_file = "testClass.txt"
weights_first_layer = "weights_first_layer.wts"
weights_deep_fidex_outfile = "weights_deep_fidex.wts"
deep_fidex_train_inputs = "deep_fidex_train_inputs.txt"
deep_fidex_test_inputs = "deep_fidex_test_inputs.txt"
train_pred_file = "train_pred.out"
test_pred_file = "test_pred.out"
rules_file = "fidex_rules.json"
final_rules_file = "final_fidex_rules.txt"
deep_test_samples_file = "deep_test_sample.txt"
test_samples_file = "test_sample.txt"
stats_file = "stats_fidex.txt"
final_stats_file = "final_stats_fidex.txt"
current_train_pred_file = "current_train_pred.out"
current_test_pred_file = "current_test_pred.out"

###############################################################
###############################################################

print("Loading data...")

train_deep   = np.loadtxt(base_folder + deep_fidex_train_inputs)

X_train_deep = train_deep.astype('float32')
print(X_train_deep.shape)

test_deep   = np.loadtxt(base_folder + deep_fidex_test_inputs)

X_test_deep = test_deep.astype('float32')
print(X_test_deep.shape)

Y_train = np.loadtxt(base_folder + train_class_file)
Y_train = Y_train.astype('int32')

Y_test  = np.loadtxt(base_folder + test_class_file)
Y_test  = Y_test.astype('int32')

test_pred = np.loadtxt(base_folder + test_pred_file)
test_pred = test_pred.astype('float32')


test   = np.loadtxt(base_folder + test_data_file)

X_test = test.astype('float32')
print(X_test.shape)

##############################################################################

nb_test_samples = 5
test_samples_deep = X_test_deep[0:nb_test_samples]
test_samples_class = Y_test[0:nb_test_samples]
test_samples_pred = test_pred[0:nb_test_samples]

test_samples = X_test[0:nb_test_samples]

try:
    with open(base_folder + deep_test_samples_file, "w") as myFile:
        for i in range(0,nb_test_samples):
            myFile.write(' '.join(map(str, test_samples_deep[i])) + " " + ' '.join(map(str, test_samples_class[i])) + "\n")
            myFile.write(' '.join(map(str,test_samples_pred[i])))
            myFile.write("\n\n")

except (FileNotFoundError):
    raise ValueError(f"Error : File {myFile} not found.")
except (IOError):
    raise ValueError(f"Error : Couldn't open file {myFile}.")

# Launch Fidex on deep layer for the test sample
nb_attributes = X_test_deep.shape[1]
#fidex.fidex()
command = (
    f"--root_folder {base_folder} "
    f"--train_data_file {deep_fidex_train_inputs} "
    f"--train_class_file {train_class_file} "
    f"--train_pred_file {train_pred_file} "
    f"--test_data_file {deep_test_samples_file} "
    f"--weights_file {weights_deep_fidex_outfile} "
    f"--rules_outfile {rules_file} "
    f"--nb_attributes {nb_attributes} "
    f"--nb_classes {nb_classes} "
    f"--stats_file {stats_file} "
    f"--nb_quant_levels {6*nbStairsPerUnit} "
    f"--hi_knot {hiknot_deep} "
    f"--dropout_hyp {dropout_hyp} "
    f"--dropout_dim {dropout_dim} "

)
fidex.fidex(command)

# Get Fidex rules
rules = [] # Need to be 100% fidel


if rules_file.endswith(".json"):
    with open(base_folder + rules_file, "r") as myFile:
        data = json.load(myFile)
        for rule_data in data['rules']:
            antecedents = []

            # Extract antecedents
            for antecedent_data in rule_data['antecedents']:
                attribute = antecedent_data['attribute']
                inequality = antecedent_data['inequality']
                value = antecedent_data['value']
                antecedents.append(Antecedent(attribute, inequality, value))

            # Extract the rule class (outputClass in your case)
            rule_class = rule_data['outputClass']

            # Create a Rule object and append it to the list
            rules.append(Rule(antecedents, rule_class))

else:
    with open(base_folder + rules_file, "r") as myFile:
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
                rules.append(Rule(antecedents, rule_class))
            line = myFile.readline()

# Create new model for each rule
for current_rule in rules:
    print(current_rule)
    current_rule = check_minimal_rule(current_rule)# Check if the rule is minimal and modify it accordingly
    current_model = ruleToIMLP(current_rule, nb_attributes)
    current_model.summary()

    # Get predictions for this model
    train_pred = current_model.predict(X_train_deep)    # Predict the response for train dataset
    test_pred = current_model.predict(X_test_deep)    # Predict the response for test dataset
    output_data(train_pred, base_folder + current_train_pred_file)
    output_data(test_pred, base_folder + current_test_pred_file)

    # Write test sample file
    try:
        with open(base_folder + test_samples_file, "w") as myFile:
            for i in range(0,nb_test_samples):
                myFile.write(' '.join(map(str, test_samples[i])) + " " + ' '.join(map(str, test_samples_class[i])) + "\n")
                myFile.write(' '.join(map(str,current_test_pred_file[i])))
                myFile.write("\n\n")
    except (FileNotFoundError):
        raise ValueError(f"Error : File {myFile} not found.")
    except (IOError):
        raise ValueError(f"Error : Couldn't open file {myFile}.")

    # Launch Fidex
    command = (
        f"--root_folder {base_folder} "
        f"--train_data_file {train_data_file} "
        f"--train_class_file {train_class_file} "
        f"--train_pred_file {current_train_pred_file} "
        f"--test_data_file {test_samples_file} "
        f"--weights_file {weights_first_layer} "
        f"--rules_outfile {final_rules_file} "
        f"--nb_attributes {X_test.shape[1]} "
        f"--nb_classes {nb_classes} "
        f"--stats_file {final_stats_file} "
        f"--nb_quant_levels {nbQuantLevels} "
        f"--dropout_hyp {dropout_hyp} "
        f"--dropout_dim {dropout_dim} "
    )
    fidex.fidex(command)
    # Les classes : que 2 classes : cover ou non
    # Récup la règle, modifier sa classe comme étant la classe de la règle abstraite

# les classes ??? 10 ou 2?

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
