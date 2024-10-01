# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:20:16 2024

@author: jean-marc.boutay
"""

import os
import sys
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from utils import *

np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
fidex.fidex()

###############################################################

start_time = time.time()

nbStairsPerUnit    = 30
#size1D             = 28    # for MNIST images
size1D             = 32    # for Cifar images
nbChannels         = 3     # for Cifar images
#nbChannels         = 1     # for MNIST images

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
deep_rules_file = "deep_fidex_rules.json"
current_rule_file = "current_rule.txt"
final_rules_file = "final_fidex_rules.txt"
deep_test_samples_file = "deep_test_sample.txt"
test_samples_file = "test_sample.txt"
deep_stats_file = "deep_stats_fidex.txt"
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
    f"--rules_outfile {deep_rules_file} "
    f"--nb_attributes {nb_attributes} "
    f"--nb_classes {nb_classes} "
    f"--stats_file {deep_stats_file} "
    f"--nb_quant_levels {6*nbStairsPerUnit} "
    f"--hi_knot {hiknot_deep} "
    f"--dropout_hyp {dropout_hyp} "
    f"--dropout_dim {dropout_dim} "
    "--lowest_min_fidelity 1 "

)
fidex.fidex(command)

# Get Fidex rules
rules = getRules(base_folder + deep_rules_file) # Need to be 100% fidel

# Create new model for each rule
final_rules = []
for sample_id, current_rule in enumerate(rules):
    current_rule = check_minimal_rule(current_rule) # Check if the rule is minimal and modify it accordingly
    current_model = ruleToIMLP(current_rule, nb_attributes)
    current_model.summary()

    # Get predictions for this model
    train_pred = current_model.predict(X_train_deep)    # Predict the response for train dataset
    test_sample = X_test_deep[sample_id].reshape(1, -1)
    test_pred = current_model.predict(test_sample)    # Predict the response for test sample
    output_data(train_pred, base_folder + current_train_pred_file)

    # Write test sample file
    try:
        with open(base_folder + test_samples_file, "w") as myFile:
            myFile.write(' '.join(map(str, test_samples[sample_id])) + " " + ' '.join(map(str, test_pred.flatten())) + "\n")
            myFile.write(' '.join(map(str, test_pred.flatten())))
            myFile.write("\n\n")
    except (FileNotFoundError):
        raise ValueError(f"Error : File {myFile} not found.")
    except (IOError):
        raise ValueError(f"Error : Couldn't open file {myFile}.")

    # Launch Fidex
    command = (
        f"--root_folder {base_folder} "
        f"--train_data_file {train_data_file} "
        f"--train_class_file {current_train_pred_file} "
        f"--train_pred_file {current_train_pred_file} "
        f"--test_data_file {test_samples_file} "
        f"--weights_file {weights_first_layer} "
        f"--rules_outfile {current_rule_file} "
        f"--nb_attributes {X_test.shape[1]} "
        "--nb_classes 2 "
        f"--stats_file {final_stats_file} "
        f"--nb_quant_levels {nbQuantLevels} "
        f"--dropout_hyp {dropout_hyp} "
        f"--dropout_dim {dropout_dim} "
        "--lowest_min_fidelity 1 "
    )
    fidex.fidex(command)
    rule = getRules(base_folder+current_rule_file)[0]
    # Change the class as being the class of the abstract rule
    rule.target_class = current_rule.target_class
    final_rules.append(rule)
with open(base_folder+current_rule_file, 'r') as myFile:
    threshold_str = myFile.readline().strip()

#Save the rules in final rule file
with open(base_folder+final_rules_file, "w") as myFile:
    myFile.write(threshold_str + "\n")
    for n, rule in enumerate(final_rules):
        myFile.write("\n\n")
        myFile.write(f"Rule for sample {n} :")
        myFile.write("\n\n")
        myFile.write(str(rule))
        myFile.write("\n\n")
        myFile.write("-------------------------------------------------")

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
