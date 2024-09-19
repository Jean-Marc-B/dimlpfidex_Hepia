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
import tensorflow as tf

np.random.seed(seed=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dimlpfidex import fidex
fidex.fidex()

###############################################################

start_time = time.time()

nbStairsPerUnit    = 30
#size1D             = 28    # for MNIST images
size1D             = 32    # for Cifar images

nb_classes = 10 # for MNIST images
hiknot = 5
hiknot_deep = 3
nbQuantLevels = 100
K_val = 1.0
dropout_hyp = 0.5
dropout_dim = 0.5

base_folder = "Cifar/"
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
rule_file = "fidex_rule.txt"
simple_rule_file = "simple_fidex_rule.txt"
deep_test_sample_file = "deep_test_sample.txt"
test_sample_file = "test_sample.txt"
stats_file = "stats_fidex.txt"
simple_stats_file = "simple_stats_fidex.txt"

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

test_sample_id = 1
test_sample_deep = X_test_deep[test_sample_id]
test_sample_class = Y_test[test_sample_id]
test_sample_pred = test_pred[test_sample_id]

test_sample = X_test[test_sample_id]


try:
    with open(base_folder + deep_test_sample_file, "w") as myFile:
        myFile.write(' '.join(map(str, test_sample_deep)) + " " + ' '.join(map(str, test_sample_class)) + "\n")
        myFile.write(' '.join(map(str,test_sample_pred)))

except (FileNotFoundError):
    raise ValueError(f"Error : File {myFile} not found.")
except (IOError):
    raise ValueError(f"Error : Couldn't open file {myFile}.")

try:
    with open(base_folder + test_sample_file, "w") as myFile:
        myFile.write(' '.join(map(str, test_sample)) + " " + ' '.join(map(str, test_sample_class)) + "\n")
        myFile.write(' '.join(map(str,test_sample_pred)))

except (FileNotFoundError):
    raise ValueError(f"Error : File {myFile} not found.")
except (IOError):
    raise ValueError(f"Error : Couldn't open file {myFile}.")

# Test : Normal Fidex :
command = (
    f"--root_folder {base_folder} "
    f"--train_data_file {train_data_file} "
    f"--train_class_file {train_class_file} "
    f"--train_pred_file {train_pred_file} "
    f"--test_data_file {test_sample_file} "
    f"--weights_file {weights_first_layer} "
    f"--rules_outfile {simple_rule_file} "
    f"--nb_attributes {X_test.shape[1]} "
    f"--nb_classes {nb_classes} "
    f"--stats_file {simple_stats_file} "
    f"--nb_quant_levels {nbQuantLevels} "
    f"--dropout_hyp {dropout_hyp} "
    f"--dropout_dim {dropout_dim} "
)

fidex.fidex(command)

# Launch Fidex on deep layer for the test sample

#fidex.fidex()
command = (
    f"--root_folder {base_folder} "
    f"--train_data_file {deep_fidex_train_inputs} "
    f"--train_class_file {train_class_file} "
    f"--train_pred_file {train_pred_file} "
    f"--test_data_file {deep_test_sample_file} "
    f"--weights_file {weights_deep_fidex_outfile} "
    f"--rules_outfile {rule_file} "
    f"--nb_attributes {X_test_deep.shape[1]} "
    f"--nb_classes {nb_classes} "
    f"--stats_file {stats_file} "
    f"--nb_quant_levels {6*nbStairsPerUnit} "
    f"--hi_knot {hiknot_deep} "
    f"--dropout_hyp {dropout_hyp} "
    f"--dropout_dim {dropout_dim} "

)
print(command)
fidex.fidex(command)

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
