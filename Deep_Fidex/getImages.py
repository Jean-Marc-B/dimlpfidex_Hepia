import os
import sys
import time
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from utils import *



start_time = time.time()

#dataset = "MNIST"
dataset = "CIFAR"

if dataset == "MNIST":     # for MNIST images
    size1D             = 28
    nbChannels         = 1
    nb_classes = 10
    base_folder = "Mnist/"
    normalized = False
    classes = {
        1:"1",
        2:"2",
        3:"3",
        4:"4",
        5:"5",
        6:"6",
        7:"7",
        8:"8",
        9:"9",
    }

elif dataset == "CIFAR":     # for Cifar images
    size1D             = 32
    nbChannels         = 3
    nb_classes = 10
    base_folder = "Cifar/"
    normalized = False
    classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }



train_data_file = "trainData.txt"
train_class_file = "trainClass.txt"
test_data_file = "testData.txt"
test_class_file = "testClass.txt"
deep_fidex_train_inputs = "deep_fidex_train_inputs.txt"
deep_fidex_test_inputs = "deep_fidex_test_inputs.txt"

final_rules_file = "final_fidex_rules.txt"
deep_rules_file = "deep_fidex_rules.json"


deep_image_folder = base_folder + "/deep_images"
image_folder = base_folder + "/images"

show_images = False

# Create folders if it doesn't exist
if not os.path.exists(deep_image_folder):
    os.makedirs(deep_image_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Get data

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

test   = np.loadtxt(base_folder + test_data_file)

X_test = test.astype('float32')
print(X_test.shape)







# Generate test sample images with colored pixels where the rule is activated

# For deep images :
# normaliser en 0 et 255(on modifie la phase de normalisation), la rendre la plus carée possible(on modifie la dernière étape). On peut prendre le max et le min des valeurs pour normaliser...
rules = getRules(base_folder + deep_rules_file)

# For final images :

rules = getRules(base_folder + final_rules_file)
image_save_folder = image_folder

for id_sample, rule in enumerate(rules):

    baseimage = X_test[id_sample]

    if nbChannels == 1:
        if normalized:
            colorimage = [[int(255*float(v)),int(255*float(v)),int(255*float(v))] for v in baseimage]
        else:
            colorimage = [[v,v,v] for v in baseimage]
    else:
        colorimage = baseimage
        if normalized:
            colorimage = [int(float(v) * 255) for v in colorimage]

    for antecedent in rule.antecedents:
        if antecedent.inequality == False:
            if nbChannels == 1:
                colorimage[antecedent.attribute]=[255,0,0]
            else:
                colorimage[antecedent.attribute - (antecedent.attribute % 3)]=255
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+1]=0
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+2]=0
        else:
            if nbChannels == 1:
                colorimage[antecedent.attribute]=[0,255,0]
            else:
                colorimage[antecedent.attribute - (antecedent.attribute % 3)]=0
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+1]=255
                colorimage[antecedent.attribute - (antecedent.attribute % 3)+2]=0

    colorimage_array = np.array(colorimage).reshape(size1D, size1D, 3)
    colorimage = Image.fromarray(colorimage_array.astype('uint8'))

    image_path = image_save_folder + '/img_'+ str(id_sample) + '_' + classes[rule.target_class] + '_out.png'
    colorimage.save(image_path)
    if show_images:
        colorimage.show()




end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")
