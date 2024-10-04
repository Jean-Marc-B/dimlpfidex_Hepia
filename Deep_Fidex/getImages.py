import os
import time
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from utils import *
import shutil



start_time = time.time()

dataset = "MNIST"
#dataset = "CIFAR"

if dataset == "MNIST":     # for MNIST images
    size1D             = 28
    nbChannels         = 1
    nb_classes = 10
    base_folder = "Mnist/"
    normalized = False
    classes = {
        0:"0",
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

# Create folders
if os.path.exists(deep_image_folder):
    shutil.rmtree(deep_image_folder)
os.makedirs(deep_image_folder)
if os.path.exists(image_folder):
    shutil.rmtree(image_folder)
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

train   = np.loadtxt(base_folder + train_data_file)

X_train = train.astype('float32')

Y_test  = np.loadtxt(base_folder + test_class_file)
Y_test  = Y_test.astype('int32')

test   = np.loadtxt(base_folder + test_data_file)

X_test = test.astype('float32')
print(X_test.shape)

nb_deep_attributes = X_test_deep.shape[1]


# Generate test sample images with colored pixels where the rule is activated



#---------------------------------------------------------------------------------------

def process_rules(rules, X_test, X_train, image_save_folder, with_pad = False, nb_attributes=None, size1D=None, nbChannels=3, normalize=True):
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
            baseimage, nb_rows, nb_cols = reshape_and_pad(baseimage, nb_deep_attributes)
        image_path = os.path.join(current_dir, f"_test_img_{id_sample}_out.png")


#---------------------------------------------------------------------------------------



# For deep images :

rules = getRules(base_folder + deep_rules_file)
image_save_folder = deep_image_folder


for id_sample, rule in enumerate(rules):
    print("deep" + str(id_sample))
    # Create new folder
    current_dir = image_save_folder + "/sample_" + str(id_sample)  + '_class_' + classes[rule.target_class]
    os.makedirs(current_dir)

    covered_samples, covered_samples_ids = getCovering(rule, X_train_deep)

    baseimage = X_test_deep[id_sample]
    baseimage, nb_rows, nb_cols = reshape_and_pad(baseimage, nb_deep_attributes)
    image_path = current_dir + '/_test_img_'+ str(id_sample) + '_out.png'
    get_image(rule, baseimage, image_path, nb_rows, nb_cols, 1, normalize=True, show_images=show_images)

    for id in covered_samples_ids:
        baseimage = X_train_deep[id]
        baseimage, nb_rows, nb_cols = reshape_and_pad(baseimage, nb_deep_attributes)
        image_path = current_dir + '/_train_img_'+ str(id) + '_out.png'
        get_image(rule, baseimage, image_path, nb_rows, nb_cols, 1, normalize=True, show_images=show_images)

# For final images :

rules = getRules(base_folder + final_rules_file)
image_save_folder = image_folder

for id_sample, rule in enumerate(rules):
    print("front" + str(id_sample))
    # Create new folder
    current_dir = image_save_folder + "/sample_" + str(id_sample)  + '_class_' + classes[rule.target_class]
    os.makedirs(current_dir)

    covered_samples, covered_samples_ids = getCovering(rule, X_train)

    baseimage = X_test[id_sample]
    image_path = current_dir + '/_test_img_'+ str(id_sample) + '_out.png'
    get_image(rule, baseimage, image_path, size1D, size1D, nbChannels, normalized01=normalized, show_images=show_images)

    for id in covered_samples_ids:
        baseimage = X_train[id]
        image_path = current_dir + '/_train_img_'+ str(id) + '_out.png'
        get_image(rule, baseimage, image_path, size1D, size1D, nbChannels, normalized01=normalized, show_images=show_images)


end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")