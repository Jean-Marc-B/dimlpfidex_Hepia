# images.py
import os
import shutil
import copy
import re
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import (
    getRules,
    highlight_area_histograms,
    highlight_area_activations_sum,
    highlight_area_probability_image
)
from trainings.trnFun import get_attribute_file
from utils.constants import HISTOGRAM_ANTECEDENT_PATTERN
from utils.config import *

def generate_explaining_images(cfg, X_train, Y_train, CNNModel, intermediate_model, args):
    """
    Generate explaining images.
    """
    print("Generation of images...")

    # 1) Load rules
    global_rules = getRules(cfg["global_rules_file"])

    # 2) Load attributes
    if args.statistic == "histogram":
        attributes = get_attribute_file(cfg["attributes_file"], cfg["nb_stats_attributes"])[0]

    # 3) Create out folder
    if os.path.exists(cfg["rules_folder"]):
        shutil.rmtree(cfg["rules_folder"])
    os.makedirs(cfg["rules_folder"])

    # 4) For each rule we get filter images for train samples covering the rule
    # good_classes = [2,3,5]
    # counter = 0
    for rule_id, rule in enumerate(global_rules[0:50]):

        # if counter == 50:
        #     exit()
        # if rule.target_class not in good_classes:
        #     continue
        # else:
        #     counter += 1

        if args.statistic == "histogram":
            rule.include_X = False
            for ant in rule.antecedents:
                ant.attribute = attributes[ant.attribute] # Replace the attribute's indiex by its true name
        elif args.statistic in ["probability", "probability_multi_nets"]:
                rule.include_X = False
        # Create folder for this rule
        rule_folder = os.path.join(cfg["rules_folder"], f"rule_{rule_id}_class_{cfg['classes'][rule.target_class]}")
        if os.path.exists(rule_folder):
            shutil.rmtree(rule_folder)
        os.makedirs(rule_folder)

        # Add a readme containing the rule
        readme_file = os.path.join(rule_folder, 'Readme.md')
        rule_to_print = copy.deepcopy(rule)

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
        elif args.statistic == "probability" or args.statistic == "probability_multi_nets":
            # Change antecedent with area and class involved

            # Scales of changes of original image to reshaped image
            scale_h = cfg["size1D"] / cfg["size_Height_proba_stat"]
            scale_w = cfg["size1D"] / cfg["size_Width_proba_stat"]
            for antecedent in rule_to_print.antecedents: # TODO : handle stride, different filter sizes, etc
                # area_index (size_Height_proba_stat, size_Width_proba_stat) : 0 : (1,1), 1: (1,2), ...
                channel_id = antecedent.attribute % (cfg["nb_classes"] + cfg["nb_channels"]) # (probas of each class + image rgb concatenated)
                area_number = antecedent.attribute // (cfg["nb_classes"] + cfg["nb_channels"])
                # channel_id = attribut_de_test % (cfg["nb_classes"] + cfg["nb_channels"])
                # area_number = attribut_de_test // (cfg["nb_classes"] + cfg["nb_channels"])
                area_Height = area_number // cfg["size_Width_proba_stat"]
                area_Width = area_number % cfg["size_Width_proba_stat"]
                if channel_id < cfg["nb_classes"]: #Proba of area
                    class_name = cfg["classes"][channel_id]
                    antecedent.attribute = f"P_class_{class_name}_area_[{area_Height}-{area_Height+FILTER_SIZE[0][0]-1}]x[{area_Width}-{area_Width+FILTER_SIZE[0][1]-1}]"
                else:
                    channel = channel_id - cfg["nb_classes"] #Pixel in concatenated original rgb image
                    # Conversion of resized coordinates into originals
                    height_original = round(area_Height * scale_h)
                    width_original = round(area_Width * scale_w)
                    antecedent.attribute = f"Pixel_{height_original}x{width_original}x{channel}"

        if os.path.exists(readme_file):
            os.remove(readme_file)
        with open(readme_file, 'w') as file:
            file.write(str(rule_to_print))

        # We create and save an image for each covered sample
        for img_id in rule.covered_samples[0:10]:
            img = X_train[img_id]
            if args.statistic == "histogram":
                highlighted_image = highlight_area_histograms(CNNModel, img, FILTER_SIZE, STRIDE, rule, cfg["classes"])
            elif args.statistic == "activation_layer":
                highlighted_image = highlight_area_activations_sum(CNNModel, intermediate_model, img, rule, FILTER_SIZE, STRIDE, cfg["classes"])
            elif args.statistic == "probability" or args.statistic == "probability_multi_nets":
                highlighted_image = highlight_area_probability_image(img, rule, cfg["size1D"], cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], FILTER_SIZE, cfg["classes"], cfg["nb_channels"])
            highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png")

            highlighted_image.savefig(f"{rule_folder}/sample_{img_id}.png")
            plt.close(highlighted_image)
