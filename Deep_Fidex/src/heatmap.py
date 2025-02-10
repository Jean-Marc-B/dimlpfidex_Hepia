# heatmap.py
import os
import shutil
import matplotlib.pyplot as plt
from utils.config import *
from utils.utils import get_heat_maps

def generate_heatmaps(cfg, X_test, CNNModel):
    """
    Generate heatmaps for X_test
    """
    if os.path.exists(cfg["heat_maps_folder"]):
        shutil.rmtree(cfg["heat_maps_folder"])
    os.makedirs(cfg["heat_maps_folder"])

    for idx, img in enumerate(X_test[:100]):
        heat_maps_img = get_heat_maps(CNNModel, img, FILTER_SIZE, STRIDE,
                                      PROBABILITY_THRESHOLDS, cfg["classes"])
        heat_maps_img.savefig(f"{cfg['heat_maps_folder']}/sample_{idx}.png")
        plt.close(heat_maps_img)
