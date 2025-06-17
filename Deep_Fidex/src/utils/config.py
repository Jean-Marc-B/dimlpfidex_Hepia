import os
from .utils import getProbabilityThresholds

# ===============================
# GLOBAL PARAMETERS
# ===============================

# List of datasets allowed
AVAILABLE_DATASETS = ["Mnist", "Cifar", "Happy", "testDataset", "HAM10000"]

# List of statistics allowed
AVAILABLE_STATISTICS = ["histogram", "activation_layer", "probability", "probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]

# List of CNN models available
AVAILABLE_CNN_MODELS = ["VGG", "VGG_metadatas", "VGG_and_big", "resnet", "small", "big", "MLP", "MLP_Patch"]

# Filters
FILTER_SIZE = [[7, 7]]  # Filter size applied on the image
STRIDE = [[1, 1]]  # Shift between each filter (need to specify one per filter size)
if len(STRIDE) != len(FILTER_SIZE):
    raise ValueError("Error : There is not the same amout of strides and filter sizes.")
NB_BINS = 9  # Number of bins wanted for probabilities (ex: NProb>=0.1, NProb>=0.2, etc.)
PROBABILITY_THRESHOLDS = getProbabilityThresholds(NB_BINS)

# Some Fidex parameters
HIKNOT = 5
NB_QUANT_LEVELS = 100
K_VAL = 1.0
DROPOUT_HYP = 0.95
DROPOUT_DIM = 0.95

# ===============================
# FONCTION TO INITIALIZE PARAMETERS WITH RESPECT TO THE ARGUMENTS
# ===============================

def load_config(args, script_dir):
    """
    Initialize the parameters with respect to the dataset and to the selected statistic.
    """
    config = {}  # Dictionary to stock parameters

    # 📂 Selection of the dataset
    if args.dataset == "Mnist":
        config["size1D"] = 28
        config["nb_channels"] = 1
        config["base_folder"] = os.path.join(os.path.dirname(script_dir), "../../data", "Mnist")
        config["data_type"] = "integer"
        config["classes"] = {i: str(i) for i in range(10)}

    elif args.dataset == "Cifar":
        config["size1D"] = 32
        config["nb_channels"] = 3
        config["base_folder"] = os.path.join(os.path.dirname(script_dir), "../../data", "Cifar")
        config["data_type"] = "integer"
        config["classes"] = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }

    elif args.dataset == "Happy":
        config["size1D"] = 48
        config["nb_channels"] = 1
        config["base_folder"] = os.path.join(os.path.dirname(script_dir), "../../data", "Happy")
        config["data_type"] = "float"
        config["classes"] = {0: "happy", 1: "not happy"}

    elif args.dataset == "HAM10000":
        config["size1D"] = 33 #28
        config["nb_channels"] = 3
        config["base_folder"] = os.path.join(os.path.dirname(script_dir), "../../data", "HAM10000")
        config["data_type"] = "integer"
        config["classes"] = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "nv", 5: "vasc", 6: "mel"}

    elif args.dataset == "testDataset":
        config["size1D"] = 20
        config["nb_channels"] = 1
        config["base_folder"] = "Test/"
        config["data_type"] = "integer"
        config["classes"] = {0: "cl0", 1: "cl1"}
    else:
        raise ValueError ("Wrong dataset given.")

    config["nb_classes"] = len(config["classes"])

    if getattr(args, "train_with_patches", False) and len(FILTER_SIZE) != 1:
        raise ValueError("Error : When training with patches, only one stride and one filter size can be chosen.")

    # 📊 Definition of the scan folder
    scan_folder = "evaluation/Scan" if args.test else "evaluation/ScanFull"
    patches_sufix = ""
    if getattr(args, "train_with_patches", False):
        patches_sufix = "_train_patches"
    folder_suf = ""
    if args.folder_sufix is not None:
        folder_suf = args.folder_sufix
    STATISTIC_FOLDERS = {
        "histogram": "Histograms" + patches_sufix + folder_suf,
        "activation_layer": "Activations_Sum" + patches_sufix + folder_suf,
        "probability_multi_nets": "Probability_Multi_Nets_Images" + patches_sufix + folder_suf,
        "probability": "Probability_Images" + patches_sufix + folder_suf,
        "probability_and_image": "Probability_and_image" + patches_sufix + folder_suf,
        "probability_multi_nets_and_image": "Probability_Multi_Nets_and_image" + patches_sufix + folder_suf,
        "probability_multi_nets_and_image_in_one": "Probability_Multi_Nets_and_image_in_one" + patches_sufix + folder_suf,
        "convDimlpFilter": "Conv_DIMLP_Filter" + folder_suf
    }
    scan_folder = os.path.join(scan_folder, STATISTIC_FOLDERS.get(args.statistic, "Probability_Images"))

    # 📂 Definition of folders of files
    config["scan_folder"] = scan_folder
    config["plot_folder"] = os.path.join(config["base_folder"], scan_folder, "plots")
    config["files_folder"] = os.path.join(config["base_folder"], scan_folder, "files")
    config["data_folder"] = os.path.join(config["base_folder"], "data")
    if args.image_version:
        config["rules_folder"] = os.path.join(config["plot_folder"], "Images")
    else:
        config["rules_folder"] = os.path.join(config["plot_folder"], "Rules")
    config["heat_maps_folder"] = os.path.join(config["plot_folder"], "Heat_maps")

    # 📄 Files
    test_particle = "_test_version" if args.test else ""
    config["train_data_file"] = os.path.join(config["data_folder"], f"trainData{test_particle}.txt")
    config["train_class_file"] = os.path.join(config["data_folder"], f"trainClass{test_particle}.txt")
    config["train_meta_file"] = os.path.join(config["data_folder"], f"trainMetaData{test_particle}.txt")
    config["test_data_file"] = os.path.join(config["data_folder"], f"testData{test_particle}.txt")
    config["test_class_file"] = os.path.join(config["data_folder"], f"testClass{test_particle}.txt")
    config["test_meta_file"] = os.path.join(config["data_folder"], f"testMetaData{test_particle}.txt")
    config["model_file"] = os.path.join(config["files_folder"], "scanModel.keras")
    config["train_pred_file"] = os.path.join(config["files_folder"], "train_pred.out")
    config["test_pred_file"] = os.path.join(config["files_folder"], "test_pred.out")
    config["attributes_file"] = os.path.join(config["files_folder"], "attributes.txt")
    config["model_checkpoint_weights"] = os.path.join(config["files_folder"], "weightsModel.weights.h5")
    config["model_stats"] = os.path.join(config["files_folder"], "stats_model.txt")
    # if args.dataset == "Mnist_Guido":
    #     config["global_rules_file"] = os.path.join(config["files_folder"], "globalRules.rls")
    # else:
    config["global_rules_file"] = os.path.join(config["files_folder"], "globalRules.json")
    config["global_rules_with_test_stats"] = os.path.join(config["files_folder"], "globalRulesWithStats.json")
    config["global_rules_stats"] = os.path.join(config["files_folder"], "global_rules_stats.txt")

    # 📌 Training (model and batch size)
    if args.test:
        config["model"] = "small"
        config["nbIt"] = 4
        config["batch_size"] = 16
        config["batch_size_second_model"] = 32
    else:
        config["model"] = "VGG"
        config["nbIt"] = 80
        config["batch_size"] = 16
        config["batch_size_second_model"] = 8

    if getattr(args, "train_with_patches", False):
        if config["model"] in AVAILABLE_CNN_MODELS:
            config["model"] = "MLP_Patch"
        elif config["model"] != "RF":
            raise ValueError("Wrong model given, give one of VGG, VGG_metadatas, VGG_and_big, resnet, small, big, MLP, MLP_Patch, RF")

    if config["model"] == "RF" and args.statistic == "activation_layer":
        raise ValueError("activation_layer can't use a Random Forest to train.")
    if args.statistic == "convDimlpFilter":
        config["model"] = "small"

    # 📊 Managment of statistics
    config["with_leaky_relu"] = args.statistic == "activation_layer"

    if args.statistic == "histogram":
        config["train_stats_file"] = os.path.join(config["files_folder"], "train_hist.txt")
        config["test_stats_file"] = os.path.join(config["files_folder"], "test_hist.txt")
        config["nb_stats_attributes"] = config["nb_classes"]*NB_BINS
    elif args.statistic == "activation_layer":
        if getattr(args, "train_with_patches", False):
            raise ValueError("Not possible to use sum of activation layers stats when training with patches.")
        config["train_stats_file"] = os.path.join(config["files_folder"], "train_activation_sum.txt")
        config["test_stats_file"] = os.path.join(config["files_folder"], "test_activation_sum.txt")
    elif args.statistic in ["probability", "probability_and_image", "probability_multi_nets_and_image", "probability_multi_nets", "probability_multi_nets_and_image_in_one"]:
        config["train_stats_file"] = os.path.join(config["files_folder"], "train_probability_images.txt")
        config["test_stats_file"] = os.path.join(config["files_folder"], "test_probability_images.txt")
        config["train_stats_file_with_image"] = os.path.join(config["files_folder"], "train_probability_images_with_original_img.txt")
        config["test_stats_file_with_image"] = os.path.join(config["files_folder"], "test_probability_images_with_original_img.txt")
    elif args.statistic == "convDimlpFilter":
        config["train_feature_map_file_npy"] = os.path.join(config["files_folder"], "train_feature_map.npy")
        config["test_feature_map_file_npy"] = os.path.join(config["files_folder"], "test_feature_map.npy")
        config["train_feature_map_file"] = os.path.join(config["files_folder"], "train_feature_map.txt")
        config["test_feature_map_file"] = os.path.join(config["files_folder"], "test_feature_map.txt")
    # Parameters for second model training
    if args.statistic in ["probability", "probability_and_image"]:
        config["second_model"] = "cnn"
        #second_model = "randomForests"
    elif args.statistic in ["probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]:
        config["second_model"] = "cnn"
    elif args.statistic == "convDimlpFilter":
        config["second_model"] = "small"
    else:
        # config["second_model"] = "randomForests"
        config["second_model"] = "gradientBoosting"
        # config["second_model"] = "dimlpTrn"
        # config["second_model"] = "dimlpBT"

    config["using_decision_tree_model"] = config["second_model"] in {"randomForests", "gradientBoosting"}

    config["second_model_stats"] = os.path.join(config["files_folder"], "second_model_stats.txt")
    config["second_model_train_pred"] = os.path.join(config["files_folder"], "second_model_train_pred.txt")
    config["second_model_test_pred"] = os.path.join(config["files_folder"], "second_model_test_pred.txt")
    config["second_model_file"] = os.path.join(config["files_folder"], "scanSecondModel.keras")
    config["second_model_checkpoint_weights"] = os.path.join(config["files_folder"], "weightsSecondModel.weights.h5")
    if config["using_decision_tree_model"]:
        config["second_model_output_weights"] = os.path.join(config["files_folder"], "second_model_rules.rls")
    else:
        config["second_model_output_weights"] = os.path.join(config["files_folder"], "second_model_weights.wts")

    config["size_Height_proba_stat"] = config["size1D"] - FILTER_SIZE[0][0] + 1
    config["size_Width_proba_stat"] = config["size1D"] - FILTER_SIZE[0][1] + 1
    # 📊 Parameters specific to probabilities
    if args.statistic == "probability":
        config["nb_stats_attributes"] = config["size_Height_proba_stat"] * config["size_Width_proba_stat"] * (config["nb_classes"] + config["nb_channels"])
    elif args.statistic in ["probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]:
        config["nb_stats_attributes"] = config["size_Height_proba_stat"] * config["size_Width_proba_stat"] * config["nb_classes"] + config["size1D"] * config["size1D"] * config["nb_channels"]
    # Display parameters
    print("\n--------------------------------------------------------------------------")
    print("Parameters :")
    print("--------------------------------------------------------------------------\n")
    print(f"Dataset : {config['base_folder']}")
    print(f"Size : {config['size1D']}x{config['size1D']}x{config['nb_channels']}")
    print(f"Data type : {config['data_type']}")
    print(f"Number of attributes : {config.get('nb_stats_attributes', 'N/A')}")

    print("Statistic :")
    print(STATISTIC_FOLDERS.get(args.statistic, "UNKNOWN"))
    if getattr(args, "train_with_patches", False):
        print("Training with patches")

    print("\n-------------")
    print("Files :")
    print("-------------")
    print(f"Train data file : {config['train_data_file']}")
    print(f"Train class file : {config['train_class_file']}")
    print(f"Train prediction file : {config['train_pred_file']}")
    print(f"Test data file : {config['test_data_file']}")
    print(f"Test class file : {config['test_class_file']}")
    print(f"Test prediction file : {config['test_pred_file']}")
    print(f"Model file : {config['model_file']}")
    print(f"Attributes file : {config['attributes_file']}")
    print(f"Rules folder : {config['rules_folder']}")
    print(f"Heat maps folder : {config['heat_maps_folder']}")

    if args.train:
        print("\n-------------")
        print("Training :")
        print("-------------")
        print(f"Model checkpoint weights : {config['model_checkpoint_weights']}")
        print(f"Model stats file : {config['model_stats']}")
        if args.statistic == "convDimlpFilter":
            print("Model : small")
        else:
            print(f"Model : {config['model']}")
        print(f"Number of iterations : {config['nbIt']}")
        print(f"Batch size : {config['batch_size']}")
        if args.statistic == "activation_layer":
            print("With Leaky Relu" if config["with_leaky_relu"] else "Without Leaky Relu")

    if args.images is not None or getattr(args, "heatmap", False) or getattr(args, "stats", False):
        print("\n-------------")
        print("Statistics :")
        print("-------------")
        print(f"Filter size : {FILTER_SIZE}")
        print(f"Stride : {STRIDE}")

    if getattr(args, "stats", False):
        print(f"Train statistics file : {config['train_stats_file']}")
        print(f"Test statistics file : {config['test_stats_file']}")
        if args.statistic in ["probability", "probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]:
            print(f"Train statistics file with image : {config['train_stats_file_with_image']}")
            print(f"Test statistics file with image : {config['test_stats_file_with_image']}")
        elif args.statistic == "histogram":
            print(f"Number of bins : {NB_BINS}")
            print(f"Probability thresholds : {PROBABILITY_THRESHOLDS}")

    if args.statistic == "convDimlpFilter":
        print(f"Train data file after first conv layer : {config['train_feature_map_file']}")
        print(f"Test data file after first conv layer : {config['test_feature_map_file']}")

    if getattr(args, "heatmap", False) and not (getattr(args, "stats", False) and args.statistic == "histogram"):
        print(f"Probability thresholds : {PROBABILITY_THRESHOLDS}")

    if args.second_train:
        print("\n-------------")
        print("Second training :")
        print("-------------")
        print(f"Second model : {config['second_model']}")
        print(f"Batch size second model: {config['batch_size_second_model']}")
        print(f"Second model statistics file : {config['second_model_stats']}")
        print(f"Second model train predictions file : {config['second_model_train_pred']}")
        print(f"Second model test predictions file : {config['second_model_test_pred']}")
        print(f"Second model output weights file : {config['second_model_output_weights']}")
        print(f"Second model output file : {config['second_model_file']}")
        print(f"Second model checkpoint weights : {config['second_model_checkpoint_weights']}")

    if args.rules:
        print("\n-------------")
        print("Fidex rules generation :")
        print("-------------")
        print(f"Global rules file : {config['global_rules_file']}")
        print(f"Hiknot : {HIKNOT}")
        print(f"Number of quantization levels : {NB_QUANT_LEVELS}")
        print(f"K : {K_VAL}")
        print(f"Dropout hyperplans : {DROPOUT_HYP}")
        print(f"Dropout dimensions : {DROPOUT_DIM}")
        print(f"Global rules file with test statistics : {config['global_rules_with_test_stats']}")
        print(f"Global rules statistics : {config['global_rules_stats']}")

    print("\n--------------------------------------------------------------------------")

    return config
