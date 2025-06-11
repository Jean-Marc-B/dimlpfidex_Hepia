# second_train.py
import time
import shutil
import os
import numpy as np
import tensorflow as tf

from trainings import randForestsTrn, gradBoostTrn
from dimlpfidex import dimlp
from utils.utils import (
    output_data,
    compute_first_hidden_layer,
    trainCNN,  # if used for second training (CNN)
    gathering_predictions
)
from utils.config import *

def train_second_model(cfg, X_train, Y_train, X_test, Y_test, intermediate_model, args):
    """
    Train a second model (For ex. RandomForest, dimlpTrn, or a second CNN)
    with respect to cfg["second_model"].
    """
    start_time_train_second_model = time.time()
    nb_train_samples = len(X_train)
    nb_test_samples = len(X_test)

    if args.statistic in ["probability", "probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]:   # We create an image out of the probabilities (for each class) of cropped areas of the original image
        # Load probas of areas from file
        print("Loading probability stats...")
        train_probas = np.loadtxt(cfg["train_stats_file"]).astype('float32')
        test_probas = np.loadtxt(cfg["test_stats_file"]).astype('float32')
        print("Probability stats loaded.")
        #print(train_probas.shape) # (nb_train_samples, 4840) (22*22*10)
        #print(test_probas.shape) # (nb_test_samples, 4840)

        print("Adding original image...")

        if cfg["second_model"] == "cnn" and args.statistic == "probability":
            train_probas = train_probas.reshape(nb_train_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"])
            X_train_reshaped = tf.image.resize(X_train, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"])) # Resize original image to the proba size
            train_probas_img = np.concatenate((train_probas, X_train_reshaped[:nb_train_samples]), axis=-1) # Concatenate the probas and the original image resized
            train_probas_img = train_probas_img.reshape(nb_train_samples, -1) # flatten for export

            test_probas = test_probas.reshape(nb_test_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"])
            X_test_reshaped = tf.image.resize(X_test, (cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"])) # Resize original image to the proba size
            test_probas_img = np.concatenate((test_probas, X_test_reshaped[:nb_test_samples]), axis=-1) # Concatenate the probas and the original image resized
            test_probas_img = test_probas_img.reshape(nb_test_samples, -1) # flatten for export
            # print(train_probas_img.shape) #(nb_train_samples, 5324) (22*22*11)
            # print(test_probas_img.shape)  #(nb_test_samples, 5324)
        else:
            X_train_flatten = X_train.reshape(nb_train_samples, -1)
            X_test_flatten = X_test.reshape(nb_test_samples, -1)
            train_probas_img = np.concatenate((train_probas, X_train_flatten), axis=-1)
            test_probas_img = np.concatenate((test_probas, X_test_flatten), axis=-1)



        # Save proba stats data with original image added
        output_data(train_probas_img, cfg["train_stats_file_with_image"])
        output_data(test_probas_img, cfg["test_stats_file_with_image"])

        cfg["train_stats_file"] = cfg["train_stats_file_with_image"]
        cfg["test_stats_file"] = cfg["test_stats_file_with_image"]

        print("original image added.")

        # The we train the second model depending on the type (RF, CNN, etc.)
        if cfg["second_model"] == "cnn":

            # Pass on the DIMLP layer
            train_probas_img_h1, mu, sigma = compute_first_hidden_layer("train", train_probas_img, K_VAL, NB_QUANT_LEVELS, HIKNOT, cfg["second_model_output_weights"], activation_fct_stairobj="identity")
            test_probas_img_h1 = compute_first_hidden_layer("test", test_probas_img, K_VAL, NB_QUANT_LEVELS, HIKNOT, mu=mu, sigma=sigma, activation_fct_stairobj="identity")


            # train_probas_img_h1 = train_probas_img_h1.reshape((nb_train_samples,)+cfg["output_size"]) #(100, 26, 26, 13)
            # print("train_probas_img_h1 reshaped : ", train_probas_img_h1.shape)
            # test_probas_img_h1 = test_probas_img_h1.reshape((nb_test_samples,)+cfg["output_size"])
            #print(train_probas_img.shape)  # (nb_train_samples, 22, 22, 10)
            #print(test_probas.shape)  # (nb_train_samples, 22, 22, 10)
            if args.statistic in ["probability_and_image", "probability_multi_nets", "probability_multi_nets_and_image", "probability_multi_nets_and_image_in_one"]:
                split_id = cfg["size_Height_proba_stat"] * cfg["size_Width_proba_stat"] * cfg["nb_classes"]
                train_proba_part = train_probas_img_h1[:, :split_id]
                test_proba_part = test_probas_img_h1[:, :split_id]
                train_img_part = train_probas_img_h1[:, split_id:]
                test_img_part = test_probas_img_h1[:, split_id:]
                train_proba_part = train_proba_part.reshape(nb_train_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"]) #(100,26,26,10)
                test_proba_part = test_proba_part.reshape(nb_test_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"]) #(100,26,26,10)
                train_img_part = train_img_part.reshape(nb_train_samples, cfg["size1D"], cfg["size1D"], cfg["nb_channels"]) #(100, 32, 32, 3)
                test_img_part = test_img_part.reshape(nb_test_samples, cfg["size1D"], cfg["size1D"], cfg["nb_channels"]) #(100, 32, 32, 3)

            if args.statistic == "probability": # Train with a CNN now
                train_probas_img_h1 = train_probas_img_h1.reshape(nb_train_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"] + cfg["nb_channels"]) #(100, 26, 26, 13)
                test_probas_img_h1 = test_probas_img_h1.reshape(nb_test_samples, cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"] + cfg["nb_channels"]) #(100, 26, 26, 13)
                trainCNN(cfg["size_Height_proba_stat"], cfg["size_Width_proba_stat"], cfg["nb_classes"]+cfg["nb_channels"], cfg["nb_classes"], "big", 80, cfg["batch_size_second_model"], cfg["second_model_file"], cfg["second_model_checkpoint_weights"], train_probas_img_h1, Y_train, test_probas_img_h1, Y_test, cfg["second_model_train_pred"], cfg["second_model_test_pred"], cfg["second_model_stats"])
            elif args.statistic == "probability_and_image": #Train with VGG for image and CNN for probas
                trainCNN((cfg["size1D"], cfg["size_Height_proba_stat"]), (cfg["size1D"], cfg["size_Width_proba_stat"]), (cfg["nb_channels"], cfg["nb_classes"]), cfg["nb_classes"], "VGG_and_big", 80, cfg["batch_size_second_model"], cfg["second_model_file"], cfg["second_model_checkpoint_weights"], (train_img_part, train_proba_part), Y_train, (test_img_part, test_proba_part), Y_test, cfg["second_model_train_pred"], cfg["second_model_test_pred"], cfg["second_model_stats"])
            else:
                # Probability_multi_nets create nb_classes networks and gather best probability among them. The images keep only the probabilities of areas for one class and add B&W image (or H and S of HSL)
                # probability_multi_nets_and_image and probability_multi_nets_and_image_in_one do the same but don't add the image. The image is trained apart.
                if args.test:
                    nbIt_current = 2
                else:
                    nbIt_current = 80

                models_folder = os.path.join(cfg["files_folder"], "Models")
                # Delete and recreate models folder
                if os.path.exists(models_folder):
                    shutil.rmtree(models_folder)
                os.makedirs(models_folder)

                # Create each dataset
                if args.statistic in ["probability_multi_nets", "probability_multi_nets_and_image"]:
                    end_particle = "_with_original_img"
                    data_flatten_size = cfg["size1D"]*cfg["size1D"]*3
                    if args.statistic == "probability_multi_nets_and_image":
                        end_particle = ""
                        data_flatten_size = cfg["size_Height_proba_stat"]*cfg["size_Width_proba_stat"]*3
                    for i in range(cfg["nb_classes"]):
                        print("Creating dataset n°",i,"...")

                        if args.statistic == "probability_multi_nets_and_image":
                            built_data_train = np.stack([train_proba_part[:,:,:,i]] * 3, axis=-1)
                            built_data_test = np.stack([test_proba_part[:,:,:,i]] * 3, axis=-1)
                        else:
                            # Create train data for each model
                            built_data_train = np.empty((nb_train_samples, cfg["size1D"], cfg["size1D"], 3))
                            # Add probas on first channel
                            train_proba_part_padded = np.pad(train_proba_part[:,:,:,i], ((0,0),(0, cfg["size1D"]-cfg["size_Height_proba_stat"]), (0, cfg["size1D"]-cfg["size_Width_proba_stat"])))
                            test_proba_part_padded = np.pad(test_proba_part[:,:,:,i], ((0,0),(0, cfg["size1D"]-cfg["size_Height_proba_stat"]), (0, cfg["size1D"]-cfg["size_Width_proba_stat"])))
                            built_data_train[:,:,:,0] = train_proba_part_padded
                            # Add H and S on last 2 channels (or R and G)
                            if cfg["nb_channels"] == 3:
                                built_data_train[:,:,:,1] = train_img_part[..., 0]
                                built_data_train[:,:,:,2] = train_img_part[..., 1]

                            else: # Add 1-probas and B&W on last 2 channels
                                built_data_train[:,:,:,1] = 1-train_proba_part_padded
                                built_data_train[:,:,:,2] = train_img_part[..., 0]
                            # built_data_train :  (100, 26, 26, 3)

                            # Create test data for each model
                            built_data_test = np.empty((nb_test_samples, cfg["size1D"], cfg["size1D"], 3))
                            # Add probas on first channel
                            built_data_test[:,:,:,0] = test_proba_part_padded
                            # Add H and S on last 2 channels
                            if cfg["nb_channels"] == 3:
                                built_data_test[:,:,:,1] = test_img_part[..., 0]
                                built_data_test[:,:,:,2] = test_img_part[..., 1]
                            else: # Add 1-probas and B&W on last 2 channels
                                built_data_test[:,:,:,1] = 1-test_proba_part_padded
                                built_data_test[:,:,:,2] = test_img_part[..., 0]

                        # Create classes for these datas
                        built_Y_train = np.zeros((nb_train_samples, 2), dtype=int)
                        built_Y_train[Y_train[:, i] == 1, 0] = 1  # If condition is True, set [1, 0]
                        built_Y_train[Y_train[:, i] != 1, 1] = 1  # If condition is False, set [0, 1]
                        built_Y_test = np.zeros((nb_test_samples, 2), dtype=int)
                        built_Y_test[Y_test[:, i] == 1, 0] = 1  # If condition is True, set [1, 0]
                        built_Y_test[Y_test[:, i] != 1, 1] = 1  # If condition is False, set [0, 1]


                        current_model_train_pred = os.path.join(models_folder, f"second_model_train_pred_{i}.txt")
                        data_filename = f"train_probability_images{end_particle}_{i}.txt"
                        class_filename = f"Y_train_probability_images{end_particle}_{i}.txt"
                        built_data_train_flatten = built_data_train.reshape(nb_train_samples, data_flatten_size)

                        # output new train data
                        output_data(built_data_train_flatten, os.path.join(models_folder, data_filename))
                        output_data(built_Y_train, os.path.join(models_folder, class_filename))

                        current_model_test_pred = os.path.join(models_folder, f"second_model_test_pred_{i}.txt")
                        data_filename = f"test_probability_images{end_particle}_{i}.txt"
                        class_filename = f"Y_test_probability_images{end_particle}_{i}.txt"
                        built_data_test_flatten = built_data_test.reshape(nb_test_samples, data_flatten_size)

                        # output new test data
                        output_data(built_data_test_flatten, os.path.join(models_folder, data_filename))
                        output_data(built_Y_test, os.path.join(models_folder, class_filename))

                        current_model_stats = os.path.join(models_folder, f"second_model_stats_{i}.txt")
                        current_model_file = os.path.join(models_folder, f"scanSecondModel_{i}.keras")
                        current_model_checkpoint_weights = os.path.join(models_folder, f"weightsSecondModel_{i}.weights.h5")

                        print("Dataset n°",i," created.")
                        # Train new model
                        if args.statistic == "probability_multi_nets_and_image":
                            trainCNN((cfg["size1D"], cfg["size_Height_proba_stat"]), (cfg["size1D"], cfg["size_Width_proba_stat"]), (cfg["nb_channels"], 3), 2, "VGG_and_VGG", nbIt_current, cfg["batch_size_second_model"], current_model_file, current_model_checkpoint_weights, (train_img_part, built_data_train), built_Y_train, (test_img_part, built_data_test), built_Y_test, current_model_train_pred, current_model_test_pred, current_model_stats)
                        else:

                            trainCNN(cfg["size1D"], cfg["size1D"], 3, 2, "VGG", nbIt_current, cfg["batch_size_second_model"], current_model_file, current_model_checkpoint_weights, built_data_train, built_Y_train, built_data_test, built_Y_test, current_model_train_pred, current_model_test_pred, current_model_stats)
                        print("Dataset n°",i," trained.")
                                    # Create test and train predictions

                    train_pred_files = [
                        os.path.join(models_folder, f"second_model_train_pred_{i}.txt") for i in range(cfg["nb_classes"])
                    ]
                    test_pred_files = [
                        os.path.join(models_folder, f"second_model_test_pred_{i}.txt") for i in range(cfg["nb_classes"])
                    ]

                    # Gathering predictions for train and test
                    print("Gathering train predictions...")
                    gathering_predictions(train_pred_files, cfg["second_model_train_pred"])
                    print("Gathering test predictions...")
                    gathering_predictions(test_pred_files, cfg["second_model_test_pred"])

                    # Compute and save predictions of the second (gathering of all models) model
                    second_model_train_preds = np.argmax(np.loadtxt(cfg["second_model_train_pred"]), axis=1)
                    second_model_test_preds = np.argmax(np.loadtxt(cfg["second_model_test_pred"]), axis=1)

                    # Compute and save train and test accuracies of the second model
                    train_accuracy = 0
                    for i in range(nb_train_samples):
                        if np.argmax(Y_train[i]) == second_model_train_preds[i]:
                            train_accuracy += 1
                    train_accuracy /= nb_train_samples

                    test_accuracy = 0
                    for i in range(nb_test_samples):
                        if np.argmax(Y_test[i]) == second_model_test_preds[i]:
                            test_accuracy += 1
                    test_accuracy /= nb_test_samples

                    with open(cfg["second_model_stats"], "w") as myFile:
                        print("Train score : ", train_accuracy)
                        myFile.write(f"Train score : {train_accuracy}\n")

                        print("Test score : ", test_accuracy)
                        myFile.write(f"Test score : {test_accuracy}\n")

                    print("Data sets created and all models trained.")

                else: # probability_multi_nets and images_in_one
                    built_data_train_full = [train_img_part]
                    built_data_test_full = [test_img_part]

                    for i in range(cfg["nb_classes"]):
                        print("Creating dataset n°",i,"...")
                        built_data_train = np.stack([train_proba_part[:,:,:,i]] * 3, axis=-1)
                        built_data_test = np.stack([test_proba_part[:,:,:,i]] * 3, axis=-1)

                        built_data_train_full.append(built_data_train)
                        built_data_test_full.append(built_data_test)

                        print("Dataset n°",i," created.")

                    # Train new model
                    trainCNN((cfg["size1D"], cfg["size_Height_proba_stat"]), (cfg["size1D"], cfg["size_Width_proba_stat"]), (cfg["nb_channels"], 3), cfg["nb_classes"], "VGG_and_nClass_VGGs", nbIt_current, cfg["batch_size_second_model"], cfg["second_model_file"], cfg["second_model_checkpoint_weights"], built_data_train_full, Y_train, built_data_test_full, Y_test, cfg["second_model_train_pred"], cfg["second_model_test_pred"], cfg["second_model_stats"])



        else:
            # Execution of randomForestsTrn, gradBoostTrn, etc.
            command = (
                f'--train_data_file {cfg["train_stats_file"]} '
                f'--train_class_file {cfg["train_class_file"]} '
                f'--test_data_file {cfg["test_stats_file"]} '
                f'--test_class_file {cfg["test_class_file"]} '
                f'--stats_file {cfg["second_model_stats"]} '
                f'--train_pred_outfile {cfg["second_model_train_pred"]} '
                f'--test_pred_outfile {cfg["second_model_test_pred"]} '
                f'--nb_attributes {cfg["nb_stats_attributes"]} '
                f'--nb_classes {cfg["nb_classes"]} '
                f'--root_folder . '
                )
            command += f'--rules_outfile {cfg["second_model_output_weights"]} '
            status = randForestsTrn(command)

    else:
        # For "histogram" or "activation_layer"
        # Train model
        command = (
            f'--train_data_file {cfg["train_stats_file"]} '
            f'--train_class_file {cfg["train_class_file"]} '
            f'--test_data_file {cfg["test_stats_file"]} '
            f'--test_class_file {cfg["test_class_file"]} '
            f'--stats_file {cfg["second_model_stats"]} '
            f'--train_pred_outfile {cfg["second_model_train_pred"]} '
            f'--test_pred_outfile {cfg["second_model_test_pred"]} '
            f'--nb_attributes {cfg["nb_stats_attributes"]} '
            f'--nb_classes {cfg["nb_classes"]} '
            f'--root_folder . '
            )

        if cfg["using_decision_tree_model"]:
            command += f'--rules_outfile {cfg["second_model_output_weights"]} '
        else:
            command += f'--weights_outfile {cfg["second_model_output_weights"]} '

        print("\nTraining second model...\n")

        # match second_model:
        #     case "randomForests":
        #         status = randForestsTrn(command)
        #     case "gradientBoosting":
        #         status = gradBoostTrn(command)
        #     case "dimlpTrn":
        #         status = dimlp.dimlpTrn(command)
        #     case "dimlpBT":
        #         command += '--nb_dimlp_nets 15 '
        #         command += '--hidden_layers [25] '
        #         if args.test:
        #             command += '--nb_epochs 10 '
        #         status = dimlp.dimlpBT(command)

        if cfg["second_model"] == "randomForests":
            status = randForestsTrn(command)
        elif cfg["second_model"] == "gradientBoosting":
            status = gradBoostTrn(command)
        elif cfg["second_model"] == "dimlpTrn":
            status = dimlp.dimlpTrn(command)
        elif cfg["second_model"] == "dimlpBT":
            command += '--nb_dimlp_nets 15 '
            command += '--hidden_layers [25] '
            if args.test:
                command += '--nb_epochs 10 '
            status = dimlp.dimlpBT(command)

        if status != -1:
            print("\nSecond model trained.")

    end_time_train_second_model = time.time()
    full_time_train_second_model = end_time_train_second_model - start_time_train_second_model
    print(f"\nTrain second model time = {full_time_train_second_model:.2f} sec")
