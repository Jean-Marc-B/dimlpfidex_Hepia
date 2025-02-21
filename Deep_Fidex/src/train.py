# train.py
import time
from utils.utils import trainCNN, output_data
from utils.config import *
from trainings import randForestsTrn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

def train_model(cfg, X_train, Y_train, X_test, Y_test, args):

    start_time_train_model = time.time()

    if args.train_with_patches:
        height = FILTER_SIZE[0][0]
        width = FILTER_SIZE[0][1]
    else:
        height = cfg["size1D"]
        width = cfg["size1D"]

    if cfg["model"] in AVAILABLE_CNN_MODELS:
        trainCNN(
            height = height,
            width = width,
            nbChannels = cfg["nb_channels"],
            nb_classes = cfg["nb_classes"],
            model = cfg["model"],
            nbIt = cfg["nbIt"],
            batch_size = cfg["batch_size"],
            model_file = cfg["model_file"],
            model_checkpoint_weights = cfg["model_checkpoint_weights"],
            X_train = X_train,
            Y_train = Y_train,
            X_test = X_test,
            Y_test = Y_test,
            train_pred_file = cfg["train_pred_file"],
            test_pred_file = cfg["test_pred_file"],
            model_stats = cfg["model_stats"],
            with_leaky_relu = cfg["with_leaky_relu"]
        )

    elif cfg["model"] == "RF":
        train_random_forest(cfg, X_train, Y_train, X_test, Y_test, args)
    else:
        raise ValueError("Wrong model given, give one of VGG, resnet, small, MLP, MLP_Patch, RF")

    end_time_train_model = time.time()
    full_time_train_model = end_time_train_model - start_time_train_model
    print(f"\nTrain first model time = {full_time_train_model:.2f} sec")





def train_random_forest(cfg, X_train, Y_train, X_test, Y_test, args):
    """
    Train a Random Forest model and save predictions and model statistics.

    Parameters:
    - cfg: Configuration dictionary.
    - X_train, Y_train: Training data and labels.
    - X_test, Y_test: Test data and labels.
    """
    print("Training Random Forest...")

    # Flatten data

    if args.train_with_patches:
        X_train, coord_train = X_train
        X_test, coord_test = X_test

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    if args.train_with_patches:
        X_train = np.append(X_train, coord_train, axis=1)
        X_test = np.append(X_test, coord_test, axis=1)

    # Ensure Y_train and Y_test are in correct shape
    Y_train_labels = np.argmax(Y_train, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 100),
        max_depth=cfg.get("max_depth", None),
        random_state=cfg.get("random_state", 42),
        n_jobs=6
    )
    rf_model.fit(X_train, Y_train_labels)

    # Save the trained model
    os.makedirs(os.path.dirname(cfg["model_file"]), exist_ok=True)
    joblib.dump(rf_model, cfg["model_file"])
    print(f"Random Forest model saved to {cfg['model_file']}")

    # Generate predictions
    train_pred = rf_model.predict_proba(X_train)
    test_pred = rf_model.predict_proba(X_test)

    # Save predictions
    output_data(train_pred, cfg["train_pred_file"])
    output_data(test_pred, cfg["test_pred_file"])
    print("Predictions saved.")

    # Evaluate model
    train_acc = rf_model.score(X_train, Y_train_labels)
    test_acc = rf_model.score(X_test, Y_test_labels)

    with open(cfg["model_stats"], "w") as stats_file:
        stats_file.write(f"Train Accuracy: {train_acc}\n")
        stats_file.write(f"Test Accuracy: {test_acc}\n")
    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
