# train.py
import time
from utils.utils import trainCNN

def train_cnn(cfg, X_train, Y_train, X_test, Y_test):
    """
    Effectue l'entraînement du premier CNN.
    """
    start_time_train_cnn = time.time()
    trainCNN(
        height = cfg["size1D"],
        width = cfg["size1D"],
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
    end_time_train_cnn = time.time()
    full_time_train_cnn = end_time_train_cnn - start_time_train_cnn
    print(f"\nTrain first CNN time = {full_time_train_cnn:.2f} sec")
