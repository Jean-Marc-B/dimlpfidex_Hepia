import numpy as np
import sys
import time
from sklearn.model_selection import train_test_split
sys.path.append('../../trainings')
from trainings.trnFun import compute_first_hidden_layer, output_data, output_stats, get_data, get_data_class
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

nbStairsPerUnit = 100
nbStairsPerUnitInv = 1.0/nbStairsPerUnit

def staircaseSemiLin(x):
   return (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

start_time = time.time()

K = 1.0
nb_quant_levels = 100
hiknot = 5
weights_outfile = "newVersion/weights.wts"
nb_channels = 6
segment_size = 250
nb_attributes = nb_channels*segment_size
nb_classes = 8
nb_epochs = 80
model_checkpoint_weights = "newVersion/weightsModel.keras"
stats_file = "newVersion/stats.txt"
train_valid_pred_outfile = "newVersion/predTrain.out"
test_pred_outfile = "newVersion/predTest.out"
train_data_file = "trainData.txt"
train_class_file = "trainClass.txt"
test_data_file = "testData.txt"
test_class_file = "testClass.txt"
train_data_file_cnn = "newVersion/trainDataCNN.txt"
test_data_file_cnn = "newVersion/testDataCNN.txt"



print("Data loading ...")
x_train_full_temp, y_train_full_temp = get_data(train_data_file, nb_attributes, nb_classes)
x_train_full = np.array(x_train_full_temp)
del x_train_full_temp
y_train_full_temp = get_data_class(train_class_file, nb_classes)
y_train_full = np.array(y_train_full_temp)
del y_train_full_temp

if len(x_train_full) != len(y_train_full):
    raise ValueError('Error, there is not the same amount of train data and train class.')

x_test, y_test = get_data(test_data_file, nb_attributes, nb_classes)
y_test = get_data_class(test_class_file, nb_classes)
x_test, y_test = np.array(x_test), np.array(y_test)

# Split in train(70%), test(20%) and validation sets(10%)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=(1/8), random_state=42)

print(f"Training set: {x_train.shape}, {y_train.shape}")
print(f"Validation set: {x_val.shape}, {y_val.shape}")
print(f"Test set: {x_test.shape}, {y_test.shape}")


print("Data loaded.")

# Compute first hidden layer
x_train_h1, mu, sigma = compute_first_hidden_layer("train", x_train, K, nb_quant_levels, hiknot, weights_outfile)
x_test_h1 = compute_first_hidden_layer("test", x_test, K, nb_quant_levels, hiknot, mu=mu, sigma=sigma)
x_val_h1 = compute_first_hidden_layer("test", x_val, K, nb_quant_levels, hiknot, mu=mu, sigma=sigma)

x_train_h1 = x_train_h1.reshape(x_train_h1.shape[0], segment_size, nb_channels)

x_test_h1 = x_test_h1.reshape(x_test_h1.shape[0], segment_size, nb_channels)
x_val_h1 = x_val_h1.reshape(x_val_h1.shape[0], segment_size, nb_channels)

print(f"Training set: {x_train_h1.shape}, {y_train.shape}")
print(f"Validation set: {x_val_h1.shape}, {y_val.shape}")
print(f"Test set: {x_test_h1.shape}, {y_test.shape}")

y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)
y_val = keras.utils.to_categorical(y_val, nb_classes)

##############################################################################
print("Training model...")

def create_model(use_staircase=False):
    input_tensor = Input(shape=(250, 6))
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.5)(x)

    activation_fn = staircaseSemiLin if use_staircase else tf.keras.activations.hard_sigmoid
    x = Conv1D(1, 1, activation=activation_fn)(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization(name='last_batch_norm')(x)

    output_tensor = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    model.summary()

    return model

model = create_model(use_staircase=False)
model2 = create_model(use_staircase=True)


bestScore = float('inf')

for epoch in range(nb_epochs):
    print(f"Epoch {epoch+1}")
    
    # Train the model for 1 epoch
    model.fit(x_train_h1, y_train, batch_size=32, epochs=1, validation_data=(x_val_h1, y_val), verbose=2)
    
    # Transfer the weights to model2
    model2.set_weights(model.get_weights())
    
    # Evalueate model2 on validation
    val_score = model2.evaluate(x_val_h1, y_val, verbose=0)
    print(f"Validation score with staircaseSemiLin: {val_score}")
    
    # Save weights if the model scores better
    if val_score[0] < bestScore:
        bestScore = val_score[0]
        model.save(model_checkpoint_weights)
        print(f"*** New best validation score. Model saved at epoch {epoch+1}.")

    # Evaluate model2 on test set
    test_score = model2.evaluate(x_test_h1, y_test, verbose=0)
    print(f"Test score with staircaseSemiLin: {test_score}\n")


modelBest  = load_model(model_checkpoint_weights)
modelBest2 = create_model(use_staircase=True)
   
modelBest2.set_weights(modelBest.get_weights())

#checkpointer = ModelCheckpoint(filepath=model_checkpoint_weights, verbose=1, save_best_only=True)
#model.fit(x_train_h1, y_train, batch_size=32, epochs=nb_epochs, validation_data=(x_val_h1, y_val), callbacks=[checkpointer], verbose=2)

#model_best = load_model(model_checkpoint_weights)

#intermediate_layer_model = Model(inputs=model_best.input, outputs=model_best.get_layer('last_batch_norm').output)

# Extraction of activations for train, test and validation sets
#train_features = intermediate_layer_model.predict(x_train_h1)
#test_features = intermediate_layer_model.predict(x_test_h1)
#valid_features = intermediate_layer_model.predict(x_val_h1)
#train_valid_features = np.concatenate((train_features, valid_features))


#output_data(train_valid_features, train_data_file_cnn)
#output_data(test_features, test_data_file_cnn)


print("model trained")

train_pred = modelBest2.predict(x_train_h1)    # Predict the response for train dataset
test_pred = modelBest2.predict(x_test_h1)    # Predict the response for test dataset
valid_pred = modelBest2.predict(x_val_h1)   # Predict the response for validation dataset
train_valid_pred = np.concatenate((train_pred,valid_pred)) # We output predictions of both validation and training sets

# Output predictions
output_data(train_valid_pred, train_valid_pred_outfile)
output_data(test_pred, test_pred_outfile)

##############################################################################

print("\nBest result :")

score = modelBest2.evaluate(x_train_h1, y_train)
print("Train score : ", score)

score = modelBest2.evaluate(x_test_h1, y_test)
print("Test score : ", score)

acc_train = modelBest2.evaluate(x_train_h1, y_train, verbose=0)[1]
acc_test = modelBest2.evaluate(x_test_h1, y_test, verbose=0)[1]

formatted_acc_train = "{:.6f}".format(acc_train*100)
formatted_acc_test = "{:.6f}".format(acc_test*100)
output_stats(stats_file, formatted_acc_train, formatted_acc_test)

end_time = time.time()
full_time = end_time - start_time
full_time = "{:.6f}".format(full_time).rstrip("0").rstrip(".")

print(f"\nFull execution time = {full_time} sec")
