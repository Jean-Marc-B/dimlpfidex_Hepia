
import numpy as np
import tensorflow as tf
from stairObj import StairObj
from keras import backend as K

nbStairsPerUnit    = 30
nbStairsPerUnitInv = 1.0/nbStairsPerUnit


def output_data(data, data_file):
    try:
        with open(data_file, "w") as file:
            for var in data:
                for val in var:
                    file.write(str(val) + " ")
                file.write("\n")
            file.close()

    except (FileNotFoundError):
        raise ValueError(f"Error : File {data_file} not found.")
    except (IOError):
        raise ValueError(f"Error : Couldn't open file {data_file}.")

def staircaseUnbound(x):

   return (K.sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin(x):
#    h -> nStairsPerUnit
#    hard_sigmoid(x) =  max(0, min(1, x/6+0.5))
#    staircaseSemiLin(x, h) = hard_sigmoid(ceil(x*h)*(1/h))

   return (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))

###############################################################

def staircaseSemiLin2(x):

   a = (tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit) * nbStairsPerUnitInv))
   a = (a - 0.5)*6.0
   return a

###############################################################

def hardSigm2(x):

   a = tf.keras.activations.hard_sigmoid(x)
   a = (a - 0.5)*6.0
   return a

###############################################################

def staircaseBound(x):

   a = tf.keras.activations.hard_sigmoid(tf.math.ceil(x*nbStairsPerUnit)*0.5 * nbStairsPerUnitInv)
   a = (a - 0.5)*10.0
   return(K.sigmoid(a))

###############################################################

def compute_first_hidden_layer(step, input_data, k, nb_stairs, hiknot, weights_outfile=None, mu=None, sigma=None):
    input_data = np.array(input_data)
    if step == "train": # Train datas
        if weights_outfile is None:
            raise ValueError("Error : weights file is None during computation of first hidden layer with train data.")
        mu = np.mean(input_data, axis=0) if mu is None else mu  # mean over variables
        sigma = np.std(input_data, axis=0) if sigma is None else sigma # std over variables
        sigma[sigma == 0] = 0.001  # Prevent division by zero

        weights = k/sigma
        biais = -k*mu/sigma

        # Save weights and bias
        if weights_outfile is not None:
         try:
               with open(weights_outfile, "w") as my_file:
                  for b in biais:
                     my_file.write(str(b))
                     my_file.write(" ")
                  my_file.write("\n")
                  for w in weights:
                     my_file.write(str(w))
                     my_file.write(" ")
                  my_file.close()
         except (FileNotFoundError):
               raise ValueError(f"Error : File {weights_outfile} not found.")
         except (IOError):
               raise ValueError(f"Error : Couldn't open file {weights_outfile}.")

    # Compute new data after first hidden layer
    h = k*(input_data-mu)/sigma # With indices : hij=K*(xij-muj)/sigmaj
    stair = StairObj(nb_stairs, hiknot)
    out_data = np.vectorize(stair.funct)(h) # Apply staircase activation function

    return (out_data, mu, sigma) if step == "train" else out_data


###############################################################
###############################################################
