###############################################################################
# This library provide utilities for buiding, training, evaluating, saving
# and loading models. The actual model is passed through the parameter
# 'model_type'. The library makes use of Keras, tensorfow and sklearn
# The provided models are:
# - ANN_dense: a simple Dense neural network
# - AMN_QP: a trainable QP solver using Gradient Descent
# - AMN_Wt: a trainable RNN cell where V is updated with a weight matrix
# - MM_QP and MM_LP: non-trainable mechanistic model based on linear program
#   and gradient descent to compute all fluxes V when target objectives
#   are provided
# - RC: make use of trained AMNs (cf. previous module)
#   in reseroir computing (RC). The reservoir (non-trainable AMN)
#   is squized between two standard ANNs. The purpose of the prior ANN is to
#   transform problem features into nutrients added to media.
#   The post-ANN reads reservoir output (user predefined specific
#   reaction rates) and produce a readout to best match training set values.
#   Note that the two ANNs are trained but not the reservoir (AMN).
# Authors: Jean-loup Faulon, jfaulon@gmail.com and Bastien Mollet (LP model)
###############################################################################

from silence_tensorflow import silence_tensorflow
silence_tensorflow() # Tensorflow generates WARNINGS because of GPU unused, silence it

import copy

from Library.Build_Dataset import *
import keras.backend as K
import tensorflow as tf
print("Using tf version ", tf. __version__)
import keras
print("Using keras version ", keras. __version__)
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

# print(tf.config.list_physical_devices('GPU'))
# tf.debugging.set_log_device_placement(False)
# tf.device('/GPU:0')

from keras import initializers
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, LSTM, Dropout, Flatten, Activation
from keras.layers import Lambda, Reshape, multiply
from keras.layers import concatenate, add, subtract, dot
# from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from keras.layers import Activation
# --- depends on TF version
# from keras.utils.generic_utils import get_custom_objects
# from keras.utils.generic_utils import CustomObjectScope

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import CustomObjectScope
# ---
from keras.callbacks import EarlyStopping, CSVLogger

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from keras_core import ops

###############################################################################
# WandB class for tracking
###############################################################################
# import wandb
# print("Using W&B version ", wandb.__version__)
# from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback, WandbCallback, WandbModelCheckpoint
# wandb.login()

# # Implement your model prediction visualization callback
# class WandbClfEvalCallback(WandbEvalCallback):
#     def __init__(
#         self, validation_data, data_table_columns, pred_table_columns, num_samples=5
#     ):
#         super().__init__(data_table_columns, pred_table_columns)
#
#         self.x = validation_data[0]
#         self.y = validation_data[1]
#
#     def add_ground_truth(self, logs=None):
#         for idx, (image, label) in enumerate(zip(self.x, self.y)):
#             self.data_table.add_data(idx, wandb.Image(image), label)
#
#     def add_model_predictions(self, epoch, logs=None):
#         preds = self.model.predict(self.x, verbose=0)
#         preds = tf.argmax(preds, axis=-1)
#
#         table_idxs = self.data_table_ref.get_index()
#
#         for idx in table_idxs:
#             pred = preds[idx]
#             self.pred_table.add_data(
#                 epoch,
#                 self.data_table_ref.data[idx][0],
#                 self.data_table_ref.data[idx][1],
#                 self.data_table_ref.data[idx][2],
#                 pred,
#             )

###############################################################################
# Custom Loss function
###############################################################################
from tensorflow.keras.losses import Loss

NBR_CONSTRAINT = 3 # The number of contraints of the mechanistic models

def generate_loss_model(parameter, Vin, Vbf):
    # Create input layer
    batch_size = parameter.batch_size
    num_rxns   = len(parameter.reactions)

    yTrue = Input(shape=(batch_size, num_rxns), name="yTrue")
    yPred = Input(shape=(batch_size, num_rxns), name="yPred")
    Vin_in = Input(shape=Vin.shape, name="Vin")
    Vbf_in = Input(shape=Vbf.shape, name="Vbf")
    S_in = Input(shape=parameter.S.shape, name="S")
    Pin_in = Input(shape=parameter.Pin.shape, name="Pin")
    Pout_in = Input(shape=parameter.Pout.shape, name="Pout")

    # Apply custom operation using Lambda layer
    out_shape = (None, )
    # , output_shape=out_shape
    out = tf.keras.layers.Lambda(compute_loss_lambda) \
                    ((yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in))
    # Create model
    loss_model = tf.keras.models.Model(
                    inputs=[yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in],
                    outputs=out)

    return loss_model

# Custom loss function using model prediction
class CustomLoss(Loss):
    def __init__(self, loss_model, parameter):
        super().__init__()
        self.loss_model = loss_model
        self.parameter = parameter
        self.Vin  = None
        self.Vbf  = None
        self.end  = len(self.parameter.reactions)
        self.input_dim = self.parameter.input_dim

        self.S    = tf.expand_dims(
                        tf.convert_to_tensor(np.float32(parameter.S)),
                        axis=0)
        self.Pin  = tf.expand_dims(
                        tf.convert_to_tensor(np.float32(parameter.Pin)),
                        axis=0)
        self.Pout = tf.expand_dims(
                        tf.convert_to_tensor(np.float32(parameter.Pout)),
                        axis=0)

        # print("Vin ", self.Vin.shape, " Vbf ", self.Vbf.shape)
        print("S ", self.S.shape, " Pin ", self.Pin.shape)
        print("Pout ", self.Pout.shape)
        loss_model.summary()


    def call(self, y_true, y_pred):
        d0, d1 = \
            y_true.shape[0], y_true.shape[1]
        if d1 > self.input_dim:
            self.Vin = CROP(1, 0, self.input_dim)(y_true)
            self.Vbf = CROP(1, self.input_dim, d1-3)(y_true)
        else:
            self.Vin = tf.constant(1000, shape=(y_true.shape[0], self.input_dim))
            self.Vbf = tf.constant(1000, shape=(y_true.shape[0], self.end))



        print('before y_true ', y_true.shape)
        print('before y_pred ', y_pred.shape)
        prediction_index = self.end + NBR_CONSTRAINT
        y_true, y_pred = y_true[:,:self.end], y_pred[:,prediction_index:]
        print('after y_true ', y_true.shape)
        print('after y_pred ', y_pred.shape)

        self.Vin = tf.expand_dims(self.Vin, axis=0)
        self.Vbf = tf.expand_dims(self.Vbf, axis=0)
        y_pred   = tf.expand_dims(y_pred, axis=0)
        y_true   = tf.expand_dims(y_true, axis=0)
        y_true   = tf.convert_to_tensor(y_true)

        loss = compute_loss_lambda(y_true, y_pred, self.Vbf, self.Vin, self.S, self.Pin, self.Pout)

        print("loss type ", type(loss))
        return loss

def compute_loss_ops(y_true, y_pred, Vbf, Vin, S, Pin, Pout):
    # y_true, y_pred, Vbf, Vin, S, Pin, Pout = vals

    y_pred = ops.squeeze(y_pred)
    Pout = ops.squeeze(Pout)
    Vbf = ops.squeeze(Vbf)
    S = ops.squeeze(S)
    Vin = ops.squeeze(Vin)
    Pin = ops.squeeze(Pin)
    print(type(y_true))

    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)
    Pout   = ops.convert_to_tensor(Pout)
    Vbf    = ops.convert_to_tensor(Vbf)
    S      = ops.convert_to_tensor(S)
    Vin    = ops.convert_to_tensor(Vin)
    Pin    = ops.convert_to_tensor(Pin)

    num_rxns = y_true.shape[2]
    num_mets = S.shape[1]
    num_ins  = Pin.shape[1]
    print("y_true ", y_true.shape, " y_pred ", y_pred.shape)
    print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
    print("S ", S.shape, " Pin ", Pin.shape)
    print("Pout ", Pout.shape)

    # print(abc)


    ## Loss_Vout_constraint
    PoutV = ops.matmul(y_pred, ops.transpose(Pout)) - Vbf
    # PoutV = ops.subtract(PoutV, Vbf)
    PoutV = tf.keras.activations.relu(PoutV)
    PoutV = ops.norm(PoutV, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32)

    # PoutV = ops.square(PoutV)
    # print("type L1 --> ", type(L1))
    # print(abc)
    # return L1

    ## Loss_SV
    SV = ops.matmul(y_pred, tf.transpose(S))
    SV = ops.norm(SV, axis=1, keepdims=True) / tf.constant(num_mets, dtype=tf.float32)

    ## Loss_Vin
    PinV = ops.matmul(y_pred, tf.transpose(Pin)) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    PinV = tf.keras.activations.relu(PinV)
    PinV = ops.norm(PinV, axis=1, keepdims=True) / tf.constant(num_ins, dtype=tf.float32)

    ## Loss_Vpos
    Vpos = tf.keras.activations.relu(-y_pred)
    Vpos = ops.norm(Vpos, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32) # rescaled


    L1 = PoutV
    L2 = SV
    L3 = PinV
    L4 = Vpos

    # square sum of L1, L2, L3, L4, L5
    L1 = ops.square(PoutV)
    L2 = ops.square(SV)
    L3 = ops.square(PinV)
    L4 = ops.square(Vpos)
    # # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    # L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4, L5], axis=1), axis=1)
    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    L = L / tf.constant(num_const, dtype=tf.float32)

    print("Loss shape ", L.shape)

    return L


def compute_loss_lambda(value):
    y_true, y_pred, Vbf, Vin, S, Pin, Pout = value
    y_pred = tf.squeeze(y_pred)
    Pout = tf.squeeze(Pout)
    Vbf = tf.squeeze(Vbf)
    S = tf.squeeze(S)
    Vin = tf.squeeze(Vin)
    Pin = tf.squeeze(Pin)
    print(type(y_true))

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    Pout   = tf.convert_to_tensor(Pout)
    Vbf    = tf.convert_to_tensor(Vbf)
    S      = tf.convert_to_tensor(S)
    Vin    = tf.convert_to_tensor(Vin)
    Pin    = tf.convert_to_tensor(Pin)

    num_rxns = y_true.shape[2]
    num_mets = S.shape[1]
    num_ins  = Pin.shape[1]
    print("y_true ", y_true.shape, " y_pred ", y_pred.shape)
    print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
    print("S ", S.shape, " Pin ", Pin.shape)
    print("Pout ", Pout.shape)

    ## Loss_Vout_constraint
    PoutV    = tf.linalg.matmul(y_pred, tf.transpose(Pout), b_is_sparse=True)  - Vbf
    PoutV = tf.keras.activations.relu(PoutV)
    PoutV = tf.norm(PoutV, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32)

    ## Loss_SV
    SV = tf.linalg.matmul(y_pred, tf.transpose(S), b_is_sparse=True)
    SV = tf.norm(SV, axis=1, keepdims=True) / tf.constant(num_mets, dtype=tf.float32)
    # SV = tf.math.divide_no_nan(SV, tf.constant(num_mets, dtype=tf.float32)) # rescaled

    ## Loss_Vin
    PinV = tf.linalg.matmul(y_pred, tf.transpose(Pin), b_is_sparse=True) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    PinV = tf.keras.activations.relu(PinV)
    PinV = tf.norm(PinV, axis=1, keepdims=True) / tf.constant(num_ins, dtype=tf.float32)
    # PinV = tf.math.divide_no_nan(PinV, tf.constant(num_ins, dtype=tf.float32))

    ## Loss_Vpos
    Vpos = tf.keras.activations.relu(-y_pred)
    Vpos = tf.norm(Vpos, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32) # rescaled


    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(PoutV)
    L2 = tf.math.square(SV)
    L3 = tf.math.square(PinV)
    L4 = tf.math.square(Vpos)
    # # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    # num_const = 1.0
    L = tf.math.divide_no_nan(L, tf.constant(num_const, dtype=tf.float32))
    # print(L.shape)

    return L


def compute_loss(y_true, y_pred, Vbf, Vin, S, Pin, Pout):
    print(type(y_true))

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    Pout   = tf.convert_to_tensor(Pout)
    Vbf    = tf.convert_to_tensor(Vbf)
    S      = tf.convert_to_tensor(S)
    Vin    = tf.convert_to_tensor(Vin)
    Pin    = tf.convert_to_tensor(Pin)

    print("y_true ", y_true.shape, " y_pred ", y_pred.shape)
    print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
    print("S ", S.shape, " Pin ", Pin.shape)
    print("Pout ", Pout.shape)

    num_rxns = y_true.shape[1]
    num_mets = S.shape[1]
    num_ins  = Pin.shape[1]

    ## Loss_Vout_constraint
    PoutV    = tf.linalg.matmul(y_pred, tf.transpose(Pout), b_is_sparse=True)  - Vbf
    PoutV = tf.keras.activations.relu(PoutV)
    PoutV = tf.norm(PoutV, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32)

    ## Loss_SV
    SV = tf.linalg.matmul(y_pred, tf.transpose(S), b_is_sparse=True)
    SV = tf.norm(SV, axis=1, keepdims=True) / tf.constant(num_mets, dtype=tf.float32)
    # SV = tf.math.divide_no_nan(SV, tf.constant(num_mets, dtype=tf.float32)) # rescaled

    ## Loss_Vin
    PinV = tf.linalg.matmul(y_pred, tf.transpose(Pin), b_is_sparse=True) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    PinV = tf.keras.activations.relu(PinV)
    PinV = tf.norm(PinV, axis=1, keepdims=True) / tf.constant(num_ins, dtype=tf.float32)
    # PinV = tf.math.divide_no_nan(PinV, tf.constant(num_ins, dtype=tf.float32))

    ## Loss_Vpos
    Vpos = tf.keras.activations.relu(-y_pred)
    Vpos = tf.norm(Vpos, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32) # rescaled


    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(PoutV)
    L2 = tf.math.square(SV)
    L3 = tf.math.square(PinV)
    L4 = tf.math.square(Vpos)
    # # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    # num_const = 1.0
    L = tf.math.divide_no_nan(L, tf.constant(num_const, dtype=tf.float32))
    # print(L.shape)

    return L


###############################################################################
# Custom functions for training
###############################################################################

def sharp_sigmoid(x):
    # Custom activation function
    return K.sigmoid(10000.0 * x)
get_custom_objects().update({'sharp_sigmoid': Activation(sharp_sigmoid)})

# source: https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def biochem_loss(parameter):

    def loss(y_true, y_pred):

        end  = len(parameter.reactions)
        input_dim = parameter.input_dim
        d0, d1 = \
            y_true.shape[0], y_true.shape[1]
        if d1 > input_dim:
            Vin = CROP(1, 0, input_dim)(y_true)
            Vbf = CROP(1, input_dim, d1-3)(y_true)
        else:
            Vin = tf.constant(1000, shape=(y_true.shape[0], input_dim))
            Vbf = tf.constant(1000, shape=(y_true.shape[0], end))


        S    = tf.convert_to_tensor(np.float32(parameter.S))
        Pin  = tf.convert_to_tensor(np.float32(parameter.Pin))
        Pout = tf.convert_to_tensor(np.float32(parameter.Pout))

        prediction_index = end + NBR_CONSTRAINT
        y_true, y_pred = y_true[:,:end], y_pred[:,prediction_index:]

        # print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
        print("S ", S.shape, " Pin ", Pin.shape)
        print("Pout ", Pout.shape)

        loss = compute_loss(y_true, y_pred, Vbf, Vin, S, Pin, Pout)
        print("loss type ", type(loss))
        return loss
        # end = len(parameter.reactions)
        # y_true, y_pred = y_true[:,:end], y_pred[:,:end]
        # print('y_true ', y_true.shape)
        # print('y_pred ', y_pred.shape)
        #
        # Vin, Vbf = paramVin, paramVbf
        # # print("x ", x.shape, " V ", V.shape)
        # print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
        # print("S ", parameter.S.shape, " Pin ", parameter.Pin.shape)
        # print("Pout ", parameter.Pout.shape, " mediumbound ", parameter.mediumbound)
        # S     = tf.convert_to_tensor(np.float32(parameter.S))
        # Pin   = tf.convert_to_tensor(np.float32(parameter.Pin))
        # Pout  = tf.convert_to_tensor(np.float32(parameter.Pout))


        #
        # # # inp1 = tf.keras.layers.Input(shape=(None, 2, end), name="inp1")
        # yTrue = Input(shape=y_true.shape, name="yTrue")
        # yPred = Input(shape=y_pred.shape, name="yPred")
        # Vin_in = Input(shape=Vin.shape, name="Vin")
        # Vbf_in = Input(shape=Vbf.shape, name="Vbf")
        # S_in = Input(shape=S.shape, name="S")
        # Pin_in = Input(shape=Pin.shape, name="Pin")
        # Pout_in = Input(shape=Pout.shape, name="Pout")
        # print('Vbf ', Vbf.shape)
        # #
        # # out = tf.keras.layers.Lambda(add_numpy)((yTrue, yPred, Vin, Vbf, parameter))
        # out = tf.keras.layers.Lambda(add_numpy)((yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in))
        # # out = tf.keras.layers.Lambda(add_numpy_temp)((yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in))
        # #
        # # mdl = tf.keras.Model(input=[yTrue, yPred], output=out)
        # mdl = tf.keras.models.Model(inputs=[yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in], outputs=out)
        # # mdl = tf.keras.models.Model(inputs=[inp1, inp2], outputs=out)
        # mdl.summary()
        # y_true = tf.expand_dims(y_true, axis=0)
        # y_pred = tf.expand_dims(y_pred, axis=0)
        # Vbf = tf.expand_dims(Vbf, axis=0)
        # Vin = tf.expand_dims(Vin, axis=0)
        # S = tf.expand_dims(S, axis=0)
        # Pout = tf.expand_dims(Pout, axis=0)
        # Pin = tf.expand_dims(Pin, axis=0)
        # L =  mdl([y_true, y_pred, Vbf, Vin, S, Pin, Pout]) #mdl([y_true, y_pred])
        # print(L.shape)
        # # print(bac)
        # # mdl(input=[y_true, y_pred], output=[out])
        # # return mdl([y_true, y_pred])
        # # return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])
        # return L

    return loss

def my_mse(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    print('--- my MSE:')
    print('y_true: ', y_true[:,:end].shape)
    print('y_pred: ', y_pred[:,:end].shape)
    print('--- --- ---')
    return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])

def my_r2(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    yt, yp = y_true[:,:end], y_pred[:,:end]
    SS =  K.sum(K.square( yt-yp ))
    ST = K.sum(K.square( yt - K.mean(yt) ) )
    return 1 - SS/(ST + K.epsilon())

def CROP(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call x = crop(2,5,10)(x) to slice the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

###############################################################################
# Custom Loss functions to evaluate models and compute gradients
# Inputs:
# - V: the (predicted) flux vector
# - Pout: the matrix selecting in V measured outgoing fluxes
# - Vout: the measured outgoing fluxes
# - Pin: the matrix selecting in V measured incoming fluxes
# - Vin: the measured incoming fluxes
# - S: the stoichiometric matrix
# Outputs:
# - Loss and gradient
###############################################################################

def Loss_Vout(V, Pout, Vout, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    if not tf.is_tensor(Pout):
        Pout = tf.convert_to_tensor(np.float32(Pout))
    # print(Pout, Vout, V)
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vout_obj(V, parameter=None, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    Pout = tf.convert_to_tensor(np.float32(parameter.objPout))
    Vout = tf.convert_to_tensor(np.float32(parameter.objY))
    # print(Pout, Vout, V)

    # np.savetxt("temp_mult.csv", temp, delimiter=',')
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled

    # np.savetxt("temp_Loss.csv", Loss, delimiter=',')
    # np.savetxt("temp_LossScaled.csv", Loss_norm, delimiter=',')
    # print(abc)
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vout_obj_constraint(V, parameter=None, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    Pout = tf.convert_to_tensor(np.float32(parameter.objPout))
    Vout = tf.convert_to_tensor(np.float32(parameter.objY))
    # print(Pout, Vout, V)

    # np.savetxt("temp_mult.csv", temp, delimiter=',')
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled

    # np.savetxt("temp_Loss.csv", Loss, delimiter=',')
    # np.savetxt("temp_LossScaled.csv", Loss_norm, delimiter=',')
    # print(abc)
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vout_constraint(V, Pout, Vout, gradient=False):
    if not tf.is_tensor(Pout):
        Pout = tf.convert_to_tensor(np.float32(Pout))
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss = tf.keras.activations.relu(Loss)
    
    Loss_norm = tf.math.square(tf.norm(Loss, axis=1, keepdims=True)) / Pout.shape[0] 
    
    if gradient:
        dLoss = tf.math.divide_no_nan(Loss, Loss) 
        dLoss = tf.math.multiply(Loss, dLoss) 
        dLoss = tf.linalg.matmul(dLoss, Pout, b_is_sparse=True) 
        dLoss = (dLoss * 2.0) / Pout.shape[0]  
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vin(V, Pin, Vin, bound='UB', gradient=False):
    if not tf.is_tensor(Pin):
        Pin  = tf.convert_to_tensor(np.float32(Pin))

    Loss = tf.linalg.matmul(V, tf.transpose(Pin), b_is_sparse=True) - Vin
    Loss = tf.keras.activations.relu(Loss) if bound == 'UB' else Loss
    
    Loss_norm = tf.math.square(tf.norm(Loss, axis=1, keepdims=True)) / Pin.shape[0] 
    
    if gradient:
        dLoss = tf.math.divide_no_nan(Loss, Loss) 
        dLoss = tf.math.multiply(Loss, dLoss) 
        dLoss = tf.linalg.matmul(dLoss, Pin, b_is_sparse=True)
        dLoss = (dLoss * 2.0) / Pin.shape[0]   
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def Loss_SV(V, S, gradient=False, save=False):
    if not tf.is_tensor(S):
        S  = tf.convert_to_tensor(np.float32(S))
    Loss = tf.linalg.matmul(V, tf.transpose(S), b_is_sparse=True)
    
    Loss_norm = tf.math.square(tf.norm(Loss, axis=1, keepdims=True)) / S.shape[0] 
    
    if save:
        np.savetxt("sv_loss.csv", Loss, delimiter=',')
        np.savetxt("sv_lossNorm.csv", Loss_norm, delimiter=',')

    if gradient:
        dLoss = tf.linalg.matmul(Loss, S, b_is_sparse=True)
        dLoss = (dLoss * 2.0) / S.shape[0]  
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def Loss_Vpos(V, Vlb, gradient=False):
    if (Vlb is not None) and (Vlb.shape[0] is not None) and (Vlb.shape[0] >= 1) \
            and (Vlb.shape[1] == V.shape[1]):
        Loss = tf.keras.activations.relu(Vlb - V)
    else:
        Loss = tf.keras.activations.relu(-V)

    Loss_norm = tf.math.square(tf.norm(Loss, axis=1, keepdims=True)) / V.shape[1] 
    
    if gradient:
        dLoss = - tf.math.divide_no_nan(Loss, Loss) 
        dLoss = tf.math.multiply(Loss, dLoss) 
        dLoss = (dLoss * 2.0) / V.shape[1] 
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def Loss_constraint(V, Vin, Vlb, parameter, gradient=False):
    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient)
    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound, gradient=gradient)
    L4, dL4 = Loss_Vpos(V, Vlb, gradient=gradient)
    
    # Sum of L2, L3, L4
    L = tf.math.reduce_sum(tf.concat([L2, L3, L4], axis=1), axis=1)
    
    # divide by 3
    L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))
    return L, dL2 + dL3 + dL4

def _loop_adjacency(parameter):
    """Symmetric num_rxns x num_rxns matrix M with M[i_f,i_r]=M[i_r,i_f]=1 for
    every reversible split pair (_f/_r or _o/_i). Cached on the parameter."""
    M = getattr(parameter, "_loop_M", None)
    if M is not None:
        return M
    reactions = [str(r) for r in parameter.reactions]
    n = len(reactions); d = {}
    for j, rid in enumerate(reactions):
        if rid.endswith("_f") or rid.endswith("_o"):
            d.setdefault(rid[:-2], {})["f"] = j
        elif rid.endswith("_r") or rid.endswith("_i"):
            d.setdefault(rid[:-2], {})["r"] = j
    M = np.zeros((n, n), dtype=np.float32); npair = 0
    for _k, v in d.items():
        if "f" in v and "r" in v:
            M[v["f"], v["r"]] = 1.0; M[v["r"], v["f"]] = 1.0; npair += 1
    parameter._loop_M = M; parameter._loop_npair = npair
    return M


def Loss_loop(V, parameter, gradient=False):
    """Complementarity (loop-law) penalty that eliminates futile 2-cycles.

    The problem it solves
    ---------------------
    To keep every flux non-negative (all reactions running left-to-right), each
    reversible reaction is split into a forward (``_f``/``_o``) and a reverse
    (``_r``/``_i``) copy. Nothing then couples the two copies, so the optimizer
    is free to run BOTH at once: ``v_f = v_r = X`` carries zero *net* flux
    (net = ``v_f - v_r`` = 0) yet perfectly satisfies mass balance and any V_bf
    target. That is a thermodynamically infeasible futile 2-cycle -- the dominant
    artifact introduced by the reversible-reaction splitting, and what drives the
    large majority of reactions onto the FVA "loop wall".

    The penalty
    -----------
    Add a complementarity term that is zero iff at most one direction of each
    split pair is active::

        L_loop = 0.5 * lam_c * sum_over_pairs( v_f * v_r )

    Using the symmetric pair-adjacency matrix M (``M[i_f,i_r]=M[i_r,i_f]=1``,
    built once by ``_loop_adjacency``) this is a single mat-mul::

        L_loop      = 0.5 * lam_c * sum( V .* (V @ M) )
        dL_loop/dV  =       lam_c * (V @ M)

    (column ``i_f`` of ``V @ M`` holds ``v_r`` and vice-versa, so the gradient
    pushes each direction down in proportion to its partner's flux). Driving
    ``v_f * v_r`` toward 0 forces one direction of every pair to zero, so the net
    flux becomes well-defined and the futile cycles disappear.

    Strength is hard-coded to ``lam_c = 0.01`` (the complementarity loop-law
    penalty is always on).
    """
    lam_c = 0.01
    L = tf.zeros([tf.shape(V)[0], 1], dtype=tf.float32)
    dL = 0.0 * V
    if lam_c <= 0:
        return L, dL
    M = ops.convert_to_tensor(_loop_adjacency(parameter))
    VM = ops.matmul(V, M)                    # col i_f holds v_r, col i_r holds v_f
    L = L + 0.5 * lam_c * tf.reduce_sum(V * VM, axis=1, keepdims=True)
    if gradient:
        dL = dL + lam_c * VM
    return L, dL


def Loss_all(V, Vin, Vout, Vlb, parameter, gradient=False, wt=False, p_sv=1, save=False):
    if (Vout is None) or ((Vout.shape[0] is not None) and (Vout.shape[0] < 1)):
        L, dL = Loss_constraint(V, Vin, Vlb, parameter, gradient=gradient)
        return L, dL
    
    L1, dL1 = Loss_Vout_constraint(V, parameter.Pout, Vout, gradient=gradient)

    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient, save=save)
    # Apply penalty
    dL2 = dL2 * p_sv
    L2 = L2 * p_sv

    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound, gradient=gradient)
    L4, dL4 = Loss_Vpos(V, Vlb, gradient=gradient)

    # Total loss is the SUM of the component losses, matching the LaTeX:
    #   Loss_t = Loss_med + Loss_pos + Loss_sv + Loss_Vbf
    # The gradient dL is already the un-normalized sum below; reporting the
    # sum (rather than the mean) keeps the printed total loss consistent
    # with the gradient actually being optimized.
    if hasattr(parameter, 'objective') and parameter.objective and parameter.objPout is not None:
        L5, dL5 = Loss_Vout_obj(V, parameter=parameter, gradient=gradient)
        L5 = tf.math.square(L5) # Assuming obj loss wasn't rewritten, keep its square

        L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4, L5], axis=1), axis=1)
        dL = dL1 + dL2 + dL3 + dL4 + dL5
    else:
        L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
        dL = dL1 + dL2 + dL3 + dL4

    # Complementarity loop-law penalty. Always on, with lambda_c hard-coded to
    # 0.01 in Loss_loop; the BIOFLUX_LOOP_LAMBDA_C environment override this
    # comment used to describe no longer exists.
    Lloop, dLloop = Loss_loop(V, parameter, gradient=gradient)
    dL = dL + dLloop
    L = L + tf.reshape(Lloop, tf.shape(L))

    return L, dL, L1, L2, L3, L4

def custom_ReLU(V, Vout, Pout):
    # Vpos = tf.zeros(V.shape, dtype=tf.dtypes.float32)
    V = tf.keras.activations.relu(V)

    Pout_i = np.identity(Pout.shape[0])

    if not tf.is_tensor(Pout):
        Pout = tf.convert_to_tensor(np.float32(Pout))
        Pout_i = tf.convert_to_tensor(np.float32(Pout_i))

    V1 = Vout - tf.linalg.matmul(V, tf.transpose(Pout))
    # print(V1.shape)
    V1 = Vout - tf.keras.activations.relu(V1)
    # print(V1.shape)

    V2 = Pout_i - Pout
    # print(V2.shape)
    V2 = tf.linalg.matmul(V, V2)
    # print(V2.shape)
    Vpos = V1 + V2

    return Vpos

###############################################################################
# Dense model
###############################################################################

def input_ANN_Dense(parameter, verbose=False):
    # Shape X and Y depending on the model used
    if parameter.scaler != 0: # Normalize X
        parameter.X, parameter.scaler = MaxScaler(parameter.X)
    if verbose:
        print('ANN Dense scaler', parameter.scaler)
    return parameter.X, parameter.Y

def Dense_layers(inputs, parameter, trainable=True, verbose=False):
    # Build a dense architecture with some hidden layers

    activation=parameter.activation
    n_hidden=parameter.n_hidden
    dropout=parameter.dropout
    hidden_dim=parameter.hidden_dim
    output_dim=parameter.output_dim
    hidden = inputs
    n_hidden = 0 if hidden_dim == 0 else n_hidden
    for i in range(n_hidden):
        hidden = Dense(hidden_dim,
                       kernel_initializer='random_normal',
                       bias_initializer='zeros',
                       activation='relu', trainable=trainable) (hidden)
        hidden = Dropout(dropout)(hidden)
    if verbose:
        print('Dense layer n_hidden, hidden_dim, output_dim, activation, trainable:', \
              n_hidden, hidden_dim, output_dim, activation, trainable)
    outputs = Dense(output_dim,
                    kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation=activation, trainable=trainable) (hidden)
    return outputs

def ANN_Dense(parameter, trainable=True, verbose=False):
    # A standard Dense model with several layers

    input_dim, output_dim = parameter.input_dim, parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs = Dense_layers(inputs, parameter,
                           trainable=trainable, verbose=verbose)
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    loss = 'mse' if parameter.regression else 'binary_crossentropy'
    metrics = ['mae'] if parameter.regression else ['acc']
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    return parameter

###############################################################################
# AMN models (1)
# AMN_QP: a ANN_Dense trainable prior layer and a mechanistic layer
# making use of gradient descent
###############################################################################

def input_AMN(parameter, verbose=False):
    # Shape the IOs
    # IO: X and Y
    # For all
    # - add additional zero columns to Y
    #   the columns are used to minimize SV, Pin V ≤ Vin, V ≥ 0
    # For AMN_LP: add b_int or b_ext
    # For AMN_Wt repeat X timestep times

    np.savetxt(os.path.join(parameter.output_dir,"initialize","param_X.tsv"), parameter.X, delimiter='\t')
    np.savetxt(os.path.join(parameter.output_dir,"initialize","param_Y.tsv"), parameter.Y, delimiter='\t')
    # print(abc)

    X, Y = parameter.X, parameter.Y
    if parameter.scaler != 0: # Normalize X
        X, parameter.scaler = MaxScaler(X)
    if verbose: print('AMN scaler', parameter.scaler)
    y = np.zeros(Y.shape[0]).reshape(Y.shape[0],1)
    Y = np.concatenate((Y, y), axis=1) # SV constraint
    Y = np.concatenate((Y, y), axis=1) # Pin constraint
    Y = np.concatenate((Y, y), axis=1) # V ≥ 0 constraint
    if 'QP' in parameter.model_type:
        if verbose: print('QP input shape',X.shape,Y.shape)
    # elif 'RC' in parameter.model_type:
    #     if verbose: print('RC input shape',X.shape,Y.shape)
    elif 'Wt' in parameter.model_type:
        x = np.copy(X)
        num_batches = int(x.shape[0]/parameter.batch_size)
        X = np.zeros((parameter.batch_size * num_batches,
                  parameter.timestep, x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(parameter.timestep):
                X[i][j] = x[i]
        if verbose: print('Wt input shape', X.shape, Y.shape)
    else:
        print(parameter.model_type)
        sys.exit('This AMN type does not have input')
    parameter.input_dim = parameter.X.shape[1]

    return X, Y

def output_AMN(V, Vin, V0, Vlb, parameter, verbose=False):
    # Get output for all AMN models
    # output = PoutV + constaints = [SV + PinV + Relu(_V)] + V
    # where S and Pout are the stoichiometric and measurement matrix

    Pout     = tf.convert_to_tensor(np.float32(parameter.Pout))
    PoutV    = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True)
    SV, _    = Loss_SV(V, parameter.S) # SV const
    PinV, _  = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound) # Pin const
    Vpos, _  = Loss_Vpos(V, Vlb) # V ≥ 0 const

    # Return outputs = PoutV + SV + PinV + Vpos + V
    if V0 == None:
        print("output_AMN --")
        print("part1 PoutV -- ", PoutV.shape)
        print("part1 SV -- ", SV.shape)
        print("part1 PinV -- ", PinV.shape)
        print("part1 V -- ", V.shape)
        outputs = concatenate([PoutV, SV, PinV, Vpos, V], axis=1)
    else:
        outputs = concatenate([PoutV, SV, PinV, Vpos, V, V0], axis=1)
    parameter.output_dim = outputs.shape[1]
    if verbose:
        print('AMN output shapes for PoutV, SV, PinV, Vpos, V, outputs', \
              PoutV.shape, SV.shape, PinV.shape, Vpos.shape,\
              V.shape, outputs.shape)

    return outputs

def Gradient_Descent(V, Vin, Vout, Vlb, parameter, mask, trainable=True, history=False, V0_init=-1, svp=15, hardConst=0, verbose=False):
    # Input:
    # S [m x n]: stoichiometric matrix
    # V [n]: the reaction flux vector
    # Pin [n_in x n]: the flux to medium projection matrix
    # Vin [p]: the medium intake flux vector
    # V_out [n_out]: the measured fluxes (can be empty)
    # mask [n]: used to uddate dL
    # history: to specify if loss is computed and recorded
    # Output: Loss and updated V

    save = False
    # Not history here if trainable
    history = False if trainable else history

    # GD loop
    Loss_mean_history, Loss_std_history = [], []
    Loss_Data_history, Loss_Mass_history = [], []
    Dead_history, StdDev_history = [], []
    prev_V_val = None                    
    diff = 0 * V

    # --- EARLY STOPPING SETUP (per-condition) ---
    n_cond = int(V.shape[0])
    best_loss_vec  = np.full(n_cond, np.inf)
    patience_vec   = np.zeros(n_cond, dtype=int)
    frozen         = np.zeros(n_cond, dtype=bool)
    frozen_at_step = np.full(n_cond, -1, dtype=int)
    # CLAUDE 2026-08-10: both are overridable from the environment so a single
    # arm can be re-run with a different stopping rule without editing (and
    # having to remember to revert) the defaults. Unset env -> published values.
    patience_limit = int(os.environ.get("BF_PATIENCE", 500))
    min_delta      = float(os.environ.get("BF_MIN_DELTA", 1e-2))
    print(f"[EarlyStop] patience_limit={patience_limit}  min_delta={min_delta:g}")
    # ---------------------------------

    # for saving checkpoint files
    ckpt_dir = os.path.join(parameter.output_dir, "checkpoints")
    # drop stale per-step files from any prior run so the same dir
    # doesn't accumulate mixed-run files (which silently produces
    # inconsistent V row counts when a later run uses fewer conditions).
    if os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            if fn.startswith(("V_step_", "Losses_step_", "frozen_at_step")):
                os.remove(os.path.join(ckpt_dir, fn))
    os.makedirs(ckpt_dir, exist_ok=True)
    np.savetxt(os.path.join(ckpt_dir, "V_step_0.tsv"), V.numpy(), delimiter='\t')

    # --- INITIAL LOSS CALCULATION FOR STEP 0 ---
    # Perform a forward pass without calculating gradients
    L_0, dL_0, lVbf_0, Lsv_0, LVin_0, LPos_0 = Loss_all(
        V, Vin, Vout, Vlb, parameter, gradient=False, p_sv=svp, save=False
    )

    # Square the data and mass losses locally to see the driving math
    lVbf_sq_0 = tf.math.square(lVbf_0)
    Lsv_sq_0 = tf.math.square(Lsv_0)

    # Flatten tensors for TSV logging
    L_val_0 = L_0.numpy().flatten()
    lVbf_val_0 = lVbf_0.numpy().flatten()
    Lsv_val_0 = Lsv_0.numpy().flatten()
    lVbf_sq_val_0 = lVbf_sq_0.numpy().flatten()
    Lsv_sq_val_0 = Lsv_sq_0.numpy().flatten()
    
    loss_matrix_0 = np.column_stack((L_val_0, lVbf_val_0, Lsv_val_0, lVbf_sq_val_0, Lsv_sq_val_0))
    loss_file_0 = os.path.join(ckpt_dir, "Losses_step_0.tsv")
    
    np.savetxt(loss_file_0, loss_matrix_0, delimiter='\t', 
               header='Total_Loss\tData_Loss\tMass_Loss\tData_Loss_Sq\tMass_Loss_Sq', comments='')
    # -------------------------------------------

    for t in range(1, parameter.timestep+1):  # Update V with GD
        # Get Loss and gradient
        L, dL, lVbf, Lsv, LVin, LPos = Loss_all(V, Vin, Vout, Vlb, parameter, gradient=True, p_sv = svp, save=save)
        save = False

        # --- NEW: L1 REGULARIZATION FOR FREE REACTIONS ---
        lambda_l1 = 1e-4  # The strength of the squeeze
        
        if hasattr(parameter, 'M_enz_tf') and parameter.M_enz_tf is not None:
            # 1. Create a mask of the Enforced base targets (1.0 if enforced, 0.0 if free)
            is_enforced_base = tf.cast(parameter.Pout_enz_tf > 0.5, dtype=tf.float32)
            
            # 2. Map that mask back to the split reactions (v_pos and v_neg)
            is_enforced_split = tf.linalg.matmul(is_enforced_base, parameter.M_enz_tf, transpose_b=True)
            
            # 3. Invert it to target ONLY the Free reactions (0.0 if enforced, 1.0 if free)
            is_free_split = 1.0 - is_enforced_split
            
            # 4. Apply the L1 penalty by adding it directly to the gradient
            dL = dL + (lambda_l1 * is_free_split)
        else:
            # Fallback: Apply to all fluxes (standard pFBA) if masks are missing
            dL = dL + lambda_l1
        # -------------------------------------------------

        # Zero gradient rows for plateaued conditions so their V stops moving
        if frozen.any():
            freeze_mask = tf.constant((1.0 - frozen.astype(np.float32)).reshape(-1, 1))
            dL = dL * freeze_mask

        dL = tf.math.multiply(dL, mask) # Apply mask on dL

        # Update V with learn and decay rates
        diff = parameter.decay_rate * diff - parameter.learn_rate * dL
        
        ########### Gradient Descent ###############
        # V = V + diff
        ########### End Gradient Descent ###############

        ########### Projected Gradient Descent and Momentum Kill-Switch ###############
        V_unclipped = V + diff

        # Force any negative fluxes back to exactly 0.0
        V = tf.maximum(0.0, V_unclipped)

        # If the unclipped V was negative, it hit the wall. 
        # We must zero out its momentum (diff) so it doesn't keep pushing down next step.
        active_mask = tf.cast(V > 0.0, dtype=tf.float32)
        diff = diff * active_mask
        ########### End Projected Gradient Descent ###############

        # --- NEW: PROPORTIONAL CLAMP (10% BUFFER) ---
        if hasattr(parameter, 'M_enz_tf') and parameter.M_enz_tf is not None:
            # 1. Calculate current combined capacity: V_total = v_pos + v_neg
            V_total = tf.linalg.matmul(V, parameter.M_enz_tf)
            
            # 2. Define the hard ceiling (110% of Vbf)
            Limit = 1.1 * parameter.Vout_enz_tf
            
            # 3. Calculate scaling factor (Limit / V_total). If under limit, scale is 1.0.
            Scale_base = tf.where(V_total > Limit, Limit / (V_total + 1e-9), 1.0)
            
            # 4. Only apply the clamp to enforced reactions
            Scale_base = tf.where(parameter.Pout_enz_tf > 0.5, Scale_base, 1.0)
            
            # 5. Map the base scales back to the split fluxes (v_pos and v_neg get same scale)
            Scale_split = tf.linalg.matmul(Scale_base, parameter.M_enz_tf, transpose_b=True)
            
            # 6. Apply the clamp to V
            V_clamped = V * Scale_split
            
            # Kill momentum for components that got clamped down so they don't bounce
            ceiling_mask = tf.cast(V_clamped >= V - 1e-6, dtype=tf.float32)
            diff = diff * ceiling_mask
            
            V = V_clamped
        # ---------------------------------------------

        # Save the true, clean, non-negative state to checkpoints
        pre_relu_V = V 
        
        # Apply the Vbf Data Ceilings
        if(hardConst > 0):
            print(f"Warning: code is now ignoring hardConst variable! It can be re-purposed in the future.")
        # if hardConst == 1:
        #     V = tf.keras.activations.relu(V) # Redundant now, but mathematically safe
        # if hardConst == 2:
        #     V = custom_ReLU(V, Vout, parameter.Pout)

        # Save V every 100 steps
        if t % 100 == 0:
            # We use f-string to include the step number in the filename
            ckpt_file = os.path.join(ckpt_dir, f"V_step_{t}.tsv")
            np.savetxt(ckpt_file, pre_relu_V.numpy(), delimiter='\t')

            # --- NEW: SAVE PER-CONDITION LOSS SPLITS ---
            # Flatten the tensors into standard 1D numpy arrays 
            # (Assuming 57 conditions, these will be arrays of length 57)
            L_val = L.numpy().flatten()       # Total Loss
            lVbf_val = lVbf.numpy().flatten() # Data Loss (Vbf)
            Lsv_val = Lsv.numpy().flatten()   # Mass Balance Loss (S*v)
            
            # Stack them as columns: [Total_Loss, Data_Loss, Mass_Loss]
            # Resulting shape: (57 rows, 3 columns)
            loss_matrix = np.column_stack((L_val, lVbf_val, Lsv_val))
            
            # Save to a companion TSV file
            loss_file = os.path.join(ckpt_dir, f"Losses_step_{t}.tsv")
            
            # Adding a header ensures you know exactly which column is which during a posteriori analysis
            np.savetxt(loss_file, loss_matrix, delimiter='\t', 
                       header='Total_Loss\tData_Loss\tMass_Loss', comments='')
            # -------------------------------------------

        # Compile Loss history
        if history:
            Loss_mean, Loss_std = np.mean(L), np.std(L)
            lVbf_mean, Lsv_mean, LVin_mean, LPos_mean = np.mean(lVbf), np.mean(Lsv), np.mean(LVin), np.mean(LPos)

            Loss_mean_history.append(Loss_mean)
            Loss_std_history.append(Loss_std)

            Loss_Data_history.append(lVbf_mean) # Vbf fit
            Loss_Mass_history.append(Lsv_mean) # Mass Balance fit
            np.savetxt(os.path.join(parameter.output_dir,"PredEval_temp.csv"), V, delimiter=',')

            # --- NEW: Calculate Dead Fluxes and Ratio StdDev ---
            current_V_val = V.numpy().flatten()
            if prev_V_val is not None:
                active_mask = np.abs(prev_V_val) > 1e-6
                dead_count = len(prev_V_val) - np.sum(active_mask)
                
                if np.sum(active_mask) > 0:
                    ratios = current_V_val[active_mask] / prev_V_val[active_mask]
                    scaling_std = np.std(ratios)
                else:
                    scaling_std = 0.0
            else:
                dead_count = 0
                scaling_std = 0.0
                
            Dead_history.append(dead_count)
            StdDev_history.append(scaling_std)
            prev_V_val = current_V_val

            if verbose and (t/1.0e3 == int(t/1.0e3)):
                save = True
                print('QP-Loss', t, Loss_mean, Loss_std)

        # --- EARLY STOPPING CHECK (per-condition) ---
        cur = L.numpy().flatten()
        denom = np.where(np.isfinite(best_loss_vec), best_loss_vec, 1.0)
        improved = np.isinf(best_loss_vec) | (((best_loss_vec - cur) / denom) > min_delta)
        best_loss_vec = np.where(improved, np.minimum(best_loss_vec, cur), best_loss_vec)
        patience_vec  = np.where(improved, 0, patience_vec + 1)
        newly = (patience_vec >= patience_limit) & ~frozen
        if newly.any():
            newly_idx = np.where(newly)[0].tolist()
            print(f"Step {t}: freezing conditions {newly_idx} "
                  f"(losses {[float(cur[i]) for i in newly_idx]})")
            frozen_at_step[newly] = t
            frozen |= newly
        if frozen.all():
            print(f"All {n_cond} conditions plateaued at step {t}; exiting GD loop.")
            np.savetxt(os.path.join(ckpt_dir, "frozen_at_step.tsv"),
                       frozen_at_step, fmt='%d', delimiter='\t',
                       header='frozen_at_step', comments='')
            break
        # ---------------------------------

    return pre_relu_V, Loss_mean_history, Loss_std_history, Loss_Data_history, Loss_Mass_history, Dead_history, StdDev_history

def get_V0(inputs, parameter, targets, lower_bounds, trainable, V0_init=-1, verbose=False):
    # Get initial vector V0 from input and target
    # Return V0, Vin, Vout, mask
    # When target is not provided this function compute
    # the initial vector V0 using Dense_Layers

    Pin = tf.convert_to_tensor(np.float32(parameter.Pin))
    if targets.shape[0] > 0: # Initialize AMN when targets provided
        # Vin = inputs, V0 = (Pin)^T Vin
        Vin = inputs # tf.cast(tf.multiply(inputs, parameter.scaler), tf.float32)
        V0 = tf.linalg.matmul(inputs, Pin, b_is_sparse=True)
    else: # Initialize AMN when targets not provided
        # Vin = inputs, V0 = Dense_layers(inputs)
        param = copy.copy(parameter)
        param.output_dim = parameter.S.shape[1]
        param.activation = 'relu'
        Vin = inputs # tf.cast(tf.multiply(inputs, parameter.scaler), tf.float32)
        V0 = Dense_layers(inputs, param,
                          trainable=trainable, verbose=verbose)

    # Get a mask for EB and UB where elements in Vin are not updated in V
    ones = np.ones(parameter.S.shape[1])
    ones = tf.convert_to_tensor(np.float32(ones))
    # mask = np.matmul(np.ones(Vin.shape[1]), Pin)
    mask = tf.linalg.matvec(Pin, tf.ones([Vin.shape[1]]), transpose_a=True)
    # element in Vin are at 0 in mask others are at 1
    mask = ones - mask

    # Vin projection in V: elements not in Vin are at 0
    VinV  = tf.linalg.matmul(Vin, Pin, b_is_sparse=True)
    if parameter.mediumbound == 'UB': # we must have V ≤ Vin
        # relu = 1 when VinV > V, 0 othervise
        relu = tf.keras.activations.relu(VinV-V0)
        relu = tf.math.divide_no_nan(relu, relu) # 0/1 tensor
        # VinV = V when V < Vin, VinV = Vin when V > Vin
        VinV = relu * V0 + (ones-relu) * VinV
    V0 = tf.math.multiply(V0, mask) + VinV
    
    if V0_init < 0:
        # make intial flux equal to Vbf or avg Vbf
        # if ('vbf' in parameter.method.lower()):
        Vout = tf.convert_to_tensor(np.float32(targets))

        # Add Vout to the internal reactions (mask == 1)
        # Leave the exchange reactions (mask == 0) at their Vin values
        V0 = V0 + tf.math.multiply(Vout, mask)

        # set a threshold of the lowest Vbf to replace by the average
        threshold = 1e-6

    if V0_init == -2:
        # ==========================================================
        # --- EVIDENCE-ONLY INITIALIZATION (test mode) ---
        # ==========================================================
        # Reactions with a Vbf start at Vbf (set immediately above).
        # Reactions WITHOUT a Vbf -- no reaction score, hence no Kapp, hence
        # no row in Pout -- start at zero rather than at true_vbf_mean/2, and
        # are recruited only where the mass-balance term demands it.
        # The media layer is handled separately by the chain table applied
        # after the exchange warm start below.
        print("[Init] V0_init=-2: unscored reactions left at 0 "
              "(no mean/2 imputation)")

    elif V0_init < 0:

        # ==========================================================
        # --- BASE-REACTION CAPACITY-AWARE INITIALIZATION ---
        # ==========================================================
        # Group the target fluxes by their Base Reaction ID to avoid double-counting FVA artifacts
        base_caps = {}
        targets_np = targets  

        for i, name in enumerate(parameter.reactions):
            clean_name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
            # Strip directional suffixes to get the core enzyme ID
            base_id = clean_name.replace('_f','').replace('_r','').replace('_o','').replace('_i','').strip()
            
            if base_id not in base_caps:
                base_caps[base_id] = np.zeros(targets_np.shape[0])
            
            # Take the MAX of the directional tracks
            # If FVA says _f=1.9 and _r=1.9, the true total capacity is just 1.9!
            base_caps[base_id] = np.maximum(base_caps[base_id], targets_np[:, i])
            
        # Calculate the mean of ONLY the active, un-duplicated base capacities
        all_caps = np.array(list(base_caps.values())).flatten()
        active_caps = all_caps[all_caps > threshold]
        true_vbf_mean = np.mean(active_caps) if len(active_caps) > 0 else 0.0
        
        print(f"Computed True Base-Level Vbf Mean: {true_vbf_mean:.4f}")

        # Use Pout to identify unmeasured reactions so we don't overwrite real biological zeros
        # Pout shape is (reactions, targets). Summing across axis 1 reveals if a reaction is mapped
        pout_sum = np.sum(np.abs(parameter.Pout), axis=1)
        is_unmapped = pout_sum == 0

        # Convert to a tensor that broadcasts across all conditions
        is_unmapped_tensor = tf.convert_to_tensor(is_unmapped, dtype=tf.bool)
        is_unmapped_tensor = tf.expand_dims(is_unmapped_tensor, axis=0) 
        
        # Identify slots that are BOTH empty (< threshold) AND unmapped in Pout
        needs_imputation = tf.logical_and(V0 < threshold, is_unmapped_tensor)

        # Apply Half-Capacity to empty unmapped tracks ONLY to establish Vbf as a single value for net flux
        Vbf_half_capacity = true_vbf_mean / 2.0
        V0 = tf.where(needs_imputation, Vbf_half_capacity, V0)
        # ==========================================================

        # Replace all values in V0 less than the threshold with Vbf_mean
        # V0 = tf.where(V0 < threshold, Vbf_mean, V0)
        # # for exact match, use this: 
        # V0 = tf.where(tf.equal(V0, 0), tf.constant(Vbf_mean, dtype=V0.dtype), V0)
    
    if V0_init > 0: 
        # set everything to 1000\Vbf_mean
        V0 = tf.where(V0 >= 0, V0_init, V0)

    Vlb = tf.convert_to_tensor(np.float32(lower_bounds))
    mask = ones if parameter.mediumbound == 'UB' else mask

    # ======================================================================
    # --- DYNAMIC FVA WARM START INJECTION ---
    # ======================================================================
    # Check if your custom exchanges dictionary exists and is not empty
    if hasattr(parameter, 'exchanges') and parameter.exchanges is not None:
        print(f"\n[Init] Injecting {len(parameter.exchanges)} exchange flux ceilings into V0...")
        
        # 1. Convert V0 from a TensorFlow tensor to a numpy array for easy assignment
        V0_np = V0.numpy()
        
        # 2. Build a quick lookup dictionary to find the column index for each reaction
        col_map = {}
        for i, name in enumerate(parameter.reactions):
            # Safely handle both byte strings (from NPZ) and normal strings
            clean_name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
            col_map[clean_name.strip()] = i
            
        # 3. Iterate through your parsed dictionary and overwrite the columns
        replaced_count = 0
        for rxn_id, flux_max in parameter.exchanges.items():
            if rxn_id in col_map:
                col_idx = col_map[rxn_id]
                # Broadcasts the max flux perfectly across all 57 condition rows
                V0_np[:, col_idx] = flux_max  
                replaced_count += 1
                
        print(f"-> Successfully replaced {replaced_count} matrix columns.\n")

        # 4. Cast the array back into a TensorFlow Tensor so the GD loop can use it
        V0 = tf.convert_to_tensor(V0_np, dtype=tf.float32)
    # ======================================================================

    # ======================================================================
    # --- MEDIA CHAIN INITIALIZATION (V0_init == -2) ---
    # ======================================================================
    # Each media compound moves through a series chain: exchange -> e0/c0
    # transporter -> c0/d0 transporter. At steady state all three carry the
    # same flux, so all three are initialized to the same value (the exchange's
    # net FVA capacity) on the column matching the exchange's direction, with
    # the opposing column held at 0. Generated by make_media_chain_init.py.
    if V0_init == -2:
        # output_dir is <project>/ml[/test]/svp_X.X/ -- walk up to the project
        # folder to find integration_results/ alongside ml/
        chain_file, probe = None, os.path.abspath(parameter.output_dir)
        while probe != os.path.dirname(probe):
            candidate = os.path.join(probe, "integration_results", "media_chain_init.tsv")
            if os.path.exists(candidate):
                chain_file = candidate
                break
            probe = os.path.dirname(probe)
        if chain_file is None:
            raise FileNotFoundError(
                "V0_init=-2 requires integration_results/media_chain_init.tsv "
                f"at or above {parameter.output_dir}; run make_media_chain_init.py first")

        col_map = {}
        for i, name in enumerate(parameter.reactions):
            clean_name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
            col_map[clean_name.strip()] = i

        V0_np = V0.numpy()
        applied, missing = 0, []
        with open(chain_file) as fh:
            fh.readline()
            for line in fh:
                rxn_id, value = line.rstrip('\n').split('\t')[:2]
                if rxn_id in col_map:
                    V0_np[:, col_map[rxn_id]] = float(value)
                    applied += 1
                else:
                    missing.append(rxn_id)
        V0 = tf.convert_to_tensor(V0_np, dtype=tf.float32)
        print(f"[Init] media chain: set {applied} columns from {os.path.basename(chain_file)}"
              + (f"; {len(missing)} not in model: {missing}" if missing else ""))
    # ======================================================================

    # ======================================================================
    # --- 2. INDEPENDENT BIO1 OVERRIDE (MANDATORY) ---
    # ======================================================================
    # Always force 'bio1' to 0.0, wiping out Vbf_mean or any other init
    V0_np = V0.numpy() # Grab current state (includes FVA if it ran)
    bio_idx = -1
    
    for i, name in enumerate(parameter.reactions):
        clean_name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
        if clean_name.strip() == 'bio1':
            bio_idx = i
            break
            
    if bio_idx != -1:
        V0_np[:, bio_idx] = 0.0
        print("-> [Init] Successfully forced 'bio1' initialization to 0.0")
        V0 = tf.convert_to_tensor(V0_np, dtype=tf.float32)
    # ======================================================================

    np.savetxt(os.path.join(parameter.output_dir,"initialize","V0_startVbf.csv"), V0, delimiter=',')

    return V0, Vin, Vout, Vlb, mask

def QP_layers(inputs, parameter, targets = np.asarray([]).reshape(0,0), lower_bounds=np.asarray([]).reshape(0,0), trainable=True, history=False, V0_init=-1, svp=15, hardConst=0, verbose=False):
    # Build and return an architecture using GD
    # The function is used with and without targets
    # - With targets there is no training set and GD is run
    #   to optimize both the objective min([PV-Target]^2))
    #   and the constraints.
    # - Without target an initial vector V is calculated via training
    #   through a Dense layer, GD is only used
    #   to minimize the constrains
    # Inputs:
    # - input flux vector, targets (can be empty)
    # - flags to train, record Loss history
    # Outputs:
    # - ouput_AMN (see function, and Loss (mean and std)

    V0, Vin, Vout, Vlb, mask = get_V0(inputs, parameter, targets, lower_bounds, trainable, V0_init=V0_init, verbose=verbose)

    V, Loss_mean, Loss_std, Loss_Data, Loss_Mass, Dead_H, StdDev_H = Gradient_Descent(V0, Vin, Vout, Vlb, parameter, mask, trainable=trainable, history=history, V0_init=V0_init, svp=svp, hardConst=hardConst, verbose=verbose)
    outputs = output_AMN(V, Vin, V0, Vlb, parameter, verbose=verbose)

    return outputs, Loss_mean, Loss_std, Loss_Data, Loss_Mass, Dead_H, StdDev_H

def AMN_QP(parameter, trainable=True, verbose=False):
    # Build and return an AMN with training
    # input : problem parameter
    # output: Trainable model
    # Loss history is not recorded (already done thru tf training)

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[1], parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs, loss_h, loss_std_h, _, _, _, _ = QP_layers(inputs, parameter,
                              trainable=trainable,
                              history=False,
                              verbose=verbose)
    # Compile
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    (loss, metrics) = (my_mse, [my_r2])
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    return parameter


###############################################################################
# AMN models (3)
# AMN_Wt: An RNN where input (the medium) and flux vector V are passed
# to the recurrent cell
# M = V2M . V
# V = Win x Vin + Wrec x M2V . M
# Win and Wrec are weight matrices learned during training
# A hidden layer can be added to Win (not Wrec)
# Warning: The model AMN_Wt works only with UB training sets
###############################################################################

class RNNCell(keras.layers.Layer): # RNN Cell, as a layer subclass.
    def __init__(self, parameter):
        meta_dim = parameter.S.shape[0]
        flux_dim = parameter.S.shape[1]
        medm_dim = parameter.Pin.shape[0]
        self.input_size = medm_dim
        self.state_size = flux_dim
        self.mediumbound = parameter.mediumbound
        self.hidden_dim = parameter.hidden_dim
        self.S  = tf.convert_to_tensor(np.float32(parameter.S))
        self.V2M = tf.convert_to_tensor(np.float32(parameter.V2M))
        self.Pin = tf.convert_to_tensor(np.float32(parameter.Pin))
        # Normalize M2V
        M2V = parameter.M2V
        for i in range(flux_dim):
            if np.count_nonzero(M2V[i]) > 0:
                M2V[i] = M2V[i] / np.count_nonzero(M2V[i])
        self.M2V  = tf.convert_to_tensor(np.float32(M2V))
        self.dropout = parameter.dropout
        super(RNNCell, self).__init__(trainable=True)

    def build(self, input_shape):
        meta_dim = self.S.shape[0]
        flux_dim = self.S.shape[1]
        medm_dim = self.input_size
        hidden_dim = self.hidden_dim
        print("meta_dim:   ", meta_dim)
        print("flux_dim:   ", flux_dim)
        print("medm_dim:   ", medm_dim)
        print("hidden_dim: ", hidden_dim)
        # weigths to compute V for both input (i) and recurrent cell (r)
        if self.mediumbound == 'UB': # no kernel_Vh and kernel_Vi for EB
            if hidden_dim > 0: # plug an hidden layer upstream of Winput
                self.wh_V = self.add_weight(shape=(medm_dim, hidden_dim),
                                        name='kernel_Vh')
                self.wi_V = self.add_weight(shape=(hidden_dim, medm_dim),
                                        name='kernel_Vi')
            else:
                self.wi_V = self.add_weight(shape=(medm_dim, medm_dim),
                                        name='kernel_Vi')
        self.wr_V = self.add_weight(shape=(flux_dim, meta_dim),
                                        name='kernel_Vr')
        self.bi_V  = self.add_weight(shape=(medm_dim,),
                                        initializer='random_normal',
                                        name='bias_Vi',
                                        trainable=True)
        self.br_V  = self.add_weight(shape=(flux_dim,),
                                        initializer='random_normal',
                                        name='bias_Vr',
                                        trainable=True)
        self.built = True

    def call(self, inputs, states):
        # At steady state we have
        # M = V2M V and V = (M2V x W) M + V0
        # Keep Vin only
        # inputs = CROP(1, 0, self.input_size)(inputs)

        V = states[0]
        # print("RNN states -> ", states[0])
        if self.mediumbound == 'UB':
            if self.hidden_dim > 0:
                VH = K.dot(inputs, self.wh_V)
                V0 = K.dot(VH, self.wi_V) + self.bi_V
            else:
                V0 = K.dot(inputs, self.wi_V) + self.bi_V
        else:
            V0 = inputs # EB case
        V0 = tf.linalg.matmul(V0, self.Pin, b_is_sparse=True)
        M = tf.linalg.matmul(V,tf.transpose(self.V2M),b_is_sparse=True)
        W = tf.math.multiply(self.M2V,self.wr_V)
        V = tf.linalg.matmul(M,tf.transpose(W),b_is_sparse=True)
        V = V + V0 + self.br_V

        return V, [V]

    def get_config(self): # override tf.get_config to save RNN model
        # The code below does not work !! anyone to debug?
        config = super().get_config().copy()
        #config.update({'parameter': self.parameter.__dict__})
        return config

def split_inputs(inputs, parameter):
    d0, d1 = \
        inputs[:,0,:].shape[0], inputs[:,0,:].shape[1]
    if d1 > parameter.input_dim:
        Vin = CROP(1, 0, parameter.input_dim)(inputs[:,0,:])
        Vbf = CROP(1, parameter.input_dim, d1)(inputs[:,0,:])
        print('Vlb ', Vbf.shape)
        # print(abc)
    else:
        Vin = inputs[:,0,:]
        Vbf = inputs[:,0,:]

    return Vin, Vbf

def Wt_layers(inputs, parameter, trainable=True, verbose=False):
    # Build and return AMN layers using an RNN cell
    with CustomObjectScope({'RNNCell': RNNCell}):
        rnn = keras.layers.RNN(RNNCell(parameter))

    Vin, Vbf = split_inputs(inputs, parameter)

    rnn_input = CROP(2, 0, parameter.input_dim)(inputs)
    V = rnn(rnn_input)
    # Vin = inputs[:,0,:]
    print("inputs Wt_layers ", inputs.shape)
    print("inputs 0 Wt_layers ", inputs[:,0,:].shape)

    return output_AMN(V, Vin, None, None, parameter, verbose=verbose)

def AMN_Wt(parameter, verbose=False):
    # Build and return an AMN using an RNN cell
    # input : medium vector in parameter
    # output: experimental steaty state fluxes

    # Get dimensions and build model
    input_dim, output_dim  = parameter.X.shape[2], parameter.Y.shape[1]
    inputs = keras.Input((None, input_dim))
    outputs = Wt_layers(inputs, parameter)


    # Compile
    model = keras.models.Model(inputs, outputs)
    # (loss, metrics) = (my_mse, [my_r2])
    Vin, Vbf = split_inputs(inputs, parameter)

    # ## use lambda layer loss
    # loss_model = generate_loss_model(parameter, Vin, Vbf)
    # # Create custom loss instance
    # custom_loss = CustomLoss(loss_model, parameter)
    # loss, metrics = custom_loss, None

    # use custom loss function
    loss, metrics = biochem_loss(parameter), None

    # Compile the model with the custom loss function
    model.compile(loss=loss,  optimizer='adam', metrics=metrics)#, run_eagerly=True)
    print(model.summary())
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    return parameter

###############################################################################
# Non-trainable Mechanistic Model (MM)
# using QP 
###############################################################################

def write_loss(f_name, param, mean_history, std_history, data_history, mass_history, dead_history, stddev_history):
    if f_name is None:
        return 0
    timesteps = np.arange(1, len(mean_history) + 1)
    losses = np.array(mean_history)
    stdevs = np.array(std_history)
    data_loss = np.array(data_history)
    mass_loss = np.array(mass_history)
    dead_arr = np.array(dead_history)
    stddev_arr = np.array(stddev_history)
    
    to_write = np.concatenate([
        timesteps.reshape((len(timesteps), 1)), 
        losses.reshape((len(losses), 1)), 
        stdevs.reshape((len(stdevs), 1)),
        data_loss.reshape((len(data_loss), 1)),
        mass_loss.reshape((len(mass_loss), 1)),
        dead_arr.reshape((len(dead_arr), 1)),
        stddev_arr.reshape((len(stddev_arr), 1))
    ], axis=1)
    
    header = "Timestep,Total_Loss,Std_Dev,Data_Loss_Vbf,Mass_Loss_SV,Dead_Fluxes,Ratio_StdDev"
    np.savetxt(f_name, to_write, delimiter=',', header=header)
    return 0

def write_targets(f_name, param, Ypred):
    if f_name is None:
        return 0
    true = np.array(param.Y)
    pred = np.array(Ypred)
    to_write = np.concatenate([true.reshape((len(true), 1)), \
        pred.reshape((len(pred), 1))], axis=1)
    np.savetxt(f_name, to_write, delimiter=',')
    return 0

def get_flux_output(param, output):
    # Just getting vector V from output
    # output : PoutV (=Ypred) + SV + PinV + Vpos + V + V0
    len_fluxes = param.S.shape[1]
    if output.shape[1] > (len_fluxes+NBR_CONSTRAINT+1): # case where we get V0
        V0 = CROP(1,param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes,
                    param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes*2) (output)
        Vf = CROP(1,param.Y.shape[1]+NBR_CONSTRAINT,
                    param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes) (output)
    else: # case where we don't have V0 at the end of the output
        Vf = CROP(1,param.Y.shape[1]+NBR_CONSTRAINT,
                    param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes) (output)
    return Vf

def run_MM_QP(parameter, loss_outfile=None, targets_outfile=None, history=True, V0_init=-1, svp=15, hardConst=0, verbose=False):
    # Solve LP or QP without training
    # inputs:
    # - problem parameter, history flag
    # output:
    # - Predicted all fluxes and stats = loss history

    # inputs must be in tf format
    param = copy.copy(parameter)
    if param.X.shape[1] < param.S.shape[1]:
        # when all X provided no need to tranform
        param.X, _ = input_AMN(param, verbose=False)
    inputs  = tf.convert_to_tensor(np.float32(param.X))
    targets = param.Y
    lower_bounds = param.LB

    # run QP
    output, Loss_mean, Loss_std, Loss_Data, Loss_Mass, Dead_H, StdDev_H = QP_layers(inputs, param, targets=targets,
                    lower_bounds=lower_bounds, trainable=False, history=history, 
                    V0_init=V0_init, svp=svp, hardConst=hardConst, verbose=verbose)
    Ypred = CROP(1,0,param.Y.shape[1]) (output)
    Vf = get_flux_output(param, output)

    # compute R2 and write losses and targets
    r2 = r2_score(param.Y, Ypred.numpy(), multioutput='variance_weighted')
    write_loss(loss_outfile, parameter, Loss_mean, Loss_std, Loss_Data, Loss_Mass, Dead_H, StdDev_H)
    write_targets(None, parameter, Ypred)

    return Vf.numpy(), ReturnStats(r2, 0, Loss_mean[-1], Loss_std[-1],
                                   0, 0, 0, 0)

def MM_QP(parameter, loss_outfile=None, targets_outfile= None, history=True, V0_init=-1, svp=15, hardConst=0, verbose=False):
    # Solve QP without training
    return run_MM_QP(parameter, loss_outfile=loss_outfile, targets_outfile=targets_outfile, history=history, V0_init=V0_init, svp=svp, hardConst=hardConst, verbose=verbose)


###############################################################################
# Train and Evaluate all models
###############################################################################

class ReturnStats:
    def __init__(self, v1, v2, v3, v4, v5, v6, v7, v8):
        self.train_objective = (v1, v2)
        self.train_loss = (v3, v4)
        self.test_objective = (v5, v6)
        self.test_loss = (v7, v8)

def print_loss_evaluate(y_true, y_pred, Vin, Vlb, parameter):
    # Print all losses
    loss_out0, loss_outf = -1, -1
    loss_cst0, loss_cstf = -1, -1
    loss_all0, loss_allf = -1, -1
    end = len(parameter.reactions) # y_true.shape[1] - NBR_CONSTRAINT
    nV = parameter.S.shape[1]
    pred_index = len(parameter.reactions) + NBR_CONSTRAINT
    # print("print_loss_evaluate : y_predShape ", y_pred.shape)
    # print("print_loss_evaluate : y_trueShape ", y_true.shape)
    # print("print_loss_evaluate : y_true.shape[1] ", y_true.shape[1])
    # print("print_loss_evaluate : nV ", nV)

    # Vf = y_pred[:,y_true.shape[1]:y_true.shape[1]+nV]
    Vf = y_pred[:,pred_index:pred_index+nV]
    if y_true.shape[1] <= pred_index:
        Vout = y_true[:,:end]
    else:
        Vout = y_true[:,parameter.input_dim:end+parameter.input_dim]
        # print("print_loss_evaluate : from ", parameter.input_dim, " to ", end+parameter.input_dim)
        # print("print_loss_evaluate : Vout ", Vout.shape)

    if y_pred.shape[1] == y_true.shape[1]+nV+nV:
        V0 = y_pred[:,y_true.shape[1]+nV:y_true.shape[1]+nV+nV]
        loss_out0, _ = Loss_Vout(V0, parameter.Pout, Vout)
        loss_out0 = np.mean(loss_out0.numpy())
        loss_cst0, _ = Loss_constraint(V0, Vin, Vlb, parameter)
        loss_cst0 = np.mean(loss_cst0.numpy())
        loss_all0, _, _, _, _, _ = Loss_all(V0, Vin, Vout, Vlb, parameter)
        loss_all0 = np.mean(loss_all0.numpy())
    loss_outf, _ = Loss_Vout(Vf, parameter.Pout, Vout)
    loss_outf = np.mean(loss_outf.numpy())
    loss_cstf, _ = Loss_constraint(Vf, Vin, Vlb, parameter)
    loss_cstf = np.mean(loss_cstf.numpy())
    loss_allf, _, _, _, _, _ = Loss_all(Vf, Vin, Vout, Vlb, parameter)
    loss_allf = np.mean(loss_allf.numpy())
    print('Loss out on V0: ', loss_out0)
    print('Loss constraint on V0: ', loss_cst0)
    print('Loss all on V0: ', loss_all0)
    print('Loss out on Vf: ', loss_outf)
    print('Loss constraint on Vf: ', loss_cstf)
    print('Loss all on Vf: ',  loss_allf)
    if y_pred.shape[1] == y_true.shape[1]+nV+nV:
        d = np.linalg.norm(Vf - V0)
        print('Distance V0 to Vf %f: ' % (d))
    return

def get_loss_evaluate(x, y_true, y_pred, parameter, verbose=False):
    # Return loss on constraint for y_pred
    end = y_true.shape[1]
    pred_index = y_true.shape[1] - parameter.input_dim
    if 'AMN' in parameter.model_type:
        nV = parameter.S.shape[1]
        Vf = y_pred[:,pred_index:pred_index+nV]
        # if 'AMN_LP' in parameter.model_type:
        #     # x = Vin + bounds is truncated
        #     Vin = x[:,0: parameter.Pin.shape[0]]
        # el
        if 'AMN_Wt' in parameter.model_type:
            # The dimension (time) added to x with RNN is removed
            Vin  = x[:,0,0:parameter.Pin.shape[0]]
            Vlb = Vin.copy()
            if Vin.shape[1] > parameter.Pin.shape[0]:
                Vlb  = x[:,0,parameter.Pin.shape[0]:]
        else:
            Vin = x

        if verbose:
            print_loss_evaluate(y_true, y_pred, Vin, Vlb, parameter)
        loss, _ = Loss_constraint(Vf, Vin, Vlb, parameter)
        loss = np.mean(loss.numpy())
    else:
        loss = -1

    return loss

def evaluate_model(model, x, y_true, parameter, inputmodel= False, verbose=False):
    # Return y_pred, stats (R2/Acc) for objective
    # and error on constraints for regression and classification
    # if input model than x, y_true sent to input model

    if inputmodel:
        param = copy.copy(parameter)
        param.X, param.Y = x, y_true
        X, Y = model_input(param, verbose=verbose)
        param.X, param.Y = X, Y
    y_pred = model.predict(x) # whole y prediction

    # AMN models have NBR_CONSTRAINT constraints added to y_true
    end = y_true.shape[1] - NBR_CONSTRAINT \
    if 'AMN' in parameter.model_type else y_true.shape[1]
    print('----------', end)
    if parameter.regression:
        yt, yp = y_true[:,:end], y_pred[:,:end]
        if yt.shape[0] == 1: # LOO case
            rss, tss = (yp - yt) * (yp - yt), yt * yt
            if np.sum(tss) > 0:
                obj = 1 - np.sum(rss) / np.sum(tss)
            else:
                obj = 1 - np.sum(rss)
            print('LOO True, Pred, Q2 =', yt, yp, obj)
        else:
            obj = r2_score(yt, yp, multioutput='variance_weighted')
    else:
        end = y_true.shape[1]
        obj = keras.metrics.binary_accuracy(y_true[:,:end],
                                            y_pred[:,:end]).numpy()
        #print('evaluate acc \ny_true', y_true[:,:end], '\ny_pred', y_pred[:,:end])
        obj = np.count_nonzero(obj)/obj.shape[0]

    # compute stats on constraints
    loss = get_loss_evaluate(x, y_true, y_pred, parameter, verbose=verbose)
    stats  = ReturnStats(obj, 0, loss,  0, obj, 0, loss,  0)

    return y_pred, stats

def model_input(parameter, trainable=True, verbose=False):
    # return input for the appropriate model_type
    if   'ANN' in parameter.model_type:
        return input_ANN_Dense(parameter, verbose=verbose)
    elif 'AMN' in parameter.model_type:
        # parameter.batch_size = 10
        return input_AMN(parameter, verbose=verbose)
    elif 'MM' in parameter.model_type:
        return input_AMN(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit('no input available')

def model_type(parameter, verbose=False):
    # create the appropriate model_type
    if verbose:
        print('-----------------------------------', parameter.model_type)
    if 'ANN_Dense' in parameter.model_type:
        return ANN_Dense(parameter, verbose=verbose)
    elif 'AMN_QP' in parameter.model_type:
        return AMN_QP(parameter, verbose=verbose)
    elif 'AMN_Wt' in parameter.model_type:
        return AMN_Wt(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit('not a trainable model')

def train_model(parameter, Xtrain, Ytrain, Xtest, Ytest, verbose=False):
    # A standard function to create a model, fit, and test
    # with early stopping
    # Inptuts:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.regression (boolean) if false classification
    # Outputs:
    # - Net: the trained network
    # - ytrain, ytest: y values for training and tets sets
    # - otrain, ltrain: objective and loss for trainig set
    # - otest, ltest: objective and loss for trainig set
    # - history: tf fit histrory
    # Must have verbose=2 to verbose the fit

    Niter = 1 # maximum number of attempts to fit

    # Create model fit and evaluate
    for kiter in range(Niter): # Looping until properly trained
        if 'AMN_Wt' in parameter.model_type:
            # we have to recreate the object model with AMN-Wt
            # parameter.batch_size = 10
            model = Neural_Model(trainingfile = parameter.trainingfile,
            objective= parameter.objective, model=parameter.model,
            model_type=parameter.model_type, scaler=parameter.scaler,
            input_dim=parameter.input_dim, output_dim=parameter.output_dim,
            n_hidden=parameter.n_hidden, hidden_dim=parameter.hidden_dim,
            activation=parameter.activation, timestep=parameter.timestep,
            learn_rate=parameter.learn_rate, decay_rate=parameter.decay_rate,
            regression=parameter.regression, epochs=parameter.epochs,
            train_rate=parameter.train_rate, dropout=parameter.dropout,
            batch_size=parameter.batch_size, niter=parameter.niter,
            xfold=parameter.xfold,  es=parameter.es, verbose=verbose)
            model.X, model.Y = Xtrain, Ytrain
        else:
            model = parameter
        Net = model_type(model, verbose=verbose)
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min',
                           patience=10, verbose=verbose)
        callbacks = [es] if model.es else []
        # fit
        v = True if verbose == 2 else False
        epochs = 0.9 * model.epochs
        print('Xtrain -> ', Xtrain.shape)
        print('Ytrain -> ', Ytrain.shape)
        print('Xtest  -> ', Xtest.shape)
        print('Ytest  -> ', Ytest.shape)
        print('epochs -> ', model.epochs)
        print('batch  -> ', model.batch_size)
        # csv_logger = CSVLogger('log.csv', append=True, separator=';')
        # save_freq
        # run = wandb.init(config={"learning_rate": parameter.learn_rate,
        #                     "decay_rate": parameter.decay_rate,
        #                     "architecture": "AMM_wt",
        #                     "dataset": "secMet",
        #                     "epochs": parameter.timestep,
        #                     "bs": parameter.batch_size
        #                     })
        # os.environ["WANDB_SILENT"] = "true"

        history = Net.model.fit(Xtrain, Ytrain,
                                validation_data=(Xtest, Ytest),
                                epochs=model.epochs,
                                batch_size=model.batch_size,
                                # callbacks=callbacks+[WandbMetricsLogger(),
                                #                     WandbModelCheckpoint(filepath="models/"),]
                                #                     # csv_logger,],
                                verbose=2)
                                # , run_eagerly=True)
        # evaluate
        ytrain, stats = evaluate_model(Net.model, Xtrain, Ytrain,
                                       model, verbose=verbose)
        otrain, ltrain = stats.train_objective[0], stats.train_loss[0]
        if otrain > 0.5:
            break
        else:
            print('looping bad training iter=%d r2=%.4f' % (kiter, otrain))

    # Hopefullly fit is > 0.5 now evaluate test set
    ytest, stats  = evaluate_model(Net.model, Xtest,  Ytest,
                                   model, verbose=verbose)
    otest, ltest = stats.test_objective[0], stats.test_loss[0]

    print("train = %.2f test = %.2f loss-train = %6f loss-test = %.6f iter=%d" % \
          (otrain, otest, ltrain, ltest, kiter))

    # Close the W&B run
    # run.finish()
    return Net, ytrain, ytest, otrain, ltrain, otest, ltest, history

def train_evaluate_model(parameter, verbose=False):
    # A standard function to create a model, fit, and Kflod cross validate
    # with early stopping
    # Kfold is performed for param.xfold test sets (if param.niter = 0)
    # otherwise only for niter test sets
    # Inptuts:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.regression (boolean) if false classification
    # Outputs:
    # - the best model (highest Q2/Acc on kfold test sets)
    # - the values predicted for each fold (if param.niter = 0)
    #   or the whole set when (param.niter > 0)
    # - the mean R2/Acc on the test sets
    # - the mean constraint value on the test sets
    # Must have verbose=True to verbose the fit

    param = copy.copy(parameter)
    X, Y = model_input(param, verbose=verbose)
    param.X, param.Y = X, Y
    # print('****** ', X[0])

    # Train on all data
    if param.xfold < 2: # no cross-validation
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param, X, Y, X, Y, verbose=verbose)
        # Return Stats
        stats = ReturnStats(otrain, 0, ltrain, 0, otest, 0, ltest, 0)

        temp_pred = Net.model.predict(X)
        np.savetxt(os.path.join(parameter.output_dir,"final_PredEval.csv"), temp_pred, delimiter=',')

        return Net, ytrain, stats, history

    # Cross-validation loop
    Otrain, Otest, Ltrain, Ltest, Omax, Netmax, Ypred = \
    [], [], [], [], -1.0e32, None, np.copy(Y)
    kfold = KFold(n_splits=param.xfold, shuffle=True)
    kiter = 0
    for train, test in kfold.split(X, Y):
        if verbose: print('------- train', X[train].shape, Y[train].shape)
        if verbose: print('------- test ', X[test].shape, Y[test].shape)
        # print()
        # print(CROP(2, 4, X[train].shape[2])(X[train])[0])
        # print(abc)
        # parameter.input_dim
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param, X[train], Y[train], X[test], Y[test], verbose=verbose)
        # compile Objective (O) and Constraint (C) for train and test
        Otrain.append(otrain)
        Otest.append(otest)
        Ltrain.append(ltrain)
        Ltest.append(ltest)
        # in case y does not have the same shape than Y
        if Ypred.shape[1] != ytest.shape[1]:
            n, m = Y.shape[0], ytest.shape[1]
            Ypred = np.zeros(n*m).reshape(n,m)
        for i in range(len(test)):
            Ypred[test[i]] = ytest[i]
        # Get the best network
        (Omax, Netmax) = (otest, Net) if otest > Omax else (Omax, Netmax)
        kiter += 1
        if (param.niter > 0 and kiter >= param.niter) or kiter >= param.xfold:
                break

    # Prediction using best model on whole dataset
    Pred, _ = evaluate_model(Netmax.model, X, Y, param, verbose=verbose)
    Ypred = Pred if param.niter > 0 else Ypred

    # np.savetxt(os.path.join(parameter.output_dir,"final_Ypred.csv"), Ypred, delimiter=',')
    # np.savetxt(os.path.join(parameter.output_dir,"final_PredEval.csv"), Pred, delimiter=',')

    # Get Stats
    stats = ReturnStats(np.mean(Otrain), np.std(Otrain),
                        np.mean(Ltrain), np.std(Ltrain),
                        np.mean(Otest),  np.std(Otest),
                        np.mean(Ltest),  np.std(Ltest))

    return Netmax, Ypred, stats, history

class Neural_Model:
    # To save, load & print all kinds of models including reservoirs
    def __init__(self,
                 trainingfile=None, # training set parameter file
                 objective=None,
                 model=None, # the actual Keras model
                 model_type='', # the function called Dense, AMN, RC...
                 scaler=False, # X is not scaled by default
                 input_dim=0, output_dim=0, # model IO dimensions
                 n_hidden=0, hidden_dim=0, # default no hidden layer
                 activation='relu', # activation for last layer
                 timestep=0, learn_rate=1.0, decay_rate=0.9,# for GD in AMN
                 # for all trainable models adam default learning rate = 1e-3
                 regression=True,
                 epochs=0, train_rate=1e-3, dropout=0.25, batch_size=100,
                 niter=0, xfold=5, # Cross valisation LOO does not work
                 es=False, # early stopping
                 biomass_max=4.0,
                 output_dir = ".",
                 exchanges = None, # for initialization of exchange reactions
                 verbose=False,
                ):
        # Create empty object
        if model_type == '':
            return
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            return

        # model architecture parameters
        self.trainingfile = trainingfile
        self.model = model
        self.model_type = model_type

        if objective is not None and len(objective) > 1:
            self.objective = [objective[1]] if objective[0] else []
            target_rxn_id = objective[1]
        else:
            self.objective = []
            target_rxn_id = None

        if target_rxn_id is not None and target_rxn_id in self.reactions:
            self.bio_id = self.reactions.tolist().index(target_rxn_id)
        else:
            # Fallback if no objective provided or reaction not found
            self.bio_id = 0 
            if verbose and target_rxn_id:
                print(f"Warning: Objective '{target_rxn_id}' not found in reactions.")

        self.scaler = float(scaler) # From bool to float
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        # LP or QP parameters
        self.timestep = timestep
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate

        if(exchanges is not None):
            self.exchanges = exchanges
            print("Exchanges: ",self.exchanges)

        # Training parameters
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.es = es
        self.mediumbound = '' # initialization
        self.reactions = list()
        self.treatments = list()

        # Get additional parameters (matrices)
        self.get_parameter(biomass_max=biomass_max, verbose=verbose)

    def get_parameter(self, biomass_max=4.0, verbose=False):
        # load parameter file if provided
        if self.trainingfile == None:
            return
        if not os.path.isfile(self.trainingfile+'.npz'):
            print(self.trainingfile+'.npz')
            sys.exit('parameter file not found')
        parameter = TrainingSet()
        parameter.load(self.trainingfile)
        if not ('MM_' in self.model_type):
            if self.objective:
                print("Filtering here AMN!")
                parameter.filter_measure(measure=self.objective, verbose=verbose)
                print("measure: ", parameter.measure)
            self.Yall = parameter.Yall if self.objective else None
        self.mediumbound = parameter.mediumbound
        self.levmed = parameter.levmed
        self.valmed = parameter.valmed
        # matrices from parameter file
        self.S = parameter.S # Stoichiometric matrix
        self.Pin = parameter.Pin # Boundary matrix from reaction to medium
        self.Pout = parameter.Pout # Measure matrix from reactions to measures
        self.V2M = parameter.V2M # Reaction to metabolite matrix
        self.M2V = parameter.M2V # Metabolite to reaction matrix
        self.X = parameter.X # Training set X
        self.Y = parameter.Y # Training set Y
        self.LB = parameter.LB # Training set LB
        self.S_int = parameter.S_int
        self.S_ext = parameter.S_ext
        self.Q = parameter.Q
        self.P = parameter.P
        self.b_int = parameter.b_int
        self.b_ext = parameter.b_ext
        self.Sb = parameter.Sb
        self.c = parameter.c
        self.reactions = parameter.reactions
        self.treatments = parameter.treatments
               
        # Update input_dim and output_dim
        self.input_dim = self.input_dim if self.input_dim > 0 \
        else parameter.X.shape[1]
        if self.input_dim > len(self.reactions):
            self.input_dim = self.input_dim - len(self.reactions)
        self.output_dim = self.output_dim if self.output_dim > 0 \
        else parameter.Y.shape[1]

        self.objPout, self.objY = None, None

        ## Set a maximum value for biomass reaction
        #   -> use biomass as an objective 
        if 'MM_' in self.model_type:
            self.Yall = None

            if self.objective:
                # parameter.filter_measure(measure=self.objective, verbose=verbose)
                self.objPout, parameter.Y, self.objY, self.Yall = parameter.filter_measure_return(measure=parameter.objective, biomass_max=biomass_max, verbose=verbose)

        # --- NEW: ENZYME CAPACITY AGGREGATION SETUP ---
        try:
            rxn_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in self.reactions]
            base_names = []
            mapping = []
            
            # Suffixes that represent directional splits of a single enzyme/transporter
            directional_suffixes = ('_f', '_r', '_i', '_o')
            
            for name in rxn_names:
                # Check if the name ends with any of our directional split markers
                if name.endswith(directional_suffixes):
                    base = name[:-2]
                else:
                    base = name
                    
                if base not in base_names:
                    base_names.append(base)
                mapping.append(base_names.index(base))

            num_split = len(rxn_names)
            num_base = len(base_names)

            M_enz_np = np.zeros((num_split, num_base), dtype=np.float32)
            Vout_raw = self.Y
            Pout_raw = self.Pout

            if len(Pout_raw.shape) == 2 and Pout_raw.shape[0] == Pout_raw.shape[1]:
                Pout_1d = np.diag(Pout_raw)
            else:
                Pout_1d = Pout_raw[0] if len(Pout_raw.shape) > 1 else Pout_raw

            Vout_enz_np = np.zeros((Vout_raw.shape[0], num_base), dtype=np.float32)
            Pout_enz_np = np.zeros((1, num_base), dtype=np.float32)

            for i, j in enumerate(mapping):
                M_enz_np[i, j] = 1.0
                if Vout_raw is not None and Vout_raw.shape[0] > 0 and Vout_raw.shape[1] > i:
                    Vout_enz_np[:, j] = np.maximum(Vout_enz_np[:, j], Vout_raw[:, i])
                if Pout_1d is not None and len(Pout_1d) > i:
                    Pout_enz_np[0, j] = np.maximum(Pout_enz_np[0, j], Pout_1d[i])

            self.M_enz_tf = tf.constant(M_enz_np)
            self.Vout_enz_tf = tf.constant(Vout_enz_np)
            self.Pout_enz_tf = tf.constant(Pout_enz_np)
        except Exception as e:
            print("Notice: Could not build enzyme aggregation matrices:", e)
            self.M_enz_tf = None
        # ----------------------------------------------

    def save(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        s = str(self.trainingfile) + ","\
                    + str(self.model_type) + ","\
                    + str(self.objective) + ","\
                    + str(self.scaler) + ","\
                    + str(self.input_dim) + ","\
                    + str(self.output_dim) + ","\
                    + str(self.n_hidden) + ","\
                    + str(self.hidden_dim) + ","\
                    + str(self.activation) + ","\
                    + str(self.timestep) + ","\
                    + str(self.learn_rate) + ","\
                    + str(self.decay_rate) + ","\
                    + str(self.epochs) + ","\
                    + str(self.regression) + ","\
                    + str(self.train_rate) + ","\
                    + str(self.dropout) + ","\
                    + str(self.batch_size) + ","\
                    + str(self.niter) + ","\
                    + str(self.xfold) + ","\
                    + str(self.es)
        with open(fileparam, "w") as h:
            # print(s, file = h)
            h.write(s)

        self.model.save(filemodel)

    def load(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        if not os.path.isfile(fileparam):
            print(fileparam)
            sys.exit('parameter file not found')
        if not os.path.isfile(filemodel):
            print(filemodel)
            sys.exit('model file not found')
        # First read parameter file
        with open(fileparam, 'r') as h:
            for line in h:
                K = line.rstrip().split(',')
        # model architecture
        self.trainingfile =  str(K[0])
        self.model_type =  str(K[1])
        self.objective =  str(K[2])
        self.scaler =  float(K[3])
        self.input_dim =  int(K[4])
        self.output_dim = int(K[5])
        self.n_hidden = int(K[6])
        self.hidden_dim = int(K[7])
        self.activation = str(K[8])
        # GD parameters
        self.timestep = int(K[9])
        self.learn_rate = float(K[10])
        self.decay_rate = float(K[11])
        # Training parameters
        self.epochs = int(K[12])
        self.regression = True if K[13] == 'True' else False
        self.train_rate = float(K[14])
        self.dropout = float(K[15])
        self.batch_size = int(K[16])
        self.niter = int(K[17])
        self.xfold = int(K[18])
        self.es = True if K[19] == 'True' else False
        # Make objective a list
        self.objective = self.objective.replace('[', '')
        self.objective = self.objective.replace(']', '')
        self.objective = self.objective.replace('\'', '')
        self.objective = self.objective.replace("\"", "")
        self.objective = self.objective.split(',')
        # Get additional parameters (matrices)
        self.get_parameter(verbose=verbose)
        # Then load model
        if self.model_type == 'AMN_Wt':
            self.model = load_model(filemodel,
                                    custom_objects={'RNNCell':RNNCell,
                                                    'parameter':Neural_Model
                                                    , 'custom_loss': CustomLoss
                                                    # , 'loss_model': generate_loss_model(parameter, Vin, Vbf)
                                                    , 'loss': biochem_loss(parameter)},
                                    compile=False)
        else:
            self.model = load_model(filemodel, compile=False)

    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        print('training file:', self.trainingfile)
        print('model type:', self.model_type)
        print('model scaler:', self.scaler)
        print('model input dim:', self.input_dim)
        print('model output dim:', self.output_dim)
        print('model medium bound:', self.mediumbound)
        print('timestep:', self.timestep)
        if self.trainingfile:
            if os.path.isfile(self.trainingfile+'.npz'):
                print('training set size', self.X.shape, self.Y.shape)
        else:
             print('no training set provided')
        if self.n_hidden > 0:
            print('nbr hidden layer:', self.n_hidden)
            print('hidden layer size:', self.hidden_dim)
            print('activation function:', self.activation)
        if self.model_type == 'AMN_QP' and self.timestep > 0:
            print('gradient learn rate:', self.learn_rate)
            print('gradient decay rate:', self.decay_rate)
        if self.epochs > 0:
            print('training epochs:', self.epochs)
            print('training regression:', self.regression)
            print('training learn rate:', self.train_rate)
            print('training dropout:', self.dropout)
            print('training batch size:', self.batch_size)
            print('training validation iter:', self.niter)
            print('training xfold:', self.xfold)
            print('training early stopping:', self.es)
        if filename != '':
            sys.stdout.close()
