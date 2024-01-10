import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Dropout, Flatten, SimpleRNN, \
    LSTM

def set_my_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    rn.seed(12345)
    tf.random.set_seed(123)


def build_model(X_train, mode='LSTM', hidden_dim=[32, 16]):
    set_my_seed()
    model = Sequential()
    if mode == 'RNN':
        # RNN
        model.add(SimpleRNN(hidden_dim[0], return_sequences=True, input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(SimpleRNN(hidden_dim[1]))

    elif mode == 'MLP':
        model.add(Dense(hidden_dim[0], activation='relu', input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(Flatten())
        model.add(Dense(hidden_dim[1], activation='relu'))

    elif mode == 'LSTM':
        # LSTM
        model.add(LSTM(hidden_dim[0], return_sequences=True, input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(LSTM(hidden_dim[1]))
    elif mode == 'GRU':
        # GRU
        model.add(GRU(hidden_dim[0], return_sequences=True, input_shape=(X_train.shape[-2], X_train.shape[-1])))
        model.add(GRU(hidden_dim[1]))
    elif mode == 'CNN':
        model.add(Conv1D(hidden_dim[0], kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=1,
                         input_shape=(X_train.shape[-2], X_train.shape[-1])))
        # model.add(MaxPooling1D())
        model.add(Conv1D(hidden_dim[1], kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2))
        # model.add(MaxPooling1D())
        model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), "mape", "mae"])
    return model