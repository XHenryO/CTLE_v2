"""
Created on '2021/10/22'

@author: JY
Version: v1.3
History:
2022/09/22 v1.2: ANN_data_transfer support complex value
2022/11/01 v1.3: add Sin layer named MyLayer2

"""
# coding:utf-8
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Add, Layer, GaussianNoise
from tensorflow.python.ops import array_ops
from decimal import Decimal


# 1D to 2D==(n,taps)
def ANN_data_transfer(x, taps):
    # x = x[:, np.newaxis] #discard
    x = x.reshape(-1, 1)
    # half_tap = (taps - 1) / 2
    # z = np.int(len(x) - 2 * half_tap)
    z = int(len(x) - taps + 1)    ## 新的长度
    if x.dtype == complex:
        y = np.zeros([z, taps], dtype=complex)
    else:
        y = np.zeros([z, taps])
    for i in range(z):
        for k in range(taps):
            y[i, k] = x[i + k, 0]
    return y

# VolterraOrder2
def ANN_VolterraOrder2_transfer(x, taps, taps_lms):
    # x = x[:, np.newaxis] #discard
    x = x.reshape(-1, 1)
    # half_tap = (time_steps - 1) / 2
    # z = np.int(len(x) - 2 * half_tap)
    z = int(len(x) - taps_lms + 1)
    y = np.zeros([z, taps * (taps + 1) // 2])
    for i in range(z):
        k = 0
        for j1 in range(taps):
            for j2 in range(j1, taps):
                y[i, k] = x[i + j1 + (taps_lms-taps)//2, 0] * x[i + j2 + (taps_lms-taps)//2, 0]
                k = k+1

    return y

def ANN_data_transfer_IQ(x, taps, dim):
    # x=x[:,np.newaxis]
    z = np.int(len(x) - taps)
    z_IQ = np.int(z / dim)  # 窗的个数
    y = np.zeros([z_IQ + 1, taps])
    for i in range(0, z + dim, dim):
        for k in range(taps):
            y_IQ = np.int(i / dim)
            y[y_IQ, k] = x[i + k]
    return y


# mse loss of time and frequency
def TFLoss(y_true, y_pred):
    a = 0.5
    mse = K.mean(K.square(y_true - y_pred))
    # y_true = K.cast(y_true, dtype='complex64')
    y_true = tf.complex(y_true, y_true - y_true)
    Fy_true = tf.signal.fft(y_true)
    # y_pred = K.cast(y_pred, dtype='complex64')
    y_pred = tf.complex(y_pred, y_pred - y_pred)
    Fy_pred = tf.signal.fft(y_pred)
    F_mse = K.mean(K.square(K.abs(Fy_true - Fy_pred)))
    loss_value = a*F_mse + (1-a)*mse
    return loss_value


# MyLayer is Noise layer
class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # return GaussianNoise(self.kernel)(x)
        return x + K.random_normal(shape=array_ops.shape(x), mean=0, stddev=self.kernel, dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# 3D to 2D
def ConcatenateLayer(x, channel_in_taps):
    kvar = x[:, 0]
    for i in range(1, channel_in_taps):
        kvar = K.concatenate([kvar, x[:, i]], axis=1)
    # print(K.int_shape(kvar))
    return kvar - K.mean(kvar)   ## 去除直流分量

# split a univariate sequence into training samples
# 2D to 3D
def split_sequence(sequence1, sequence2, n_steps):   ##输入 len 5  ##输出 len2 tap1 5   len2 1 5
    x_train, y_train = list(), list()
    for i in range(len(sequence1)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence1):
            break
        # gather input and output parts of the pattern
        if len(sequence2) == 0:
            seq_x = sequence1[i:end_ix]
        else:
            seq_x, seq_y = sequence1[i:end_ix], sequence2[np.int(end_ix - n_steps / 2 - 1 / 2)]
        x_train.append(seq_x)
        if len(sequence2) != 0:
            y_train.append(seq_y)
    return np.array(x_train), np.array(y_train)

# MyLayer2 is Sin layer

class MyLayer2(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.fre = self.add_weight(name='fre',
                                      shape=(1,),
                                      initializer=keras.initializers.Constant(value=1.306367041198502e10),
                                      trainable=True)
        self.phase = self.add_weight(name='phase',
                                   shape=(1,),
                                   initializer='uniform',
                                   trainable=True)
        self.amplitude = self.add_weight(name='amplitude',
                                     shape=(1,),
                                     initializer='uniform',
                                     trainable=True)
        self.fre2 = self.add_weight(name='fre2',
                                   shape=(1,),
                                   initializer=keras.initializers.Constant(value=1.306429463171036e10),
                                   trainable=True)
        self.phase2 = self.add_weight(name='phase2',
                                     shape=(1,),
                                     initializer='uniform',
                                     trainable=True)
        self.amplitude2 = self.add_weight(name='amplitude2',
                                         shape=(1,),
                                         initializer='uniform',
                                         trainable=True)
        super(MyLayer2, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # return GaussianNoise(self.kernel)(x)
        return self.amplitude * K.sin(2*np.pi*self.fre*x + self.phase)+self.amplitude2 * K.sin(2*np.pi*self.fre2*x + self.phase2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim}
        base_config = super(MyLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))