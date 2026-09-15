import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from SelfDesign import ANN_data_transfer


SEED = 10
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)


def normalize_pair_by_tx(data_tx, data_rx, ratio, eps=1e-12):
    split = int(len(data_tx) * ratio)
    scale = np.sqrt(np.mean(np.square(data_tx[:split])) + eps)
    return data_tx / scale, data_rx / scale, scale


def build_window_data(data_tx, data_rx, taps, output_dim, ratio):
    data_tx_length = len(data_tx)
    data_rx_length = len(data_rx)
    split_tx = int(data_tx_length * ratio)
    split_rx = int(data_rx_length * ratio)

    x_train_temp = data_tx[:split_tx]
    y_train_temp = data_rx[:split_rx]
    x_test_temp = data_tx[split_tx + 1:]
    y_test_temp = data_rx[split_rx + 1:]

    x_train = ANN_data_transfer(x_train_temp, taps)
    x_test = ANN_data_transfer(x_test_temp, taps)

    trim = (taps - output_dim) // 2
    if trim > 0:
        y_train_temp = y_train_temp[trim:-trim]
        y_test_temp = y_test_temp[trim:-trim]

    y_train = ANN_data_transfer(y_train_temp, output_dim)
    y_test = ANN_data_transfer(y_test_temp, output_dim)

    return x_train, y_train, x_test, y_test


def recover_waveform(model, data_tx, taps, output_dim, batch_size):
    pad = (taps - 1) // 2
    predict_data = np.concatenate([data_tx[-pad:], data_tx, data_tx[:pad]])
    predict_data_final = ANN_data_transfer(predict_data, taps)
    data_equalization = model.predict(predict_data_final, batch_size=batch_size)
    data_recover = data_equalization[:, output_dim // 2]
    return data_recover.reshape(len(data_tx), order="C")


retrain = 1
train_gan = 1
epoch = 100
batch_size = 256
ratio = 0.8

BaudRate = 25e3
upsam = 4
taps = 157
output_dim = 85

path = "./channel/"
model_name = "channelest_TTHNet_SinM"
model_file = path + "DNN_" + model_name + ".h5"
best_weights_file = path + "DNN_" + model_name + "_best.weights.h5"

name_Tx = "D:/git_CTLE/CTLE_v2/data/Txdata_NN_baseline_wgn.txt"
name_Rx = "D:/git_CTLE/CTLE_v2/data/Rxdata_NN_baseline_wgn.txt"
name_Sa = "Txdata_afterNN.txt"

os.makedirs(path, exist_ok=True)
os.makedirs("./NN/forEpoch", exist_ok=True)

data_Tx = np.loadtxt(name_Tx).astype(np.float32)
data_Rx = np.loadtxt(name_Rx).astype(np.float32)
# data_Tx, data_Rx, train_rms = normalize_pair_by_tx(data_Tx, data_Rx, ratio)
# print("train RMS =", train_rms)

x_train, y_train, x_test, y_test = build_window_data(
    data_Tx,
    data_Rx,
    taps,
    output_dim,
    ratio,
)

print("data_Tx length =", len(data_Tx))
print("data_Rx length =", len(data_Rx))
print("x_train shape =", x_train.shape)
print("y_train shape =", y_train.shape)
print("x_test shape =", x_test.shape)
print("y_test shape =", y_test.shape)

reduce_lr = ReduceLROnPlateau(
    monitor="val_mse",
    factor=0.9,
    verbose=1,
    patience=5,
    min_lr=0,
    mode="auto",
)
early_stop = EarlyStopping(
    monitor="val_mse",
    min_delta=1e-11,
    patience=epoch * 0.7,
    verbose=1,
    mode="auto",
)
checkpoint = ModelCheckpoint(
    filepath=best_weights_file,
    save_weights_only=True,
    monitor="val_mse",
    save_best_only=True,
)

if retrain:
    NNinput = Input(shape=(taps,), name="channel_input")
    output = Dense(
        output_dim,
        activation="linear",
        use_bias=False,
        name="channel_linear",
    )(NNinput)

    model = Model(inputs=NNinput, outputs=output, name="linear_nn_channel")
    model.summary()

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        metrics=["mse"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epoch,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stop, checkpoint],
    )
    model.load_weights(best_weights_file)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("test score =", score)
    model.save(model_file)
else:
    model = keras.models.load_model(model_file, compile=False)

data_recover = recover_waveform(model, data_Tx, taps, output_dim, batch_size)
np.savetxt(name_Sa, data_recover, fmt="%1.7e")
print("File is", name_Sa)

m = np.shape(data_recover)[0]
xfre = ((np.arange(m) - m / 2) * BaudRate * upsam / 1e9) / m
fft_figure = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data_recover))) + 1e-12)
plt.figure()
plt.plot(xfre, fft_figure, "-g")
plt.title("Tx after NN/Channel")

fft_figure_error = np.abs(
    np.fft.fftshift(np.fft.fft(data_Rx)) -
    np.fft.fftshift(np.fft.fft(data_recover))
)
plt.figure()
plt.plot(fft_figure_error, "-g")
plt.title("fft error")

if retrain:
    np.savetxt("epoch.txt", history.epoch, fmt="%1.7e")
    np.savetxt("train_loss.txt", history.history["loss"], fmt="%1.7e")
    np.savetxt("val_loss.txt", history.history["val_loss"], fmt="%1.7e")
    plt.figure()
    plt.plot(history.epoch, np.log10(history.history["loss"]), "r")
    plt.plot(history.epoch, np.log10(history.history["val_loss"]), "b")
    plt.xlabel("Epoch")
    plt.ylabel("loss(log10)")
    plt.legend(["Train loss", "Validation loss"])

print("deterministic channel part finished")

if train_gan:
    os.makedirs("./Noise", exist_ok=True)
    rx_simu = np.reshape(data_recover, [-1, 1])
    rx_real = np.reshape(data_Rx, [-1, 1])
    noise_nor = rx_real - rx_simu
    np.savetxt("./Noise/noise.txt", noise_nor)

    print("normalization check")
    print("output energy")
    print(np.mean(np.sum(rx_simu ** 2)))
    print("noise energy")
    print(np.mean(np.sum(noise_nor ** 2)))
    print("direct snr ratio")
    print(np.mean(np.sum(rx_simu ** 2)) / np.mean(np.sum(noise_nor ** 2)))
    np.savetxt(
        "./z_zhijiesnr.txt",
        np.expand_dims(np.mean(np.sum(rx_simu ** 2)) / np.mean(np.sum(noise_nor ** 2)), axis=0),
    )

    print("noise mean and std")
    print(np.mean(noise_nor))
    print(np.sqrt(np.var(noise_nor)))
    plt.figure()
    plt.hist(noise_nor, bins=500, density=True, color="lightgreen")
    plt.title("noise_nor distribution")
