# import numpy as np
# import tensorflow as tf
from tensorflow import keras
from SelfDesign import MyLayer, MyLayer2

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from Layer import (CTLELayer, CTLELayer2, CTLELayer4, CTLELayer_model_v12_v4,
                   CTLEBypassGate, FixedFIRChannelLayer, NormalizeRMSLayer)
from equalizer_group_delay import print_maximum_link_group_delay
from experiment_reporting import configured_retrain, configured_snr, report_experiment_result


SEED = 10
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def normalize_rms(data, eps=1e-12):
    data = np.asarray(data, dtype=np.float32)
    return data / np.sqrt(np.mean(np.square(data)) + eps)


def to_symbol_batch(data, segment_length=1024):
    usable_length = len(data) // segment_length * segment_length
    data = data[:usable_length]
    return data.reshape(-1, segment_length).astype(np.float32)


def upsample_batch(data, sps):
    out = np.zeros((data.shape[0], data.shape[1] * sps, data.shape[2]), dtype=np.float32)
    out[:, ::sps, :] = data
    return out


def zero_upsample_tensor(x, sps):
    shape = tf.shape(x)
    x = tf.expand_dims(x, axis=2)
    zeros = tf.zeros([shape[0], shape[1], sps - 1], dtype=x.dtype)
    x = tf.concat([x, zeros], axis=2)
    return tf.reshape(x, [shape[0], shape[1] * sps])


def repeat_upsample_tensor(x, sps):
    shape = tf.shape(x)
    x = tf.expand_dims(x, axis=2)
    x = tf.repeat(x, repeats=sps, axis=2)
    return tf.reshape(x, [shape[0], shape[1] * sps])


def downsample_batch(data, sps, offset=None):
    if offset is None:
        offset = 0
    return data[:, offset::sps]


def downsample_tensor(x, sps, offset=0):
    return x[:, offset::sps]


def normalize_tensor_rms(x):
    power = tf.reduce_mean(tf.square(x), axis=1, keepdims=True)
    return x / tf.sqrt(power + tf.keras.backend.epsilon())


def add_awgn(x, snr):
    signal_power = tf.reduce_mean(tf.square(x), axis=1, keepdims=True)
    noise_power = signal_power / tf.pow(10.0, snr / 10.0)
    noise = tf.random.normal(tf.shape(x), dtype=x.dtype) * tf.sqrt(noise_power)
    return x + noise


def channel_squeeze(x):
    return tf.squeeze(x, axis=-1)


def channel_max_normalize(x):
    return x / (tf.reduce_max(tf.abs(x), axis=1, keepdims=True) + tf.keras.backend.epsilon())


def channel_circular_pad(x):
    pad = channel_in_taps // 2
    return tf.concat([x[:, -pad:], x, x[:, :pad]], axis=1)


def channel_make_window(x):
    return tf.signal.frame(x, frame_length=channel_in_taps, frame_step=1, axis=1)


def channel_flatten_window(x):
    return tf.reshape(x, [-1, channel_in_taps])


def channel_take_center(x):
    return x[:, channel_output_dim // 2]


def channel_restore_waveform(inputs):
    x, ref = inputs
    return tf.reshape(x, [tf.shape(ref)[0], tf.shape(ref)[1]])


def calculate_raw_metrics(output, target, constellation, gray_bits):
    output_flat = output.reshape(-1)
    target_flat = target.reshape(-1)
    output_index = np.argmin(
        np.abs(output_flat[:, np.newaxis] - constellation[np.newaxis, :]),
        axis=1,
    )
    target_index = np.argmin(
        np.abs(target_flat[:, np.newaxis] - constellation[np.newaxis, :]),
        axis=1,
    )
    mse = np.mean(np.square(output - target))
    ber = np.mean(gray_bits[output_index] != gray_bits[target_index])
    return mse, ber


def get_model_offset(model):
    arguments = model.get_layer("rx_downsampling").get_config().get("arguments", {})
    offset = arguments.get("offset", 0)
    return int(np.asarray(offset).item())


def save_before_downsample_probe(model, data, label):
    probe_model = Model(
        inputs=model.input,
        outputs=model.get_layer("rx_downsampling").input,
        name="before_downsample_probe",
    )
    before_downsample = probe_model.predict(data, batch_size=batch_size)
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    np.savetxt(path + model_name + "_" + label + "_before_downsample.txt", before_downsample.reshape(-1), fmt="%1.7e")
    print(label, "probe offset =", get_model_offset(model))


def print_ctle_weights(weight_file):
    if not os.path.exists(weight_file):
        print("weight file not found:", weight_file)
        return

    import h5py

    with h5py.File(weight_file, "r") as f:
        ctle_prefixes = []

        def visit(name, obj):
            if name.endswith("/vars/0") and "ctle_layer" in name:
                ctle_prefixes.append(name[:-2])

        f.visititems(visit)

        for prefix in sorted(ctle_prefixes):
            log_tau_z = float(f[prefix + "/0"][()])
            log_tau_p1 = float(f[prefix + "/1"][()])
            log_tau_p2 = float(f[prefix + "/2"][()])
            log_A0 = float(f[prefix + "/3"][()])

            tau_z = np.clip(np.exp(log_tau_z), 1e-15, 1e-6)
            tau_p1 = np.clip(np.exp(log_tau_p1), 1e-15, 1e-6)
            tau_p2 = np.clip(np.exp(log_tau_p2), 1e-15, 1e-6)
            A0 = np.clip(np.exp(log_A0), 1e-3, 1e3)

            print(prefix + " weights:")
            print("log_tau_z =", log_tau_z, "tau_z =", tau_z)
            print("log_tau_p1 =", log_tau_p1, "tau_p1 =", tau_p1)
            print("log_tau_p2 =", log_tau_p2, "tau_p2 =", tau_p2)
            print("log_A0 =", log_A0, "A0 =", A0)


retrain = configured_retrain(1)
epoch = 500
batch_size = 16
ratio = 0.8
Rs = 25e9
sps = 4
Fs = Rs * sps
snr = 15

snr_tag = "snr" + str(snr)
# offset_candidates = [1]
offset_candidates = list(range(sps))
path = "data_model/"
name_Tx = "./data/Txdata.txt"
channel_path = "channel/"
channel_in_taps = 157
channel_output_dim = 85
# model_file = path + "joint_ctle_nn_channel_model.keras"
# final_weights_file = path + "joint_ctle_nn_channel_final.weights.h5"
model_file = path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_model.keras"
final_weights_file = path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_final.weights.h5"

os.makedirs(path, exist_ok=True)

channel = keras.models.load_model(
    channel_path + "DNN_" + "channelest_TTHNet_Sin" + ".h5",
    custom_objects={
        "MyLayer": MyLayer,
        "MyLayer2": MyLayer2,
        "time_steps": channel_in_taps,
    },
    compile=False,
    safe_mode=False,
)
channel.trainable = False

data_raw = np.loadtxt(name_Tx).astype(np.float32)
split = int(len(data_raw) * ratio)
train_rms = np.sqrt(np.mean(np.square(data_raw[:split])) + 1e-12)
data_Tx = data_raw / train_rms
print("train RMS =", train_rms)
# np.savetxt("./data/rms.txt", data_Tx, fmt="%1.7e")

x_train = data_Tx[:split]
y_train = data_Tx[:split]
x_test = data_Tx[split:]
y_test = data_Tx[split:]


# 1      -> batch size
# 81600  -> time
# 1      -> channel
x_train = to_symbol_batch(x_train)
y_train = to_symbol_batch(y_train)
x_test = to_symbol_batch(x_test)
y_test = to_symbol_batch(y_test)

def build_model(current_offset):
    NNinput = Input(shape=(None,), name="tx_input")

    data_upsampled = Lambda(repeat_upsample_tensor, arguments={"sps": sps}, name="tx_upsampling")(NNinput)
    # data_tx_ctle = CTLELayer(
    #     Fs=Fs,
    #     name="tx_ctle",
    # )(data_upsampled)
    data_tx_ctle_filtered = CTLELayer_model_v12_v4(
        Fs=Fs,
        prewarp_frequency_hz=Rs / 2.0,
        name="tx_ctle",
    )(data_upsampled)
    data_tx_ctle_filtered = tf.keras.layers.Rescaling(
        scale=-1.0,
        name="tx_ctle_polarity_compensation",
    )(data_tx_ctle_filtered)
    # alpha starts effectively at zero: before the TX path is enabled, this
    # model is the same topology as the validated Post model.
    data_tx_ctle = CTLEBypassGate(
        init_alpha=1e-6,
        filtered_polarity=-1.0,
        name="tx_ctle_bypass_gate",
    )([data_upsampled, data_tx_ctle_filtered])

    Tx = data_tx_ctle
    Tx = Lambda(channel_circular_pad, name="channel_circular_pad")(Tx)
    Tx = Lambda(channel_make_window, name="channel_make_window")(Tx)
    Tx = Lambda(channel_flatten_window, name="channel_flatten_window")(Tx)

    Rx = channel(Tx)
    Rx = Lambda(channel_take_center, name="channel_take_center")(Rx)
    data_ch = Lambda(channel_restore_waveform, name="channel_restore_waveform")([Rx, data_tx_ctle])
    data_ch = Lambda(add_awgn, arguments={"snr": snr}, name="awgn_noise")(data_ch)

    # data_CTLE = CTLELayer(
    #     Fs=Fs,
    #     name="rx_ctle",
    # )(data_ch)
    data_CTLE = CTLELayer_model_v12_v4(
        Fs=Fs,
        prewarp_frequency_hz=Rs / 2.0,
        name="rx_ctle",
    )(data_ch)
    data_CTLE = tf.keras.layers.Rescaling(
        scale=-1.0,
        name="rx_ctle_polarity_compensation",
    )(data_CTLE)

    data_symbol = Lambda(
        downsample_tensor,
        arguments={"sps": sps, "offset": current_offset},
        name="rx_downsampling",
    )(data_CTLE)
    # data_nor = NormalizeRMSLayer(name="output_normalization")(data_symbol)

    model = Model(inputs=NNinput, outputs=data_symbol, name="joint_equalization_ctle")

    return model


def compile_joint_model(model, learning_rate):
    """Recompile after changing which TX-path variables are trainable."""
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mse"],
    )


def fit_joint_stage(model, current_offset, stage_name, epochs, learning_rate, checkpoint_path):
    """Train one phase, then restore its own best validation checkpoint."""
    compile_joint_model(model, learning_rate)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_mse", factor=0.4, verbose=1, patience=4,
        min_lr=1e-9, mode="auto",
    )
    early_stop = EarlyStopping(
        monitor="val_mse", min_delta=1e-11, patience=26, verbose=1,
        mode="auto", restore_best_weights=True,
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, monitor="val_mse",
        save_best_only=True,
    )
    print(
        f"offset={current_offset}, {stage_name}: epochs={epochs}, "
        f"learning_rate={learning_rate:g}"
    )
    history = model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stop, checkpoint],
    )
    model.load_weights(checkpoint_path)
    return history


if retrain:
    offset_results = []

    for current_offset in offset_candidates:
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        # current_model_file = path + "joint_ctle_nn_channel_offset" + str(current_offset) + ".keras"
        # current_final_weights_file = path + "joint_ctle_nn_channel_offset" + str(current_offset) + "_final.weights.h5"
        # current_best_weights_file = path + "joint_ctle_nn_channel_offset" + str(current_offset) + "_best.weights.h5"
        current_model_file = path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_offset" + str(current_offset) + ".keras"
        current_final_weights_file = path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_offset" + str(current_offset) + "_final.weights.h5"
        current_warmup_weights_file = path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_offset" + str(current_offset) + "_warmup.weights.h5"
        current_best_weights_file = path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_offset" + str(current_offset) + "_best.weights.h5"

        model = build_model(current_offset)
        model.summary()

        tx_ctle = model.get_layer("tx_ctle")
        tx_gate = model.get_layer("tx_ctle_bypass_gate")

        # For exactly 25 epochs, keep the added TX branch closed.  This gives
        # the RX path a Post-like starting point without a long staged run.
        tx_ctle.trainable = False
        tx_gate.trainable = False
        compile_joint_model(model, 4e-2)
        print(f"offset={current_offset}, warmup / RX only: epochs=25, learning_rate=0.04")
        model.fit(
            x_train, y_train, epochs=25, batch_size=batch_size,
            validation_data=(x_test, y_test),
        )
        model.save_weights(current_warmup_weights_file)

        # Then open the complete TX path and use the same LR for normal joint
        # optimization.  ReduceLROnPlateau and EarlyStopping decide where to
        # stop; the total maximum remains 500 epochs.
        tx_ctle.trainable = True
        tx_gate.trainable = True
        fit_joint_stage(
            model, current_offset, "joint optimization", epoch - 25, 4e-2,
            current_best_weights_file,
        )

        # Keep the warmup solution only as a safety checkpoint; normally the
        # selected result will be the full jointly optimized model.
        stage_results = []
        for stage_name, stage_weights_file in (
            ("warmup", current_warmup_weights_file),
            ("joint", current_best_weights_file),
        ):
            model.load_weights(stage_weights_file)
            tf.random.set_seed(SEED + 100 + current_offset)
            stage_output = model.predict(x_test, batch_size=batch_size)
            stage_mse, stage_ber = calculate_raw_metrics(
                stage_output, y_test,
                np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) / train_rms,
                np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32),
            )
            stage_results.append((stage_mse, stage_ber, stage_name, stage_weights_file))
            print("offset =", current_offset, stage_name, "val MSE =", stage_mse, "val BER =", stage_ber)

        val_mse, val_ber, best_stage_name, best_stage_weights_file = min(
            stage_results, key=lambda item: item[0]
        )
        model.load_weights(best_stage_weights_file)
        print("offset =", current_offset, "selected", best_stage_name)
        model.save_weights(current_final_weights_file)
        model.save(current_model_file)
        offset_results.append((current_offset, val_mse, val_ber, current_model_file, current_final_weights_file))
        print("offset =", current_offset, "val MSE =", val_mse, "val BER =", val_ber)

    best_offset, best_mse, best_ber, best_model_file, best_offset_weights_file = min(
        offset_results,
        key=lambda item: item[1],
    )
    model = tf.keras.models.load_model(
        best_model_file,
        custom_objects={
            "CTLELayer": CTLELayer,
            "CTLELayer2": CTLELayer2,
            "CTLELayer4": CTLELayer4,
            "CTLELayer_model_v12_v4": CTLELayer_model_v12_v4,
            "CTLEBypassGate": CTLEBypassGate,
            "FixedFIRChannelLayer": FixedFIRChannelLayer,
            "NormalizeRMSLayer": NormalizeRMSLayer,
            "MyLayer": MyLayer,
            "MyLayer2": MyLayer2,
            "add_awgn": add_awgn,
            "zero_upsample_tensor": zero_upsample_tensor,
            "repeat_upsample_tensor": repeat_upsample_tensor,
            "downsample_tensor": downsample_tensor,
            "channel_squeeze": channel_squeeze,
            "channel_max_normalize": channel_max_normalize,
            "channel_circular_pad": channel_circular_pad,
            "channel_make_window": channel_make_window,
            "channel_flatten_window": channel_flatten_window,
            "channel_take_center": channel_take_center,
            "channel_restore_waveform": channel_restore_waveform,
        },
        safe_mode=False,
    )
    model.save_weights(final_weights_file)
    model.save(model_file)
    np.savetxt(
        path + "pw_joint_gated_ctle_nn_channel_" + snr_tag + "_offset_results.txt",
        np.array([[item[0], item[1], item[2]] for item in offset_results], dtype=np.float64),
        fmt="%1.7e",
        header="offset val_mse val_ber",
    )
    print("best offset =", best_offset, "best val MSE =", best_mse, "best val BER =", best_ber)
    # np.savetxt(path + "joint_ctle_nn_channel_train_loss.txt", history.history["loss"], fmt="%1.7e")
    # np.savetxt(path + "joint_ctle_nn_channel_val_loss.txt", history.history["val_loss"], fmt="%1.7e")

    # Legacy four-parameter CTLELayer4 decoder is not valid for v12_v4.

model = tf.keras.models.load_model(
    model_file,
    custom_objects={
        "CTLELayer": CTLELayer,
        "CTLELayer2": CTLELayer2,
        "CTLELayer4": CTLELayer4,
        "CTLELayer_model_v12_v4": CTLELayer_model_v12_v4,
        "CTLEBypassGate": CTLEBypassGate,
        "FixedFIRChannelLayer": FixedFIRChannelLayer,
        "NormalizeRMSLayer": NormalizeRMSLayer,
        "MyLayer": MyLayer,
        "MyLayer2": MyLayer2,
        "add_awgn": add_awgn,
        "zero_upsample_tensor": zero_upsample_tensor,
            "repeat_upsample_tensor": repeat_upsample_tensor,
        "downsample_tensor": downsample_tensor,
        "channel_squeeze": channel_squeeze,
        "channel_max_normalize": channel_max_normalize,
        "channel_circular_pad": channel_circular_pad,
        "channel_make_window": channel_make_window,
        "channel_flatten_window": channel_flatten_window,
        "channel_take_center": channel_take_center,
        "channel_restore_waveform": channel_restore_waveform,
    },
    safe_mode=False,
)
output_train = model.predict(x_train, batch_size=batch_size)
output_symbol = output_train

constellation_raw = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
# constellation_norm = constellation_raw / np.sqrt(np.mean(np.square(constellation_raw)))
constellation_norm = constellation_raw / train_rms
gray_bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32)

output_flat_original = output_symbol.reshape(-1)
target_flat = y_train.reshape(-1)
train_mse_raw = np.mean(np.square(output_symbol - y_train))

output_index = np.argmin(
    np.abs(output_flat_original[:, np.newaxis] - constellation_norm[np.newaxis, :]),
    axis=1,
)
target_index = np.argmin(np.abs(target_flat[:, np.newaxis] - constellation_norm[np.newaxis, :]), axis=1)

output_bits = gray_bits[output_index]
target_bits = gray_bits[target_index]

ber = np.mean(output_bits != target_bits)
print("train MSE raw =", train_mse_raw)
print("train BER raw =", ber)

output_test = model.predict(x_test, batch_size=batch_size)
if output_test.shape[1] == y_test.shape[1]:
    output_test_symbol = output_test
else:
    output_test_symbol = downsample_batch(output_test, sps)

output_test_flat = output_test_symbol.reshape(-1)
target_test_flat = y_test.reshape(-1)
test_mse_raw = np.mean(np.square(output_test_symbol - y_test))

output_test_index = np.argmin(
    np.abs(output_test_flat[:, np.newaxis] - constellation_norm[np.newaxis, :]),
    axis=1,
)
target_test_index = np.argmin(
    np.abs(target_test_flat[:, np.newaxis] - constellation_norm[np.newaxis, :]),
    axis=1,
)

output_test_bits = gray_bits[output_test_index]
target_test_bits = gray_bits[target_test_index]

test_ber = np.mean(output_test_bits != target_test_bits)
print("test MSE raw =", test_mse_raw)
print("test BER raw =", test_ber)
report_experiment_result(
    model=model,
    model_name="Model_ch_Joint",
    snr=snr,
    train_mse=train_mse_raw,
    train_ber=ber,
    test_mse=test_mse_raw,
    test_ber=test_ber,
    layer_names=["tx_ctle", "tx_ctle_bypass_gate", "linear_nn_channel", "rx_ctle"],
    Fs=Fs,
)

tf.random.set_seed(SEED)
save_before_downsample_probe(model, x_train, "train")
tf.random.set_seed(SEED + 1)
save_before_downsample_probe(model, x_test, "test")

# output_flat = output_flat_original.copy()
# scale = np.sum(output_flat * target_flat) / (np.sum(output_flat ** 2) + 1e-12)
# output_flat = output_flat * scale
#
# output_flat_rms = output_flat_original.copy()
# scale_rms = np.sqrt(np.mean(target_flat ** 2)) / (np.sqrt(np.mean(output_flat_rms ** 2)) + 1e-12)
# output_flat_rms = output_flat_rms * scale_rms
#
# output_flat_affine = output_flat_original.copy()
# output_centered = output_flat_affine - np.mean(output_flat_affine)
# target_centered = target_flat - np.mean(target_flat)
# gain_affine = np.sum(output_centered * target_centered) / (np.sum(output_centered ** 2) + 1e-12)
# bias_affine = np.mean(target_flat) - gain_affine * np.mean(output_flat_affine)
# output_flat_affine = output_flat_affine * gain_affine + bias_affine
#
# output_index = np.argmin(np.abs(output_flat[:, np.newaxis] - constellation_norm[np.newaxis, :]), axis=1)
# output_index_rms = np.argmin(np.abs(output_flat_rms[:, np.newaxis] - constellation_norm[np.newaxis, :]), axis=1)
# output_index_affine = np.argmin(np.abs(output_flat_affine[:, np.newaxis] - constellation_norm[np.newaxis, :]), axis=1)
#
# output_bits = gray_bits[output_index]
# output_bits_rms = gray_bits[output_index_rms]
# output_bits_affine = gray_bits[output_index_affine]
#
# ber = np.mean(output_bits != target_bits)
# ber_rms = np.mean(output_bits_rms != target_bits)
# ber_affine = np.mean(output_bits_affine != target_bits)
#
# print("train BER LS scale =", ber)
# print("train BER RMS scale =", ber_rms)
# print("train BER affine LS =", ber_affine)
# print("LS scale =", scale)
# print("RMS gain =", scale_rms)
# print("affine LS gain =", gain_affine, "bias =", bias_affine)
