# import numpy as np
# import tensorflow as tf
from tensorflow import keras
from SelfDesign import MyLayer, MyLayer2
from tensorflow.keras.layers import Conv1D

import inspect
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from Layer import CTLELayer2, CTLELayer4, CTLELayer_model_v12_v4, FixedFIRChannelLayer, NormalizeRMSLayer
from equalizer_group_delay import print_maximum_link_group_delay
from experiment_reporting import configured_retrain, configured_snr, report_experiment_result


SEED = 10
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_saved_model(filepath, **kwargs):
    """Load local experiment models with either legacy or newer Keras."""
    if "safe_mode" in inspect.signature(keras.models.load_model).parameters:
        kwargs["safe_mode"] = False
    return keras.models.load_model(filepath, **kwargs)


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


def rrc_upsample_tensor(x, sps, span_symbols=8, beta=0.25):
    x_up = zero_upsample_tensor(x, sps)
    dtype = x_up.dtype

    half_len = span_symbols * sps // 2
    n = tf.range(-half_len, half_len + 1, dtype=dtype)
    t = n / tf.cast(sps, dtype)
    beta = tf.cast(beta, dtype)
    pi = tf.constant(np.pi, dtype=dtype)
    eps = tf.constant(1e-7, dtype=dtype)

    numerator = (
        tf.sin(pi * t * (1.0 - beta))
        + 4.0 * beta * t * tf.cos(pi * t * (1.0 + beta))
    )
    denominator = pi * t * (1.0 - tf.square(4.0 * beta * t))
    denominator_safe = tf.where(tf.abs(denominator) < eps, tf.ones_like(denominator), denominator)
    h_general = numerator / denominator_safe

    h_zero = 1.0 + beta * (4.0 / pi - 1.0)
    h_singular = (
        beta / tf.sqrt(tf.cast(2.0, dtype))
        * (
            (1.0 + 2.0 / pi) * tf.sin(pi / (4.0 * beta))
            + (1.0 - 2.0 / pi) * tf.cos(pi / (4.0 * beta))
        )
    )

    h = tf.where(tf.abs(t) < eps, h_zero, h_general)
    h = tf.where(tf.abs(tf.abs(t) - 1.0 / (4.0 * beta)) < eps, h_singular, h)
    h = h / (tf.reduce_max(tf.abs(h)) + eps)
    h = tf.reshape(h, [-1, 1, 1])

    y = tf.nn.conv1d(x_up[:, :, tf.newaxis], h, stride=1, padding="SAME")
    return tf.squeeze(y, axis=-1)


def downsample_batch(data, sps, offset=None):
    if offset is None:
        offset = 0
    return data[:, offset::sps]


def downsample_tensor(x, sps, offset=0):
    return x[:, offset::sps]


def ffe_identity_initializer(shape, dtype=None):
    w = np.zeros(shape, dtype=np.float32)
    center = shape[0] // 2
    w[center, 0, 0] = 1.0
    return w


def ffe_circular_pad(x, kernel_size):
    pad = (kernel_size - 1) // 2
    return tf.concat([x[:, -pad:, :], x, x[:, :pad, :]], axis=1)

def expand_channel_dim(x):
    return x[:, :, tf.newaxis]


def squeeze_channel_dim(x):
    return tf.squeeze(x, axis=-1)



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
    # Avoid UNPACK_SEQUENCE bytecode: Keras 2.10's Windows Lambda serializer
    # replaces byte 0x5c, which is this opcode in Python 3.9.
    x = inputs[0]
    ref = inputs[1]
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
    waveform_file = os.path.join(waveforms_dir, model_name + "_" + label + "_before_downsample.txt")
    np.savetxt(waveform_file, before_downsample.reshape(-1), fmt="%1.7e")
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
# The runner may override this without editing the experiment script.
snr = configured_snr(15)
snr_tag = "snr" + str(snr)
offset_candidates = list(range(sps))
# ffe_kernel_size = int(os.environ.get("CTLE_FFE_KERNEL_SIZE", "17"))
ffe_kernel_size = 9
# if ffe_kernel_size < 3 or ffe_kernel_size % 2 == 0:
#     raise ValueError("CTLE_FFE_KERNEL_SIZE must be an odd integer >= 3")
tap_tag = "tap" + str(ffe_kernel_size)
# Resolve input and output paths relative to this script.
output_root = os.path.dirname(os.path.abspath(__file__))
name_Tx = os.path.join(os.path.dirname(output_root), "data", "Txdata.txt")
channel_file = os.path.join(output_root, "channel", "DNN_channelest_TTHNet_SinM.h5")
channel_in_taps = 157
channel_output_dim = 85
experiment_name = "repeat_fc_nn_channel_" + snr_tag + "_" + tap_tag
models_dir = os.path.join(output_root, "model_trained")
offset_models_dir = os.path.join(models_dir, experiment_name + "_offsets")
weights_dir = os.path.join(output_root, "model_weights", experiment_name)
results_dir = os.path.join(output_root, "result")
waveforms_dir = os.path.join(results_dir, experiment_name + "_waveforms")

model_file = os.path.join(models_dir, experiment_name + "_model.keras")
final_weights_file = os.path.join(weights_dir, experiment_name + "_final.weights.h5")
offset_results_file = os.path.join(results_dir, experiment_name + "_offset_results.txt")

for output_dir in (models_dir, offset_models_dir, weights_dir, results_dir, waveforms_dir):
    os.makedirs(output_dir, exist_ok=True)

channel = load_saved_model(
    channel_file,
    custom_objects={
        "MyLayer": MyLayer,
        "MyLayer2": MyLayer2,
        "time_steps": channel_in_taps,
    },
    compile=False,
)
channel.trainable = False

data_raw = np.loadtxt(name_Tx).astype(np.float32)
split = int(len(data_raw) * ratio)
train_rms = np.sqrt(np.mean(np.square(data_raw[:split])) + 1e-12)
data_Tx = data_raw / train_rms
print("train RMS =", train_rms)

x_train = data_Tx[:split]
y_train = data_Tx[:split]
x_test = data_Tx[split:]
y_test = data_Tx[split:]


# 1      -> batch size閿涘矁銆冪粈铏瑰箛閸︺劌褰ч張澶夌閺佸瓨娼▔銏犺埌
# 81600  -> time閿涘矁銆冪粈楦跨箹閺夆剝灏濊ぐ銏℃箒 81600 娑擃亪鍣伴弽椋庡仯
# 1      -> channel閿涘矁銆冪粈鐑樼槨娑擃亪鍣伴弽椋庡仯閸欘亝婀佹稉鈧稉顏勭杽閺佹澘绠欐惔锕€鈧?x_train = to_symbol_batch(x_train)
x_train = to_symbol_batch(x_train)
y_train = to_symbol_batch(y_train)
x_test = to_symbol_batch(x_test)
y_test = to_symbol_batch(y_test)

def build_model(current_offset):
    NNinput = Input(shape=(None,), name="tx_input")

    data_upsampled = Lambda(repeat_upsample_tensor, arguments={"sps": sps}, name="tx_upsampling")(NNinput)

    data_ffe_input = Lambda(expand_channel_dim, name="tx_ffe_expand")(data_upsampled)

    data_ffe_input = Lambda(

        ffe_circular_pad,

        arguments={"kernel_size": ffe_kernel_size},

        name="tx_ffe_circular_pad",

    )(data_ffe_input)
    data_tx_ffe = Conv1D(
        filters=1,
        kernel_size=ffe_kernel_size,
        padding="valid",
        use_bias=False,
        activation="linear",
        kernel_initializer=ffe_identity_initializer,
        name="tx_ffe",
    )(data_ffe_input)
    data_tx_ffe = Lambda(squeeze_channel_dim, name="tx_ffe_squeeze")(data_tx_ffe)

    Tx = data_tx_ffe
    Tx = Lambda(channel_circular_pad, name="channel_circular_pad")(Tx)
    Tx = Lambda(channel_make_window, name="channel_make_window")(Tx)
    Tx = Lambda(channel_flatten_window, name="channel_flatten_window")(Tx)

    Rx = channel(Tx)
    Rx = Lambda(channel_take_center, name="channel_take_center")(Rx)
    data_ch = Lambda(channel_restore_waveform, name="channel_restore_waveform")([Rx, data_tx_ffe])
    data_ch = Lambda(add_awgn, arguments={"snr": snr}, name="awgn_noise")(data_ch)

    # data_CTLE = CTLELayer2(
    #     Fs=Fs,
    #     name="rx_ctle",
    # )(data_ch)
    # data_CTLE = CTLELayer_model_v12_v4(
    #     Fs=Fs,
    #     prewarp_frequency_hz=Rs / 2.0,
    #     name="rx_ctle",
    # )(data_ch)
    data_CTLE = CTLELayer4(
        Fs=Fs,
        name="rx_ctle",
    )(data_ch)
    # data_CTLE = tf.keras.layers.Rescaling(
    #     scale=-1.0,
    #     name="rx_ctle_polarity_compensation",
    # )(data_CTLE)
    data_symbol = Lambda(
        downsample_tensor,
        arguments={"sps": sps, "offset": current_offset},
        name="rx_downsampling",
    )(data_CTLE)
    # data_nor = NormalizeRMSLayer(name="output_normalization")(data_symbol)

    model = Model(inputs=NNinput, outputs=data_symbol, name="pre_ffe_post_ctle")
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=4e-2),
                  metrics=['mse'])

    return model


if retrain:
    offset_results = []

    for current_offset in offset_candidates:
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        offset_name = experiment_name + "_offset" + str(current_offset)
        current_model_file = os.path.join(offset_models_dir, offset_name + ".keras")
        current_final_weights_file = os.path.join(weights_dir, offset_name + "_final.weights.h5")
        current_best_weights_file = os.path.join(weights_dir, offset_name + "_best.weights.h5")

        model = build_model(current_offset)
        model.summary()

        current_reduce_lr = ReduceLROnPlateau(
            monitor="val_mse",
            factor=0.4,
            verbose=1,
            patience=6,
            min_lr=1e-9,
            mode="auto",
        )
        current_early_stop = EarlyStopping(
            monitor="val_mse",
            min_delta=1e-11,
            patience=26,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
        current_checkpoint = ModelCheckpoint(
            filepath=current_best_weights_file,
            save_weights_only=True,
            monitor="val_mse",
            save_best_only=True,
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=epoch,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[current_reduce_lr, current_early_stop, current_checkpoint],
        )

        model.load_weights(current_best_weights_file)
        model.save_weights(current_final_weights_file)
        model.save(current_model_file)

        output_val = model.predict(x_test, batch_size=batch_size)
        val_mse, val_ber = calculate_raw_metrics(
            output_val,
            y_test,
            np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) / train_rms,
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32),
        )
        offset_results.append((current_offset, val_mse, val_ber, current_model_file, current_final_weights_file))
        print("offset =", current_offset, "val MSE =", val_mse, "val BER =", val_ber)

    best_offset, best_mse, best_ber, best_model_file, best_offset_weights_file = min(
        offset_results,
        key=lambda item: item[1],
    )
    model = load_saved_model(
        best_model_file,
        custom_objects={
            "CTLELayer2": CTLELayer2,
            "CTLELayer4": CTLELayer4,
            "CTLELayer_model_v12_v4": CTLELayer_model_v12_v4,
            "FixedFIRChannelLayer": FixedFIRChannelLayer,
            "NormalizeRMSLayer": NormalizeRMSLayer,
            "MyLayer": MyLayer,
            "MyLayer2": MyLayer2,
            "ffe_identity_initializer": ffe_identity_initializer,
            "ffe_circular_pad": ffe_circular_pad,
            "expand_channel_dim": expand_channel_dim,
            "squeeze_channel_dim": squeeze_channel_dim,
            "add_awgn": add_awgn,
            "zero_upsample_tensor": zero_upsample_tensor,
            "repeat_upsample_tensor": repeat_upsample_tensor,
            "rrc_upsample_tensor": rrc_upsample_tensor,
            "downsample_tensor": downsample_tensor,
            "channel_squeeze": channel_squeeze,
            "channel_max_normalize": channel_max_normalize,
            "channel_circular_pad": channel_circular_pad,
            "channel_make_window": channel_make_window,
            "channel_flatten_window": channel_flatten_window,
            "channel_take_center": channel_take_center,
            "channel_restore_waveform": channel_restore_waveform,
        },
    )
    model.save_weights(final_weights_file)
    model.save(model_file)
    np.savetxt(
        offset_results_file,
        np.array([[item[0], item[1], item[2]] for item in offset_results], dtype=np.float64),
        fmt="%1.7e",
        header="offset val_mse val_ber",
    )
    print("best offset =", best_offset, "best val MSE =", best_mse, "best val BER =", best_ber)

    # Legacy four-parameter CTLELayer4 decoder is not valid for v12_v4.

model = load_saved_model(
    model_file,
    custom_objects={
        "CTLELayer2": CTLELayer2,
        "CTLELayer4": CTLELayer4,
        "CTLELayer_model_v12_v4": CTLELayer_model_v12_v4,
        "FixedFIRChannelLayer": FixedFIRChannelLayer,
        "NormalizeRMSLayer": NormalizeRMSLayer,
        "MyLayer": MyLayer,
        "MyLayer2": MyLayer2,
        "ffe_identity_initializer": ffe_identity_initializer,
        "ffe_circular_pad": ffe_circular_pad,
        "expand_channel_dim": expand_channel_dim,
        "squeeze_channel_dim": squeeze_channel_dim,
        "add_awgn": add_awgn,
        "zero_upsample_tensor": zero_upsample_tensor,
            "repeat_upsample_tensor": repeat_upsample_tensor,
        "rrc_upsample_tensor": rrc_upsample_tensor,
        "downsample_tensor": downsample_tensor,
        "channel_squeeze": channel_squeeze,
        "channel_max_normalize": channel_max_normalize,
        "channel_circular_pad": channel_circular_pad,
        "channel_make_window": channel_make_window,
        "channel_flatten_window": channel_flatten_window,
        "channel_take_center": channel_take_center,
        "channel_restore_waveform": channel_restore_waveform,
    },
)
constellation_raw = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
constellation_norm = constellation_raw / train_rms
gray_bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32)

output_train = model.predict(x_train, batch_size=batch_size)
train_mse, train_ber = calculate_raw_metrics(
    output_train,
    y_train,
    constellation_norm,
    gray_bits,
)
print("train MSE raw =", train_mse)
print("train BER raw =", train_ber)

output_test = model.predict(x_test, batch_size=batch_size)
test_mse, test_ber = calculate_raw_metrics(
    output_test,
    y_test,
    constellation_norm,
    gray_bits,
)
print("test MSE raw =", test_mse)
print("test BER raw =", test_ber)
report_experiment_result(
    model=model,
    model_name="Model_ch_FC",
    snr=snr,
    train_mse=train_mse,
    train_ber=train_ber,
    test_mse=test_mse,
    test_ber=test_ber,
    layer_names=["tx_ffe", "linear_nn_channel", "rx_ctle"],
    Fs=Fs,
)

tf.random.set_seed(SEED)
# save_before_downsample_probe(model, x_train, "train")
tf.random.set_seed(SEED + 1)
# save_before_downsample_probe(model, x_test, "test")
