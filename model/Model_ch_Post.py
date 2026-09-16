# import numpy as np
# import tensorflow as tf
from tensorflow import keras
# import CTLELayer
from SelfDesign import MyLayer, MyLayer2

import inspect
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from Layer import CTLELayer, FixedFIRChannelLayer, NormalizeRMSLayer, CTLELayer2, CTLELayer3, CTLELayer4, \
    CTLELayer_model, CTLELayer_model2, CTLELayer_model_v12, CTLELayer_model_v12_v2, \
    CTLELayer_model_v12_v3, CTLELayer_model_v12_v4
from equalizer_group_delay import print_maximum_link_group_delay
from experiment_reporting import configured_retrain, configured_snr, report_experiment_result

# from Layer_tau import CTLELayer


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
    out = np.zeros((data.shape[0], data.shape[1] * sps), dtype=np.float32)
    out[:, ::sps] = data
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
    # Avoid Python 3.9 UNPACK_SEQUENCE bytecode in Keras 2.10's Windows
    # Lambda serializer, which replaces byte 0x5c while saving functions.
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

        if ctle_prefixes:
            prefix = sorted(ctle_prefixes)[0]
            values = []
            index = 0
            while prefix + "/" + str(index) in f:
                values.append(float(f[prefix + "/" + str(index)][()]))
                index += 1
        else:
            # Keras 2 saves CTLELayer4 variables by name rather than index.
            variable_names = ("log_tau_z", "log_tau_p1", "log_tau_p2", "log_A0")
            named_values = {}

            def visit_legacy(name, obj):
                variable_name = name.rsplit("/", 1)[-1].split(":", 1)[0]
                if (isinstance(obj, h5py.Dataset)
                        and "rx_ctle" in name.split("/")[:-1]
                        and variable_name in variable_names):
                    named_values[variable_name] = float(obj[()])

            f.visititems(visit_legacy)
            if not all(name in named_values for name in variable_names):
                print("CTLELayer weights not found in:", weight_file)
                return
            values = [named_values[name] for name in variable_names]

    if len(values) == 5:
        # Raw CTLE variables must be decoded by the same layer version that
        # wrote them because v11 and v12 use different conditional bounds.
        if "ctle_layer_model_v12" in prefix:
            layer_class = CTLELayer_model_v12
        else:
            layer_class = CTLELayer_model2

        layer = layer_class(Fs=Fs)
        layer.build((None, 64, 1))
        for variable, value in zip(layer.trainable_variables, values):
            variable.assign(value)
        parameters = {
            name: float(value.numpy())
            for name, value in layer.physical_parameters().items()
        }
        print(layer_class.__name__ + " raw weights:")
        for variable, value in zip(layer.trainable_variables, values):
            print(variable.name, "=", value)
        print(layer_class.__name__ + " physical parameters:")
        print("RS_ohm =", parameters["RS"])
        print("CS_F =", parameters["CS"])
        print("RL_ohm =", parameters["RL"])
        print("CL_F =", parameters["CL"])
        print("ibias_A =", parameters["ibias"])
        return

    if len(values) < 4:
        print("unexpected CTLE weight count:", len(values), "in", weight_file)
        return

    log_tau_z, log_tau_p1, log_tau_p2, log_A0 = values[:4]

    tau_z = np.clip(np.exp(log_tau_z), 1e-15, 1e-6)
    tau_p1 = np.clip(np.exp(log_tau_p1), 1e-15, 1e-6)
    tau_p2 = np.clip(np.exp(log_tau_p2), 1e-15, 1e-6)
    A0 = np.clip(np.exp(log_A0), 1e-6, 1e3)

    print("CTLELayer weights:")
    print("log_tau_z =", log_tau_z, "tau_z =", tau_z)
    print("log_tau_p1 =", log_tau_p1, "tau_p1 =", tau_p1)
    print("log_tau_p2 =", log_tau_p2, "tau_p2 =", tau_p2)
    print("log_A0 =", log_A0, "A0 =", A0)


retrain = configured_retrain(1)
VM = 0
epoch = 500
batch_size = 16
ratio = 0.8
Rs = 25e9
sps = 4
Fs = Rs * sps
snr = 10
snr_tag = "snr" + str(snr)
offset_candidates = list(range(sps))
# Resolve inputs and outputs independently of the launch directory.
output_root = os.path.dirname(os.path.abspath(__file__))
name_Tx = os.path.join(os.path.dirname(output_root), "data", "Txdata.txt")
channel_file = os.path.join(output_root, "channel", "DNN_channelest_TTHNet_SinM.h5")
channel_in_taps = 157
channel_output_dim = 85
experiment_name = "pw_post_ctle_nn_channel_" + snr_tag
models_dir = os.path.join(output_root, "model_trained")
offset_models_dir = os.path.join(models_dir, experiment_name + "_offsets")
weights_dir = os.path.join(output_root, "model_weights", experiment_name)
results_dir = os.path.join(output_root, "result")
waveforms_dir = os.path.join(results_dir, experiment_name + "_waveforms")

model_file = os.path.join(models_dir, experiment_name + "_model.keras")
final_weights_file = os.path.join(weights_dir, experiment_name + "_final.weights.h5")
offset_results_file = os.path.join(results_dir, experiment_name + "_offset_results.txt")
vm_offset = 0
vm_A0 = 1.21575
vm_fz = 4.7798e9
vm_fp1 = 5.9057e9
vm_fp2 = 22.4565e9

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

def build_model(current_offset, use_vm_ctle=False):
    NNinput = Input(shape=(None,), name="tx_input")


    data_upsampled = Lambda(repeat_upsample_tensor, arguments={"sps": sps}, name="tx_upsampling")(NNinput)

    Tx = data_upsampled
    Tx = Lambda(channel_circular_pad, name="channel_circular_pad")(Tx)
    Tx = Lambda(channel_make_window, name="channel_make_window")(Tx)
    Tx = Lambda(channel_flatten_window, name="channel_flatten_window")(Tx)

    Rx = channel(Tx)
    Rx = Lambda(channel_take_center, name="channel_take_center")(Rx)
    data_ch = Lambda(channel_restore_waveform, name="channel_restore_waveform")([Rx, data_upsampled])
    data_ch = Lambda(add_awgn, arguments={"snr": snr}, name="awgn_noise")(data_ch)

    # data_CTLE = CTLELayer(
    #     Fs=Fs,
    #     gain=1.0,
    #     fz=10e9,
    #     fp1=70e9,
    #     fp2=80e9,
    #     name="rx_ctle",
    # )(data_ch)
    # data_CTLE = CTLELayer2(
    #     Fs=Fs,
    #     name="rx_ctle",
    # )(data_ch)
    # 原来的零极点 CTLE 调用暂时停用；需要恢复时取消下面这段注释。
    # data_CTLE = CTLELayer_model_v12_v4(
    #     Fs=Fs,
    #     prewarp_frequency_hz=Rs / 2.0,
    #     name="rx_ctle",
    # )(data_ch)

    data_CTLE = CTLELayer(
        Fs=Fs,
        name="rx_ctle",
    )(data_ch)



    # v12 physical IIR CTLE with prewarped Tustin discretization.
    # data_CTLE = CTLELayer_model_v12_v4(
    #     Fs=Fs,
    #     name="rx_ctle",
    # )(data_ch)

    # CTLELayer4 is defined with a positive low-frequency gain A0.  Do not
    # apply the Cadence common-source polarity compensation used by the
    # physical v12 model; doing so makes the positive-only A0 collapse to zero.
    # data_CTLE = tf.keras.layers.Rescaling(
    #     scale=-1.0,
    #     name="ctle_polarity_compensation",
    # )(data_CTLE)

    # Rx_ctle = CTLELayer(
    #     Fs=192e9,
    #     init_tau_z=2e-12,
    #     init_tau_p1=1e-11,
    #     init_tau_p2=5e-11,
    #     init_A0=1.0,
    #     name="rx_ctle"
    # )(Rx2)
    data_symbol = Lambda(
        downsample_tensor,
        arguments={"sps": sps, "offset": current_offset},
        name="rx_downsampling",
    )(data_CTLE)
    # data_nor = NormalizeRMSLayer(name="output_normalization")(data_symbol)

    model = Model(inputs=NNinput, outputs=data_symbol, name="post_equalization_ctle")
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=5e-2),
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
            "CTLELayer": CTLELayer,
            "CTLELayer2": CTLELayer2,
            "CTLELayer3": CTLELayer3,
            "CTLELayer4": CTLELayer4,
            "CTLELayer_model": CTLELayer_model,
            "CTLELayer_model2": CTLELayer_model2,
            "CTLELayer_model_v12": CTLELayer_model_v12,
            "CTLELayer_model_v12_v2": CTLELayer_model_v12_v2,
            "CTLELayer_model_v12_v3": CTLELayer_model_v12_v3,
            "CTLELayer_model_v12_v4": CTLELayer_model_v12_v4,
            "FixedFIRChannelLayer": FixedFIRChannelLayer,
            "NormalizeRMSLayer": NormalizeRMSLayer,
            "MyLayer": MyLayer,
            "MyLayer2": MyLayer2,
            "add_awgn": add_awgn,
            "zero_upsample_tensor": zero_upsample_tensor,
            "repeat_upsample_tensor": repeat_upsample_tensor,
            "downsample_tensor": downsample_tensor,
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

if VM == 0:
    print_ctle_weights(final_weights_file)
    model = load_saved_model(
        model_file,
        custom_objects={
            "CTLELayer": CTLELayer,
            "CTLELayer2": CTLELayer2,
            "CTLELayer3": CTLELayer3,
            "CTLELayer4": CTLELayer4,
            "CTLELayer_model": CTLELayer_model,
            "CTLELayer_model2": CTLELayer_model2,
            "CTLELayer_model_v12": CTLELayer_model_v12,
            "CTLELayer_model_v12_v2": CTLELayer_model_v12_v2,
            "CTLELayer_model_v12_v3": CTLELayer_model_v12_v3,
            "CTLELayer_model_v12_v4": CTLELayer_model_v12_v4,
            "FixedFIRChannelLayer": FixedFIRChannelLayer,
            "NormalizeRMSLayer": NormalizeRMSLayer,
            "MyLayer": MyLayer,
            "MyLayer2": MyLayer2,
            "add_awgn": add_awgn,
            "zero_upsample_tensor": zero_upsample_tensor,
            "repeat_upsample_tensor": repeat_upsample_tensor,
            "downsample_tensor": downsample_tensor,
            "channel_max_normalize": channel_max_normalize,
            "channel_circular_pad": channel_circular_pad,
            "channel_make_window": channel_make_window,
            "channel_flatten_window": channel_flatten_window,
            "channel_take_center": channel_take_center,
            "channel_restore_waveform": channel_restore_waveform,
        },
    )
    print("VM = 0, load trained model:", model_file)
else:
    model = build_model(vm_offset, use_vm_ctle=True)
    print("VM = 1, use fitted CTLE parameters")
    print("vm offset =", vm_offset)
    print("vm A0 =", vm_A0)
    print("vm fz =", vm_fz, "fp1 =", vm_fp1, "fp2 =", vm_fp2)


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
# report_experiment_result(
#     model=model,
#     model_name="Model_ch_Post",
#     snr=snr,
#     train_mse=train_mse,
#     train_ber=train_ber,
#     test_mse=test_mse,
#     test_ber=test_ber,
#     layer_names=["linear_nn_channel", "rx_ctle"],
#     Fs=Fs,
# )

tf.random.set_seed(SEED)
save_before_downsample_probe(model, x_train, "train")
tf.random.set_seed(SEED + 1)
save_before_downsample_probe(model, x_test, "test")
