import argparse

import numpy as np
import tensorflow as tf

from SelfDesign import MyLayer, MyLayer2


def load_linear_channel_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "MyLayer": MyLayer,
            "MyLayer2": MyLayer2,
            "time_steps": 157,
        },
        compile=False,
        safe_mode=False,
    )


def channel_polynomials(channel_model, output_index=None):
    """Extract the centered FIR used by the sliding-window channel model."""
    if len(channel_model.inputs) != 1 or len(channel_model.outputs) != 1:
        raise ValueError("The channel model must have exactly one input and one output")

    dense_layers = [
        layer
        for layer in channel_model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    if len(dense_layers) != 1:
        raise ValueError(
            "Expected one Dense layer in the linear channel, found "
            + str(len(dense_layers))
        )
    dense = dense_layers[0]
    activation = tf.keras.activations.serialize(dense.activation)
    if activation != "linear" or dense.use_bias:
        raise ValueError(
            "Channel is not strictly linear: Dense must use linear activation "
            "and use_bias=False"
        )

    kernel = np.asarray(dense.kernel.numpy(), dtype=np.float64)
    input_taps, output_dim = kernel.shape
    if output_index is None:
        output_index = output_dim // 2
    output_index = int(output_index)
    if not 0 <= output_index < output_dim:
        raise ValueError("output_index is outside the channel output dimension")

    coefficients = kernel[:, output_index]
    center = (input_taps - 1) / 2.0
    # The model receives x[n-center:n+center].  Dense kernel row k therefore
    # multiplies x[n-center+k], equivalent to delay center-k.
    delays = center - np.arange(input_taps, dtype=np.float64)
    return coefficients, delays, np.array([1.0]), np.array([0.0])


def maximum_channel_group_delay_ps(
    channel_model,
    Fs,
    f_min=100e3,
    f_max=50e9,
    points=65536,
    magnitude_floor_ratio=1e-10,
):
    coefficients, delays, _, _ = channel_polynomials(channel_model)
    Fs = float(Fs)
    lower = float(f_min)
    upper = min(float(f_max), Fs / 2.0)
    if not 0.0 < lower < upper:
        raise ValueError("Require 0 < f_min < min(f_max, Fs/2)")

    frequency = np.linspace(lower, upper, int(points), endpoint=False, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / Fs
    phase_matrix = np.exp(-1j * delays[:, np.newaxis] * omega[np.newaxis, :])
    response = coefficients @ phase_matrix
    moment = (coefficients * delays) @ phase_matrix
    delay_ps = np.real(moment / response) / Fs * 1e12

    magnitude = np.abs(response)
    floor = float(magnitude_floor_ratio) * np.max(magnitude)
    valid = np.isfinite(delay_ps) & np.isfinite(magnitude) & (magnitude > floor)
    if not np.any(valid):
        raise ValueError("No finite channel group-delay samples above the magnitude floor")
    return float(np.max(delay_ps[valid]))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate maximum group delay of the saved linear channel"
    )
    parser.add_argument(
        "--model",
        default="channel/DNN_channelest_TTHNet_Sin.h5",
        help="Path to the saved Keras channel model",
    )
    parser.add_argument("--fs", type=float, default=100e9, help="Sample rate in Hz")
    parser.add_argument("--f-min", type=float, default=100e3)
    parser.add_argument("--f-max", type=float, default=50e9)
    args = parser.parse_args()

    channel = load_linear_channel_model(args.model)
    value = maximum_channel_group_delay_ps(
        channel,
        Fs=args.fs,
        f_min=args.f_min,
        f_max=args.f_max,
    )
    print("Maximum channel group delay (100 kHz to 50 GHz) =", value, "ps")


if __name__ == "__main__":
    main()
