import numpy as np
import tensorflow as tf

from channel_group_delay import channel_polynomials


def _as_numpy(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _ctle_polynomials(layer):
    """Return the complete v12_v4 digital CTLE numerator/denominator."""
    if not hasattr(layer, "digital_iir_coefficients"):
        raise TypeError(
            "Layer " + layer.name + " does not expose digital_iir_coefficients()"
        )
    core_b, core_a, transport_b, transport_a = (
        layer.digital_iir_coefficients()
    )
    b = np.convolve(_as_numpy(core_b), _as_numpy(transport_b)).astype(np.float64)
    a = np.convolve(_as_numpy(core_a), _as_numpy(transport_a)).astype(np.float64)
    return b, np.arange(len(b), dtype=np.float64), a, np.arange(len(a), dtype=np.float64)


def _conv1d_polynomials(layer):
    """Return the centered FIR represented by a same-length padded Conv1D."""
    kernel = _as_numpy(layer.kernel)
    if kernel.ndim != 3 or kernel.shape[1:] != (1, 1):
        raise ValueError(
            "Group-delay utility expects a single-input/single-output Conv1D; "
            + layer.name
            + " has kernel shape "
            + str(kernel.shape)
        )
    b = kernel[:, 0, 0].astype(np.float64)
    center = (len(b) - 1) / 2.0
    # Keras Conv1D is cross-correlation.  With symmetric circular padding,
    # tap k multiplies x[n-center+k], equivalent to delay center-k.
    delays = center - np.arange(len(b), dtype=np.float64)
    return b, delays, np.array([1.0]), np.array([0.0])


def _layer_polynomials(layer):
    if hasattr(layer, "digital_iir_coefficients"):
        return _ctle_polynomials(layer)
    if isinstance(layer, tf.keras.layers.Conv1D):
        return _conv1d_polynomials(layer)
    if isinstance(layer, tf.keras.Model):
        return channel_polynomials(layer)
    raise TypeError(
        "Unsupported equalizer layer " + layer.name + " (" + layer.__class__.__name__ + ")"
    )


def _bypass_blended_polynomials(ctle_layer, gate_layer):
    """Return the transfer function of ``(1-a)I + a*H_ctle``.

    The Joint topology feeds the CTLE output through a polarity compensation
    layer before blending it with the original input.  A constant polarity
    inversion has no standalone group delay, but its sign is essential inside
    the sum.  ``filtered_polarity`` records that sign for analysis.
    """
    b, _, a, _ = _ctle_polynomials(ctle_layer)
    if len(b) != len(a):
        raise ValueError("CTLE numerator and denominator orders must match for bypass analysis")
    alpha_value = gate_layer.alpha()
    alpha = float(_as_numpy(alpha_value))
    polarity = float(getattr(gate_layer, "filtered_polarity", 1.0))
    blended_b = (1.0 - alpha) * a + alpha * polarity * b
    delays = np.arange(len(a), dtype=np.float64)
    return blended_b, delays, a, delays


def maximum_equalizer_group_delay_ps(
    model,
    layer_names,
    Fs,
    f_min=100e3,
    f_max=50e9,
    points=65536,
    magnitude_floor_ratio=1e-10,
):
    """Maximum group delay of cascaded CTLE/FFE layers, in picoseconds.

    The upper endpoint is excluded because 50 GHz is Nyquist for Fs=100 GHz
    and the bilinear CTLE has a structural zero there.  Frequencies whose
    cascade magnitude is effectively zero are also excluded because their
    phase and group delay are undefined.
    """
    Fs = float(Fs)
    lower = float(f_min)
    upper = min(float(f_max), Fs / 2.0)
    if not 0.0 < lower < upper:
        raise ValueError("Require 0 < f_min < min(f_max, Fs/2)")
    if not layer_names:
        raise ValueError("layer_names must contain at least one equalizer layer")

    frequency = np.linspace(lower, upper, int(points), endpoint=False, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / Fs
    total_response = np.ones_like(omega, dtype=np.complex128)
    total_delay_samples = np.zeros_like(omega)

    names = list(layer_names)
    index = 0
    while index < len(names):
        layer_name = names[index]
        layer = model.get_layer(layer_name)
        next_layer = (
            model.get_layer(names[index + 1]) if index + 1 < len(names) else None
        )
        if (
            hasattr(layer, "digital_iir_coefficients")
            and next_layer is not None
            and next_layer.__class__.__name__ == "CTLEBypassGate"
        ):
            b, b_delay, a, a_delay = _bypass_blended_polynomials(layer, next_layer)
            index += 2
        else:
            b, b_delay, a, a_delay = _layer_polynomials(layer)
            index += 1
        b_phase = np.exp(-1j * b_delay[:, np.newaxis] * omega[np.newaxis, :])
        a_phase = np.exp(-1j * a_delay[:, np.newaxis] * omega[np.newaxis, :])
        numerator = b @ b_phase
        denominator = a @ a_phase
        numerator_moment = (b * b_delay) @ b_phase
        denominator_moment = (a * a_delay) @ a_phase

        response = numerator / denominator
        delay_samples = np.real(
            numerator_moment / numerator - denominator_moment / denominator
        )
        total_response *= response
        total_delay_samples += delay_samples

    magnitude = np.abs(total_response)
    magnitude_floor = float(magnitude_floor_ratio) * np.max(magnitude)
    delay_ps = total_delay_samples / Fs * 1e12
    valid = (
        np.isfinite(delay_ps)
        & np.isfinite(magnitude)
        & (magnitude > magnitude_floor)
    )
    if not np.any(valid):
        raise ValueError("No finite group-delay samples above the magnitude floor")
    return float(np.max(delay_ps[valid]))


def print_maximum_equalizer_group_delay(model, layer_names, Fs):
    value = maximum_equalizer_group_delay_ps(model, layer_names, Fs)
    print(
        "Maximum equalizer group delay (100 kHz to 50 GHz) =",
        value,
        "ps",
    )
    return value


def maximum_link_group_delay_ps(model, layer_names, Fs, **kwargs):
    """Alias emphasizing that layer_names may include the channel model."""
    return maximum_equalizer_group_delay_ps(
        model, layer_names, Fs, **kwargs
    )


def print_maximum_link_group_delay(model, layer_names, Fs):
    value = maximum_link_group_delay_ps(model, layer_names, Fs)
    print(
        "Maximum channel + equalizer group delay (100 kHz to 50 GHz) =",
        value,
        "ps",
    )
    return value
