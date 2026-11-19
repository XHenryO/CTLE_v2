"""Stable initialization and optimizer settings for cascaded Post CTLEs."""

import numpy as np


BASE_LEARNING_RATE = 5e-2


def ctle_stage_initialization(Fs, stage_index):
    """Return a deterministic, symmetry-breaking initialization.

    The first CTLE keeps the original 5/25/35-GHz initialization.  Extra
    stages start close to unity inside the 0--12.5-GHz signal band, with
    slightly different zero frequencies.  This avoids both an Nth-power
    initial response and the identical-gradient symmetry that otherwise
    keeps every stage equal throughout training.
    """
    if stage_index < 1:
        raise ValueError("stage_index must start at 1")
    if stage_index == 1:
        return {}

    Fs = float(Fs)
    fz_fraction = 0.36 + 0.02 * (stage_index - 2)
    if fz_fraction > 0.44:
        raise ValueError("near-unity initialization is defined for at most 6 stages")
    fz = fz_fraction * Fs
    fp1 = 0.47 * Fs
    # Smooth logit mapping requires a strictly interior initial value.
    fp2 = 0.485 * Fs
    return {
        "init_tau_z": 1.0 / (2.0 * np.pi * fz),
        "init_tau_p1": 1.0 / (2.0 * np.pi * fp1),
        "init_tau_p2": 1.0 / (2.0 * np.pi * fp2),
        "init_A0": 1.0,
    }


def cascaded_ctle_learning_rate(stage_count):
    """Scale Adam's step because log-gain changes add across a cascade."""
    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    return BASE_LEARNING_RATE / float(stage_count)


def print_trained_ctle_stages(model, stage_count):
    """Report effective parameters from a loaded model, not raw u weights."""
    for stage_index in range(1, stage_count + 1):
        name = "rx_ctle" if stage_index == 1 else f"rx_ctle_stage{stage_index}"
        layer = model.get_layer(name)
        fz, fp1, fp2, A0 = [
            float(value.numpy()) for value in layer.frequencies_and_gain()
        ]
        print(f"{name}: A0={A0:.8g}, fz={fz/1e9:.6g} GHz, "
              f"fp1={fp1/1e9:.6g} GHz, fp2={fp2/1e9:.6g} GHz")
