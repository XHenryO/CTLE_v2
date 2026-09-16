
"""Shared configuration and machine-readable reporting for SNR sweeps."""

import json
import os

from equalizer_group_delay import maximum_link_group_delay_ps


RESULT_PREFIX = "CTLE_EXPERIMENT_RESULT="


def _environment_number(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    number = float(value)
    return int(number) if number.is_integer() else number


def configured_snr(default=5):
    """Return the SNR requested by the batch runner, or the script default."""
    return _environment_number("CTLE_SNR", default)


def configured_retrain(default=1):
    """Return 0/1 retraining mode requested by the batch runner."""
    value = int(_environment_number("CTLE_RETRAIN", default))
    if value not in (0, 1):
        raise ValueError("CTLE_RETRAIN must be 0 or 1")
    return value


def report_experiment_result(
    *,
    model,
    model_name,
    snr,
    train_mse,
    train_ber,
    test_mse,
    test_ber,
    layer_names,
    Fs,
):
    """Print requested metrics and one JSON record for the batch runner."""
    bands = (
        ("gd_100k_12p5g_ps", 12.5e9),
        ("gd_100k_25g_ps", 25.0e9),
        ("gd_100k_50g_ps", 50.0e9),
    )
    # group_delays = {}
    # for field_name, upper_frequency in bands:
    #     value = maximum_link_group_delay_ps(
    #         model,
    #         layer_names,
    #         Fs,
    #         f_min=100e3,
    #         f_max=upper_frequency,
    #     )
    #     group_delays[field_name] = value
    #     print(
    #         "Maximum channel + equalizer group delay "
    #         f"(100 kHz to {upper_frequency / 1e9:g} GHz) = {value} ps"
    #     )

    result = {
        "model": str(model_name),
        "snr_db": float(snr),
        "train_mse": float(train_mse),
        "train_ber": float(train_ber),
        "test_mse": float(test_mse),
        "test_ber": float(test_ber),

    }
    # ** group_delays,
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return result
