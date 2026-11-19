"""Train CTLELayer_u Post stages 1-6 at SNR 5/10/15.

Each completed run is appended to summary.csv, so --resume can continue an
interrupted sweep without retraining completed combinations.  FPost is
unchanged and its existing model and measurements are included for reference.
"""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "result" / "post_stage_sweep_u"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"
FIELDS = (
    "snr_db",
    "model",
    "ctle_stages",
    "ffe_taps",
    "best_offset",
    "val_mse",
    "val_ber",
    "elapsed_seconds",
    "model_file",
    "log_file",
)
RUNS = (
    ("1-stage CTLE", "Model_ch_Post.py", 1, 0, "pw_post_ctle_nn_channel_"),
    ("2-stage CTLE", "Model_ch_Post_2stage.py", 2, 0, "pw_post_2stage_ctle_nn_channel_"),
    ("3-stage CTLE", "Model_ch_Post_3stage.py", 3, 0, "pw_post_3stage_ctle_nn_channel_"),
    ("4-stage CTLE", "Model_ch_Post_4stage.py", 4, 0, "pw_post_4stage_ctle_nn_channel_"),
    ("5-stage CTLE", "Model_ch_Post_5stage.py", 5, 0, "pw_post_5stage_ctle_nn_channel_"),
    ("6-stage CTLE", "Model_ch_Post_6stage.py", 6, 0, "pw_post_6stage_ctle_nn_channel_"),
    ("FPost", "Model_ch_FPost.py", 0, 9, "fpost_nn_channel_"),
)


def read_completed():
    if not SUMMARY_FILE.is_file():
        return {}
    with SUMMARY_FILE.open(newline="", encoding="utf-8") as stream:
        return {
            (float(row["snr_db"]), row["model"]): row
            for row in csv.DictReader(stream)
        }


def append_result(row):
    write_header = not SUMMARY_FILE.is_file()
    with SUMMARY_FILE.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def output_paths(model_name, stages, taps, snr):
    snr_tag = "snr" + str(snr)
    if stages:
        prefix = "pw_post_ctle_nn_channel_" if stages == 1 else f"pw_post_{stages}stage_ctle_nn_channel_"
        experiment = prefix + snr_tag + "_u"
    else:
        experiment = f"fpost_nn_channel_{snr_tag}_tap{taps}"
    return (
        MODEL_DIR / "result" / f"{experiment}_offset_results.txt",
        MODEL_DIR / "model_trained" / f"{experiment}_model.keras",
        OUTPUT_DIR / f"{experiment}.log",
    )


def parse_best_offset(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        offset, mse, ber = (float(value) for value in line.split())
        rows.append((int(offset), mse, ber))
    if len(rows) != 4 or {row[0] for row in rows} != {0, 1, 2, 3}:
        raise RuntimeError(f"Expected four phase results in {path}, got {rows}")
    return min(rows, key=lambda row: row[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        completed = read_completed()
    else:
        # A non-resume run is a fresh benchmark.  Avoid mixing stale rows
        # from a previous implementation with the newly trained models.
        SUMMARY_FILE.unlink(missing_ok=True)
        completed = {}
    total = len(RUNS) * 3
    run_number = 0
    for snr in (5, 10, 15):
        for model_name, script, stages, taps, _ in RUNS:
            run_number += 1
            key = (float(snr), model_name)
            if key in completed:
                print(f"[{run_number}/{total}] SKIP SNR={snr} {model_name}", flush=True)
                continue
            result_file, model_file, log_file = output_paths(model_name, stages, taps, snr)
            if not stages:
                # FPost has no CTLELayer, so the previously trained reference
                # remains valid and should not be overwritten by this sweep.
                old_log = (MODEL_DIR / "result" / "post_stage_sweep"
                           / f"fpost_nn_channel_snr{snr}_tap{taps}.log")
                best_offset, val_mse, val_ber = parse_best_offset(result_file)
                if not model_file.is_file():
                    raise FileNotFoundError(model_file)
                row = {
                    "snr_db": snr, "model": model_name,
                    "ctle_stages": stages, "ffe_taps": taps,
                    "best_offset": best_offset,
                    "val_mse": f"{val_mse:.10g}",
                    "val_ber": f"{val_ber:.10g}",
                    "elapsed_seconds": "0",
                    "model_file": str(model_file),
                    "log_file": str(old_log),
                }
                append_result(row)
                print(f"[{run_number}/{total}] REUSE SNR={snr} {model_name}", flush=True)
                continue
            env = os.environ.copy()
            env["CTLE_SNR"] = str(snr)
            env["CTLE_RETRAIN"] = "1"
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            started = time.monotonic()
            print(f"[{run_number}/{total}] START SNR={snr} {model_name}", flush=True)
            with log_file.open("w", encoding="utf-8", errors="replace") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(MODEL_DIR / script)],
                    cwd=str(MODEL_DIR),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                return_code = process.wait()
            elapsed = time.monotonic() - started
            if return_code:
                print(f"FAILED: {log_file}", flush=True)
                return return_code
            best_offset, val_mse, val_ber = parse_best_offset(result_file)
            if not model_file.is_file():
                raise FileNotFoundError(model_file)
            row = {
                "snr_db": snr,
                "model": model_name,
                "ctle_stages": stages,
                "ffe_taps": taps,
                "best_offset": best_offset,
                "val_mse": f"{val_mse:.10g}",
                "val_ber": f"{val_ber:.10g}",
                "elapsed_seconds": f"{elapsed:.3f}",
                "model_file": str(model_file),
                "log_file": str(log_file),
            }
            append_result(row)
            print(
                f"[{run_number}/{total}] DONE  SNR={snr} {model_name}: "
                f"offset={best_offset}, MSE={val_mse:.7g}, BER={val_ber:.7g}, {elapsed:.1f}s",
                flush=True,
            )
    print(f"Sweep complete: {SUMMARY_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
