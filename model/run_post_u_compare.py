"""Train representative CTLELayer_u models alongside preserved clip results.

Run with the project's Python 3.9 interpreter from any directory.  Results,
logs, and a side-by-side CSV are written to result/post_stage_u_compare.
"""

import csv
import os
from pathlib import Path
import subprocess
import sys
import time


MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "result" / "post_stage_u_compare"
BASELINE_FILE = MODEL_DIR / "result" / "post_stage_sweep" / "summary.csv"
CASES = (
    (10, 1, "Model_ch_Post.py", "pw_post_ctle_nn_channel_"),
    (10, 4, "Model_ch_Post_4stage.py", "pw_post_4stage_ctle_nn_channel_"),
    (10, 6, "Model_ch_Post_6stage.py", "pw_post_6stage_ctle_nn_channel_"),
)
FIELDS = (
    "snr_db", "stages", "old_offset", "new_offset", "old_val_mse",
    "new_val_mse", "old_val_ber", "new_val_ber", "elapsed_seconds",
    "new_model_file", "new_log_file",
)


def offset_metrics(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        offset, mse, ber = (float(value) for value in line.split())
        rows.append((int(offset), mse, ber))
    if len(rows) != 4 or {row[0] for row in rows} != {0, 1, 2, 3}:
        raise ValueError(f"Expected offsets 0-3 in {path}")
    return min(rows, key=lambda row: row[1])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with BASELINE_FILE.open(newline="", encoding="utf-8") as stream:
        baseline = {
            (int(float(row["snr_db"])), int(row["ctle_stages"])): row
            for row in csv.DictReader(stream)
        }
    rows = []
    for index, (snr, stages, script, prefix) in enumerate(CASES, 1):
        experiment = f"{prefix}snr{snr}_u"
        log_file = OUTPUT_DIR / f"{experiment}.log"
        result_file = MODEL_DIR / "result" / f"{experiment}_offset_results.txt"
        model_file = MODEL_DIR / "model_trained" / f"{experiment}_model.keras"
        environment = os.environ.copy()
        environment.update({
            "CTLE_SNR": str(snr), "CTLE_RETRAIN": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        })
        print(f"[{index}/{len(CASES)}] Training SNR={snr}, {stages}-stage", flush=True)
        start = time.monotonic()
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            code = subprocess.call(
                [sys.executable, "-u", str(MODEL_DIR / script)],
                cwd=str(MODEL_DIR), env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        if code:
            raise RuntimeError(f"Training failed ({code}); see {log_file}")
        if not model_file.is_file():
            raise FileNotFoundError(model_file)
        new_offset, new_mse, new_ber = offset_metrics(result_file)
        old = baseline[(snr, stages)]
        row = {
            "snr_db": snr, "stages": stages,
            "old_offset": old["best_offset"], "new_offset": new_offset,
            "old_val_mse": old["val_mse"], "new_val_mse": new_mse,
            "old_val_ber": old["val_ber"], "new_val_ber": new_ber,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "new_model_file": str(model_file), "new_log_file": str(log_file),
        }
        rows.append(row)
        with (OUTPUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[{index}/{len(CASES)}] MSE {float(old['val_mse']):.6g} -> {new_mse:.6g}; "
            f"BER {float(old['val_ber']):.6g} -> {new_ber:.6g}",
            flush=True,
        )
    print(f"Comparison saved to {OUTPUT_DIR / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
