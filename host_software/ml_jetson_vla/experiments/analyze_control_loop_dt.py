"""Cheap, no-hardware first step for `docs/TODO.md`'s `core/control_net.py` dt-mismatch item
(flagged 2026-10-14): compares the real per-tick dt distribution of the host loop that produced
`ControlNet`'s training data (`session_jetson_track4_*/telemetry.csv`) against the host loop
of a real deployment run (`run_policy_live.py`/`run_jetson_standalone.py` telemetry CSV -- same
`host_timestamp_ms` column, confirmed by reading both writers directly, so one tool handles
either without format-specific handling).

Correction vs. this item's original framing: `session_jetson_track4_*` sessions are NOT an
STM32-internal/hardware-rate signal -- `host_timestamp_ms` is written once per iteration of the
host-side collection loop in `data_collection/collect_vla_data.py`, which is itself bound by
`cap.read()` on a `cv2.CAP_PROP_FPS=30` webcam (confirmed by reading that script directly). So
both sides of this comparison are "host loop that calls `ControlNet.infer()` once per camera
frame" -- the real open question is whether a Jetson deployment run's camera-bound loop dt
distribution differs from the laptop-webcam collection loop's, not "hardware loop vs. software
loop."

Why this matters regardless: `ControlNet.infer()`'s `VEL_FILTER_ALPHA` EMA (see `core/control_net.py`)
is applied identically regardless of `actual_dt`, but a fixed-alpha EMA's effective time constant
is dt-dependent -- if the Jetson deployment loop's dt distribution differs meaningfully from the
training-data collection loop's, the filtered velocity the policy sees in deployment could lag
more (or less) than whatever it was implicitly exposed to during training.

This script only computes and reports real dt statistics -- it does not decide whether a given
mismatch is acceptable; that's a judgment call for whoever reads the numbers, informed by how
`ControlNet`'s policy was actually trained.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def dt_stats_ms(csv_path: str) -> dict[str, float]:
    """Real per-tick dt (ms) from a telemetry CSV's `host_timestamp_ms` column -- works for
    both STM32-native (`session_jetson_track4_*/telemetry.csv`) and Jetson-loop
    (`run_policy_live.py`/`run_jetson_standalone.py` output) files, confirmed to share this exact
    column name by reading both writers directly."""
    df = pd.read_csv(csv_path)
    if "host_timestamp_ms" not in df.columns:
        raise ValueError(
            f"{csv_path}: no 'host_timestamp_ms' column (columns: {list(df.columns)})"
        )
    ts = df["host_timestamp_ms"].to_numpy(dtype=np.float64)
    if len(ts) < 2:
        raise ValueError(f"{csv_path}: fewer than 2 rows, cannot compute dt")
    dt = np.diff(ts)
    return {
        "n_ticks": len(ts),
        "n_dt_samples": len(dt),
        "mean_ms": float(np.mean(dt)),
        "std_ms": float(np.std(dt)),
        "min_ms": float(np.min(dt)),
        "p50_ms": float(np.percentile(dt, 50)),
        "p90_ms": float(np.percentile(dt, 90)),
        "p99_ms": float(np.percentile(dt, 99)),
        "max_ms": float(np.max(dt)),
        "implied_hz_mean": float(1000.0 / np.mean(dt)) if np.mean(dt) > 0 else float("nan"),
    }


def _print_stats(label: str, stats: dict[str, float]) -> None:
    print(f"\n{label}:")
    print(f"  n_ticks={int(stats['n_ticks'])}  n_dt_samples={int(stats['n_dt_samples'])}")
    print(f"  mean={stats['mean_ms']:.2f}ms  std={stats['std_ms']:.2f}ms  "
          f"implied_hz={stats['implied_hz_mean']:.1f}")
    print(f"  min={stats['min_ms']:.2f}ms  p50={stats['p50_ms']:.2f}ms  "
          f"p90={stats['p90_ms']:.2f}ms  p99={stats['p99_ms']:.2f}ms  max={stats['max_ms']:.2f}ms")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-data", required=True,
        help="Path to a session_jetson_track4_*/telemetry.csv -- the host-loop (webcam "
             "collect_vla_data.py) timing that ControlNet's training data was collected under.",
    )
    parser.add_argument(
        "--deployment", default=None,
        help="Path to a run_policy_live.py/run_jetson_standalone.py telemetry CSV "
             "(real Jetson deployment's camera-bound loop). Optional -- if omitted, only the "
             "training-data side is reported (e.g. no deployment telemetry file available "
             "locally yet).",
    )
    args = parser.parse_args(argv)

    train_stats = dt_stats_ms(args.training_data)
    _print_stats(f"Training-data collection loop ({args.training_data})", train_stats)

    if args.deployment is None:
        print(
            "\nNo --deployment file given -- cannot compare yet. Copy a real "
            "run_policy_live.py *.telemetry.csv (or run_jetson_standalone.py's own telemetry "
            "output) back from the Jetson and re-run with --deployment <path> to complete the "
            "comparison this script exists for."
        )
        return 0

    deploy_stats = dt_stats_ms(args.deployment)
    _print_stats(f"Deployment loop ({args.deployment})", deploy_stats)

    ratio = deploy_stats["mean_ms"] / train_stats["mean_ms"] if train_stats["mean_ms"] > 0 else float("nan")
    print(
        f"\nDeployment/training-data mean-dt ratio: {ratio:.2f}x "
        f"({deploy_stats['mean_ms']:.2f}ms vs {train_stats['mean_ms']:.2f}ms)"
    )
    print(
        "Interpretation left to the reader: a fixed-alpha EMA's effective time constant is "
        "dt-dependent, so a ratio far from 1.0x means ControlNet's velocity filtering is "
        "operating at a materially different effective smoothing timescale in deployment than "
        "whatever generated the training data's own as-collected behavior."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
