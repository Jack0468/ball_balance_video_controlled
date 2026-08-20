"""Estimates the R (measurement noise) and Q (process noise) matrices a
constant-velocity Kalman filter would use, from a ground-truth telemetry CSV
collected via --scripted-sequence (see scripted_command_sequencer.py).

Splits the recording by elapsed time at --phase-split-s (default 30.0,
matching ScriptedCommandSequencer's default schedule):
  - Phase 1 (before the split): the ball should be sitting still near center
    (no command was issued, TargetStateMachine defaults to "center"). This
    window gives R -- checked earlier this session that a MOVING ball
    contaminates the vision-vs-touch error with pure timing-skew between the
    two sensors' unsynchronized clocks, not genuine sensor noise, so R must
    come from a still window specifically, not from any recording where the
    ball's in motion.
  - Phase 2 (after the split): the scripted forward/backward/left/right
    sequence sweeps the target through varied motion. Differentiating
    touch_x_mm/touch_y_mm twice against mcu_ms gives an empirical
    acceleration signal; its spread is what sets Q (how much real motion
    deviates from the constant-velocity model between measurements).

Run as a module from the repo root:
    python -m host_software.ml_vision.evaluations.estimate_kalman_noise_params \
        --csv host_software/data/01_bronze/evaluation/ground_truth_<timestamp>.csv
"""

import argparse
import json

import numpy as np
import pandas as pd


def estimate_r(df_still: pd.DataFrame) -> np.ndarray:
    # raw_err_x/y_mm (raw CNN vs touch), NOT err_x/y_mm (fully processed
    # vision -- post-gate/dead-band/MLP -- vs touch). The filter being designed
    # is meant to REPLACE that processing stack, so R must characterize the
    # raw sensor it will actually operate on, not the current stack's already-
    # smoothed output. Requires a recording taken after the touch_logger.py
    # raw-signal logging change (PROJECT_LOGBOOK.md 20/08) -- older recordings
    # have these columns blank and will report 0 valid rows here.
    valid = df_still[df_still["touch_valid"] == 1].dropna(subset=["raw_err_x_mm", "raw_err_y_mm"])
    if len(valid) < 10:
        print(f"WARNING: only {len(valid)} valid still-phase rows with raw_err_x/y_mm populated -- "
              f"R estimate will be unreliable. (0 here means this recording predates the raw-signal "
              f"logging fix -- re-record, don't trust a 0-row estimate.)")
    err_x = valid["raw_err_x_mm"].to_numpy()
    err_y = valid["raw_err_y_mm"].to_numpy()

    print(f"\n=== R (measurement noise, from RAW CNN vs touch) -- {len(valid)} still-phase rows ===")
    print(f"mean raw_err_x_mm: {err_x.mean():+.3f} (bias check -- should be small vs. std below)" if len(valid) else "no data")
    print(f"mean raw_err_y_mm: {err_y.mean():+.3f}" if len(valid) else "")
    r = np.cov(err_x, err_y) if len(valid) > 1 else np.eye(2)
    print(f"R matrix (mm^2):\n{r}")
    return r


def estimate_q(dfs_moving) -> np.ndarray:
    # dfs_moving: a single moving-phase DataFrame, or a list of them (one per
    # recording, when pooling multiple runs). mcu_ms is a per-recording MCU
    # clock (resets each session/boot) -- differentiation MUST happen within
    # a single recording's own rows. Pooling raw rows across recordings
    # before differentiating would interleave unrelated timelines and
    # produce garbage accelerations at the seams, so each recording is
    # differentiated independently here and only the resulting acceleration
    # *samples* (not the raw position rows) are pooled across recordings.
    if isinstance(dfs_moving, pd.DataFrame):
        dfs_moving = [dfs_moving]

    all_ax, all_ay, all_dt2, total_valid_rows = [], [], [], 0
    for df_moving in dfs_moving:
        valid = df_moving[df_moving["touch_valid"] == 1].dropna(subset=["touch_x_mm", "touch_y_mm", "mcu_ms"]).copy()
        valid = valid.sort_values("mcu_ms")
        if len(valid) < 20:
            continue
        total_valid_rows += len(valid)

        t = valid["mcu_ms"].to_numpy(dtype=float) / 1000.0  # seconds
        x = valid["touch_x_mm"].to_numpy(dtype=float)
        y = valid["touch_y_mm"].to_numpy(dtype=float)

        dt = np.diff(t)
        # Guard against any zero/negative dt from duplicate/out-of-order samples.
        ok = dt > 0.001
        dt = dt[ok]
        vx = np.diff(x)[ok] / dt
        vy = np.diff(y)[ok] / dt

        dt2 = dt[1:]
        ok2 = dt2 > 0.001
        all_ax.append(np.diff(vx)[ok2] / dt2[ok2])
        all_ay.append(np.diff(vy)[ok2] / dt2[ok2])
        all_dt2.append(dt2[ok2])

    if not all_ax or sum(len(a) for a in all_ax) < 20:
        print(f"WARNING: only {total_valid_rows} valid moving-phase rows across {len(dfs_moving)} recording(s) -- Q estimate will be unreliable.")
        return np.eye(4)

    ax = np.concatenate(all_ax)
    ay = np.concatenate(all_ay)
    dt_pooled = np.concatenate(all_dt2)

    print(f"\n=== Q (process noise) -- {len(ax)} acceleration samples from {total_valid_rows} moving-phase rows across {len(dfs_moving)} recording(s) ===")
    print(f"accel_x: mean={ax.mean():+.1f} std={ax.std():.1f} mm/s^2")
    print(f"accel_y: mean={ay.mean():+.1f} std={ay.std():.1f} mm/s^2")

    # Discretized constant-velocity process noise (standard van Loan-style
    # approximation): Q for state [x, y, vx, vy] driven by acceleration
    # variance sigma_a^2, at the median dt across all pooled recordings.
    dt_med = float(np.median(dt_pooled))
    sigma_ax2 = float(ax.var())
    sigma_ay2 = float(ay.var())
    q = np.zeros((4, 4))
    q[0, 0] = sigma_ax2 * dt_med**4 / 4
    q[0, 2] = q[2, 0] = sigma_ax2 * dt_med**3 / 2
    q[2, 2] = sigma_ax2 * dt_med**2
    q[1, 1] = sigma_ay2 * dt_med**4 / 4
    q[1, 3] = q[3, 1] = sigma_ay2 * dt_med**3 / 2
    q[3, 3] = sigma_ay2 * dt_med**2
    print(f"Suggested Q matrix for state [x, y, vx, vy], dt={dt_med * 1000:.1f}ms:\n{q}")
    return q


def load_phases(csv_path: str, phase_split_s: float):
    """Loads one recording and splits it into (still, moving) phase frames,
    each still indexed against that recording's own t0 -- elapsed time isn't
    comparable across recordings, so this must be done per-file, before any
    pooling across multiple runs."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["host_recv_ts"]).sort_values("host_recv_ts").reset_index(drop=True)
    t0 = df["host_recv_ts"].iloc[0]
    df["elapsed_s"] = df["host_recv_ts"] - t0
    still = df[df["elapsed_s"] < phase_split_s]
    moving = df[df["elapsed_s"] >= phase_split_s]
    print(f"{csv_path}: span={df['elapsed_s'].iloc[-1]:.1f}s -- still phase: {len(still)} rows (<{phase_split_s:.0f}s), moving phase: {len(moving)} rows")
    return still, moving


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Kalman filter R/Q from one or more scripted-sequence recordings")
    parser.add_argument("--csv", required=True, nargs="+", help="One or more recording paths -- when multiple are given, their still/moving phases are pooled (each split against its own recording's t0) for a tighter, multi-run estimate")
    parser.add_argument("--phase-split-s", type=float, default=30.0, help="Elapsed seconds separating the still (R) and moving (Q) phases -- match --scripted-sequence's schedule")
    parser.add_argument("--output-json", type=str, default=None, help="Write the estimated R/Q matrices to this path as JSON ({'r': [[...]], 'q': [[...]]}), for main_onnx_shared_vision_audio.py's --kalman-params to consume directly")
    args = parser.parse_args()

    still_parts, moving_parts = [], []
    for csv_path in args.csv:
        still, moving = load_phases(csv_path, args.phase_split_s)
        still_parts.append(still)
        moving_parts.append(moving)
    still = pd.concat(still_parts, ignore_index=True)
    if len(args.csv) > 1:
        print(f"\nPooled across {len(args.csv)} recordings: {len(still)} still-phase rows "
              f"(safe to pool directly -- each row's raw_err is independent, no cross-row timing involved)")

    r = estimate_r(still)
    q = estimate_q(moving_parts)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"r": r.tolist(), "q": q.tolist()}, f, indent=2)
        print(f"\nWrote R/Q to {args.output_json} -- pass it to main_onnx_shared_vision_audio.py via --kalman --kalman-params {args.output_json}")


if __name__ == "__main__":
    main()
