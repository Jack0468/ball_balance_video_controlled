"""
Reproduces the vision_calibration.py validation numbers (2026-09-24) from a pooled
per-frame CSV (session, touch_x, touch_y, vision_x, vision_y columns required --
matches the schema an ml-vision Track 4 re-inference pass produces).

Calls the real core/vision_calibration.py functions (fit_affine_correction,
apply_correction, blend_corrections, fit_global_correction,
generate_calibration_grid) rather than re-implementing the math here, so this is
an actual test of that module's code, not a parallel implementation that could
silently drift from it.

IMPORTANT -- this reproduces the "oracle" validation, not a real-deployment
simulation. Read before trusting the numbers this prints:

Tonight's validation (and this script) fits AND applies the correction using
TOUCH position (touch_x, touch_y) at every step, not vision position -- i.e. it
assumes continuous access to touch-sensor ground truth, not just during an
initial calibration window. That's a fair way to characterize how good this
CLASS of correction could be, but the project's own architecture treats the
touchscreen as a sensor-only, evaluation-independent reference that "never feeds
the controller" (see firmware/stm32_ml_control_and_vision/BallBalancingBot/
BallBalancingBot.ino's own header comment) -- continuously using it to correct
vision would be a real departure from that principle, not just an
implementation detail, and is a design decision this script does not make.

A real deployment following that principle would calibrate ONCE at session
start (while touch is available) and then apply the fitted correction using only
VISION position for the rest of the session (touch no longer read for this
purpose). That variant -- fit on touch, apply on vision -- has NOT been tested
here or anywhere else. See vision_calibration.py's module docstring, gap 1, and
VISION_CALIBRATION_PROPOSAL.md's "Two deployment variants" section for the two
options and what each would need before it could be trusted.

Usage:
    python validate_vision_calibration.py --csv path/to/ALL_frames_pooled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from vision_calibration import (  # noqa: E402
    AffineCorrection,
    apply_correction,
    blend_corrections,
    fit_affine_correction,
    fit_global_correction,
    generate_calibration_grid,
)

PLATFORM_WIDTH_MM = 187.5
PLATFORM_HEIGHT_MM = 142.0
POINTS_PER_CELL = 15
ERR_MM_TRIM = 50.0  # drop frames beyond this -- matches the outlier trim used throughout the 2026-09-24 investigation


def select_grid_calibration_frames(session_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Nearest-neighbor selection of up to POINTS_PER_CELL real recorded frames
    per generate_calibration_grid() point, simulating "the ball visited this
    grid position and these frames were recorded there" against data that was
    actually collected freely (Track 4 sessions), not on a real calibration
    routine -- an approximation of what a real grid calibration pass would
    collect, not identical to it."""
    grid = generate_calibration_grid(PLATFORM_WIDTH_MM, PLATFORM_HEIGHT_MM, grid_size=3, margin_frac=0.15)
    touch_xy = session_df[["touch_x", "touch_y"]].to_numpy()
    chosen_positions: set = set()
    for gx, gy in grid:
        dist = np.hypot(touch_xy[:, 0] - gx, touch_xy[:, 1] - gy)
        order = np.argsort(dist)
        take = 0
        for idx in order:
            if session_df.index[idx] in chosen_positions:
                continue
            chosen_positions.add(session_df.index[idx])
            take += 1
            if take >= POINTS_PER_CELL:
                break
    return session_df.loc[list(chosen_positions)]


def touch_positions_and_errors(df: pd.DataFrame) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    positions = list(zip(df["touch_x"].tolist(), df["touch_y"].tolist()))
    errors = list(zip((df["vision_x"] - df["touch_x"]).tolist(), (df["vision_y"] - df["touch_y"]).tolist()))
    return positions, errors


def mean_err_mm_raw(df: pd.DataFrame) -> float:
    return float(np.hypot(df["vision_x"] - df["touch_x"], df["vision_y"] - df["touch_y"]).mean())


def apply_and_score(df: pd.DataFrame, correction: AffineCorrection) -> float:
    """err_mm after correction, calling the real apply_correction() -- not a
    reimplementation of its arithmetic -- evaluated the same (oracle,
    touch-position) way as the 2026-09-24 validation.

    apply_correction(raw_x, raw_y, correction) always evaluates the model AND
    subtracts at the SAME point it's given (that's the real-deployment contract:
    you only have one reading to both evaluate the model at and correct). To
    reproduce the oracle case (evaluate the model at touch position, but correct
    the VISION reading, not touch itself) without duplicating apply_correction's
    formula, we call it once AT touch position to recover what error it predicts
    there (touch_x - apply_correction(touch_x, touch_y)[0] == predicted_err_x by
    construction), then apply that same predicted error to vision instead."""
    predicted_err_x = np.empty(len(df))
    predicted_err_y = np.empty(len(df))
    for i, (tx, ty) in enumerate(zip(df["touch_x"], df["touch_y"])):
        pseudo_x, pseudo_y = apply_correction(tx, ty, correction)
        predicted_err_x[i] = tx - pseudo_x
        predicted_err_y[i] = ty - pseudo_y
    corrected_vx = df["vision_x"].to_numpy() - predicted_err_x
    corrected_vy = df["vision_y"].to_numpy() - predicted_err_y
    return float(np.hypot(corrected_vx - df["touch_x"].to_numpy(), corrected_vy - df["touch_y"].to_numpy()).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Pooled per-frame CSV: session, touch_x, touch_y, vision_x, vision_y columns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    err_mm = np.hypot(df["vision_x"] - df["touch_x"], df["vision_y"] - df["touch_y"])
    df = df[err_mm <= ERR_MM_TRIM].copy()

    rng = np.random.default_rng(args.seed)
    calib_sets: Dict[str, pd.DataFrame] = {}
    test_sets: Dict[str, pd.DataFrame] = {}
    for session, sub in df.groupby("session"):
        sub = sub.reset_index(drop=True)
        calib = select_grid_calibration_frames(sub, rng)
        test = sub.drop(index=calib.index)
        if len(calib) < 20 or len(test) < 30:
            continue
        calib_sets[session] = calib
        test_sets[session] = test

    if not calib_sets:
        print("No session had enough frames for calibration+test split -- nothing to validate.")
        return

    no_correction = [mean_err_mm_raw(test_sets[s]) for s in test_sets]

    per_session_results = []
    per_session_corrections: Dict[str, AffineCorrection] = {}
    for session in calib_sets:
        positions, errors = touch_positions_and_errors(calib_sets[session])
        correction = fit_affine_correction(positions, errors)
        per_session_corrections[session] = correction
        per_session_results.append(apply_and_score(test_sets[session], correction))

    global_correction = fit_global_correction(
        [touch_positions_and_errors(calib_sets[s]) for s in calib_sets]
    )
    global_results = [apply_and_score(test_sets[s], global_correction) for s in calib_sets]

    blended_results = []
    for session in calib_sets:
        blended = blend_corrections(per_session_corrections[session], global_correction)
        blended_results.append(apply_and_score(test_sets[session], blended))

    print(f"n_sessions={len(calib_sets)}  median_calib_points={int(np.median([len(c) for c in calib_sets.values()]))}")
    print()
    print(f"{'method':32s} {'mean err_mm':>12s}")
    print(f"{'no correction':32s} {np.mean(no_correction):12.2f}")
    print(f"{'per-session affine correction':32s} {np.mean(per_session_results):12.2f}")
    print(f"{'global averaged correction':32s} {np.mean(global_results):12.2f}")
    print(f"{'blended (session+global)':32s} {np.mean(blended_results):12.2f}")


if __name__ == "__main__":
    main()
