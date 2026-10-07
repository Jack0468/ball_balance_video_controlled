"""Frame-convention round-trip + touch_stale tests for the LeRobot converter (2026-10-06, risk #2 and #3
in `docs/MULTI_HEAD_ARCHITECTURE_SPEC.md`).

Checks:
  (a) `convert_to_lerobot.py` writes `observation.state` == raw telemetry `touch_x`/`touch_y`, no transform
      (run through the real `convert()` on the first 50 rows of one real Track 4 session).
  (b) `score_minimal_baseline_offline.touch_frame_to_manifest_mm` and its inverse round-trip every
      ground-truth manifest marker centre. The scorer has no inverse, so the inverse is defined here.
  (c) The state frame is the telemetry frame the scorer assumes: the scorer's map, applied to the state,
      lands inside the manifest platform rectangle, and the auto-labeler uses the same touch->mm formula.
  (T3) `touch_stale` flag: pure-function checks and an end-to-end synthetic session through `convert()`.

Repo convention: `deployment/test_*.py` are standalone scripts with exit codes; these tests follow that
(`python tests/test_lerobot_frame_roundtrip.py`) and are also pytest-discoverable. No new dependency.

Usage (from host_software/, pinned interpreter):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe ml_jetson_vla/tests/test_lerobot_frame_roundtrip.py
Exit code 0 only if every test passes.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR: str = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

BRONZE_DIR: str = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
REAL_SESSION: str = "session_jetson_track4_20260915_151627"
REAL_ROWS: int = 50
GROUND_TRUTH_MANIFEST: str = os.path.join(_REPO_ROOT_DIR, "hardware", "platform_templates", "ground_truth_manifest.json")
CONVERTER_PATH: str = os.path.join(_ML_JETSON_VLA_DIR, "data_processing", "convert_to_lerobot.py")
SCORER_PATH: str = os.path.join(_ML_JETSON_VLA_DIR, "deployment", "score_minimal_baseline_offline.py")
AUTO_LABEL_PATH: str = os.path.join(_REPO_ROOT_DIR, "host_software", "ml_vision", "data_processing", "auto_label_shared_vision.py")
PLATFORM_W_MM: float = 187.5
PLATFORM_H_MM: float = 142.0


def _load_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _converter() -> ModuleType:
    return _load_module("convert_to_lerobot_under_test", CONVERTER_PATH)


def _scorer() -> ModuleType:
    return _load_module("score_minimal_baseline_offline_under_test", SCORER_PATH)


def scorer_inverse_touch_frame(x_manifest_mm: float, y_manifest_mm: float,
                               platform_w_mm: float = PLATFORM_W_MM,
                               platform_h_mm: float = PLATFORM_H_MM) -> Tuple[float, float]:
    """Inverse of the scorer's `touch_frame_to_manifest_mm` (which maps telemetry -> manifest):
    x_m = W/2 - x_t  =>  x_t = W/2 - x_m ;  y_m = H/2 + y_t  =>  y_t = y_m - H/2."""
    return platform_w_mm / 2.0 - float(x_manifest_mm), float(y_manifest_mm) - platform_h_mm / 2.0


def _real_session_rows(n: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(BRONZE_DIR, REAL_SESSION, "telemetry.csv"))
    return df.iloc[:n].reset_index(drop=True)


# --- (a) converter state == raw telemetry touch, through the real convert() ------------------------

def test_a_converter_state_is_raw_touch() -> None:
    conv = _converter()
    tmp = tempfile.mkdtemp(prefix="lerobot_rt_a_")
    try:
        out_root = os.path.join(tmp, "ds")  # must not pre-exist (LeRobotDataset.create contract)
        conv.convert(
            bronze_dir=BRONZE_DIR, out_root=out_root, repo_id="test/roundtrip_a", fps=30,
            session_pattern=REAL_SESSION, include_track4_sessions=True, include_pid_sessions=False,
            max_frames_per_session=REAL_ROWS,
        )
        parquet_files = sorted(glob.glob(os.path.join(out_root, "data", "**", "*.parquet"), recursive=True))
        if not parquet_files:
            raise AssertionError(f"No parquet written under {out_root}")
        out = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        if len(out) != REAL_ROWS:
            raise AssertionError(f"Expected {REAL_ROWS} converted rows, got {len(out)}")
        state = np.stack(out["observation.state"].to_numpy()).astype(np.float32)

        raw = _real_session_rows(REAL_ROWS)
        expected = raw[["touch_x", "touch_y"]].to_numpy(dtype=np.float32)
        if not np.array_equal(state, expected):
            diff = np.abs(state - expected).max()
            raise AssertionError(
                f"observation.state differs from raw telemetry touch_x/touch_y (max abs diff {diff}). "
                f"The converter applies a transform the scorer does not."
            )
        names = _converter()._build_features(480, 640)["observation.state"]["names"]
        assert names == ["touch_x_mm", "touch_y_mm"], names
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- (b) scorer forward map and its inverse round-trip known manifest points ------------------------

def _marker_centres_mm() -> List[Tuple[int, float, float]]:
    with open(GROUND_TRUTH_MANIFEST, "r") as f:
        manifest = json.load(f)
    return [(int(m["id"]), float(m["center_mm"][0]), float(m["center_mm"][1])) for m in manifest["aruco_markers"]]


def test_b_scorer_roundtrip_manifest_points() -> None:
    sc = _scorer()
    markers = _marker_centres_mm()
    assert markers, "ground_truth_manifest.json has no markers"
    for marker_id, x_m, y_m in markers:
        x_t, y_t = scorer_inverse_touch_frame(x_m, y_m)
        x_back, y_back = sc.touch_frame_to_manifest_mm(x_t, y_t, PLATFORM_W_MM, PLATFORM_H_MM)
        if abs(x_back - x_m) > 1e-9 or abs(y_back - y_m) > 1e-9:
            raise AssertionError(
                f"marker {marker_id}: manifest ({x_m}, {y_m}) -> telemetry ({x_t}, {y_t}) -> "
                f"manifest ({x_back}, {y_back}) does not round-trip"
            )
    # Known point: telemetry origin is the platform centre.
    cx, cy = sc.touch_frame_to_manifest_mm(0.0, 0.0, PLATFORM_W_MM, PLATFORM_H_MM)
    assert (cx, cy) == (PLATFORM_W_MM / 2.0, PLATFORM_H_MM / 2.0), (cx, cy)


def test_b2_scorer_roundtrip_converter_rows() -> None:
    """Round-trip real converter-input rows: telemetry -> manifest -> telemetry is identity."""
    sc = _scorer()
    raw = _real_session_rows(REAL_ROWS)
    for tx, ty in raw[["touch_x", "touch_y"]].to_numpy(dtype=np.float64):
        xm, ym = sc.touch_frame_to_manifest_mm(tx, ty, PLATFORM_W_MM, PLATFORM_H_MM)
        tx2, ty2 = scorer_inverse_touch_frame(xm, ym, PLATFORM_W_MM, PLATFORM_H_MM)
        assert abs(tx2 - tx) < 1e-9 and abs(ty2 - ty) < 1e-9, (tx, ty, tx2, ty2)


# --- (c) state frame == the telemetry frame the scorer assumes ---------------------------------------

def test_c_state_frame_matches_scorer_convention() -> None:
    sc = _scorer()
    raw = _real_session_rows(REAL_ROWS)
    # The scorer's map must place every sampled touch inside the physical platform rectangle. If the
    # state were in the manifest frame (top-left origin) instead, values would be shifted by ~W/2, H/2.
    for tx, ty in raw[["touch_x", "touch_y"]].to_numpy(dtype=np.float64):
        xm, ym = sc.touch_frame_to_manifest_mm(tx, ty, PLATFORM_W_MM, PLATFORM_H_MM)
        if not (0.0 <= xm <= PLATFORM_W_MM and 0.0 <= ym <= PLATFORM_H_MM):
            raise AssertionError(
                f"touch ({tx}, {ty}) maps to manifest ({xm:.2f}, {ym:.2f}) outside the "
                f"{PLATFORM_W_MM}x{PLATFORM_H_MM} mm platform -- frame convention mismatch"
            )
    # The auto-labeler (the other consumer of touch_x/touch_y) must use the same formula as the scorer.
    with open(AUTO_LABEL_PATH, "r") as f:
        src = f.read()
    has_x = re.search(r"ball_x_mm\s*=\s*\(TOUCHPAD_W\s*/\s*2\.0\)\s*-\s*touch_x", src) is not None
    has_y = re.search(r"ball_y_mm\s*=\s*\(TOUCHPAD_H\s*/\s*2\.0\)\s*\+\s*touch_y", src) is not None
    if not (has_x and has_y):
        raise AssertionError("auto_label_shared_vision.py no longer uses the touch->mm formula the scorer assumes")
    # Cross-check the scorer's forward map against that formula at a sample point.
    tx, ty = float(raw.loc[0, "touch_x"]), float(raw.loc[0, "touch_y"])
    xm, ym = sc.touch_frame_to_manifest_mm(tx, ty, PLATFORM_W_MM, PLATFORM_H_MM)
    assert (xm, ym) == (PLATFORM_W_MM / 2.0 - tx, PLATFORM_H_MM / 2.0 + ty), (xm, ym)


# --- (T3) touch_stale ----------------------------------------------------------------------------------

def test_t3_compute_touch_stale_pure() -> None:
    conv = _converter()
    df = pd.DataFrame({
        "touch_x": [1.0, 1.0, 3.0, 3.0, 3.0, 5.0, 5.0, np.nan, 5.0],
        "touch_y": [2.0, 2.0, 4.0, 4.0, 4.0, 6.0, 6.0, np.nan, 6.0],
    })
    flags = conv.compute_touch_stale(df).tolist()
    # first row 0; repeats flagged; NaN row and the row after it never flagged
    assert flags == [0, 1, 0, 1, 1, 0, 1, 0, 0], flags
    # a change in only one axis is not a repeat
    df2 = pd.DataFrame({"touch_x": [1.0, 1.0], "touch_y": [2.0, 2.5]})
    assert conv.compute_touch_stale(df2).tolist() == [0, 0]


def _write_synthetic_session(bronze: str, name: str, touch: List[Tuple[float, float]]) -> None:
    sdir = os.path.join(bronze, name)
    os.makedirs(sdir, exist_ok=True)
    rows = []
    for i, (tx, ty) in enumerate(touch):
        rows.append({
            "frame_index": i, "host_timestamp_ms": 1_000_000 + 33 * i,
            "target_x": 0.0, "target_y": 0.0, "touch_x": tx, "touch_y": ty,
            "theta_a": 0.0, "theta_b": 0.0, "theta_c": 0.0, "audio_command": "",
        })
    pd.DataFrame(rows).to_csv(os.path.join(sdir, "telemetry.csv"), index=False)
    writer = cv2.VideoWriter(os.path.join(sdir, "rgb_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4v output on this machine")
    for i in range(len(touch)):
        writer.write(np.full((480, 640, 3), 40 + 10 * i, dtype=np.uint8))
    writer.release()


def test_t3_end_to_end_synthetic_session() -> None:
    conv = _converter()
    tmp = tempfile.mkdtemp(prefix="lerobot_rt_t3_")
    try:
        bronze = os.path.join(tmp, "bronze")
        name = "session_jetson_track4_20990101_000000"
        touch = [(1.0, 2.0), (1.0, 2.0), (3.0, 4.0), (3.0, 4.0), (3.0, 4.0), (5.0, 6.0), (5.0, 6.0)]
        _write_synthetic_session(bronze, name, touch)
        out_root = os.path.join(tmp, "ds")
        conv.convert(
            bronze_dir=bronze, out_root=out_root, repo_id="test/touch_stale", fps=30,
            session_pattern=name, include_track4_sessions=True, include_pid_sessions=False,
        )
        parquet_files = sorted(glob.glob(os.path.join(out_root, "data", "**", "*.parquet"), recursive=True))
        out = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        flags = [int(v[0]) if hasattr(v, "__len__") else int(v) for v in out["touch_stale"].to_numpy()]
        assert flags == [0, 1, 0, 1, 1, 0, 1], flags
        assert flags[0] == 0
        assert "touch_stale" in conv._build_features(480, 640)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- (T4) touch_glitch -----------------------------------------------------------------------------
# 2026-10-07: companion to T3's touch_stale -- see compute_touch_glitch's own docstring in
# convert_to_lerobot.py for GLITCH_JUMP_THRESHOLD_MM/GLITCH_LOOKAHEAD_K justification
# (reports/touch_glitch_analysis_2026_10_07.md has the full real-corpus analysis).

def test_t4_compute_touch_glitch_pure() -> None:
    conv = _converter()
    # Realistic shape (mirrors the 5 manually-confirmed real examples): a slowly-drifting
    # baseline, a 2-frame spike far outside the threshold, then a return close to where the
    # baseline left off, followed by more small steps -- no second large jump within the
    # look-ahead window, so only the spike itself (not the recovery) is flagged.
    df = pd.DataFrame({
        "touch_x": [-30.0, -29.0, -28.0, -26.0, 85.0, 85.0, -22.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0],
        "touch_y": [0.0] * 13,
    })
    flags = conv.compute_touch_glitch(df).tolist()
    assert flags == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], flags
    # First row is always 0 (no previous row to jump from).
    assert flags[0] == 0
    # A single-frame spike (entry and return both present, no intervening repeat) is also caught.
    df2 = pd.DataFrame({
        "touch_x": [1.0, 1.0, 1.0, 72.0, 1.0, 1.0],
        "touch_y": [2.0, 2.0, 2.0, 63.0, 2.0, 2.0],
    })
    assert conv.compute_touch_glitch(df2).tolist() == [0, 0, 0, 1, 0, 0]
    # A big jump that never comes back within the look-ahead window is NOT flagged (no evidence
    # of "snaps back" -- distinguishes a real large excursion from a glitch).
    df3 = pd.DataFrame({
        "touch_x": [1.0, 1.0, 1.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0],
        "touch_y": [2.0, 2.0, 2.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0],
    })
    assert conv.compute_touch_glitch(df3).tolist() == [0] * 10


def test_t4_compute_touch_glitch_nan_and_boundary() -> None:
    conv = _converter()
    # NaN never sets or absorbs the flag: the entry jump right after a NaN row, and the row
    # that follows it, both read as NaN (not > threshold), so they can't be detected as a
    # candidate -- same rule compute_touch_stale documents for exact-repeat comparisons.
    df = pd.DataFrame({
        "touch_x": [1.0, 1.0, 3.0, 3.0, 3.0, np.nan, 90.0, 90.0, -20.0, -19.0],
        "touch_y": [2.0, 2.0, 4.0, 4.0, 4.0, np.nan, 65.0, 65.0, -15.0, -14.0],
    })
    flags = conv.compute_touch_glitch(df).tolist()
    assert flags == [0] * 10, flags
    # End-of-session boundary: a big jump with fewer than GLITCH_LOOKAHEAD_K frames remaining
    # can't have its look-ahead window fully evaluated -- treated conservatively (not flagged),
    # the same outcome as "jumped but never came back".
    df2 = pd.DataFrame({
        "touch_x": [-10.0, -9.0, -8.0, -7.0, 80.0],
        "touch_y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    assert conv.compute_touch_glitch(df2).tolist() == [0, 0, 0, 0, 0]


def test_t4_end_to_end_synthetic_session() -> None:
    conv = _converter()
    tmp = tempfile.mkdtemp(prefix="lerobot_rt_t4_")
    try:
        bronze = os.path.join(tmp, "bronze")
        name = "session_jetson_track4_20990101_000001"
        touch = [
            (-30.0, 0.0), (-29.0, 0.0), (-28.0, 0.0), (-26.0, 0.0), (85.0, 0.0), (85.0, 0.0),
            (-22.0, 0.0), (-21.0, 0.0), (-19.0, 0.0), (-17.0, 0.0), (-15.0, 0.0), (-13.0, 0.0), (-11.0, 0.0),
        ]
        _write_synthetic_session(bronze, name, touch)
        out_root = os.path.join(tmp, "ds")
        conv.convert(
            bronze_dir=bronze, out_root=out_root, repo_id="test/touch_glitch", fps=30,
            session_pattern=name, include_track4_sessions=True, include_pid_sessions=False,
        )
        parquet_files = sorted(glob.glob(os.path.join(out_root, "data", "**", "*.parquet"), recursive=True))
        out = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        flags = [int(v[0]) if hasattr(v, "__len__") else int(v) for v in out["touch_glitch"].to_numpy()]
        assert flags == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], flags
        assert "touch_glitch" in conv._build_features(480, 640)
        # touch_stale must be unaffected by this addition -- still present, still correct (no
        # exact repeats other than the glitch run's own duplicate value at indices 4/5).
        stale = [int(v[0]) if hasattr(v, "__len__") else int(v) for v in out["touch_stale"].to_numpy()]
        assert stale == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], stale
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS: Dict[str, Callable[[], None]] = {
    "test_a_converter_state_is_raw_touch": test_a_converter_state_is_raw_touch,
    "test_b_scorer_roundtrip_manifest_points": test_b_scorer_roundtrip_manifest_points,
    "test_b2_scorer_roundtrip_converter_rows": test_b2_scorer_roundtrip_converter_rows,
    "test_c_state_frame_matches_scorer_convention": test_c_state_frame_matches_scorer_convention,
    "test_t3_compute_touch_stale_pure": test_t3_compute_touch_stale_pure,
    "test_t3_end_to_end_synthetic_session": test_t3_end_to_end_synthetic_session,
    "test_t4_compute_touch_glitch_pure": test_t4_compute_touch_glitch_pure,
    "test_t4_compute_touch_glitch_nan_and_boundary": test_t4_compute_touch_glitch_nan_and_boundary,
    "test_t4_end_to_end_synthetic_session": test_t4_end_to_end_synthetic_session,
}


def main() -> int:
    failures = 0
    for name, fn in TESTS.items():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
