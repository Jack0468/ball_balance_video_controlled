"""Offline scoring of the Arm 2 minimal-baseline prompt/parse path (`core/minimal_vlm_policy.py`)
against real logged Track 4 session data -- the "achievable now, offline" item of
`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`'s run/test plan.

## 2026-09-19 CORRECTION -- ground-truth coordinate frame (read this first)

Every `error_mm` / `hit` / `mean_error_mm` this script produced BEFORE 2026-09-19 is invalid.
The predicted pixel was mapped to mm through the ArUco homography, whose mm side is the
MANIFEST frame (origin at the sheet's top-left corner, 187.5x142.0mm, x right / y down), and
then compared directly against telemetry.csv's `target_x`/`target_y`. Those columns are NOT in
that frame: they are CENTER-origin touch/telemetry-frame coordinates,
`target_x = W/2 - x_manifest`, `target_y = y_manifest - H/2` (same relationship the repo's
`auto_label_shared_vision.py` documents for `touch_x`/`touch_y`; re-verified empirically
2026-09-19 -- HSV-detected marker centroids match to ~1mm median, vs 78-170mm under the old
comparison). Consequence: a PERFECT prediction would have scored ~78-170mm error, so "0/60 hits"
was guaranteed regardless of the model. `touch_frame_to_manifest_mm()` below is the fix.
The old comparison is kept as `error_mm_legacy` (and `mean_error_mm_legacy`) ONLY so a re-run
elsewhere (the Colab sweep) can be checked against the historical 183.8/170.3mm numbers, which
depend only on the model's raw pixel outputs. `--rescore-from` recomputes an old result file
under the corrected frame without re-running any model. Output JSON now carries
`"scoring_frame_version": 2`.

**Deliberately NOT `docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md`'s Metric A**, even though it looks
similar (same 20mm tolerance, same color-command-only scope, same Track 4 data) -- that plan's
Metric A scores a frozen-backbone GROUNDING CAPABILITY probe (a `generate()` call asking for a
point, evaluated purely on localization accuracy). This script scores the actual MINIMAL-
BASELINE PATH end to end: the real prompt template, the real parser (including its documented
parse failures), and the real backend abstraction from `core/minimal_vlm_policy.py` -- so a
frame here can fail for a reason Metric A's methodology doesn't even model (the model answered
in an unparseable format). Two numbers this script reports that Metric A does not:
`parse_success_rate` and a per-failure-reason breakdown.

## What this measures (and what it deliberately does not)

- **Parse success rate**: over color-command frames, how often `parse_minimal_baseline_output()`
  extracts ANY point at all, regardless of accuracy.
- **Mean/median localization error (mm)**, among parsed frames only: predicted pixel point ->
  mm via a real per-frame ArUco homography (`ml_vision.data_processing.auto_label_shared_vision
  .estimate_homography_from_aruco`, read-only reuse -- `ml_vision` is not this agent's territory
  to modify, same precedent `BOOTSTRAP_MODEL_COMPARISON_PLAN.md` S2.1 already used for the same
  reason) inverted to map the predicted raw-camera-pixel point back to real platform mm, compared
  against `telemetry.csv`'s own real logged `target_x`/`target_y` (mm).
- **Hit rate**: fraction of parsed frames within `--tolerance-mm` (default 20.0mm, reusing
  `EVALUATION_STRATEGY.md`'s settling-time tolerance radius, matching Metric A's own reuse of the
  same ready-made threshold rather than inventing a new one).

**Explicitly NOT measured here** (per `docs/ARM2_MINIMAL_BASELINE_SCOPE.md` S3/S4 and this
task's own scope): real closed-loop control quality, the project's standard four metrics, or
anything requiring firmware execution -- those stay blocked on the firmware decision, stated
plainly, not attempted. Directional/hold/stop commands are out of scope for hit-rate scoring
(no well-defined target point -- same reasoning as `BOOTSTRAP_MODEL_COMPARISON_PLAN.md` S2.4);
this script only scores the 4 color commands.

**Known imprecision, flagged not fixed**: frame seeking uses `cv2.CAP_PROP_POS_FRAMES`, which is
not guaranteed frame-exact on compressed video (keyframe-interval rounding) -- for a handful of
sampled frames per session this is an acceptable approximation for a smoke-level offline score,
but it means `error_mm` here should not be read as precise to the single frame the way a
sequential full-video decode (as `convert_to_lerobot.py` does) would be.

**Small-sample discipline (`model-iteration-constraints` skill S4)**: this script's own default
settings sample a handful of frames per session for a cheap smoke-level run, not a statistically
powered comparison -- report per-session results, never a single pooled number treated as a
verdict, and do not rank candidates from one run of this script. See
`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`'s verification section for what was actually run
and how small it was.

## 2026-09-18 follow-up: multi-variant scoring + a real video-decode fix

Two real changes made for the "get a statistically real accuracy signal" follow-up task
(`core/minimal_vlm_policy.py`'s new `prompt_variant` option, `PROMPT_TEMPLATE_ORIENTED`):

1. **`--prompt-variants`** (plural, list) replaces the implicit single-baseline-prompt
   behavior. For each sampled frame, EVERY requested variant is scored against the exact same
   decoded frame + homography (computed once, reused across variants) -- this is what makes
   the baseline-vs-oriented comparison apples-to-apples rather than two separately-sampled
   runs that could land on different frames. All requested variants share one loaded backend
   (`VLMBackend.load()` is documented idempotent), so the model is loaded once regardless of
   how many variants are scored.
2. **Video decoding switched from `cv2.VideoCapture` to PyAV (`av` package)**, discovered
   necessary while building this run: on this machine, `cv2.VideoCapture` cannot open this
   project's `rgb_video.mp4` files at all (`isOpened()` False under both the default backend
   and an explicit `cv2.CAP_FFMPEG` -- this OpenCV build has GStreamer but is missing the
   Quicktime/mp4 demuxer plugin GStreamer needs, confirmed by direct testing 2026-09-18). PyAV
   (already an installed dependency) opens and decodes these files without issue. Frames are
   now read via one sequential forward decode per session (`_iter_frames_bgr()`) rather than
   `CAP_PROP_POS_FRAMES` seeks -- this is also a precision improvement, not just a
   compatibility fix: the old approach's own docstring already flagged seek-based sampling as
   "not guaranteed frame-exact on compressed video"; sequential decode with an exact frame-index
   counter has no such imprecision. Cost is small: ~2600 frames at 640x480 decodes in
   single-digit seconds, negligible next to this candidate's multi-minute-per-frame `generate()`
   cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, Optional

import av
import cv2
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.minimal_vlm_policy import (  # noqa: E402
    MinimalVLMPolicy, PROMPT_VARIANTS, to_raw_px,
)
from ml_jetson_vla.core.vlm_backends import BACKEND_REGISTRY, qwen_model_input_hw  # noqa: E402
from host_software.ml_vision.data_processing.auto_label_shared_vision import (  # noqa: E402
    estimate_homography_from_aruco,
    load_manifest_full,
)

GROUND_TRUTH_MANIFEST = os.path.join(
    _REPO_ROOT_DIR, "hardware", "platform_templates", "ground_truth_manifest.json"
)
DEFAULT_TOLERANCE_MM = 20.0
COLOR_COMMANDS = {"go_red", "go_green", "go_yellow", "go_black"}


def raw_px_to_mm(homography_mm_to_px: np.ndarray, x_px: float, y_px: float) -> tuple:
    """`estimate_homography_from_aruco()` returns the mm->px direction (see its own
    docstring). This script needs the inverse (predicted pixel -> mm) -- inverting a real,
    per-frame-computed homography rather than assuming a fixed mapping, since the camera can
    move slightly between sessions (or even isn't guaranteed static within one, though this
    project's rig is a fixed tripod in practice)."""
    m_inv = np.linalg.inv(homography_mm_to_px)
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    mm_pt = cv2.perspectiveTransform(pt, m_inv.astype(np.float32))
    return float(mm_pt[0][0][0]), float(mm_pt[0][0][1])


def touch_frame_to_manifest_mm(
    target_x_tel: float, target_y_tel: float,
    platform_w_mm: float = 187.5, platform_h_mm: float = 142.0,
) -> tuple:
    """telemetry `target_x`/`target_y` (center-origin, x mirrored) -> manifest/homography mm
    (top-left origin, x right, y down). Same relation `auto_label_shared_vision.py` uses for
    `touch_x`/`touch_y` (`ball_x_mm = W/2 - touch_x`, `ball_y_mm = H/2 + touch_y`); verified
    against HSV-detected marker centroids 2026-09-19 (see module docstring)."""
    return platform_w_mm / 2.0 - float(target_x_tel), platform_h_mm / 2.0 + float(target_y_tel)


TRANSITION_TOL_MM = 3.0


def compute_transition_flags(df: pd.DataFrame, tol_mm: float = TRANSITION_TOL_MM) -> pd.Series:
    """True for frames whose logged target has not yet settled for the current command.

    The logged `target_x/target_y` is a setpoint that moves to a new marker over a few frames
    after a command starts (e.g. session ...160509 has go_red target_x running 0.0 -> 31.2), so
    the FIRST frames of a command carry the previous location, not the commanded marker. A frame
    is flagged when it lies in a contiguous run of one `audio_command` and its target is more
    than `tol_mm` from the median target of that run's last half. Diagnostic only -- sampling is
    unchanged (same frames as the 2026-09-18 run); metrics are reported both with and without
    flagged frames."""
    flags = pd.Series(False, index=df.index)
    cmd = df["audio_command"]
    run_id = ((cmd != cmd.shift()) | cmd.isna()).cumsum()
    for _rid, grp in df.groupby(run_id):
        if grp["audio_command"].isna().all() or len(grp) < 4:
            continue
        tail = grp.iloc[len(grp) // 2:]
        sx, sy = tail["target_x"].median(), tail["target_y"].median()
        dist = np.hypot(grp["target_x"] - sx, grp["target_y"] - sy)
        flags.loc[grp.index] = (dist > tol_mm).values
    return flags


def score_prediction(
    homography_mm_to_px: np.ndarray, px: float, py: float,
    true_target_x_tel: float, true_target_y_tel: float,
    tolerance_mm: float, platform_w_mm: float = 187.5, platform_h_mm: float = 142.0,
    coord_space: str = "raw_image", model_input_hw: Optional[tuple] = None,
    raw_hw: tuple = (480, 640),
) -> dict:
    """Corrected scoring of ONE predicted point (`px`,`py` = coordinates exactly as parsed).

    Primary `error_mm`/`hit`: the point is first mapped into RAW-frame pixels per the backend's
    declared `coord_space` ("model_input" backends -- Qwen2.5-VL -- answer in the resized image
    the vision tower sees; see `to_raw_px`), then through the per-frame homography to manifest
    mm, then compared with the ground truth converted into that same frame. The other error
    fields exist so a wrong coordinate-space assumption is visible instead of silently baked in:
      - `error_mm_alt_raw`: as-parsed coordinates taken as raw-frame pixels (= what "as prompted"
        means; equals `error_mm` for raw_image backends),
      - `error_mm_alt_model_input`: taken as living in `model_input_hw` (only if known),
      - `error_mm_alt_norm1000`: taken as 0-1000 normalized over the raw frame (InternVL's own
        native grounding convention),
      - `error_mm_legacy`: the pre-2026-09-19 value (as-parsed raw pixels vs. telemetry target
        in the WRONG frame) -- kept only so a re-run can be checked against the historical
        183.8/170.3mm numbers.
    """
    raw_h, raw_w = raw_hw
    true_mx, true_my = touch_frame_to_manifest_mm(
        true_target_x_tel, true_target_y_tel, platform_w_mm, platform_h_mm)

    def _err(rx: float, ry: float) -> tuple:
        mx, my = raw_px_to_mm(homography_mm_to_px, rx, ry)
        return mx, my, float(np.hypot(mx - true_mx, my - true_my))

    rx, ry = to_raw_px(px, py, coord_space, model_input_hw, raw_hw)
    pred_x_mm, pred_y_mm, error_mm = _err(rx, ry)
    raw_mx, raw_my, err_raw = _err(px, py)
    out = {
        "target_point_px": [px, py],
        "target_point_raw_px": [rx, ry],
        "coord_space": coord_space,
        "model_input_hw": list(model_input_hw) if model_input_hw else None,
        "pred_target_x_mm": pred_x_mm,           # manifest frame, from the PRIMARY raw-px point
        "pred_target_y_mm": pred_y_mm,
        "true_target_manifest_x_mm": true_mx,    # ground truth in the SAME frame
        "true_target_manifest_y_mm": true_my,
        "error_mm": error_mm,                    # PRIMARY (v2)
        "hit": error_mm <= tolerance_mm,
        "error_mm_alt_raw": err_raw,
        "error_mm_alt_norm1000": _err(px / 1000.0 * raw_w, py / 1000.0 * raw_h)[2],
        "error_mm_legacy": float(np.hypot(raw_mx - true_target_x_tel, raw_my - true_target_y_tel)),
    }
    if model_input_hw:
        out["error_mm_alt_model_input"] = _err(
            *to_raw_px(px, py, "model_input", model_input_hw, raw_hw))[2]
    return out


def score_frame_with_policy(
    policy: "MinimalVLMPolicy", name: str, frame_bgr: np.ndarray, instruction: str,
    homography: np.ndarray, true_x_tel: float, true_y_tel: float, tolerance_mm: float,
    frame_index: int, target_transition: Optional[bool] = None,
    platform_w_mm: float = 187.5, platform_h_mm: float = 142.0,
) -> dict:
    """One policy call on one frame -> one per-frame result entry (the schema this script has
    always written, plus the v2 fields). Shared by `score_session()` and the Colab sweep engine
    (`deployment/colab_sweep.py`) so there is a single implementation of the per-frame logic."""
    t0 = time.time()
    policy.act(frame_bgr, instruction, state={})
    act_s = time.time() - t0
    debug = policy.last_debug
    entry = {
        "frame_index": frame_index,
        "instruction": instruction,
        "prompt_variant": debug.get("prompt_variant", name),
        "parse_ok": debug.get("parse_ok"),
        "parse_reason": debug.get("parse_reason"),
        "raw_text": debug.get("raw_text"),
        "act_s": act_s,
        "generate_latency_s": debug.get("generate_latency_s"),
        "true_target_x_mm": true_x_tel,   # NAME KEPT for schema compat; telemetry (center-origin) frame
        "true_target_y_mm": true_y_tel,
        "target_transition": target_transition,
    }
    entry["dtype"] = debug.get("dtype")
    entry["peak_memory_gb"] = debug.get("peak_memory_gb")
    if debug.get("parse_ok"):
        px, py = debug["target_point_px"]
        entry.update(score_prediction(
            homography, px, py, true_x_tel, true_y_tel, tolerance_mm, platform_w_mm, platform_h_mm,
            coord_space=debug.get("coord_space", "raw_image"),
            model_input_hw=debug.get("model_input_hw"),
            raw_hw=tuple(debug.get("raw_image_hw", frame_bgr.shape[:2])),
        ))
    return entry


def atomic_write_json(path: str, obj: dict) -> None:
    """Write-to-temp-then-`os.replace()` so a crash mid-write never leaves `path` truncated
    or corrupted -- part of this module's checkpointing (see `main()`'s "2026-09-18: project
    convention" note)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def read_frames_at_indices(video_path: str, wanted_indices: set) -> Dict[int, np.ndarray]:
    """Sequential forward decode via PyAV, picking out exactly the frame indices requested.
    Replaces `cv2.VideoCapture` + `CAP_PROP_POS_FRAMES` seeking -- see this module's docstring
    ("2026-09-18 follow-up") for why: `cv2.VideoCapture` cannot open this project's mp4 files
    on this machine at all, and sequential decode is frame-exact besides (no seek-rounding
    imprecision on compressed video). Returns {frame_index: frame_bgr}; missing indices
    (beyond end of stream) are simply absent from the result, not an error."""
    found: Dict[int, np.ndarray] = {}
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        remaining = set(wanted_indices)
        for i, frame in enumerate(container.decode(stream)):
            if i in remaining:
                found[i] = frame.to_ndarray(format="bgr24")
                remaining.discard(i)
                if not remaining:
                    break
    finally:
        container.close()
    return found


def sample_frame_indices(df: pd.DataFrame, max_frames: int) -> list:
    """Frames where a color command is active AND ground truth exists, evenly subsampled to
    `max_frames` -- not the first N (which would bias toward one command/early-session
    conditions)."""
    candidates = df[
        df["audio_command"].isin(COLOR_COMMANDS)
        & df["target_x"].notna() & df["target_y"].notna()
    ]
    if len(candidates) == 0:
        return []
    if len(candidates) <= max_frames:
        return candidates.index.tolist()
    step = len(candidates) / max_frames
    picked = [candidates.index[int(i * step)] for i in range(max_frames)]
    return picked


def summarize_variant_frames(results: list) -> dict:
    """Shared by the final per-session return AND the per-frame checkpoint writer below, so a
    partial (in-progress) write and the final write use identical summary math."""
    parsed = [r for r in results if r.get("parse_ok")]
    n_scored = len([r for r in results if "error" not in r])
    # Frames whose logged target had not settled yet (see compute_transition_flags) -- secondary
    # view only; the primary numbers below use every parsed frame, like the original run.
    settled = [r for r in parsed if r.get("target_transition") is False]
    legacy = [r["error_mm_legacy"] for r in parsed if "error_mm_legacy" in r]
    alt_raw = [r["error_mm_alt_raw"] for r in parsed if "error_mm_alt_raw" in r]
    return {
        "mean_error_mm_alt_raw": float(np.mean(alt_raw)) if alt_raw else None,
        "n_frames_sampled": len(results),
        "n_frames_scored": n_scored,  # excludes video-read/homography failures, not parse failures
        "n_parsed": len(parsed),
        "parse_success_rate": (len(parsed) / n_scored) if n_scored else None,
        "hit_rate": (sum(1 for r in parsed if r["hit"]) / len(parsed)) if parsed else None,
        "mean_error_mm": (float(np.mean([r["error_mm"] for r in parsed]))) if parsed else None,
        "median_error_mm": (float(np.median([r["error_mm"] for r in parsed]))) if parsed else None,
        "n_settled": len(settled),
        "hit_rate_settled": (sum(1 for r in settled if r["hit"]) / len(settled)) if settled else None,
        "mean_error_mm_settled": (float(np.mean([r["error_mm"] for r in settled]))) if settled else None,
        "mean_error_mm_legacy": float(np.mean(legacy)) if legacy else None,  # wrong-frame, validation only
        "mean_generate_latency_s": (
            float(np.mean([r["generate_latency_s"] for r in results if r.get("generate_latency_s") is not None]))
            if results else None
        ),
        "frames": results,
    }


def score_session(
    session_dir: str,
    policies: Dict[str, MinimalVLMPolicy],
    aruco_lookup: dict,
    max_frames: int,
    tolerance_mm: float,
    checkpoint_cb=None,
) -> dict:
    """Scores every requested prompt variant (`policies`: {variant_name: MinimalVLMPolicy},
    all sharing one loaded backend) against the SAME sampled frames -- one video decode, one
    ArUco homography per frame, reused across variants, so a baseline-vs-oriented comparison
    is apples-to-apples rather than two independently-sampled runs.

    **Checkpointing** (project convention, added 2026-09-18: any process expected to run >30
    minutes must persist partial progress periodically, not only write output at the end): if
    `checkpoint_cb` is given, it is called after EVERY sampled frame finishes (all requested
    variants for that frame) with `(session_basename, partial_variant_summaries)` -- the caller
    (`main()`) uses this to flush the full run's current state to `--out` after each frame, so
    a crash mid-session loses at most one frame's worth of generate() calls (a few minutes),
    not the whole multi-hour run."""
    session_name = os.path.basename(session_dir)
    csv_path = os.path.join(session_dir, "telemetry.csv")
    video_path = os.path.join(session_dir, "rgb_video.mp4")
    if not os.path.exists(csv_path) or not os.path.exists(video_path):
        return {"session": session_name, "error": "missing telemetry.csv or rgb_video.mp4"}

    df = pd.read_csv(csv_path)
    frame_indices = sample_frame_indices(df, max_frames)
    if not frame_indices:
        return {
            "session": session_name,
            "error": "no color-command frames with ground truth found",
        }

    rows = [df.loc[idx] for idx in frame_indices]
    wanted = {int(row["frame_index"]) for row in rows}
    frames_by_index = read_frames_at_indices(video_path, wanted)
    transition_flags = compute_transition_flags(df)

    results_by_variant: Dict[str, list] = {name: [] for name in policies}
    for row in rows:
        frame_index = int(row["frame_index"])
        instruction = str(row["audio_command"])
        true_x_mm, true_y_mm = float(row["target_x"]), float(row["target_y"])

        frame_bgr = frames_by_index.get(frame_index)
        if frame_bgr is None:
            for name in policies:
                results_by_variant[name].append(
                    {"frame_index": frame_index, "error": "video_read_failed"}
                )
            if checkpoint_cb:
                checkpoint_cb(session_name, {n: summarize_variant_frames(r) for n, r in results_by_variant.items()})
            continue

        homography = estimate_homography_from_aruco(frame_bgr, aruco_lookup)
        if homography is None:
            for name in policies:
                results_by_variant[name].append(
                    {"frame_index": frame_index, "error": "no_aruco_homography"}
                )
            if checkpoint_cb:
                checkpoint_cb(session_name, {n: summarize_variant_frames(r) for n, r in results_by_variant.items()})
            continue

        for name, policy in policies.items():
            entry = score_frame_with_policy(
                policy, name, frame_bgr, instruction, homography, true_x_mm, true_y_mm,
                tolerance_mm, frame_index,
                target_transition=bool(transition_flags.loc[int(row.name)]),
            )
            results_by_variant[name].append(entry)

            # Checkpoint after EACH variant's generate() call, not just each frame -- the
            # oriented variant alone can take ~2.5 min on this CPU, so per-variant is the
            # actual safe granularity, not just per-frame.
            if checkpoint_cb:
                checkpoint_cb(session_name, {n: summarize_variant_frames(r) for n, r in results_by_variant.items()})

    variant_summaries = {name: summarize_variant_frames(results) for name, results in results_by_variant.items()}

    return {
        "session": session_name,
        "variants": variant_summaries,
    }


def rescore_results_file(
    in_path: str, out_path: str, bronze_dir: str, tolerance_mm: Optional[float] = None,
) -> dict:
    """Recomputes an existing (v1, wrong-frame) results file under the corrected ground-truth
    frame WITHOUT re-running any model: the raw model outputs / predicted pixels in the file are
    frame-independent, so only the pixel->mm homography (recomputed per frame from the video) and
    the error/hit fields change. Also adds `target_transition` flags and the settled-only summary.
    Writes `out_path` (never overwrites `in_path`) with `scoring_frame_version: 2`."""
    with open(in_path) as fh:
        old = json.load(fh)
    tol = tolerance_mm if tolerance_mm is not None else float(old.get("tolerance_mm", DEFAULT_TOLERANCE_MM))
    aruco_markers, _f, plat_w, plat_h = load_manifest_full(GROUND_TRUTH_MANIFEST)
    aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}

    new_sessions = []
    for sess in old.get("sessions", []):
        name = sess["session"]
        if "variants" not in sess:
            new_sessions.append(sess)
            continue
        sdir = os.path.join(bronze_dir, name)
        df = pd.read_csv(os.path.join(sdir, "telemetry.csv"))
        flags = compute_transition_flags(df)
        by_fi = df.set_index("frame_index")
        wanted = {f["frame_index"] for v in sess["variants"].values() for f in v["frames"]}
        frames = read_frames_at_indices(os.path.join(sdir, "rgb_video.mp4"), wanted)
        homographies = {fi: estimate_homography_from_aruco(img, aruco_lookup) for fi, img in frames.items()}
        raw_hw = next(iter(frames.values())).shape[:2] if frames else (480, 640)
        # The 2026-09-18 file did not record the model-input size; Qwen's is a deterministic
        # function of the frame size + this backend's (default, unchanged) pixel budget.
        if old.get("backend") == "qwen2_5_vl_3b_instruct":
            coord_space, model_hw = "model_input", qwen_model_input_hw(*raw_hw)
        else:
            coord_space, model_hw = "raw_image", None
        variants_out = {}
        for vname, v in sess["variants"].items():
            new_frames = []
            for f in v["frames"]:
                f = dict(f)
                fi = f["frame_index"]
                row = by_fi.loc[fi]
                f["target_transition"] = bool(flags.loc[df.index[df["frame_index"] == fi][0]])
                H = homographies.get(fi)
                if f.get("parse_ok") and f.get("target_point_px") and H is not None:
                    px, py = f["target_point_px"]
                    f.update(score_prediction(H, px, py, float(row["target_x"]), float(row["target_y"]),
                                              tol, plat_w, plat_h, coord_space=coord_space,
                                              model_input_hw=model_hw, raw_hw=tuple(raw_hw)))
                elif f.get("parse_ok"):
                    f.pop("hit", None)
                    f["error"] = "no_aruco_homography"
                new_frames.append(f)
            variants_out[vname] = summarize_variant_frames(new_frames)
        new_sessions.append({"session": name, "variants": variants_out})

    out = {
        "backend": old.get("backend"),
        "prompt_variants": old.get("prompt_variants"),
        "tolerance_mm": tol,
        "scoring_frame_version": 2,
        "rescored_from": os.path.basename(in_path),
        "rescore_note": "Model outputs unchanged; error/hit recomputed with (1) telemetry "
                        "target_x/y converted to the manifest mm frame and (2) Qwen's answers "
                        "interpreted in the resized model-input space (see module docstring and "
                        "score_prediction()). error_mm_alt_raw = as-parsed pixels taken as raw-frame "
                        "pixels; error_mm_legacy = the original 2026-09-18 value.",
        "sessions": new_sessions,
    }
    atomic_write_json(out_path, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline parse/localization scoring of the Arm 2 minimal-baseline "
                     "prompt+parse path against real Track 4 session data."
    )
    parser.add_argument("--backend", type=str, default="qwen2_5_vl_3b_instruct",
                         choices=list(BACKEND_REGISTRY.keys()))
    parser.add_argument("--model-dir-or-repo", type=str, default=None,
                         help="Local checkpoint dir (Qwen) or HF repo id (other candidates). "
                              "Defaults to each backend's own default.")
    parser.add_argument("--bronze-dir", type=str,
                         default=os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze"))
    parser.add_argument("--session-pattern", type=str, default="session_jetson_track4_*")
    parser.add_argument("--sessions", type=str, nargs="*", default=None,
                         help="Explicit session directory names (under --bronze-dir) instead "
                              "of globbing --session-pattern -- for a small, fast smoke run.")
    parser.add_argument("--max-frames-per-session", type=int, default=3,
                         help="Kept small by default -- see module docstring's small-sample "
                              "discipline note. Raise this deliberately for a real run, don't "
                              "treat the default as a statistically powered setting.")
    parser.add_argument("--tolerance-mm", type=float, default=DEFAULT_TOLERANCE_MM)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-variants", type=str, nargs="*", default=["baseline"],
                         choices=list(PROMPT_VARIANTS.keys()),
                         help="One or more prompt templates from core/minimal_vlm_policy.py's "
                              "PROMPT_VARIANTS to score, e.g. --prompt-variants baseline "
                              "oriented. All requested variants are scored against the SAME "
                              "sampled frames (one backend load, one video decode, one "
                              "homography per frame) for an apples-to-apples comparison.")
    parser.add_argument("--out", type=str, default=None, help="Write full per-frame JSON results here.")
    parser.add_argument("--rescore-from", type=str, default=None,
                         help="Path to an OLD (pre-2026-09-19, wrong-ground-truth-frame) results "
                              "JSON. Recomputes it under the corrected frame without running any "
                              "model and writes --out (required). Does not overwrite the input.")
    args = parser.parse_args()

    if args.rescore_from:
        if not args.out:
            parser.error("--rescore-from requires --out")
        if os.path.abspath(args.out) == os.path.abspath(args.rescore_from):
            parser.error("--out must differ from --rescore-from (the original is kept as-is)")
        res = rescore_results_file(args.rescore_from, args.out, args.bronze_dir, args.tolerance_mm)
        print(f"Rescored {args.rescore_from} -> {args.out} (scoring_frame_version=2)")
        for variant in res["prompt_variants"] or []:
            frames = [f for s in res["sessions"] if "variants" in s for f in s["variants"][variant]["frames"]]
            summ = summarize_variant_frames(frames)
            print(f"  [{variant}] pooled n={summ['n_frames_sampled']} parsed={summ['n_parsed']} "
                  f"hits(<= {res['tolerance_mm']}mm)={sum(1 for f in frames if f.get('hit'))} "
                  f"hit_rate={summ['hit_rate']:.3f} mean_err={summ['mean_error_mm']:.1f} "
                  f"median_err={summ['median_error_mm']:.1f} | settled-only n={summ['n_settled']} "
                  f"hit_rate={summ['hit_rate_settled']:.3f} mean_err={summ['mean_error_mm_settled']:.1f} | "
                  f"as-raw-px(GT frame fixed only) mean_err={summ['mean_error_mm_alt_raw']:.1f} | "
                  f"legacy(both bugs) mean_err={summ['mean_error_mm_legacy']:.1f}")
        return

    if args.sessions:
        session_dirs = [os.path.join(args.bronze_dir, s) for s in args.sessions]
    else:
        session_dirs = sorted(glob.glob(os.path.join(args.bronze_dir, args.session_pattern)))
        session_dirs = [d for d in session_dirs if not d.endswith(".dvc") and os.path.isdir(d)]

    if not session_dirs:
        print("No sessions found -- nothing to score.")
        return

    backend_cls = BACKEND_REGISTRY[args.backend]
    if args.backend == "qwen2_5_vl_3b_instruct":
        backend = backend_cls(model_dir=args.model_dir_or_repo, max_new_tokens=args.max_new_tokens)
    else:
        backend = backend_cls(model_dir_or_repo=args.model_dir_or_repo)
    # One policy per requested prompt variant, all sharing the SAME backend instance --
    # VLMBackend.load() is documented idempotent, so the model is loaded once regardless of
    # how many variants are scored (see this module's "2026-09-18 follow-up" docstring note).
    policies = {
        variant: MinimalVLMPolicy(backend=backend, pixel_to_mm=None, prompt_variant=variant)
        for variant in args.prompt_variants
    }  # per-frame homography supplied below, not via pixel_to_mm

    aruco_markers, _features, _w, _h = load_manifest_full(GROUND_TRUTH_MANIFEST)
    aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}

    # --- Checkpointing + resume (project convention, added 2026-09-18: any process expected
    # to run >30 minutes must persist partial progress periodically -- per-frame/per-session,
    # to a JSON/log file -- not only write output at the end, with skip-already-scored-items
    # on resume). This run is a multi-hour CPU-only VLM scoring job, squarely in scope.
    #
    # `--out`, if given, doubles as BOTH the final results file AND the checkpoint/resume
    # file: if it already exists when this script starts, sessions already fully scored for
    # every requested prompt variant (at >= the requested frame count) are loaded from it and
    # skipped rather than re-run; everything else is scored fresh. During scoring,
    # `score_session()`'s `checkpoint_cb` fires after every single generate() call and
    # atomically rewrites `--out` with the run's current state (all sessions done so far, plus
    # the in-progress session's partial frame results) -- so an interruption at any point loses
    # at most the single generate() call in flight, never the whole run.
    session_results_by_name: Dict[str, dict] = {}
    if args.out and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                prior = json.load(f)
            if prior.get("scoring_frame_version") != 2:
                raise KeyError(
                    "existing --out file is scoring_frame_version != 2 (pre-2026-09-19 wrong "
                    "ground-truth frame); refusing to resume into it -- use --rescore-from to "
                    "convert it, or choose a new --out")
            for entry in prior.get("sessions", []):
                session_results_by_name[entry["session"]] = entry
            print(f"Resuming from existing --out file: {args.out} "
                  f"({len(session_results_by_name)} session(s) recorded)")
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            print(f"Could not parse existing --out file for resume ({exc}) -- starting fresh.")
            session_results_by_name = {}

    def flush_checkpoint() -> None:
        if not args.out:
            return
        ordered = [
            session_results_by_name[os.path.basename(d)]
            for d in session_dirs
            if os.path.basename(d) in session_results_by_name
        ]
        atomic_write_json(
            args.out,
            {
                "backend": args.backend,
                "prompt_variants": args.prompt_variants,
                "tolerance_mm": args.tolerance_mm,
                "scoring_frame_version": 2,
                "sessions": ordered,
            },
        )

    def is_session_complete(entry: Optional[dict]) -> bool:
        if not entry or "error" in entry or entry.get("in_progress"):
            return False
        variants = entry.get("variants", {})
        return all(
            v in variants and variants[v].get("n_frames_sampled", 0) >= args.max_frames_per_session
            for v in args.prompt_variants
        )

    print(f"Scoring backend={args.backend} prompt_variants={args.prompt_variants} over "
          f"{len(session_dirs)} session(s), max {args.max_frames_per_session} frames/session, "
          f"tolerance={args.tolerance_mm}mm")

    all_results = []
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        print(f"\n--- {session_name} ---")

        if is_session_complete(session_results_by_name.get(session_name)):
            print(f"  SKIPPING (already fully scored for {args.prompt_variants} in {args.out})")
            all_results.append(session_results_by_name[session_name])
            continue

        def checkpoint_cb(sess_name: str, partial_variants: dict, _name=session_name) -> None:
            session_results_by_name[_name] = {
                "session": sess_name,
                "variants": partial_variants,
                "in_progress": True,
            }
            flush_checkpoint()

        result = score_session(
            session_dir, policies, aruco_lookup, args.max_frames_per_session, args.tolerance_mm,
            checkpoint_cb=checkpoint_cb,
        )
        session_results_by_name[session_name] = result  # final write replaces the in_progress one
        flush_checkpoint()
        all_results.append(result)
        if "error" in result:
            print(f"  SKIPPED: {result['error']}")
            continue
        for variant, summary in result["variants"].items():
            print(f"  [{variant}] frames sampled={summary['n_frames_sampled']} "
                  f"scored={summary['n_frames_scored']} parsed={summary['n_parsed']} "
                  f"parse_rate={summary['parse_success_rate']}")
            print(f"  [{variant}] hit_rate={summary['hit_rate']} mean_error_mm={summary['mean_error_mm']} "
                  f"mean_generate_latency_s={summary['mean_generate_latency_s']}")

    # Cross-session, per-variant pooled summary (frames pooled across all scored sessions) --
    # printed IN ADDITION TO per-session numbers above, never instead of them, per the
    # model-iteration-constraints skill's "report per-session, don't treat one pooled number
    # as a verdict" guidance restated below. This pooled view is what actually answers
    # "baseline vs oriented, at a real n" -- per-session n is too small on its own.
    print("\n=== Pooled across all scored sessions, per prompt variant ===")
    for variant in args.prompt_variants:
        pooled_frames = []
        for result in all_results:
            if "error" in result:
                continue
            pooled_frames.extend(result["variants"][variant]["frames"])
        parsed = [f for f in pooled_frames if f.get("parse_ok")]
        n_scored = len([f for f in pooled_frames if "error" not in f])
        hit_rate = (sum(1 for f in parsed if f["hit"]) / len(parsed)) if parsed else None
        mean_err = float(np.mean([f["error_mm"] for f in parsed])) if parsed else None
        median_err = float(np.median([f["error_mm"] for f in parsed])) if parsed else None
        parse_rate = (len(parsed) / n_scored) if n_scored else None
        print(f"  [{variant}] n_scored={n_scored} n_parsed={len(parsed)} "
              f"parse_success_rate={parse_rate} hit_rate(<= {args.tolerance_mm}mm)={hit_rate} "
              f"mean_error_mm={mean_err} median_error_mm={median_err}")

    if args.out:
        # Already atomically checkpointed after every generate() call during the loop above
        # (`flush_checkpoint()`) -- this final call is just the last, complete write, using
        # the exact same atomic-write path, not a separate/duplicate write mechanism.
        flush_checkpoint()
        print(f"\nFull per-frame results written to {args.out}")

    print(
        "\nReminder (model-iteration-constraints skill): report per-session AND pooled numbers "
        "together, don't treat either alone as a final verdict without checking session-to-"
        "session spread, and don't rank prompt variants from a single run of this script "
        "without re-running to check stability."
    )


if __name__ == "__main__":
    main()
