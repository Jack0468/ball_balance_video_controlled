"""Coordinate-space calibration probe for the Arm 2 minimal-baseline candidates.

WHY (2026-09-22). The scorer already hit one real pixel-space mix-up: Qwen2.5-VL answers in its
resized 504x364 input space, not in raw 640x480 pixels, which invalidated the whole 2026-09-18
result until `to_raw_px` was added. The other candidates' conventions (InternVL 0-1000 normalized,
PaliGemma2 `<loc>` 1024 grid with Y first, Moondream2 normalized 0-1) come from documentation only
and have never been observed from a real run. A model that crops or letterboxes internally would
not even map linearly onto the raw frame, which looks exactly like "poor accuracy" in the sweep but
is a mapping error. This module settles the mapping per candidate BEFORE a long sweep, on frames
where the truth is known exactly.

WHAT. Synthetic 640x480 frames (the real raw frame size): uniform mid-grey background, one large
(30 px) saturated red disc at a KNOWN position. Positions are deliberately asymmetric and include
near-edge ones (`PROBE_POSITIONS`, checked by `validate_positions`) so x/y swaps, aspect
distortion, offsets and axis-order differences are distinguishable. Each frame is driven through
the SAME path as the real sweep: `MinimalVLMPolicy(backend, prompt_variant=...).act(frame, "go_red")`
-- same prompt text, same backend call, same parser -- and the parsed point is compared with the
truth under every plausible hypothesis (the names `minimal_vlm_policy.to_raw_px` understands:
`raw_image`, `model_input`, `norm1000`, `norm1`, each also with `_yx` = row-first answers), plus a
free per-axis linear fit `true = a*pred + b`. A pure resize/rescale gives b ~ 0; a crop or letterbox
gives a non-zero offset even when a and the axis order are right.

Because the hypotheses ARE `to_raw_px` coordinate-space names, the fix this module recommends for a
mismatched backend is a `BackendOutput.coord_space` string the scorer already understands.

VERDICTS (thresholds are module constants, judgement calls, exposed for the caller):
  INSUFFICIENT          fewer than MIN_PARSED positions parsed -- nothing can be concluded.
  NO_FIT                best explanation still worse than NO_FIT_PX: the model cannot localize even a
                        large saturated red dot. A CAPABILITY finding, not a mapping error.
  CONSISTENT            the backend's declared coord_space reproduces the truth about as well as the
                        best explanation (not materially worse).
  MISMATCH_NAMED        a different named hypothesis fits clearly and materially better than the
                        declared one -> the sweep would measure a mapping error. Recommendation names
                        the coord_space to declare instead.
  MISMATCH_UNSUPPORTED  no named hypothesis fits, but a tight linear map does (crop/letterbox/offset
                        or a mixed axis order): the answers are structured but `to_raw_px` cannot
                        express them.
"Materially worse" = declared error > FIT_GOOD_PX AND >= best error + MATERIAL_ABS_PX AND >=
MATERIAL_RATIO x best error. (A declared mapping that is itself within FIT_GOOD_PX is never refused,
however tight the best explanation; a realistic letterbox to a square at this aspect ratio is ~28 px.)

This module never imports torch/transformers and does not import the sweep engine: it takes a
constructed backend object, so the Colab notebook can adopt it later unchanged.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.minimal_vlm_policy import COORD_SPACES, MinimalVLMPolicy, to_raw_px  # noqa: E402

PROBE_FORMAT = "arm2_coord_probe_v1"

# --- the synthetic frames -------------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480            # the real raw frame size (Track 4 camera)
DISC_DIAMETER_PX = 30
BACKGROUND_GRAY = 128
DISC_BGR = (0, 0, 255)                 # saturated red (BGR); HSV (0, 255, 255)
INSTRUCTION = "go_red"                 # same instruction, hence same prompt/parser path, as the sweep
# (x, y) disc centres in continuous raw-pixel coordinates (pixel i spans [i, i+1)). Asymmetric on
# purpose: no pair is a mirror/point-reflection of another about the frame centre (checked below),
# the set reaches near every edge, and every position has |x - y| > 60 so an x/y swap is visible.
PROBE_POSITIONS: Tuple[Tuple[int, int], ...] = (
    (52, 131), (548, 96), (301, 402), (598, 431), (139, 288),
    (457, 187), (34, 452), (605, 233), (233, 41),
)

# --- decision thresholds (raw pixels of the 640x480 frame; 20 mm is roughly 40+ px on the rig) ----
MIN_PARSED = 5           # positions that must parse before any conclusion is drawn
FIT_GOOD_PX = 20.0       # mean error at or under this: a hypothesis "fits" (about 1.3 disc radii)
NO_FIT_PX = 40.0         # best explanation worse than this: the model does not localize the dot
MATERIAL_ABS_PX = 15.0   # declared must be worse than the best explanation by at least this ...
MATERIAL_RATIO = 2.0     # ... and by at least this factor, to count as a mismatch
LINEAR_MIN_CORR = 0.9    # a free linear map only counts as an explanation if both axes correlate this well
NAMED_PREFERENCE_PX = 10.0  # a named hypothesis is preferred over the linear fit if within this many px
MAX_CONSECUTIVE_ERRORS = 3

VERDICT_CONSISTENT = "CONSISTENT"
VERDICT_NO_FIT = "NO_FIT"
VERDICT_INSUFFICIENT = "INSUFFICIENT"
VERDICT_MISMATCH_NAMED = "MISMATCH_NAMED"
VERDICT_MISMATCH_UNSUPPORTED = "MISMATCH_UNSUPPORTED"
REFUSING_VERDICTS = (VERDICT_MISMATCH_NAMED, VERDICT_MISMATCH_UNSUPPORTED)

# Where a human changes the declared coordinate space, per backend registry key. The declared value
# is the `coord_space=` argument of the `BackendOutput(...)` a backend's generate() returns.
BACKEND_FIELD_HINTS: Dict[str, str] = {
    "qwen2_5_vl_3b_instruct": "core/vlm_backends.py, QwenVLBackend.generate(): BackendOutput(coord_space=...)",
    "internvl2_5_4b": "core/vlm_backends.py, InternVLBackend.generate(): add coord_space=<name> to the BackendOutput(...)",
    "internvl3_5_4b_hf": "core/vlm_backends.py, InternVLNativeBackend.generate(): add coord_space=<name> to the BackendOutput(...)",
    "paligemma2_3b_mix": ("core/vlm_backends.py, PaliGemma2Backend.generate(): coord_space=<name> on the BackendOutput -- BUT "
                          "if the answers are <loc> tokens they are already denormalised by the parser "
                          "(core/minimal_vlm_policy.py parse_minimal_baseline_output, branch 4: Y-first, /1024), so a "
                          "wrong axis order or grid is fixed THERE, not via coord_space"),
    "moondream2": ("core/vlm_backends.py, Moondream2Backend.generate(): ':query' -> BackendOutput(coord_space=...); "
                   "':point' -> the p['x']*img_w denormalisation of native_points inside generate() (a value that is "
                   "already denormalised there cannot be re-declared via coord_space)"),
    "mock": "core/mock_vlm_backends.py (test double)",
}


# ------------------------------------------------------------------------------------------
# Frames
# ------------------------------------------------------------------------------------------

def make_probe_frame(cx: float, cy: float, w: int = FRAME_W, h: int = FRAME_H,
                     diameter: float = DISC_DIAMETER_PX) -> np.ndarray:
    """BGR uint8 (h, w, 3): grey background, one saturated red disc centred at (cx, cy) in continuous
    pixel coordinates. BGR because `MinimalVLMPolicy.act()` takes camera-convention frames and
    converts to RGB itself, exactly as the real sweep's decoded frames are passed."""
    frame = np.full((h, w, 3), BACKGROUND_GRAY, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= (diameter / 2.0) ** 2
    frame[inside] = DISC_BGR
    return frame


def frame_sha1(frame: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()[:12]


def validate_positions(positions: Sequence[Tuple[float, float]], w: int = FRAME_W, h: int = FRAME_H,
                       diameter: float = DISC_DIAMETER_PX, tol: float = 12.0) -> List[str]:
    """Design checks for a set of probe positions; returns human-readable problems (empty = OK)."""
    problems: List[str] = []
    r = diameter / 2.0
    pts = [(float(x), float(y)) for x, y in positions]
    if len(pts) < 6:
        problems.append(f"only {len(pts)} positions (need >= 6)")
    for x, y in pts:
        if x - r < 2 or y - r < 2 or x + r > w - 2 or y + r > h - 2:
            problems.append(f"disc at ({x:g}, {y:g}) is not fully inside the frame")
    for i, (x1, y1) in enumerate(pts):
        for x2, y2 in pts[i + 1:]:
            if abs(x1 + x2 - w) < tol and abs(y1 + y2 - h) < tol:
                problems.append(f"({x1:g},{y1:g}) and ({x2:g},{y2:g}) are point-symmetric about the centre")
            if abs(x1 + x2 - w) < tol and abs(y1 - y2) < tol:
                problems.append(f"({x1:g},{y1:g}) and ({x2:g},{y2:g}) mirror each other in x")
            if abs(y1 + y2 - h) < tol and abs(x1 - x2) < tol:
                problems.append(f"({x1:g},{y1:g}) and ({x2:g},{y2:g}) mirror each other in y")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if min(xs) > 0.15 * w or max(xs) < 0.85 * w or min(ys) > 0.15 * h or max(ys) < 0.85 * h:
        problems.append("positions do not reach near all four edges")
    if sum(1 for x, y in pts if abs(x - y) > 60) < len(pts) - 1:
        problems.append("too many positions near the x=y diagonal (an x/y swap would be invisible there)")
    return problems


def frame_spec(positions: Sequence[Tuple[float, float]] = PROBE_POSITIONS) -> Dict[str, Any]:
    """JSON-able description of the synthetic frame set (recorded next to the raw answers)."""
    return {"width": FRAME_W, "height": FRAME_H, "disc_diameter_px": DISC_DIAMETER_PX,
            "background_gray": BACKGROUND_GRAY, "disc_bgr": list(DISC_BGR), "instruction": INSTRUCTION,
            "positions_xy": [[float(x), float(y)] for x, y in positions],
            "frame_sha1": [frame_sha1(make_probe_frame(x, y)) for x, y in positions]}


def write_probe_frames(out_dir: str, positions: Sequence[Tuple[float, float]] = PROBE_POSITIONS) -> List[str]:
    """Writes the synthetic frames as PNGs (so a human can look at exactly what the model saw).
    Returns the paths written; [] (with a printed note) if no image writer is available."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    try:
        import cv2
    except ImportError:
        print("  (cv2 unavailable: probe frames not written as PNGs; they are deterministic, see frame_spec)")
        return paths
    for i, (x, y) in enumerate(positions):
        path = os.path.join(out_dir, f"probe_{i:02d}_x{int(x)}_y{int(y)}.png")
        cv2.imwrite(path, make_probe_frame(x, y))
        paths.append(path)
    return paths


# ------------------------------------------------------------------------------------------
# Running the probe
# ------------------------------------------------------------------------------------------

def run_probe_variant(
    backend: Any, variant: str, positions: Sequence[Tuple[float, float]] = PROBE_POSITIONS,
    on_observation: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """One prompt variant over every probe position, through the real policy path. Per-call failures
    are recorded, not raised; MAX_CONSECUTIVE_ERRORS in a row abort the rest (marked not_run).
    KeyboardInterrupt and other BaseExceptions propagate."""
    policy = MinimalVLMPolicy(backend=backend, pixel_to_mm=None, prompt_variant=variant)
    out: List[Dict[str, Any]] = []
    consecutive = 0
    for i, (cx, cy) in enumerate(positions):
        obs: Dict[str, Any] = {"index": i, "true_xy": [float(cx), float(cy)], "variant": variant}
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            obs.update(parse_ok=False, error="not_run: aborted after consecutive call failures")
            out.append(obs)
            continue
        frame = make_probe_frame(cx, cy)
        obs["frame_sha1"] = frame_sha1(frame)
        try:
            policy.act(frame, INSTRUCTION, state={})
        except Exception as exc:  # noqa: BLE001 -- fault isolation, same policy as the sweep engine
            consecutive += 1
            obs.update(parse_ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        else:
            consecutive = 0
            dbg = policy.last_debug
            pt = dbg.get("target_point_px")
            obs.update(
                raw_text=dbg.get("raw_text"), parse_ok=bool(dbg.get("parse_ok")), parse_reason=dbg.get("parse_reason"),
                pred_xy=[float(pt[0]), float(pt[1])] if pt else None,
                declared_space=dbg.get("coord_space", "raw_image"),
                model_input_hw=list(dbg["model_input_hw"]) if dbg.get("model_input_hw") else None,
                raw_image_hw=list(dbg.get("raw_image_hw", (FRAME_H, FRAME_W))),
                latency_s=dbg.get("generate_latency_s"), error=None)
        out.append(obs)
        if on_observation is not None:
            on_observation(obs)
    return out


# ------------------------------------------------------------------------------------------
# Analysis (pure functions of the observations)
# ------------------------------------------------------------------------------------------

def _hypothesis_names(has_model_input_hw: bool) -> List[str]:
    names: List[str] = []
    for base in COORD_SPACES:
        if base == "model_input" and not has_model_input_hw:
            continue
        names.extend([base, base + "_yx"])
    return names


def _map(pred: np.ndarray, hypothesis: str, model_input_hw: Optional[Tuple[int, int]],
         raw_hw: Tuple[int, int]) -> np.ndarray:
    return np.array([to_raw_px(float(x), float(y), hypothesis, model_input_hw, raw_hw) for x, y in pred])


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])


def _fit_1d(src: np.ndarray, dst: np.ndarray) -> Dict[str, float]:
    if len(src) < 2 or float(np.ptp(src)) < 1e-9:
        a, b, corr = 0.0, float(np.mean(dst)), 0.0
    else:
        a, b = (float(v) for v in np.polyfit(src, dst, 1))
        corr = float(np.corrcoef(src, dst)[0, 1]) if float(np.ptp(dst)) > 1e-9 else 0.0
    resid = dst - (a * src + b)
    return {"a": a, "b": b, "resid_rms_px": float(np.sqrt(np.mean(resid ** 2))), "corr": corr}


def fit_linear(pred: np.ndarray, true: np.ndarray) -> Dict[str, Any]:
    """Per-axis linear map `true_axis = a * pred_source + b`. The source axis is the same axis unless
    the other one fits at least twice as tightly (an x/y swap). `valid` = both axes correlate >=
    LINEAR_MIN_CORR, from two DIFFERENT source axes, on >= MIN_PARSED points (i.e. the answers are
    genuinely a structured function of the truth, whatever that function is)."""
    result: Dict[str, Any] = {"n": int(len(pred))}
    chosen: List[int] = []
    resid_xy: List[np.ndarray] = []
    for j, axis in enumerate(("x", "y")):
        same = _fit_1d(pred[:, j], true[:, j])
        other = _fit_1d(pred[:, 1 - j], true[:, j])
        use_other = other["resid_rms_px"] < 0.5 * same["resid_rms_px"]
        fit = other if use_other else same
        src = 1 - j if use_other else j
        chosen.append(src)
        resid_xy.append(true[:, j] - (fit["a"] * pred[:, src] + fit["b"]))
        result[axis] = {**{k: round(v, 4) for k, v in fit.items()}, "source": "xy"[src]}
    result["mean_err_px"] = float(np.mean(np.hypot(resid_xy[0], resid_xy[1])))
    result["valid"] = bool(len(pred) >= MIN_PARSED and chosen[0] != chosen[1]
                           and min(abs(result["x"]["corr"]), abs(result["y"]["corr"])) >= LINEAR_MIN_CORR)
    return result


def analyze(
    observations: Sequence[Dict[str, Any]], raw_hw: Tuple[int, int] = (FRAME_H, FRAME_W),
    model_input_hw_hint: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Verdict for one candidate x prompt-variant from its observations. `model_input_hw_hint`
    (h, w) enables the `model_input` hypotheses for a backend that does not report its input size at
    run time (only Qwen does); it is recorded as a hint, never presented as a measurement."""
    good = [o for o in observations if o.get("parse_ok") and o.get("pred_xy")]
    res: Dict[str, Any] = {"n_positions": len(observations), "n_parsed": len(good), "raw_hw": list(raw_hw),
                           "thresholds": {"fit_good_px": FIT_GOOD_PX, "no_fit_px": NO_FIT_PX,
                                          "material_abs_px": MATERIAL_ABS_PX, "material_ratio": MATERIAL_RATIO,
                                          "min_parsed": MIN_PARSED}}
    if len(good) < MIN_PARSED:
        res.update(verdict=VERDICT_INSUFFICIENT, agrees_with_declared=None, enforce=False,
                   declared_space=(good[0]["declared_space"] if good else None),
                   detail=f"only {len(good)}/{len(observations)} positions produced a parseable answer "
                          f"(need >= {MIN_PARSED}); nothing can be concluded about the coordinate space")
        return res

    pred = np.array([o["pred_xy"] for o in good], dtype=float)
    true = np.array([o["true_xy"] for o in good], dtype=float)
    declared = collections.Counter(o["declared_space"] for o in good).most_common(1)[0][0]
    runtime_hw = next((tuple(o["model_input_hw"]) for o in good if o.get("model_input_hw")), None)
    mi_hw = runtime_hw or model_input_hw_hint
    res.update(declared_space=declared, model_input_hw=list(mi_hw) if mi_hw else None,
               model_input_hw_source=("runtime" if runtime_hw else ("hint" if model_input_hw_hint else None)))

    # What the sweep would do: each answer mapped with the backend's own declared space + reported size.
    declared_err = _dist(np.array([to_raw_px(*o["pred_xy"], o["declared_space"], o.get("model_input_hw"), raw_hw)
                                   for o in good]), true)
    res["declared_mean_err_px"] = float(declared_err.mean())

    scores: Dict[str, Dict[str, float]] = {}
    for name in _hypothesis_names(mi_hw is not None):
        e = _dist(_map(pred, name, mi_hw, raw_hw), true)
        scores[name] = {"mean_err_px": float(e.mean()), "median_err_px": float(np.median(e)), "max_err_px": float(e.max())}
    res["hypotheses"] = scores
    order = sorted(scores, key=lambda n: (round(scores[n]["mean_err_px"], 1), n != declared))  # ties -> declared
    best_name = order[0]
    res["best_hypothesis"] = best_name
    res["best_hypothesis_mean_err_px"] = scores[best_name]["mean_err_px"]

    lin_raw = fit_linear(pred, true)                                   # true = a * (parsed) + b
    lin_best = fit_linear(_map(pred, best_name, mi_hw, raw_hw), true)  # true = a' * (best-mapped px) + b'
    res["linear_fit_vs_parsed"] = lin_raw
    res["linear_fit_vs_best_hypothesis"] = lin_best

    named_err = scores[best_name]["mean_err_px"]
    linear_ok = lin_raw["valid"]
    if linear_ok and named_err > lin_raw["mean_err_px"] + NAMED_PREFERENCE_PX:
        explanation, best_err = "linear_fit", lin_raw["mean_err_px"]
    else:
        explanation, best_err = best_name, named_err
    res["explanation"] = explanation
    res["explanation_mean_err_px"] = best_err
    res["agrees_with_declared"] = bool(declared_err.mean() <= FIT_GOOD_PX or explanation == declared)

    d = float(declared_err.mean())
    if best_err > NO_FIT_PX:
        res.update(verdict=VERDICT_NO_FIT, enforce=False,
                   detail=(f"NO hypothesis fits: the best explanation ({explanation}) is still {best_err:.0f} px off on "
                           f"average (> {NO_FIT_PX:g} px), and a free linear map does not explain it "
                           f"({'valid' if linear_ok else 'not valid'}). The model cannot localize even a large saturated "
                           f"red disc on a plain background -- a CAPABILITY finding, not a mapping error."))
    elif d > FIT_GOOD_PX and d >= best_err + MATERIAL_ABS_PX and d >= MATERIAL_RATIO * best_err:
        unsupported = explanation == "linear_fit"
        res.update(verdict=VERDICT_MISMATCH_UNSUPPORTED if unsupported else VERDICT_MISMATCH_NAMED, enforce=True)
        if unsupported:
            res["detail"] = (f"declared coord_space {declared!r} is {d:.0f} px off, no named hypothesis fits (best "
                             f"{best_name!r}: {named_err:.0f} px), but a linear map reproduces the answers to "
                             f"{best_err:.1f} px (x: a={lin_raw['x']['a']:.3f} b={lin_raw['x']['b']:+.1f} from "
                             f"{lin_raw['x']['source']}; y: a={lin_raw['y']['a']:.3f} b={lin_raw['y']['b']:+.1f} from "
                             f"{lin_raw['y']['source']}). Structured but not expressible by to_raw_px: a crop/letterbox "
                             f"offset or a mixed axis order. Needs a backend/parser change, not just a coord_space name.")
        else:
            res["detail"] = (f"declared coord_space {declared!r} reproduces the truth to {d:.0f} px; {best_name!r} "
                             f"reproduces it to {best_err:.1f} px. The sweep would be measuring a mapping error. "
                             f"Declare coord_space={best_name!r} instead.")
    else:
        res.update(verdict=VERDICT_CONSISTENT, enforce=False,
                   detail=(f"declared coord_space {declared!r}: {d:.1f} px mean error; best explanation "
                           f"({explanation}) {best_err:.1f} px -- no material disagreement."))
        if d > FIT_GOOD_PX:
            res["detail"] += f" NOTE: the declared mapping itself is {d:.0f} px off (> {FIT_GOOD_PX:g}); see fitted offsets."
    return res


def worst_verdict(verdicts: Sequence[str]) -> str:
    rank = [VERDICT_MISMATCH_NAMED, VERDICT_MISMATCH_UNSUPPORTED, VERDICT_NO_FIT, VERDICT_INSUFFICIENT, VERDICT_CONSISTENT]
    for v in rank:
        if v in verdicts:
            return v
    return VERDICT_INSUFFICIENT


def recommendation(candidate: str, backend_key: str, variant: str, analysis: Dict[str, Any]) -> str:
    """Human text for a refusing verdict: the evidence and the field to change."""
    where = BACKEND_FIELD_HINTS.get(backend_key, f"the backend class registered as {backend_key!r} in core/vlm_backends.py")
    return (f"[{candidate} / {variant}] {analysis['verdict']}: {analysis['detail']}\n"
            f"    field to change: {where}")


# ------------------------------------------------------------------------------------------
# Reporting / persistence
# ------------------------------------------------------------------------------------------

def _fmt_fit(fit: Dict[str, Any]) -> str:
    parts = []
    for axis in ("x", "y"):
        f = fit[axis]
        parts.append(f"{axis}: a={f['a']:.3f} b={f['b']:+.1f}px resid_rms={f['resid_rms_px']:.1f}px corr={f['corr']:.3f}"
                     + (f" (from {f['source']})" if f["source"] != axis else ""))
    return " | ".join(parts) + f"   [{'valid' if fit['valid'] else 'not a valid linear structure'}]"


def format_report(candidate: str, variant: str, a: Dict[str, Any]) -> str:
    lines = [f"[{candidate} / {variant}]  parsed {a['n_parsed']}/{a['n_positions']}"]
    if a["verdict"] == VERDICT_INSUFFICIENT:
        lines.append(f"  VERDICT: {a['verdict']} -- {a['detail']}")
        return "\n".join(lines)
    lines.append(f"  declared coord_space: {a['declared_space']}"
                 + (f" (model_input_hw={a['model_input_hw']}, {a['model_input_hw_source']})" if a.get("model_input_hw") else "")
                 + f" -> mean error {a['declared_mean_err_px']:.1f} px")
    lines.append(f"  best named hypothesis: {a['best_hypothesis']} -> {a['best_hypothesis_mean_err_px']:.1f} px; "
                 f"agrees with declared: {'YES' if a['agrees_with_declared'] else 'NO'}; explanation: {a['explanation']}")
    lines.append("  all hypotheses (mean px error): " + " | ".join(
        f"{n} {s['mean_err_px']:.0f}" for n, s in sorted(a["hypotheses"].items(), key=lambda kv: kv[1]["mean_err_px"])))
    lines.append("  linear fit true = a*parsed + b   : " + _fmt_fit(a["linear_fit_vs_parsed"]))
    lines.append("  linear fit true = a*best-mapped+b: " + _fmt_fit(a["linear_fit_vs_best_hypothesis"]) + "   (b = residual offset in raw px)")
    lines.append(f"  VERDICT: {a['verdict']} -- {a['detail']}")
    return "\n".join(lines)


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def load_probe_file(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path) as fh:
                doc = json.load(fh)
            if doc.get("format") == PROBE_FORMAT:
                return doc
        except (OSError, json.JSONDecodeError):
            pass
    return {"format": PROBE_FORMAT, "frame_spec": frame_spec(), "candidates": {}}


def save_candidate_record(path: str, key: str, record: Dict[str, Any]) -> None:
    """Merge one candidate's raw answers + analysis into the shared probe file (atomic). Separate
    invocations from different environments share the file, one candidate at a time."""
    doc = load_probe_file(path)
    doc["frame_spec"] = frame_spec()
    doc["candidates"][key] = {**record, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _atomic_write_json(path, doc)


def summarize_for_checkpoint(analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compact per-variant outcome stored in the sweep checkpoint (the raw answers stay in the probe file)."""
    out: Dict[str, Any] = {}
    for variant, a in analyses.items():
        s: Dict[str, Any] = {k: a.get(k) for k in ("verdict", "declared_space", "best_hypothesis", "explanation",
                                                    "agrees_with_declared", "n_parsed", "n_positions", "detail")}
        s["declared_mean_err_px"] = a.get("declared_mean_err_px")
        s["best_hypothesis_mean_err_px"] = a.get("best_hypothesis_mean_err_px")
        lf = a.get("linear_fit_vs_best_hypothesis")
        if lf:
            s["fit_vs_best"] = {ax: {k: lf[ax][k] for k in ("a", "b", "resid_rms_px")} for ax in ("x", "y")}
        out[variant] = s
    return out
