"""Transformers 4.57.6 vs 5.17.0 parity check for Qwen2.5-VL-3B grounding (Arm 2).

Re-runs the baseline prompt on the exact (session, frame_index) pairs in the reference sweep JSON
(produced on the Jetson under transformers 5.17.0, bf16, pinned revision) and compares, per frame, the raw
text and the parsed point. Intended to run inside the arm2-lerobot image (transformers 4.57.6). The
reference file alone decides which frames are compared; nothing is freshly sampled for the comparison set.

Reuses, does not copy: `colab_sweep.prepare_frames` (frame sampling + decode + ArUco homography),
`colab_sweep.default_candidate_specs` / `build_backend` (the Qwen candidate and its pinned revision),
`score_minimal_baseline_offline.score_frame_with_policy` (the per-frame policy call, same as the sweep) and
`atomic_write_json`. The prompt is built inside `MinimalVLMPolicy` from `PROMPT_VARIANTS["baseline"]`.

Model and heavy imports happen only inside `run_parity()`, so this module and its CPU tests import with the
standard library alone.

Checkpointing (CLAUDE.md): the output JSON is rewritten atomically after every frame. A re-run without
`--force` resumes an incomplete checkpoint whose config matches; a completed output, or an incomplete one from
a different config, is refused rather than overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR: str = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

FORMAT_VERSION: str = "qwen_transformers_parity_v1"
PROMPT_VARIANT: str = "baseline"
FRAMES_PER_SESSION: int = 6  # same per-session sampling as the reference run (n_frames_sampled = 6)
MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION: str = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DTYPE: str = "bf16"
MAX_NEW_TOKENS: int = 24
MIN_PIXELS: int = 50176
MAX_PIXELS: int = 200704
TOLERANCE_MM: float = 20.0  # scoring tolerance passed through to the policy call; does not affect parity
EQUIVALENT_PX: float = 4.0
REFERENCE_CONFIG_KEYS: Tuple[str, ...] = ("model", "revision", "dtype", "max_new_tokens", "min_pixels", "max_pixels")

DEFAULT_REFERENCE: str = os.path.join(
    _REPO_ROOT_DIR, "host_software", "data", "arm2_jetson_sweep_results",
    "scorer_format_qwen2_5_vl_3b_jetson_run1.json")
DEFAULT_BRONZE_DIR: str = os.path.join(_REPO_ROOT_DIR, "host_software", "data", "01_bronze")
DEFAULT_OUTPUT: str = os.path.join(_ML_JETSON_VLA_DIR, "reports", "qwen_parity", "qwen_transformers_parity.json")


def frame_key(session: str, frame_index: int) -> str:
    return f"{session}|{frame_index}"


def _point_or_none(pt: Any) -> Optional[Tuple[float, float]]:
    if pt is None:
        return None
    return (float(pt[0]), float(pt[1]))


def load_reference_frames(reference_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns (reference config block, baseline-variant frames in file order)."""
    with open(reference_path, encoding="utf-8") as fh:
        ref = json.load(fh)
    frames: List[Dict[str, Any]] = []
    for sess in ref["sessions"]:
        base = sess.get("variants", {}).get(PROMPT_VARIANT)
        if base is None:
            continue
        for f in base["frames"]:
            frames.append({
                "session": str(sess["session"]),
                "frame_index": int(f["frame_index"]),
                "instruction": str(f["instruction"]),
                "raw_text": f.get("raw_text"),
                "parse_ok": bool(f.get("parse_ok")),
                "target_point_px": _point_or_none(f.get("target_point_px")),
            })
    if not frames:
        raise ValueError(f"no '{PROMPT_VARIANT}' frames found in {reference_path}")
    return dict(ref.get("config", {})), frames


def build_run_config(reference_basename: str) -> Dict[str, Any]:
    """The identity of one parity run. A checkpoint is only resumed when this matches exactly."""
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "dtype": DTYPE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "prompt_variant": PROMPT_VARIANT,
        "reference_file": reference_basename,
    }


def config_mismatches(reference_config: Dict[str, Any], run_config: Dict[str, Any]) -> List[str]:
    """Human-readable differences over the model-defining keys; empty list means they agree."""
    return [
        f"{k}: reference={reference_config.get(k)!r} run={run_config.get(k)!r}"
        for k in REFERENCE_CONFIG_KEYS
        if reference_config.get(k) != run_config.get(k)
    ]


def exact_match(new_text: Optional[str], ref_text: Optional[str]) -> bool:
    return new_text is not None and ref_text is not None and new_text == ref_text


def pixel_abs_diff(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[float]:
    """Euclidean distance in model-input pixels between two parsed points; None if either is missing."""
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_row(
    ref: Dict[str, Any],
    new_text: Optional[str],
    new_parse_ok: bool,
    new_point: Optional[Tuple[float, float]],
    latency_s: Optional[float],
) -> Dict[str, Any]:
    pt_new = new_point if new_parse_ok else None
    pt_ref = ref["target_point_px"] if ref["parse_ok"] else None
    return {
        "session": ref["session"],
        "frame_index": ref["frame_index"],
        "instruction": ref["instruction"],
        "raw_text": new_text,
        "raw_text_reference": ref["raw_text"],
        "exact_match": exact_match(new_text, ref["raw_text"]),
        "parse_ok_new": new_parse_ok,
        "parse_ok_reference": ref["parse_ok"],
        "target_point_px_new": list(pt_new) if pt_new is not None else None,
        "target_point_px_reference": list(pt_ref) if pt_ref is not None else None,
        "pixel_abs_diff": pixel_abs_diff(pt_new, pt_ref),
        "generate_latency_s": latency_s,
    }


def within_tolerance(row: Dict[str, Any], tol_px: float = EQUIVALENT_PX) -> bool:
    d = row.get("pixel_abs_diff")
    return d is not None and d <= tol_px


def summarize(rows: Sequence[Dict[str, Any]], n_reference: int) -> Dict[str, Any]:
    n = len(rows)
    diffs = [float(r["pixel_abs_diff"]) for r in rows if r["pixel_abs_diff"] is not None]
    n_parsed_new = sum(1 for r in rows if r["parse_ok_new"])
    return {
        "n_reference_frames": n_reference,
        "n_compared": n,
        "n_exact_match": sum(1 for r in rows if r["exact_match"]),
        "n_parse_ok_new": n_parsed_new,
        "n_parse_ok_reference": sum(1 for r in rows if r["parse_ok_reference"]),
        "parse_rate_new": (n_parsed_new / n) if n else None,
        "n_within_tolerance": sum(1 for r in rows if within_tolerance(r)),
        "mean_pixel_abs_diff": (sum(diffs) / len(diffs)) if diffs else None,
        "max_pixel_abs_diff": max(diffs) if diffs else None,
    }


def verdict(rows: Sequence[Dict[str, Any]], n_reference: int) -> Tuple[str, int]:
    """Returns (label, n_divergent). Frames not yet run, unparsed, or beyond 4 px count as divergent."""
    n_divergent = n_reference - sum(1 for r in rows if within_tolerance(r))
    if n_divergent > 0:
        return "DIVERGENT", n_divergent
    if len(rows) == n_reference and all(r["exact_match"] for r in rows):
        return "IDENTICAL", 0
    return "EQUIVALENT", 0


def verdict_line(label: str, n_divergent: int, n_reference: int) -> str:
    if label == "IDENTICAL":
        return f"VERDICT: IDENTICAL ({n_reference}/{n_reference} frames match exactly)"
    if label == "EQUIVALENT":
        return (f"VERDICT: EQUIVALENT (all {n_reference} points within {EQUIVALENT_PX:g} px; "
                f"raw text differs on at least one frame)")
    return (f"VERDICT: DIVERGENT ({n_divergent} of {n_reference} frames not within {EQUIVALENT_PX:g} px, "
            f"unparsed, or not run)")


def pending_reference_frames(ref_frames: Sequence[Dict[str, Any]], done_keys: Set[str]) -> List[Dict[str, Any]]:
    return [f for f in ref_frames if frame_key(f["session"], f["frame_index"]) not in done_keys]


def check_resumable(existing: Dict[str, Any], run_config: Dict[str, Any]) -> str:
    """Decides what to do with an existing output file (called only when one exists and --force is off).
    Returns 'resume'; raises FileExistsError for a completed run and ValueError for a config mismatch."""
    if existing.get("complete") is True:
        raise FileExistsError("output file holds a completed run; pass --force to overwrite it")
    if existing.get("config") != run_config:
        raise ValueError("output file is an incomplete checkpoint from a different config; "
                         "pass --force to discard it")
    return "resume"


def _read_existing(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} exists but is not valid JSON ({exc}); pass --force to overwrite") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not hold a parity checkpoint object; pass --force to overwrite")
    return obj


def _write_checkpoint(
    output_path: str,
    run_config: Dict[str, Any],
    reference_path: str,
    environment: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    skipped: Sequence[Dict[str, Any]],
    n_reference: int,
    atomic_write: Callable[[str, dict], None],
) -> Dict[str, Any]:
    label, n_div = verdict(rows, n_reference)
    doc: Dict[str, Any] = {
        "format": FORMAT_VERSION,
        "complete": len(rows) == n_reference,
        "config": run_config,
        "reference_file": reference_path,
        "environment": dict(environment),
        "pixel_abs_diff_definition": "Euclidean distance (model-input px) between target_point_px_new "
                                     "and target_point_px_reference",
        "summary": summarize(rows, n_reference),
        "verdict": label,
        "n_divergent": n_div,
        "skipped": list(skipped),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "frames": list(rows),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    atomic_write(output_path, doc)
    return doc


def run_parity(
    reference_path: str,
    bronze_dir: str,
    output_path: str,
    force: bool = False,
    verbose: bool = True,
    use_fast_processor: Optional[bool] = None,
) -> Dict[str, Any]:
    """Runs (or resumes) the parity check and returns the final checkpoint document."""
    ref_config, ref_frames = load_reference_frames(reference_path)
    run_config = build_run_config(os.path.basename(reference_path))
    mismatches = config_mismatches(ref_config, run_config)
    if mismatches:
        raise ValueError("reference config disagrees with the pinned run config: " + "; ".join(mismatches))

    rows: List[Dict[str, Any]] = []
    if os.path.exists(output_path) and not force:
        # Refusal happens here, before any model import, so an existing result costs nothing.
        existing = _read_existing(output_path)
        if check_resumable(existing, run_config) == "resume":
            rows = list(existing.get("frames", []))
    skipped: List[Dict[str, Any]] = []
    pending = pending_reference_frames(ref_frames, {frame_key(r["session"], r["frame_index"]) for r in rows})
    environment: Dict[str, Any] = {}
    from ml_jetson_vla.deployment.score_minimal_baseline_offline import atomic_write_json

    if pending:
        import torch
        import transformers

        from ml_jetson_vla.core.minimal_vlm_policy import MinimalVLMPolicy
        from ml_jetson_vla.core.vlm_backends import free_gpu_memory
        from ml_jetson_vla.deployment.colab_sweep import build_backend, default_candidate_specs, prepare_frames
        from ml_jetson_vla.deployment.score_minimal_baseline_offline import score_frame_with_policy

        environment = {"transformers": transformers.__version__, "torch": torch.__version__,
                       "cuda_available": bool(torch.cuda.is_available())}
        sessions = list(dict.fromkeys(f["session"] for f in ref_frames))
        frames, prep_skipped = prepare_frames(bronze_dir, frames_per_session=FRAMES_PER_SESSION,
                                              sessions=sessions, verbose=verbose)
        skipped.extend(prep_skipped)
        by_key = {fr.ident: fr for fr in frames}

        spec = default_candidate_specs(max_new_tokens=MAX_NEW_TOKENS, dtype=DTYPE)[0]
        spec.kwargs.update(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        spec_cfg = {"model": spec.kwargs.get("model_dir"), "revision": spec.kwargs.get("revision"),
                    "dtype": spec.kwargs.get("dtype"), "max_new_tokens": spec.kwargs.get("max_new_tokens"),
                    "min_pixels": spec.kwargs.get("min_pixels"), "max_pixels": spec.kwargs.get("max_pixels")}
        spec_mismatch = config_mismatches(spec_cfg, run_config)
        if spec_mismatch:
            raise ValueError("sweep candidate spec disagrees with the pinned run config: " + "; ".join(spec_mismatch))

        backend = build_backend(spec, MAX_NEW_TOKENS)
        # None leaves the backend's own default (preserves prior parity-run behavior); set only when
        # the CLI asks us to pin a specific processor path, e.g. to test whether forcing the historical
        # "slow" processor closes the 2026-10-09 DIVERGENT gap against transformers>=5's default.
        if use_fast_processor is not None:
            backend.use_fast = use_fast_processor
        policy = MinimalVLMPolicy(backend=backend, pixel_to_mm=None, prompt_variant=PROMPT_VARIANT)
        n_ref = len(ref_frames)
        try:
            backend.load()
            environment["dtype_resolved"] = getattr(backend, "dtype_name", None)
            environment["use_fast_processor"] = getattr(backend, "use_fast", None)
            for ref in pending:
                fr = by_key.get(frame_key(ref["session"], ref["frame_index"]))
                if fr is None:
                    skipped.append({"session": ref["session"], "frame_index": ref["frame_index"],
                                    "reason": "not decoded by prepare_frames"})
                    continue
                if fr.instruction != ref["instruction"]:
                    skipped.append({"session": ref["session"], "frame_index": ref["frame_index"],
                                    "reason": f"instruction {fr.instruction!r} != reference {ref['instruction']!r}"})
                    continue
                entry = score_frame_with_policy(
                    policy, PROMPT_VARIANT, fr.frame_bgr, fr.instruction, fr.homography,
                    fr.true_x_tel, fr.true_y_tel, TOLERANCE_MM, fr.frame_index,
                    target_transition=fr.target_transition,
                )
                new_pt = tuple(entry["target_point_px"]) if entry.get("parse_ok") else None
                rows.append(build_row(ref, entry.get("raw_text"), bool(entry.get("parse_ok")), new_pt,
                                      entry.get("generate_latency_s")))
                _write_checkpoint(output_path, run_config, reference_path, environment, rows, skipped,
                                  n_ref, atomic_write_json)
                if verbose:
                    last = rows[-1]
                    d = last["pixel_abs_diff"]
                    print(f"  [{len(rows)}/{n_ref}] {ref['session'][-15:]} f{ref['frame_index']:<5} "
                          f"exact={'Y' if last['exact_match'] else 'N'} "
                          f"diff={'%.2fpx' % d if d is not None else '  -  '} raw={str(last['raw_text'])[:70]!r}")
        finally:
            try:
                backend.unload()
            except Exception:  # cleanup must not mask the real result
                pass
            free_gpu_memory()

    return _write_checkpoint(output_path, run_config, reference_path, environment, rows, skipped,
                             len(ref_frames), atomic_write_json)


def _fmt_px(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B transformers 4.57.6 vs 5.17.0 grounding parity.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="reference sweep JSON (frames + outputs)")
    parser.add_argument("--bronze-dir", default=DEFAULT_BRONZE_DIR, help="directory holding session_jetson_track4_*")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output JSON (refused if it exists, unless --force)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    parser.add_argument("--use-fast-processor", default="default", choices=["default", "true", "false"],
                         help="pin AutoProcessor's use_fast instead of this transformers version's own "
                              "default; 'default' preserves prior behavior exactly")
    args = parser.parse_args(argv)
    use_fast_processor = {"default": None, "true": True, "false": False}[args.use_fast_processor]

    doc = run_parity(args.reference, args.bronze_dir, args.output, force=args.force,
                      use_fast_processor=use_fast_processor)
    s = doc["summary"]
    print(f"compared {s['n_compared']}/{s['n_reference_frames']} frames | exact text {s['n_exact_match']} | "
          f"parse {s['n_parse_ok_new']}/{s['n_compared']} | within {EQUIVALENT_PX:g}px {s['n_within_tolerance']} | "
          f"mean px diff {_fmt_px(s['mean_pixel_abs_diff'])} | max px diff {_fmt_px(s['max_pixel_abs_diff'])}")
    if doc["skipped"]:
        print(f"skipped {len(doc['skipped'])} frame(s); see 'skipped' in {args.output}")
    print(verdict_line(doc["verdict"], doc["n_divergent"], s["n_reference_frames"]))
    print(f"wrote {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
