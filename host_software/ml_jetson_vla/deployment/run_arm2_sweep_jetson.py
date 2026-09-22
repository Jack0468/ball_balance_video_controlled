"""Plain local-execution driver for the Arm 2 minimal-baseline multi-candidate sweep, for running
directly on the Jetson AGX Orin instead of via Colab
(`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`, `deployment/arm2_colab_sweep.ipynb`).
Runs from a FULL repo checkout (`git clone`/`git pull`), not a pruned bundle. Read
`docs/JETSON_ARM2_SWEEP_LOCAL.md` first: it has the environment-isolation decision and the exact
ordered commands; this docstring covers only what the script itself does.

Reuses -- imports, does not copy -- the exact engine the notebook drives:
`deployment/colab_sweep.py` (frame prep, candidate specs, checkpointed sweep runner, validation
gate, aggregation/export) and, transitively, `core/minimal_vlm_policy.py`, `core/vlm_backends.py`,
`deployment/score_minimal_baseline_offline.py` (the corrected v2 scorer). This script is a thin,
linear stage sequencer over that engine (notebook cells -> CLI stages). What genuinely differs
from the Colab version, and how each difference is handled:

- **No Drive I/O, no zip bundle.** `--bronze-dir`/`--results-dir` are plain local paths; the 10
  `session_jetson_track4_*` sessions live on this device already.
- **Session set locked to the reference run's 10 sessions by default.** The Colab bundle shipped
  exactly those 10; a Jetson bronze dir can hold MORE `session_jetson_track4_*` directories
  (collection is ongoing), which would silently change the frame set and void the validation gate.
  So when `--frames-per-session 6` and no `--sessions` is given, the session list is read from
  `--reference-json` and the sampled (session, frame_index) set is asserted identical to the
  reference's (`--all-matching-sessions` opts out, at the cost of the gate being not applicable).
- **HF auth for gated PaliGemma2 without `google.colab.userdata`.** `colab_sweep.get_hf_token()`
  already falls back to the `HF_TOKEN` env var outside Colab; and when neither is set,
  `check_hf_access()` passes `token=None`, which `huggingface_hub` resolves to the cached login
  from a prior `login()` -- so a one-time interactive login is enough, no code path needed. Details
  in the doc.
- **This script never pip-installs anything.** The notebook's setup cell does; here that is
  exactly what silently upgraded numpy underneath Track 1 earlier (`lerobot`). Dependencies are
  the user's explicit, constraint-frozen step (doc). This script only PROBES what is importable
  (`--stage probe`, and automatically before every real run), auto-skips a candidate that is
  statically known not to be able to run in the current environment (loudly, e.g. Moondream2 on
  transformers>=5) rather than downloading several GB and failing at load, and refuses to start
  if the shared plumbing (cv2+aruco, PyAV, pandas) is missing.
- **Validation gate is enforced across separate invocations, not just convention.** The gate's
  outcome is recorded in the checkpoint (`stages["_validation_gate"]`); a later `--stage sweep`
  invocation (typically from the OTHER venv, without Qwen in its `--candidates`) refuses to burn a
  long run unless a passing (or explicitly forced) gate record exists for that `--run-label`
  (`--skip-validation-gate` bypasses knowingly).
- **Coordinate-space calibration (`--stage calibrate`, logic in `deployment/coord_space_probe.py`).**
  Before any long sweep, every candidate is shown synthetic 640x480 frames with one large saturated
  red disc at KNOWN, asymmetric positions, through the same prompt/backend/parser path as the sweep,
  and its answers are tested against every coordinate-space hypothesis (`to_raw_px`'s names: raw px,
  model-input px, 0-1000, 0-1, each also axis-swapped) plus a free per-axis linear fit. The outcome
  is recorded in the checkpoint (`stages["_coord_calibration"]`), like the validation gate's. A
  later sweep DROPS (loudly, exit code 2) any candidate whose declared `coord_space` is materially
  worse than what fits, or that was never calibrated; `--allow-coord-mismatch KEY` overrides for a
  named candidate, `--skip-coord-calibration` bypasses the whole mechanism. A candidate that cannot
  localize even the big red dot only gets a warning (capability finding, not a mapping error). The
  raw answers go to `<results-dir>/coord_probe_<run-label>.json`.
- **Local-disk checkpointing.** `CheckpointStore` (unchanged) does atomic temp-file + `os.replace`
  writes after every single `generate()` call and skip-done resume: the CLAUDE.md >30-minute
  checkpointing rule. Separate invocations from different environments share one checkpoint file
  through the same `--results-dir --run-label` (run them one at a time -- one GPU, and the store is
  not a concurrent-writer design). Each candidate's stage record also gets the environment that
  actually produced it (transformers/torch/... versions), since `meta` is last-writer-wins.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

# Light import only (numpy): the heavy engine (`colab_sweep`: av/cv2/pandas/torch-adjacent) is loaded
# lazily by `_load_engine()` AFTER the environment probe, so `--stage probe` still works in an
# environment that is missing some of those packages -- which is exactly when you want it.
from ml_jetson_vla.core.minimal_vlm_policy import PROMPT_VARIANTS  # noqa: E402
from ml_jetson_vla.deployment import coord_space_probe as csp  # noqa: E402  (numpy + policy only; light)

cs: Any = None  # the colab_sweep module, set by _load_engine()

DEFAULT_BRONZE_DIR = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
DEFAULT_REFERENCE_JSON = os.path.join(
    _THIS_DIR, "arm2_minimal_baseline_prompt_ab_scoring_20260918_RESCORED_v2.json"
)
DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, "arm2_jetson_sweep_results")
STAGES = ("probe", "frames", "hfauth", "smoke", "calibrate", "validate", "sweep", "export", "all")
GATE_REQUIRED_VARIANTS = ("baseline", "oriented")
GATE_STAGE_KEY = "_validation_gate"
CAL_STAGE_KEY = "_coord_calibration"
# Documented input resolutions for backends that do not report one at run time (only Qwen does).
# PaliGemma2-mix-448 resizes to a 448x448 square. A HINT: enables the `model_input` hypotheses in
# the probe and is recorded as a hint, never as a measurement. (h, w).
MODEL_INPUT_HW_HINTS = {"paligemma2_3b_mix": (448, 448)}
QWEN_KEY = "qwen2_5_vl_3b"
INTERNVL35_FALLBACK_KEY = "internvl3_5_4b_hf"
PROBE_PACKAGES = (
    "numpy", "pandas", "av", "cv2", "PIL", "torch", "torchvision", "transformers", "tokenizers",
    "huggingface_hub", "safetensors", "accelerate", "qwen_vl_utils", "sentencepiece", "timm", "einops",
)
# Plumbing every candidate needs (decode + ArUco homography + scoring live in colab_sweep's imports).
GLOBAL_REQUIRED = ("numpy", "pandas", "av", "cv2")


# ------------------------------------------------------------------------------------------
# Environment probe (never imports the engine; tolerates missing packages)
# ------------------------------------------------------------------------------------------

def _try_import(name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # ImportError, or an ABI failure raised as AttributeError/ImportError
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def _major(version: Optional[str]) -> Optional[int]:
    if not version:
        return None
    head = "".join(ch for ch in version.split(".")[0] if ch.isdigit())
    return int(head) if head else None


def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            return fh.readline().decode("utf-8", "replace").strip("\x00\n ") or None
    except OSError:
        return None


def probe_environment() -> Dict[str, Any]:
    """What is actually importable in THIS interpreter, plus Jetson/GPU facts. Read-only."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix), "packages": {},
    }
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    if os.path.exists(cfg):
        try:
            with open(cfg) as fh:
                for line in fh:
                    if line.lower().startswith("include-system-site-packages"):
                        info["include_system_site_packages"] = line.split("=", 1)[1].strip()
        except OSError:
            pass
    mods: Dict[str, Any] = {}
    for name in PROBE_PACKAGES:
        mod, err = _try_import(name)
        mods[name] = mod
        info["packages"][name] = {"version": (getattr(mod, "__version__", "present") if mod else None), "error": err}
    cv2 = mods.get("cv2")
    if cv2 is not None:
        info["cv2_aruco"] = bool(hasattr(cv2, "aruco") and (
            hasattr(cv2.aruco, "ArucoDetector") or hasattr(cv2.aruco, "detectMarkers")))
    torch = mods.get("torch")
    if torch is not None:
        info["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    tr = mods.get("transformers")
    if tr is not None:
        info["transformers_has_qwen2_5_vl"] = hasattr(tr, "Qwen2_5_VLForConditionalGeneration")
        info["transformers_has_paligemma"] = hasattr(tr, "PaliGemmaForConditionalGeneration")
    info["jetson"] = {
        "model": _read_first_line("/proc/device-tree/model"),
        "l4t_release": _read_first_line("/etc/nv_tegra_release"),
    }
    if shutil.which("nvpmodel"):
        try:  # may need sudo on some images; a failure is just "unknown", never an error
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
            info["jetson"]["nvpmodel"] = (out.stdout or out.stderr).strip().replace("\n", " | ")[:200]
        except Exception:
            info["jetson"]["nvpmodel"] = "unavailable"
    return info


def compact_env(info: Dict[str, Any]) -> Dict[str, Any]:
    """The subset recorded next to each candidate's results (who produced these numbers)."""
    pk = info["packages"]
    keep = {k: pk[k]["version"] for k in ("torch", "torchvision", "transformers", "tokenizers", "huggingface_hub",
                                          "accelerate", "numpy", "cv2", "av", "pandas") if k in pk}
    keep.update(python=info["python"], prefix=info["prefix"], cuda_available=info.get("cuda_available"),
                gpu=info.get("gpu"), gpu_capability=info.get("gpu_capability"), jetson=info.get("jetson"))
    return keep


def global_blockers(info: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for name in GLOBAL_REQUIRED:
        p = info["packages"][name]
        if p["version"] is None:
            out.append(f"{name} not importable ({p['error']})")
    if info["packages"]["cv2"]["version"] is not None and not info.get("cv2_aruco"):
        out.append("cv2 has no aruco module (need apt python3-opencv or pip opencv-contrib-python-headless)")
    return out


def candidate_issues(backend: str, info: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Static, repo-documented reasons a candidate cannot / may not run in this environment.
    ("BLOCKED", msg) -> the driver skips the candidate (before any multi-GB download); ("WARN", msg)
    -> runs, but a load failure is likely to be this. Nothing here is guessed at runtime: every rule
    cites a repo doc (ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md S8.4) or is a plain import check."""
    pk, issues = info["packages"], []

    def have(name: str) -> bool:
        return pk[name]["version"] is not None

    if not have("torch"):
        return [("BLOCKED", "torch not importable")]
    if backend in ("qwen2_5_vl_3b_instruct", "paligemma2_3b_mix", "moondream2", "internvl2_5_4b", "internvl3_5_4b_hf") \
            and not have("accelerate"):
        issues.append(("BLOCKED", "accelerate missing (from_pretrained(device_map=...) needs it)"))
    if not have("transformers"):
        return issues + [("BLOCKED", "transformers not importable")]
    if backend == "qwen2_5_vl_3b_instruct":
        if not info.get("transformers_has_qwen2_5_vl"):
            issues.append(("BLOCKED", "installed transformers has no Qwen2_5_VLForConditionalGeneration"))
        if not have("qwen_vl_utils"):
            issues.append(("BLOCKED", "qwen_vl_utils missing"))
    elif backend == "moondream2":
        # Verified 2026-09-22 by reading the pinned revision's files (9a7d402...): vision_encoder.py
        # imports einops.rearrange and torchvision.transforms.v2 at module top level; transformers'
        # remote-code loader refuses to load a repo whose top-level imports are missing.
        for req in ("einops", "torchvision"):
            if not have(req):
                issues.append(("BLOCKED", f"{req} missing (Moondream2's remote code imports it unconditionally)"))
        if (_major(pk["transformers"]["version"]) or 0) >= 5:
            issues.append(("BLOCKED", f"transformers {pk['transformers']['version']} >= 5: Moondream2's remote code "
                                      "(revision 2025-06-21) is reported broken there (all_tied_weights_keys); "
                                      "needs transformers<5 -- run it from the transformers<5 environment"))
    elif backend == "internvl2_5_4b":
        if not have("torchvision"):
            issues.append(("BLOCKED", "torchvision missing (InternVL dynamic-tiling preprocessing imports it)"))
        for req in ("timm", "einops"):
            if not have(req):  # verified 2026-09-22 from OpenGVLab/InternVL2_5-4B's modeling files
                issues.append(("BLOCKED", f"{req} missing (InternVL2.5's modeling files import "
                                          f"{'timm.models.layers.DropPath' if req == 'timm' else 'einops.rearrange'} unconditionally)"))
    elif backend == "paligemma2_3b_mix":
        if not info.get("transformers_has_paligemma"):
            issues.append(("BLOCKED", "installed transformers has no PaliGemmaForConditionalGeneration"))
        if not have("sentencepiece"):
            issues.append(("WARN", "sentencepiece not installed (may be needed by the Gemma tokenizer; precautionary)"))
    elif backend == "internvl3_5_4b_hf":
        if not have("torchvision"):
            issues.append(("BLOCKED", "torchvision missing"))
    return issues


def print_probe(info: Dict[str, Any]) -> None:
    print(f"python {info['python']} at {info['executable']}")
    print(f"  prefix={info['prefix']}  venv={info['in_venv']}  "
          f"include-system-site-packages={info.get('include_system_site_packages', 'n/a')}")
    for name in PROBE_PACKAGES:
        p = info["packages"][name]
        print(f"  {name:<16} {p['version'] if p['version'] else 'MISSING'}"
              + (f"   [{p['error']}]" if p["error"] and p["version"] is None else ""))
    if "cv2_aruco" in info:
        print(f"  cv2.aruco        {info['cv2_aruco']}")
    print(f"  cuda_available={info.get('cuda_available')} torch_cuda_build={info.get('torch_cuda_build')} "
          f"gpu={info.get('gpu')} capability={info.get('gpu_capability')} mem_gb={info.get('gpu_mem_gb')}")
    print(f"  jetson={info['jetson']}")


# ------------------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bronze-dir", default=DEFAULT_BRONZE_DIR)
    ap.add_argument("--sessions", nargs="*", default=None,
                     help="Explicit session directory names under --bronze-dir (disables the "
                          "reference-locked session set and the parity assertion).")
    ap.add_argument("--all-matching-sessions", action="store_true",
                     help="Glob --session-pattern instead of using the reference run's 10 sessions. "
                          "The validation gate is then not applicable.")
    ap.add_argument("--session-pattern", default="session_jetson_track4_*")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                     help="Local directory for checkpoints/smoke files/exports. Shared by separate "
                          "invocations from different venvs when given the same --run-label.")
    ap.add_argument("--run-label", default="jetson_run1",
                     help="Same label across invocations = resume/extend that checkpoint; a new "
                          "label starts a fresh run.")
    ap.add_argument("--reference-json", default=DEFAULT_REFERENCE_JSON)
    ap.add_argument("--frames-per-session", type=int, default=6,
                     help="6 = identical frames to the local reference run (required for the gate).")
    ap.add_argument("--tolerance-mm", type=float, default=20.0)
    ap.add_argument("--max-new-tokens", type=int, default=24,
                     help="24 = matches the local reference run (required for the gate).")
    ap.add_argument("--dtype", default="auto",
                     help="auto = bf16 on the Orin's Ampere GPU (compute capability 8.7); see "
                          "vlm_backends.resolve_torch_dtype.")
    ap.add_argument("--qwen-model-dir", default=None,
                     help="Local Qwen2.5-VL-3B checkpoint dir. Default: models/qwen2_5_vl_3b_instruct "
                          "if it exists on this checkout, else the HF repo id at the pinned commit. Use "
                          "the SAME value in every invocation of a run (it is part of the checkpoint "
                          "identity).")
    ap.add_argument("--smoke-frames", type=int, default=2)
    ap.add_argument("--prompt-variants", nargs="*", default=["baseline", "oriented", "oriented_aruco"],
                     choices=list(PROMPT_VARIANTS.keys()))
    ap.add_argument("--candidates", nargs="*", default=None,
                     help="Restrict to these CandidateSpec keys. Default: the 6 standard specs. The "
                          f"optional fallback '{INTERNVL35_FALLBACK_KEY}' (InternVL3.5-4B via native "
                          "transformers -- a DIFFERENT generation, not InternVL2.5) is only available "
                          "when named here.")
    ap.add_argument("--use-mock", action="store_true",
                     help="Mock backends (no GPU/weights/downloads) to dry-run the plumbing. Skips "
                          "the environment candidate checks and the validation gate.")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-validation-gate", action="store_true",
                     help="Bypass the gate knowingly (both running it and requiring a recorded pass "
                          "before a sweep). Different from --force-continue.")
    ap.add_argument("--force-continue", action="store_true",
                     help="Run the gate but proceed even if it FAILED (recorded as forced). Leave off: "
                          "a failed gate means something is wrong on this machine.")
    ap.add_argument("--calibrate-variants", nargs="*", default=["baseline"], choices=list(PROMPT_VARIANTS.keys()),
                    help="Prompt variants shown to each prompt-taking candidate in --stage calibrate "
                         "(native-API specs always run once). Default: baseline only -- the oriented_aruco "
                         "text describes ArUco markers a synthetic frame does not contain. A refusal from "
                         "any variant that IS probed counts.")
    ap.add_argument("--recalibrate", action="store_true",
                    help="Re-run calibration even where a non-refusing record for the same candidate "
                         "config already exists (refusing/failed records are always re-run).")
    ap.add_argument("--allow-coord-mismatch", nargs="+", default=None, metavar="KEY",
                    help="Knowingly sweep these candidates although their calibration refuses them (or is "
                         "missing). The override is recorded in the checkpoint. Prefer fixing the backend's "
                         "declared coord_space (the calibration output names the field).")
    ap.add_argument("--skip-coord-calibration", action="store_true",
                    help="Bypass calibration entirely (neither run in --stage all nor required before a sweep).")
    ap.add_argument("--stage", choices=STAGES, default="all")
    return ap


# ------------------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------------------

def _load_engine() -> None:
    global cs
    if cs is None:
        from ml_jetson_vla.deployment import colab_sweep as _engine

        cs = _engine


def reference_sessions(path: str) -> List[str]:
    with open(path) as fh:
        ref = json.load(fh)
    return [s["session"] for s in ref["sessions"]]


def resolve_session_lock(args: argparse.Namespace) -> Optional[List[str]]:
    """The reference run's session list when this run is meant to be comparable to it."""
    if args.sessions or args.all_matching_sessions or args.frames_per_session != 6:
        return None
    if not os.path.exists(args.reference_json):
        print(f"WARNING: reference JSON {args.reference_json} not found -- cannot lock the session set "
              f"or run the validation gate.")
        return None
    return reference_sessions(args.reference_json)


def stage_frames(args: argparse.Namespace) -> List[Any]:
    lock = resolve_session_lock(args)
    sessions = args.sessions or lock
    banner = f"reference-locked, {len(lock)} sessions" if lock else f"sessions={args.sessions or args.session_pattern}"
    print(f"\n=== frames: {banner} ===")
    frames, skipped = cs.prepare_frames(
        args.bronze_dir, frames_per_session=args.frames_per_session,
        sessions=sessions, session_pattern=args.session_pattern,
    )
    print(f"{len(frames)} frames prepared" + (f", {len(skipped)} skipped: {skipped}" if skipped else ""))
    if not frames:
        raise SystemExit(f"No frames prepared from {args.bronze_dir} -- check --bronze-dir / --sessions.")
    if lock:
        par = cs.sampling_parity(frames, args.reference_json)
        print("sampling parity with the local reference run:", par)
        if not par["identical"]:
            raise SystemExit(
                "The sampled (session, frame_index) set differs from the local reference run (missing "
                "session data, or different telemetry/video?). Results would not be comparable and the "
                "validation gate would be meaningless. Fix the data, or pass --all-matching-sessions / "
                "--sessions to run knowingly without the gate."
            )
    return frames


def build_specs(args: argparse.Namespace) -> List[Any]:
    base = (cs.mock_candidate_specs() if args.use_mock
            else cs.default_candidate_specs(max_new_tokens=args.max_new_tokens, dtype=args.dtype))
    optional: List[Any] = []
    if not args.use_mock:
        optional.append(cs.CandidateSpec(
            INTERNVL35_FALLBACK_KEY, "internvl3_5_4b_hf", dict(max_new_tokens=args.max_new_tokens, dtype=args.dtype),
            note="OPTIONAL fallback: InternVL3.5-4B (a later generation than InternVL2.5) via native transformers"))
        qwen_dir = args.qwen_model_dir
        if qwen_dir is None:
            # Same path as qwen_vl_smoke_test.DEFAULT_MODEL_DIR, recomputed here rather than imported:
            # that module imports torch/transformers/qwen_vl_utils at load time, which an environment
            # that does not run Qwen (the transformers<5 one) need not have.
            default_dir = os.path.join(_ML_JETSON_VLA_DIR, "models", "qwen2_5_vl_3b_instruct")
            qwen_dir = default_dir if os.path.exists(default_dir) else None
        for s in base:
            if s.key == QWEN_KEY and qwen_dir:
                s.kwargs = dict(s.kwargs, model_dir=qwen_dir)  # same weights as the pinned commit; no re-download
                s.note += f"; using LOCAL checkpoint {qwen_dir}"
    universe = {s.key: s for s in base + optional}
    if args.candidates:
        unknown = [k for k in args.candidates if k not in universe]
        if unknown:
            raise SystemExit(f"unknown --candidates {unknown}; known: {sorted(universe)}")
        return [universe[k] for k in args.candidates if k in universe]
    return list(base)


def preflight(args: argparse.Namespace, info: Dict[str, Any], specs: Sequence[Any]) -> List[Any]:
    """Global blockers stop the run; per-candidate BLOCKED issues drop that candidate LOUDLY (before
    any download) so one environment-incompatible candidate does not waste a multi-GB fetch."""
    if args.use_mock:
        return list(specs)
    keep: List[Any] = []
    for s in specs:
        issues = candidate_issues(s.backend, info)
        for level, msg in issues:
            print(f"  [{level}] {s.key}: {msg}")
        if any(level == "BLOCKED" for level, _ in issues):
            print(f"  -> SKIPPING {s.key} in this environment.")
        else:
            keep.append(s)
    if not keep:
        raise SystemExit("No candidate can run in this environment (see BLOCKED lines above). Run the "
                         "missing candidates from the environment the doc assigns them to.")
    return keep


def stage_hf_auth(args: argparse.Namespace, specs: Sequence[Any]) -> Optional[str]:
    token, how = cs.get_hf_token()
    gated = sorted({s.hf_gated_repo for s in specs if s.hf_gated_repo})
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print(f"HF auth via {how}")
    elif gated:
        print(f"No HF_TOKEN env var ({how}); relying on a cached `login()` if one exists.")
    if gated and not args.use_mock:
        for repo in gated:
            ok, msg = cs.check_hf_access(repo, token)
            print(("GATED ACCESS OK: " if ok else f"GATED ACCESS UNAVAILABLE -> {repo} candidates will be "
                                                    "skipped.\n") + msg)
    return token


def _store(args: argparse.Namespace, prefix: str, with_meta: bool = True) -> Any:
    path = os.path.join(args.results_dir, f"{prefix}_{args.run_label}.json")
    return cs.CheckpointStore(path, meta=cs.collect_env_meta() if with_meta else None)


def _record_env(store: Any, status: Dict[str, str], env: Dict[str, Any]) -> None:
    """Attach the producing environment to each candidate that actually loaded in THIS process."""
    for key, st in status.items():
        if st not in ("nothing_to_do", "gated_skipped"):
            store.set_stage(key, env=env)


def stage_smoke(args: argparse.Namespace, specs: Sequence[Any], frames: Sequence[Any],
                hf_token: Optional[str], env: Dict[str, Any]) -> List[str]:
    store = _store(args, "smoke")
    smoke_frames = cs.pick_smoke_frames(frames, args.smoke_frames)
    print(f"\n=== smoke: {len(smoke_frames)} frame(s) x {len(specs)} candidate(s) ===")
    status = cs.run_sweep(specs, smoke_frames, args.prompt_variants, store, tolerance_mm=args.tolerance_mm,
                          max_new_tokens=args.max_new_tokens, hf_token=hf_token)
    _record_env(store, status, env)
    print(cs.smoke_report(store, specs, smoke_frames, args.prompt_variants, args.max_new_tokens).to_string())
    seen: set = set()
    for e in store.errors:
        if e["candidate"] not in seen:
            seen.add(e["candidate"])
            print(f"  FIRST ERROR [{e['candidate']}] {e['stage']}: {e['error_type']}: {e['error'][:300]}")
    runnable = [s.key for s in specs if status.get(s.key) in ("complete", "nothing_to_do")
                and any(e.get("parse_ok") for e in store.items.values() if e["candidate"] == s.key)]
    print(f"\nwill run in the full sweep: {runnable}\nskipped (failed/unparseable/gated): "
          f"{[s.key for s in specs if s.key not in runnable]}")
    return runnable


def spec_config_hash(spec: Any) -> str:
    """Identity of a candidate's configuration for calibration records (same ingredients as the sweep's
    item keys: constructor kwargs minus mock fault-injection switches, and the effective dtype)."""
    kwargs = {k: v for k, v in spec.kwargs.items() if k not in cs._NON_IDENTITY_KWARGS}
    return cs._sha({"backend": spec.backend, "kwargs": kwargs, "dtype": cs.effective_dtype_name(spec)})


def _calibrate_one(args: argparse.Namespace, store: Any, spec: Any, chash: str, hf_token: Optional[str],
                   probe_path: str, env: Dict[str, Any]) -> Dict[str, Any]:
    """Loads one candidate, runs the probe for each of its variants, returns its checkpoint record."""
    base: Dict[str, Any] = {"config_hash": chash, "backend": spec.backend, "env": env,
                            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if spec.hf_gated_repo:
        ok, msg = cs.check_hf_access(spec.hf_gated_repo, hf_token)
        if not ok:
            print(f"  SKIPPED (gated / no access): {msg[:300]}")
            return {**base, "verdict": "GATED_SKIPPED", "detail": msg[:300]}
    backend = None
    analyses: Dict[str, Dict[str, Any]] = {}
    raw: Dict[str, Any] = {}
    try:
        try:
            backend = cs.build_backend(spec, args.max_new_tokens)
            backend.load()
        except Exception as exc:
            print(f"  LOAD FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            store.record_error(spec.key, "calibrate_load", exc)
            return {**base, "verdict": "LOAD_FAILED", "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}
        for variant in cs.spec_variants(spec, args.calibrate_variants):
            obs = csp.run_probe_variant(backend, variant)
            analysis = csp.analyze(obs, (csp.FRAME_H, csp.FRAME_W), MODEL_INPUT_HW_HINTS.get(spec.backend))
            analyses[variant] = analysis
            raw[variant] = {"observations": obs, "analysis": analysis}
            print(csp.format_report(spec.key, variant, analysis))
            csp.save_candidate_record(probe_path, spec.key, {"config_hash": chash, "backend": spec.backend,
                                                             "variants": raw})
    finally:
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                pass
        backend = None
        cs.free_gpu_memory()
    refusing = [csp.recommendation(spec.key, spec.backend, v, a) for v, a in analyses.items()
                if a["verdict"] in csp.REFUSING_VERDICTS]
    return {**base, "verdict": csp.worst_verdict([a["verdict"] for a in analyses.values()]),
            "variants": csp.summarize_for_checkpoint(analyses), "recommendations": refusing}


def stage_calibrate(args: argparse.Namespace, specs: Sequence[Any], hf_token: Optional[str],
                    env: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Coordinate-space calibration for every spec, recorded in the checkpoint. Never raises for a bad
    verdict (enforcement happens at sweep time, see `enforce_calibration`)."""
    if args.skip_coord_calibration:
        print("\n=== coordinate-space calibration: skipped (--skip-coord-calibration) ===")
        return {}
    problems = csp.validate_positions(csp.PROBE_POSITIONS)
    if problems:
        raise SystemExit(f"probe position set is badly designed: {problems}")
    print(f"\n=== coordinate-space calibration: {len(csp.PROBE_POSITIONS)} synthetic frames "
          f"({csp.FRAME_W}x{csp.FRAME_H}, {csp.DISC_DIAMETER_PX}px red disc) x {len(specs)} candidate(s) ===")
    store = _store(args, "checkpoint")
    probe_path = os.path.join(args.results_dir, f"coord_probe_{args.run_label}.json")
    csp.write_probe_frames(os.path.join(args.results_dir, "coord_probe_frames"))
    records: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        chash = spec_config_hash(spec)
        prior = store.stages.get(CAL_STAGE_KEY, {}).get(spec.key)
        print(f"\n--- {spec.key}  [{spec.backend}] ---")
        if (prior and prior.get("config_hash") == chash and not args.recalibrate
                and prior.get("verdict") in (csp.VERDICT_CONSISTENT, csp.VERDICT_NO_FIT, csp.VERDICT_INSUFFICIENT)):
            print(f"  cached calibration record from {prior.get('checked_at')}: {prior['verdict']} "
                  f"(--recalibrate to redo)")
            records[spec.key] = prior
            continue
        rec = _calibrate_one(args, store, spec, chash, hf_token, probe_path, env)
        store.set_stage(CAL_STAGE_KEY, **{spec.key: rec})
        records[spec.key] = rec
    print("\ncalibration summary:")
    for spec in specs:
        rec = records.get(spec.key)
        if rec is None:
            continue
        print(f"  {spec.key:<28} {rec['verdict']}")
        for msg in rec.get("recommendations", []):
            print("    " + msg.replace("\n", "\n    "))
    print(f"raw answers: {probe_path}   synthetic frames: {os.path.join(args.results_dir, 'coord_probe_frames')}")
    return records


def enforce_calibration(args: argparse.Namespace, store: Any, specs: Sequence[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Splits `specs` into (allowed, refused). A candidate is refused when its calibration record (same
    config hash) has a MISMATCH verdict, or when no record exists (never calibrated) -- except under
    --use-mock, where a missing record is tolerated so the plumbing tests need not calibrate first.
    `--allow-coord-mismatch KEY` overrides for a named candidate; `--skip-coord-calibration` bypasses
    everything. Every decision is written into the candidate's stage record."""
    if args.skip_coord_calibration:
        for s in specs:
            store.set_stage(s.key, coord_calibration={"skipped": True})
        return list(specs), []
    allowed_keys = set(args.allow_coord_mismatch or [])
    unknown = sorted(allowed_keys - {s.key for s in specs})
    if unknown:
        print(f"  note: --allow-coord-mismatch names candidates not in this invocation: {unknown}")
    recs = store.stages.get(CAL_STAGE_KEY, {})
    keep: List[Any] = []
    refused: List[Dict[str, Any]] = []
    print("\n=== coordinate-space calibration check ===")
    for s in specs:
        rec = recs.get(s.key)
        fresh = rec is not None and rec.get("config_hash") == spec_config_hash(s)
        reason: Optional[str] = None
        verdict = rec["verdict"] if fresh else None
        if not fresh:
            if not args.use_mock:
                reason = ("no calibration record for this candidate/config" if rec is None else
                          "calibration record is for a different candidate configuration (stale)")
                reason += f" -- run `--stage calibrate --candidates {s.key}` first"
        elif verdict in csp.REFUSING_VERDICTS:
            reason = "\n      ".join(rec.get("recommendations") or [f"calibration verdict {verdict}"])
        if reason is None:
            print(f"  ok       {s.key:<28} {verdict or '(not calibrated; --use-mock)'}")
            store.set_stage(s.key, coord_calibration={"verdict": verdict, "override": False})
            keep.append(s)
        elif s.key in allowed_keys:
            print(f"  OVERRIDE {s.key:<28} {verdict or 'no record'} -- sweeping anyway (--allow-coord-mismatch); "
                  f"its numbers may measure a mapping error:\n      {reason}")
            store.set_stage(s.key, coord_calibration={"verdict": verdict, "override": True, "reason": reason[:600]})
            keep.append(s)
        else:
            print(f"  REFUSED  {s.key:<28} {verdict or 'no record'}\n      {reason}\n      "
                  f"(knowing override: --allow-coord-mismatch {s.key})")
            store.set_stage(s.key, status="calibration_refused", detail=reason[:600])
            refused.append({"key": s.key, "verdict": verdict, "reason": reason})
    return keep, refused


def gate_applicable(args: argparse.Namespace, specs: Sequence[Any]) -> Tuple[bool, str]:
    if not any(s.key == QWEN_KEY for s in specs):
        return False, f"{QWEN_KEY} is not in this invocation's candidates"
    if args.frames_per_session != 6 or args.max_new_tokens != 24 or not set(GATE_REQUIRED_VARIANTS) <= set(args.prompt_variants):
        return False, "config differs from the reference run (needs --frames-per-session 6, --max-new-tokens 24, baseline+oriented)"
    if args.sessions or args.all_matching_sessions:
        return False, "session set differs from the reference run's"
    return True, ""


def print_gate_hits(args: argparse.Namespace, store: Any, qwen_spec: Any, frames: Sequence[Any]) -> None:
    """Human-readable headline next to the gate's pass/fail: hits and mean error on this machine vs the
    local CPU reference (both read from data -- the reference numbers are not hardcoded here; they are
    40/60 hits, 27.9 mm for baseline and 21/60, 43.7 mm for oriented per the multi-candidate doc S8.2)."""
    with open(args.reference_json) as fh:
        ref = json.load(fh)
    for v in GATE_REQUIRED_VARIANTS:
        ref_f = [f for s in ref["sessions"] for f in s["variants"][v]["frames"] if f.get("parse_ok")]
        mine = [e for fr in frames for e in [store.items.get(cs.item_key(qwen_spec, v, fr, args.max_new_tokens))]
                if e is not None and e.get("parse_ok")]
        def _fmt(rows: List[Dict[str, Any]]) -> str:
            errs = [float(r["error_mm"]) for r in rows]
            return (f"{sum(1 for x in errs if x <= args.tolerance_mm)}/{len(rows)} hits, mean "
                    f"{sum(errs) / len(errs):.1f} mm" if errs else "n/a")
        print(f"  [{v}] this machine: {_fmt(mine)}  |  local CPU reference: {_fmt(ref_f)}")


def stage_validate(args: argparse.Namespace, specs: Sequence[Any], frames: Sequence[Any],
                   hf_token: Optional[str], env: Dict[str, Any]) -> None:
    if args.skip_validation_gate or args.use_mock:
        print(f"\n=== validation gate: skipped ({'--skip-validation-gate' if args.skip_validation_gate else '--use-mock'}) ===")
        return
    ok, why = gate_applicable(args, specs)
    if not ok:
        print(f"\n=== validation gate: not applicable here ({why}) -- not run in this invocation ===")
        return
    qwen_spec = next(s for s in specs if s.key == QWEN_KEY)
    print("\n=== validation gate ===")
    store = _store(args, "checkpoint")
    st = cs.run_sweep([qwen_spec], frames, list(GATE_REQUIRED_VARIANTS), store, tolerance_mm=args.tolerance_mm,
                      max_new_tokens=args.max_new_tokens, hf_token=hf_token)
    _record_env(store, st, env)
    if st.get(QWEN_KEY) not in ("complete", "nothing_to_do"):
        raise RuntimeError(f"Qwen validation run did not complete ({st}); errors: "
                           f"{[e['error'][:200] for e in store.errors[-3:]]}")
    report = cs.validate_against_reference(store, qwen_spec, frames, args.reference_json,
                                           max_new_tokens=args.max_new_tokens)
    cs.print_validation_report(report)
    print_gate_hits(args, store, qwen_spec, frames)
    forced = bool(not report["passed"] and args.force_continue)
    store.set_stage(GATE_STAGE_KEY, passed=bool(report["passed"]), forced=forced,
                    checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"), env=env,
                    variants=report["variants"], failed_checks=[c["check"] for c in report["checks"] if not c["ok"]])
    if not report["passed"] and not args.force_continue:
        raise cs.PipelineValidationError(
            "The Jetson's Qwen output does not match the local CPU reference closely enough. Do NOT trust "
            "a long run on this config. Suspects, in order: (1) dtype (try --dtype bf16 explicitly); "
            "(2) transformers version / processor resize (see the model-input-size check above); "
            "(3) frame decoding. Pass --force-continue only to proceed knowingly."
        )


def require_gate_record(args: argparse.Namespace, store: Any) -> None:
    """Refuse a long sweep unless this run label has a passing (or knowingly forced) gate record."""
    if args.use_mock or args.skip_validation_gate:
        return
    rec = store.stages.get(GATE_STAGE_KEY)
    if rec and (rec.get("passed") or rec.get("forced")):
        print(f"validation gate record: passed={rec.get('passed')} forced={rec.get('forced')} at {rec.get('checked_at')}")
        return
    raise SystemExit(
        f"No passing validation-gate record for run label {args.run_label!r} in "
        f"{os.path.join(args.results_dir, 'checkpoint_' + args.run_label + '.json')}. Run the gate first "
        f"(from the environment that has Qwen: `--candidates {QWEN_KEY} --stage validate`), or pass "
        f"--skip-validation-gate to proceed knowingly without it."
    )


def stage_sweep(args: argparse.Namespace, specs: Sequence[Any], frames: Sequence[Any],
                hf_token: Optional[str], env: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
    """Returns `(store, refused)`; `refused` = candidates dropped by the calibration check."""
    store = _store(args, "checkpoint")
    require_gate_record(args, store)
    specs, refused = enforce_calibration(args, store, specs)
    if not specs:
        print("\nNo candidate passed the calibration check; nothing swept.")
        return store, refused
    print(f"\n=== full sweep: {[s.key for s in specs]} ===")
    t0 = time.time()
    status = cs.run_sweep(specs, frames, args.prompt_variants, store, tolerance_mm=args.tolerance_mm,
                          max_new_tokens=args.max_new_tokens, hf_token=hf_token)
    _record_env(store, status, env)
    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min: {status}")
    return store, refused


def stage_export(args: argparse.Namespace, specs: Sequence[Any], frames: Sequence[Any]) -> str:
    store = _store(args, "checkpoint", with_meta=False)
    table = cs.aggregate(store, specs, frames, args.prompt_variants, args.tolerance_mm, args.max_new_tokens)
    print("\n=== headline table (candidate x variant) ===")
    print(table.to_string())
    coord = cs.aggregate_coord_space(store, specs, frames, args.prompt_variants, args.tolerance_mm, args.max_new_tokens)
    if len(coord):
        print("\n=== coordinate-space diagnostic ===")
        print(coord.to_string())
    path = cs.export_all(store, specs, frames, args.prompt_variants, args.results_dir, args.tolerance_mm,
                         args.max_new_tokens, run_label=args.run_label)
    print(f"\ncombined results written to: {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.results_dir, exist_ok=True)

    info = probe_environment()
    print_probe(info)
    blockers = global_blockers(info)
    for b in blockers:
        print(f"  [BLOCKED-ALL] {b}")

    if args.stage == "probe":
        if blockers:
            print("\nPROBE: shared plumbing incomplete -- fix the [BLOCKED-ALL] items before running anything.")
            return 1
        _load_engine()
        print("\nPer-candidate feasibility in THIS environment:")
        for s in build_specs(args):
            issues = candidate_issues(s.backend, info) if not args.use_mock else []
            tag = "BLOCKED" if any(lv == "BLOCKED" for lv, _ in issues) else ("ok (with warnings)" if issues else "ok")
            print(f"  {s.key:<28} {tag}")
            for level, msg in issues:
                print(f"      [{level}] {msg}")
        return 0

    if blockers and not args.use_mock:
        raise SystemExit("Shared plumbing missing in this environment (see [BLOCKED-ALL]); nothing was run.")
    _load_engine()
    env = compact_env(info)
    specs = build_specs(args)
    print(f"candidates requested this invocation: {[s.key for s in specs]}")

    runs_models = args.stage in ("smoke", "calibrate", "validate", "sweep", "all")
    if runs_models:
        specs = preflight(args, info, specs)
        print(f"candidates that can run here: {[s.key for s in specs]}")
        # Fail fast, before any download/model load: a sweep needs a passing gate record, and this
        # invocation will only produce one itself for `--stage all` with Qwen present.
        gate_runs_here = (args.stage == "all" and not args.skip_validation_gate and not args.use_mock
                          and gate_applicable(args, specs)[0])
        if args.stage in ("sweep", "all") and not gate_runs_here:
            require_gate_record(args, _store(args, "checkpoint", with_meta=False))

    frames: List[Any] = [] if args.stage in ("hfauth", "calibrate") else stage_frames(args)
    hf_token = stage_hf_auth(args, specs) if (runs_models or args.stage == "hfauth") else None

    if args.stage in ("smoke", "all") and not args.skip_smoke:
        stage_smoke(args, specs, frames, hf_token, env)
    if args.stage in ("calibrate", "all"):
        stage_calibrate(args, specs, hf_token, env)
    if args.stage in ("validate", "all"):
        stage_validate(args, specs, frames, hf_token, env)
    refused: List[Dict[str, Any]] = []
    if args.stage in ("sweep", "all"):
        _store_unused, refused = stage_sweep(args, specs, frames, hf_token, env)
    if args.stage in ("export", "all"):
        stage_export(args, specs, frames)
    if refused:
        print(f"\nEXIT 2: {len(refused)} candidate(s) were refused by the coordinate-space calibration check and NOT "
              f"swept: {[r['key'] for r in refused]}. Fix the declared coord_space (see the evidence above), re-run "
              f"`--stage calibrate`, then the sweep -- or override knowingly with --allow-coord-mismatch.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
