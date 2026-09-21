"""Sweep engine for the Arm 2 minimal-baseline Colab notebook (`arm2_colab_sweep.ipynb`):
candidates x prompt variants x fixed frame set, scored offline against real Track 4 sessions.

The notebook is a thin driver over THIS module (config cells + calls), so an experiment (change a
candidate, a prompt, the frame count, the tolerance) never touches the plumbing, and the same
plumbing is what `deployment/test_colab_sweep_mock.py` dry-runs locally with mock backends.

It reuses -- imports, does not copy -- the repo's real logic: `core/minimal_vlm_policy.py` (prompt
variants, parser, policy), `core/vlm_backends.py` (the candidates), `deployment/
score_minimal_baseline_offline.py` (frame sampling, PyAV decoding, corrected pixel->mm scoring,
per-frame result schema, summaries, atomic writes), and the read-only ArUco homography helper in
`ml_vision/data_processing/auto_label_shared_vision.py`.

## Checkpointing (CLAUDE.md: >30 min processes must persist progress)
Every single `generate()` result is written to a JSON checkpoint (atomic temp-file +
`os.replace`) before the next call starts; on re-run, items whose key is already present are
skipped, and a candidate with nothing left to do is skipped WITHOUT loading its model. The key is
`candidate | variant | prompt-text hash | run-config hash | session | frame_index`, so editing a
prompt template or changing dtype/max_new_tokens/etc. automatically invalidates the stale items
instead of silently reusing them. Failed calls are never stored as done (a re-run retries them).

## Fault isolation
Load, gate-check and each call are wrapped in `except Exception`; the full traceback is recorded
against that candidate and the sweep continues. `BaseException`s (KeyboardInterrupt / a Colab
stop) are deliberately NOT swallowed -- they propagate after the `finally` frees GPU memory, and
the checkpoint already holds everything finished. After `MAX_CONSECUTIVE_CALL_ERRORS` consecutive
failing calls a candidate is aborted (recorded) rather than burning the run on a broken backend.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.minimal_vlm_policy import MinimalVLMPolicy, PROMPT_VARIANTS  # noqa: E402
from ml_jetson_vla.core.vlm_backends import BACKEND_REGISTRY, free_gpu_memory, resolve_torch_dtype  # noqa: E402
from ml_jetson_vla.deployment.score_minimal_baseline_offline import (  # noqa: E402
    GROUND_TRUTH_MANIFEST,
    atomic_write_json,
    compute_transition_flags,
    read_frames_at_indices,
    sample_frame_indices,
    score_frame_with_policy,
    summarize_variant_frames,
    touch_frame_to_manifest_mm,
)
from host_software.ml_vision.data_processing.auto_label_shared_vision import (  # noqa: E402
    estimate_homography_from_aruco,
    load_manifest_full,
)

CHECKPOINT_FORMAT = "arm2_colab_sweep_checkpoint_v1"
MAX_CONSECUTIVE_CALL_ERRORS = 3
NATIVE_VARIANT_LABEL = "(native API - prompt ignored)"

# Pinned upstream weights for the Qwen candidate: the exact commit of the local checkpoint
# `models/qwen2_5_vl_3b_instruct` (from its .cache/huggingface/download/*.metadata), so the Colab
# numbers use the same weights as the local 2026-09-18 run.
QWEN_REPO_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_LOCAL_COMMIT = "66285546d2b821cf421d4f5eb2576359d3770cd3"


class PipelineValidationError(RuntimeError):
    """Raised by the notebook's validation gate when Colab's Qwen output does not match the local
    reference closely enough to trust a long run."""


# ------------------------------------------------------------------------------------------
# Frames
# ------------------------------------------------------------------------------------------

@dataclasses.dataclass
class FrameItem:
    session: str
    frame_index: int
    instruction: str
    true_x_tel: float           # telemetry target_x (center-origin frame)
    true_y_tel: float
    target_transition: bool
    frame_bgr: np.ndarray
    homography: np.ndarray      # mm(manifest) -> raw px, computed per frame from the ArUco markers

    @property
    def ident(self) -> str:
        return f"{self.session}|{self.frame_index}"


def prepare_frames(
    bronze_dir: str,
    frames_per_session: int = 6,
    sessions: Optional[Sequence[str]] = None,
    session_pattern: str = "session_jetson_track4_*",
    verbose: bool = True,
) -> Tuple[List[FrameItem], List[Dict[str, object]]]:
    """Same sampling as the local 2026-09-18 run (`sample_frame_indices`: color-command frames with
    ground truth, evenly subsampled to `frames_per_session`), decoded with the same PyAV routine.
    Returns `(frames, skipped)`; `skipped` lists any frame dropped for a decode/homography failure
    (never silently)."""
    aruco_markers, _feat, _w, _h = load_manifest_full(GROUND_TRUTH_MANIFEST)
    aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}
    if sessions:
        dirs = [os.path.join(bronze_dir, s) for s in sessions]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(bronze_dir, session_pattern))
                      if os.path.isdir(d) and not d.endswith(".dvc"))
    frames: List[FrameItem] = []
    skipped: List[Dict[str, object]] = []
    for sdir in dirs:
        name = os.path.basename(sdir)
        csv_path, vid_path = os.path.join(sdir, "telemetry.csv"), os.path.join(sdir, "rgb_video.mp4")
        if not (os.path.exists(csv_path) and os.path.exists(vid_path)):
            skipped.append({"session": name, "reason": "missing telemetry.csv or rgb_video.mp4"})
            continue
        df = pd.read_csv(csv_path)
        idxs = sample_frame_indices(df, frames_per_session)
        rows = [df.loc[i] for i in idxs]
        flags = compute_transition_flags(df)
        decoded = read_frames_at_indices(vid_path, {int(r["frame_index"]) for r in rows})
        n_ok = 0
        for r in rows:
            fi = int(r["frame_index"])
            img = decoded.get(fi)
            if img is None:
                skipped.append({"session": name, "frame_index": fi, "reason": "video_read_failed"})
                continue
            hom = estimate_homography_from_aruco(img, aruco_lookup)
            if hom is None:
                skipped.append({"session": name, "frame_index": fi, "reason": "no_aruco_homography"})
                continue
            frames.append(FrameItem(
                session=name, frame_index=fi, instruction=str(r["audio_command"]),
                true_x_tel=float(r["target_x"]), true_y_tel=float(r["target_y"]),
                target_transition=bool(flags.loc[int(r.name)]), frame_bgr=img, homography=hom,
            ))
            n_ok += 1
        if verbose:
            print(f"  {name}: {n_ok}/{len(rows)} frames ready")
    return frames, skipped


def sampling_parity(frames: Sequence[FrameItem], reference_path: str) -> Dict[str, object]:
    """Compares the sampled (session, frame_index) set with the local reference run's. Identical
    sets are what make Colab numbers directly comparable to the local ones."""
    with open(reference_path) as fh:
        ref = json.load(fh)
    ref_set = set()
    for s in ref["sessions"]:
        variants = s.get("variants", {})
        if variants:
            first = next(iter(variants.values()))
            ref_set |= {(s["session"], f["frame_index"]) for f in first["frames"]}
    mine = {(f.session, f.frame_index) for f in frames}
    return {"identical": mine == ref_set, "n_mine": len(mine), "n_reference": len(ref_set),
            "only_mine": sorted(mine - ref_set)[:10], "only_reference": sorted(ref_set - mine)[:10]}


def pick_smoke_frames(frames: Sequence[FrameItem], n: int) -> List[FrameItem]:
    """`n` frames spread across the set (different sessions where possible)."""
    if n >= len(frames):
        return list(frames)
    stride = max(1, len(frames) // n)
    return [frames[i * stride] for i in range(n)]


# ------------------------------------------------------------------------------------------
# Candidates / variants
# ------------------------------------------------------------------------------------------

@dataclasses.dataclass
class CandidateSpec:
    """One row of the candidate table. `key` is the id used everywhere (checkpoint keys, tables);
    `backend` is a `BACKEND_REGISTRY` key (or a mock); `kwargs` go to its constructor.
    `variants=None` -> use the run's prompt variants; a fixed list (native-API modes) overrides it.
    `prompt_independent` marks specs whose real API ignores the prompt text."""

    key: str
    backend: str
    kwargs: Dict[str, object] = dataclasses.field(default_factory=dict)
    variants: Optional[List[str]] = None
    prompt_independent: bool = False
    hf_gated_repo: Optional[str] = None
    note: str = ""


def default_candidate_specs(max_new_tokens: int = 24, dtype: str = "auto") -> List[CandidateSpec]:
    """The four requested candidates (+ the two native-API modes that make the prompt-variant
    comparison honest for PaliGemma2/Moondream2, see their backend docstrings). Order matters:
    Qwen first, because its result doubles as the pipeline-validation gate."""
    return [
        CandidateSpec("qwen2_5_vl_3b", "qwen2_5_vl_3b_instruct", dict(
            model_dir=QWEN_REPO_ID, revision=QWEN_LOCAL_COMMIT, max_new_tokens=max_new_tokens, dtype=dtype),
            note="reference candidate; weights pinned to the local checkpoint's commit"),
        CandidateSpec("internvl2_5_4b", "internvl2_5_4b", dict(
            max_new_tokens=max_new_tokens, dtype=dtype),
            note="trust_remote_code + model.chat(); never executed before this sweep"),
        CandidateSpec("paligemma2_3b_mix:prompt", "paligemma2_3b_mix", dict(
            max_new_tokens=max_new_tokens, dtype=dtype, mode="prompt"),
            hf_gated_repo="google/paligemma2-3b-mix-448",
            note="GATED (Gemma license). Full prompt text sent to the model"),
        CandidateSpec("moondream2:query", "moondream2", dict(mode="query"),
            note="README-recommended revision pinned; requires transformers<5. Prompt text sent via .query()"),
        CandidateSpec("paligemma2_3b_mix:detect", "paligemma2_3b_mix", dict(
            max_new_tokens=max_new_tokens, dtype=dtype, mode="detect"),
            variants=["baseline"], prompt_independent=True, hf_gated_repo="google/paligemma2-3b-mix-448",
            note="native 'detect <label>' task prefix; prompt variants do not apply"),
        CandidateSpec("moondream2:point", "moondream2", dict(mode="point"),
            variants=["baseline"], prompt_independent=True,
            note="native .point() API; prompt variants do not apply"),
    ]


def mock_candidate_specs() -> List[CandidateSpec]:
    """Mock stand-ins in each real candidate's native output format (see mock_vlm_backends)."""
    return [
        CandidateSpec("qwen2_5_vl_3b", "mock", dict(flavor="json", coord_space="model_input",
                                                    model_input_hw=(364, 504), dtype="mock-bf16")),
        CandidateSpec("internvl2_5_4b", "mock", dict(flavor="qwen_point2d")),
        CandidateSpec("paligemma2_3b_mix:prompt", "mock", dict(flavor="paligemma_loc")),
        CandidateSpec("moondream2:query", "mock", dict(flavor="json")),
        CandidateSpec("paligemma2_3b_mix:detect", "mock", dict(flavor="paligemma_loc"),
                      variants=["baseline"], prompt_independent=True),
        CandidateSpec("moondream2:point", "mock", dict(flavor="moondream_native"),
                      variants=["baseline"], prompt_independent=True),
    ]


def validate_prompt_template(template: str) -> None:
    """Fails early, with a readable message, on a template `build_prompt()` could not format
    (the usual culprit: literal JSON braces must be doubled, i.e. `{{"target_point_xy": [x, y]}}`)."""
    try:
        template.format(target_label="green marker", width=640, height=480,
                        platform_w_mm=187.5, platform_h_mm=142.0)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"prompt template is not a valid str.format() template ({exc!r}). Available fields: "
            "{target_label} {width} {height} {platform_w_mm} {platform_h_mm}. Literal braces (the "
            'JSON example) must be doubled: {{"target_point_xy": [x, y]}}.') from exc


BUILTIN_VARIANTS = tuple(PROMPT_VARIANTS)  # baseline / oriented / oriented_aruco, captured at import


def register_prompt_overrides(overrides: Dict[str, str]) -> List[str]:
    """Adds/replaces named prompt variants in the live `PROMPT_VARIANTS` (no re-zip needed).
    Returns the names registered. Re-registering a name with new text is safe: checkpoint keys
    include a hash of the template text, so old results for that name are not reused. The three
    built-in variants cannot be overridden (the pipeline-validation gate and every comparison
    against the local run depend on their exact text) -- give an experiment its own name."""
    for name, text in overrides.items():
        if name in BUILTIN_VARIANTS:
            raise ValueError(f"{name!r} is a built-in variant and cannot be overridden; pick another name.")
        validate_prompt_template(text)
        PROMPT_VARIANTS[name] = text
    return list(overrides)


def _sha(obj: object) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def effective_dtype_name(spec: CandidateSpec) -> str:
    if spec.backend == "mock":
        return str(spec.kwargs.get("dtype", "mock"))
    return resolve_torch_dtype(str(spec.kwargs.get("dtype", "auto")))[1]


# Mock-only fault-injection switches: they change WHEN a run fails, never what a finished item
# contains, so they must not be part of a result's identity (else a resume after an injected
# interrupt would not recognise its own earlier items).
_NON_IDENTITY_KWARGS = frozenset({"fail_on_load", "raise_on_call_indices", "interrupt_after_calls", "sleep_s"})


def item_key(spec: CandidateSpec, variant: str, frame: FrameItem, max_new_tokens: int) -> str:
    identity_kwargs = {k: v for k, v in spec.kwargs.items() if k not in _NON_IDENTITY_KWARGS}
    cfg = _sha({"kwargs": identity_kwargs, "dtype": effective_dtype_name(spec), "max_new_tokens": max_new_tokens})
    prompt = _sha(PROMPT_VARIANTS[variant])
    return f"{spec.key}|{variant}|{prompt}|{cfg}|{frame.session}|{frame.frame_index}"


def spec_variants(spec: CandidateSpec, run_variants: Sequence[str]) -> List[str]:
    return list(spec.variants) if spec.variants is not None else list(run_variants)


def build_backend(spec: CandidateSpec, max_new_tokens: int):
    if spec.backend == "mock":
        from ml_jetson_vla.core.mock_vlm_backends import MOCK_REGISTRY

        return MOCK_REGISTRY["mock"](**spec.kwargs)
    cls = BACKEND_REGISTRY[spec.backend]
    kwargs = dict(spec.kwargs)
    if spec.backend in ("qwen2_5_vl_3b_instruct",):
        kwargs.setdefault("max_new_tokens", max_new_tokens)
    return cls(**kwargs)


# ------------------------------------------------------------------------------------------
# Hugging Face auth / gating
# ------------------------------------------------------------------------------------------

def get_hf_token() -> Tuple[Optional[str], str]:
    """Colab secret `HF_TOKEN` (google.colab.userdata), else env `HF_TOKEN`. Returns (token, how)."""
    try:
        from google.colab import userdata  # type: ignore

        try:
            tok = userdata.get("HF_TOKEN")
            if tok:
                return tok, "Colab secret HF_TOKEN"
        except Exception as exc:  # SecretNotFoundError / NotebookAccessError
            reason = f"Colab secret HF_TOKEN unavailable ({type(exc).__name__}: {exc})"
            env = os.environ.get("HF_TOKEN")
            return (env, "env HF_TOKEN") if env else (None, reason)
    except ImportError:
        pass
    env = os.environ.get("HF_TOKEN")
    return (env, "env HF_TOKEN") if env else (None, "no Colab secret / env HF_TOKEN found")


def check_hf_access(repo_id: str, token: Optional[str]) -> Tuple[bool, str]:
    """Probes a (possibly gated) repo by downloading its tiny config.json. Never raises."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False, "huggingface_hub is not installed"
    try:
        hf_hub_download(repo_id, "config.json", token=token)
        return True, "access confirmed"
    except Exception as exc:
        name = type(exc).__name__
        msg = (
            f"cannot access {repo_id} ({name}). If this is a gated model: (1) open "
            f"https://huggingface.co/{repo_id} while logged in and accept the license, (2) create a "
            f"read token at https://huggingface.co/settings/tokens, (3) add it as a Colab secret "
            f"named HF_TOKEN (key icon in the left sidebar) and enable 'Notebook access', then "
            f"re-run. Underlying error: {str(exc)[:300]}"
        )
        return False, msg


# ------------------------------------------------------------------------------------------
# Checkpoint store
# ------------------------------------------------------------------------------------------

class CheckpointStore:
    """JSON-file-backed store of finished items + per-candidate stage records + errors. Every
    mutation is flushed atomically (temp file + os.replace)."""

    MAX_ERRORS_PER_CANDIDATE = 50

    def __init__(self, path: str, meta: Optional[Dict[str, object]] = None) -> None:
        self.path = path
        self.items: Dict[str, Dict[str, object]] = {}
        self.stages: Dict[str, Dict[str, object]] = {}
        self.errors: List[Dict[str, object]] = []
        self.meta: Dict[str, object] = dict(meta or {})
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
                if data.get("format") != CHECKPOINT_FORMAT:
                    raise ValueError(f"unexpected checkpoint format {data.get('format')!r}")
                self.items = data.get("items", {})
                self.stages = data.get("stages", {})
                self.errors = data.get("errors", [])
                self.meta = {**data.get("meta", {}), **self.meta}
                print(f"Resuming checkpoint {path}: {len(self.items)} finished item(s), "
                      f"{len(self.errors)} recorded error(s).")
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                bad = f"{path}.corrupt-{int(time.time())}"
                os.replace(path, bad)
                print(f"Could not read checkpoint ({exc}); moved it to {bad} and starting fresh.")

    def flush(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        atomic_write_json(self.path, {"format": CHECKPOINT_FORMAT, "meta": self.meta,
                                      "stages": self.stages, "errors": self.errors, "items": self.items})

    def has(self, key: str) -> bool:
        return key in self.items

    def put(self, key: str, entry: Dict[str, object]) -> None:
        self.items[key] = entry
        self.flush()

    def set_stage(self, cand: str, **fields: object) -> None:
        self.stages.setdefault(cand, {}).update(fields)
        self.flush()

    def record_error(self, cand: str, stage: str, exc: BaseException, extra: Optional[Dict[str, object]] = None) -> None:
        rec = {"candidate": cand, "stage": stage, "error_type": type(exc).__name__, "error": str(exc)[:2000],
               "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:],
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
        self.errors.append(rec)
        mine = [e for e in self.errors if e["candidate"] == cand]
        if len(mine) > self.MAX_ERRORS_PER_CANDIDATE:
            self.errors.remove(mine[0])
        self.flush()


# ------------------------------------------------------------------------------------------
# The sweep
# ------------------------------------------------------------------------------------------

def run_sweep(
    specs: Sequence[CandidateSpec],
    frames: Sequence[FrameItem],
    variants: Sequence[str],
    store: CheckpointStore,
    tolerance_mm: float = 20.0,
    max_new_tokens: int = 24,
    hf_token: Optional[str] = None,
    only: Optional[Iterable[str]] = None,
    log_every_call: bool = True,
) -> Dict[str, str]:
    """Runs (or resumes) every spec; returns `{candidate_key: status}` with status one of
    complete / partial / load_failed / gated_skipped / aborted / nothing_to_do. Never raises for an
    ordinary candidate failure; KeyboardInterrupt propagates (after cleanup)."""
    missing = [v for v in variants if v not in PROMPT_VARIANTS]
    if missing:
        raise ValueError(f"unknown prompt variant(s) {missing}; known: {list(PROMPT_VARIANTS)}")
    wanted = set(only) if only is not None else None
    statuses: Dict[str, str] = {}
    for spec in specs:
        if wanted is not None and spec.key not in wanted:
            continue
        cand_variants = spec_variants(spec, variants)
        todo = [(fr, v) for fr in frames for v in cand_variants
                if not store.has(item_key(spec, v, fr, max_new_tokens))]
        total = len(frames) * len(cand_variants)
        print(f"\n=== {spec.key}  [{spec.backend}]  {total - len(todo)}/{total} already done ===")
        if not todo:
            statuses[spec.key] = "nothing_to_do"
            print("  nothing left to do -- skipping (model not loaded).")
            continue

        if spec.hf_gated_repo:
            ok, msg = check_hf_access(spec.hf_gated_repo, hf_token)
            if not ok:
                print(f"  SKIPPED (gated / no access): {msg}")
                store.set_stage(spec.key, status="gated_skipped", detail=msg)
                statuses[spec.key] = "gated_skipped"
                continue

        backend = None
        try:
            try:
                backend = build_backend(spec, max_new_tokens)
                t0 = time.time()
                backend.load()
                load_s = time.time() - t0
            except Exception as exc:
                print(f"  LOAD FAILED: {type(exc).__name__}: {str(exc)[:400]}")
                store.record_error(spec.key, "load", exc)
                store.set_stage(spec.key, status="load_failed", detail=f"{type(exc).__name__}: {str(exc)[:400]}")
                statuses[spec.key] = "load_failed"
                continue

            cfg = backend.config_summary() if hasattr(backend, "config_summary") else {}
            store.set_stage(spec.key, status="running", load_time_s=load_s, config=cfg,
                            dtype=getattr(backend, "dtype_name", None), note=spec.note)
            print(f"  loaded in {load_s:.1f}s  config={cfg}")
            policies = {v: MinimalVLMPolicy(backend=backend, pixel_to_mm=None, prompt_variant=v)
                        for v in cand_variants}
            consecutive = 0
            aborted = False
            done_now = 0
            for i, (fr, v) in enumerate(todo, 1):
                key = item_key(spec, v, fr, max_new_tokens)
                try:
                    entry = score_frame_with_policy(
                        policies[v], v, fr.frame_bgr, fr.instruction, fr.homography,
                        fr.true_x_tel, fr.true_y_tel, tolerance_mm, fr.frame_index,
                        target_transition=fr.target_transition,
                    )
                except Exception as exc:
                    consecutive += 1
                    print(f"  [{i}/{len(todo)}] {v} {fr.ident}: CALL FAILED ({type(exc).__name__}: {str(exc)[:200]})")
                    store.record_error(spec.key, "generate", exc, {"variant": v, "frame": fr.ident})
                    if consecutive >= MAX_CONSECUTIVE_CALL_ERRORS:
                        print(f"  ABORTING candidate after {consecutive} consecutive failures.")
                        aborted = True
                        break
                    continue
                consecutive = 0
                entry.update({"candidate": spec.key, "variant": v, "session": fr.session,
                              "backend": spec.backend, "prompt_independent": spec.prompt_independent})
                store.put(key, entry)
                done_now += 1
                if log_every_call:
                    err = entry.get("error_mm")
                    print(f"  [{i}/{len(todo)}] {v:<15} {fr.session[-15:]} f{fr.frame_index:<5} {fr.instruction:<9} "
                          f"parse={'Y' if entry.get('parse_ok') else 'N'} "
                          f"err={'%.1fmm' % err if err is not None else '  -  '} "
                          f"{'HIT' if entry.get('hit') else '   '} "
                          f"{(entry.get('generate_latency_s') or 0):.1f}s raw={str(entry.get('raw_text'))[:60]!r}")
            remaining = sum(1 for fr, v in todo if not store.has(item_key(spec, v, fr, max_new_tokens)))
            status = "aborted" if aborted else ("complete" if remaining == 0 else "partial")
            store.set_stage(spec.key, status=status, detail=f"{done_now} new, {remaining} remaining")
            statuses[spec.key] = status
        finally:
            if backend is not None:
                try:
                    backend.unload()
                except Exception:
                    pass
            backend = None
            free_gpu_memory()
    return statuses


# ------------------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------------------

def _entries_for(store: CheckpointStore, spec: CandidateSpec, variant: str,
                 frames: Sequence[FrameItem], max_new_tokens: int) -> List[Dict[str, object]]:
    out = []
    for fr in frames:
        e = store.items.get(item_key(spec, variant, fr, max_new_tokens))
        if e is not None:
            out.append(e)
    return out


def _stats(entries: Sequence[Dict[str, object]], n_expected: int, tol: float) -> Dict[str, object]:
    parsed = [e for e in entries if e.get("parse_ok") and e.get("error_mm") is not None]
    err = np.array([float(e["error_mm"]) for e in parsed])
    settled = [e for e in parsed if e.get("target_transition") is False]
    lat = [float(e["generate_latency_s"]) for e in entries if e.get("generate_latency_s") is not None]
    mem = [float(e["peak_memory_gb"]) for e in entries if e.get("peak_memory_gb") is not None]
    return {
        "n_expected": n_expected, "n_done": len(entries), "n_parsed": len(parsed),
        "parse_rate": (len(parsed) / len(entries)) if entries else np.nan,
        f"hit@{tol:g}mm": (float((err <= tol).mean()) if len(err) else np.nan),
        "hits": int((err <= tol).sum()) if len(err) else 0,
        "mean_err_mm": float(err.mean()) if len(err) else np.nan,
        "median_err_mm": float(np.median(err)) if len(err) else np.nan,
        "mean_err_settled_mm": (float(np.mean([e["error_mm"] for e in settled])) if settled else np.nan),
        "mean_gen_s": float(np.mean(lat)) if lat else np.nan,
        "peak_mem_gb": float(max(mem)) if mem else np.nan,
    }


def reference_baseline_rows(frames: Sequence[FrameItem], tol: float, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """No-model reference rows on the SAME frames, so accuracy numbers have a yardstick: a
    constant guess at the platform centre, and a uniformly random point on the platform. Computed
    from ground truth only (manifest-frame mm)."""
    _a, _f, w, h = load_manifest_full(GROUND_TRUTH_MANIFEST)
    truth = np.array([touch_frame_to_manifest_mm(f.true_x_tel, f.true_y_tel, w, h) for f in frames])
    rows = []
    e_c = np.hypot(*(truth - np.array([w / 2, h / 2])).T)
    rows.append({"candidate": "[ref] constant platform-centre guess", "variant": "-", "n_done": len(frames),
                 "n_parsed": len(frames), "parse_rate": 1.0, f"hit@{tol:g}mm": float((e_c <= tol).mean()),
                 "hits": int((e_c <= tol).sum()), "mean_err_mm": float(e_c.mean()), "median_err_mm": float(np.median(e_c))})
    rng = np.random.default_rng(seed)
    errs = np.stack([np.hypot(*(truth - rng.uniform([0, 0], [w, h], size=truth.shape)).T) for _ in range(draws)])
    rows.append({"candidate": "[ref] uniform-random point on platform", "variant": "-", "n_done": len(frames),
                 "n_parsed": len(frames), "parse_rate": 1.0, f"hit@{tol:g}mm": float((errs <= tol).mean()),
                 "hits": float((errs <= tol).sum(axis=1).mean()), "mean_err_mm": float(errs.mean()),
                 "median_err_mm": float(np.median(errs))})
    return pd.DataFrame(rows)


def aggregate(
    store: CheckpointStore, specs: Sequence[CandidateSpec], frames: Sequence[FrameItem],
    variants: Sequence[str], tolerance_mm: float = 20.0, max_new_tokens: int = 24,
    include_references: bool = True,
) -> pd.DataFrame:
    """Candidate x variant table. Hit rate is recomputed from `error_mm` with the CURRENT
    tolerance, so changing `tolerance_mm` needs no re-run."""
    rows = []
    for spec in specs:
        stage = store.stages.get(spec.key, {})
        for v in spec_variants(spec, variants):
            entries = _entries_for(store, spec, v, frames, max_new_tokens)
            row = {"candidate": spec.key,
                   "variant": NATIVE_VARIANT_LABEL if spec.prompt_independent else v,
                   **_stats(entries, len(frames), tolerance_mm),
                   "dtype": stage.get("dtype") or effective_dtype_name(spec),
                   "status": stage.get("status", "not run")}
            rows.append(row)
    df = pd.DataFrame(rows)
    if include_references and len(frames):
        df = pd.concat([df, reference_baseline_rows(frames, tolerance_mm)], ignore_index=True)
    return df


def aggregate_coord_space(
    store: CheckpointStore, specs: Sequence[CandidateSpec], frames: Sequence[FrameItem],
    variants: Sequence[str], tolerance_mm: float = 20.0, max_new_tokens: int = 24,
) -> pd.DataFrame:
    """DIAGNOSTIC: mean error / hit rate under each way of reading the model's coordinates
    (primary = the backend's declared space; alt_raw = as raw-frame pixels; alt_model_input;
    alt_norm1000). A backend whose declared space is wrong shows up as another column being far
    better. Diagnostic only -- picking the best column per model after the fact would be forking
    paths; the headline table uses the declared convention."""
    rows = []
    for spec in specs:
        for v in spec_variants(spec, variants):
            entries = [e for e in _entries_for(store, spec, v, frames, max_new_tokens) if e.get("parse_ok")]
            if not entries:
                continue
            row = {"candidate": spec.key, "variant": NATIVE_VARIANT_LABEL if spec.prompt_independent else v,
                   "declared_space": entries[0].get("coord_space")}
            for col, label in [("error_mm", "primary"), ("error_mm_alt_raw", "as_raw_px"),
                               ("error_mm_alt_model_input", "as_model_input"), ("error_mm_alt_norm1000", "as_norm1000")]:
                vals = np.array([float(e[col]) for e in entries if e.get(col) is not None])
                row[f"{label}_mean_mm"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{label}_hit@{tolerance_mm:g}"] = float((vals <= tolerance_mm).mean()) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def per_session_table(
    store: CheckpointStore, spec: CandidateSpec, frames: Sequence[FrameItem], variants: Sequence[str],
    max_new_tokens: int = 24, tolerance_mm: float = 20.0,
) -> pd.DataFrame:
    """Mean error and hits per session for one candidate -- per the `model-iteration-constraints`
    skill: report session-to-session spread, never only a pooled number."""
    rows = []
    for sess in sorted({f.session for f in frames}):
        fs = [f for f in frames if f.session == sess]
        row = {"session": sess[-15:]}
        for v in spec_variants(spec, variants):
            es = [e for e in _entries_for(store, spec, v, fs, max_new_tokens) if e.get("parse_ok")]
            errs = [float(e["error_mm"]) for e in es]
            row[f"{v}: mean_mm"] = float(np.mean(errs)) if errs else np.nan
            row[f"{v}: hits"] = f"{sum(1 for x in errs if x <= tolerance_mm)}/{len(fs)}"
        rows.append(row)
    return pd.DataFrame(rows)


def export_scorer_format(
    store: CheckpointStore, spec: CandidateSpec, frames: Sequence[FrameItem], variants: Sequence[str],
    tolerance_mm: float = 20.0, max_new_tokens: int = 24,
) -> Dict[str, object]:
    """One candidate's results in the local scorer's own output format
    (`score_minimal_baseline_offline.py --out`: backend / prompt_variants / tolerance_mm /
    sessions[].variants[].frames[] with `summarize_variant_frames` summaries), so it can be pasted
    back and compared/diffed against the local files."""
    cand_variants = spec_variants(spec, variants)
    sessions = []
    for sess in sorted({f.session for f in frames}):
        fs = [f for f in frames if f.session == sess]
        vmap = {}
        for v in cand_variants:
            es = _entries_for(store, spec, v, fs, max_new_tokens)
            for e in es:  # keep `hit` consistent with the requested tolerance
                if e.get("error_mm") is not None:
                    e["hit"] = bool(float(e["error_mm"]) <= tolerance_mm)
            vmap[v] = summarize_variant_frames(es)
        sessions.append({"session": sess, "variants": vmap})
    return {"backend": spec.key, "backend_class": spec.backend, "prompt_variants": cand_variants,
            "tolerance_mm": tolerance_mm, "scoring_frame_version": 2,
            "config": store.stages.get(spec.key, {}).get("config"), "sessions": sessions}


def export_all(
    store: CheckpointStore, specs: Sequence[CandidateSpec], frames: Sequence[FrameItem],
    variants: Sequence[str], out_dir: str, tolerance_mm: float = 20.0, max_new_tokens: int = 24,
    run_label: str = "run",
) -> str:
    """Writes one scorer-format JSON per candidate plus a combined file and the aggregate table
    (CSV). Returns the combined file's path (this is the one to paste back)."""
    os.makedirs(out_dir, exist_ok=True)
    combined: Dict[str, object] = {"run_label": run_label, "meta": store.meta, "stages": store.stages,
                                   "errors": store.errors, "candidates": {}}
    for spec in specs:
        if not any(store.has(item_key(spec, v, fr, max_new_tokens))
                   for v in spec_variants(spec, variants) for fr in frames):
            continue
        doc = export_scorer_format(store, spec, frames, variants, tolerance_mm, max_new_tokens)
        safe = spec.key.replace(":", "_")
        atomic_write_json(os.path.join(out_dir, f"scorer_format_{safe}_{run_label}.json"), doc)
        combined["candidates"][spec.key] = doc
    table = aggregate(store, specs, frames, variants, tolerance_mm, max_new_tokens)
    table.to_csv(os.path.join(out_dir, f"aggregate_table_{run_label}.csv"), index=False)
    combined["aggregate_table"] = json.loads(table.to_json(orient="records"))
    path = os.path.join(out_dir, f"arm2_colab_sweep_results_{run_label}.json")
    atomic_write_json(path, combined)
    return path


# ------------------------------------------------------------------------------------------
# Pipeline validation against the local reference run
# ------------------------------------------------------------------------------------------

def validate_against_reference(
    store: CheckpointStore, spec: CandidateSpec, frames: Sequence[FrameItem], reference_path: str,
    variants: Sequence[str] = ("baseline", "oriented"), max_new_tokens: int = 24,
    max_legacy_mean_delta_mm: float = 25.0, max_primary_mean_delta_mm: float = 12.0,
    max_median_px_dev: float = 25.0, min_parse_rate: float = 0.95,
) -> Dict[str, object]:
    """Compares this run's Qwen results with the local 2026-09-18 CPU run (rescored copy of the
    reference file, `..._RESCORED_v2.json`). Checks, per variant: parse rate; mean LEGACY error vs
    the historical 183.8/170.3mm (depends only on raw model outputs, so it validates the pipeline
    independent of the scoring fix); mean primary (corrected) error; median pixel distance between
    Colab's and local's as-parsed points on the same frames (the sharpest test -- a mean error near
    chance can hide a broken image/resize path, per-frame agreement cannot); and that the measured
    model-input size equals the locally-derived one (catches a transformers-version resize
    difference). Thresholds are judgement calls, exposed as arguments.

    NOT expected to be bit-exact: local ran bf16 on CPU, Colab runs bf16/fp16 on a GPU with
    different kernels; greedy decoding can still flip a digit or two."""
    with open(reference_path) as fh:
        ref = json.load(fh)
    ref_frames: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for s in ref["sessions"]:
        for v, summ in s.get("variants", {}).items():
            for f in summ["frames"]:
                ref_frames[(v, s["session"], f["frame_index"])] = f

    report: Dict[str, object] = {"variants": {}, "checks": [], "passed": True}
    for v in variants:
        pairs, legacy, primary = [], [], []
        entries = _entries_for(store, spec, v, frames, max_new_tokens)
        n_parsed = sum(1 for e in entries if e.get("parse_ok"))
        hw_ok = True
        for fr in frames:
            e = store.items.get(item_key(spec, v, fr, max_new_tokens))
            r = ref_frames.get((v, fr.session, fr.frame_index))
            if e is None or r is None or not e.get("parse_ok") or not r.get("parse_ok"):
                continue
            (ex, ey), (rx, ry) = e["target_point_px"], r["target_point_px"]
            pairs.append(float(np.hypot(ex - rx, ey - ry)))
            legacy.append((float(e["error_mm_legacy"]), float(r["error_mm_legacy"])))
            primary.append((float(e["error_mm"]), float(r["error_mm"])))
            if r.get("model_input_hw") and e.get("model_input_hw") and list(e["model_input_hw"]) != list(r["model_input_hw"]):
                hw_ok = False
        vr: Dict[str, object] = {
            "n_compared": len(pairs), "n_done": len(entries),
            "parse_rate": (n_parsed / len(entries)) if entries else float("nan"),
            "median_px_dev": float(np.median(pairs)) if pairs else float("nan"),
            "frac_within_10px": float(np.mean(np.array(pairs) <= 10)) if pairs else float("nan"),
            "legacy_mean_colab": float(np.mean([a for a, _ in legacy])) if legacy else float("nan"),
            "legacy_mean_local": float(np.mean([b for _, b in legacy])) if legacy else float("nan"),
            "primary_mean_colab": float(np.mean([a for a, _ in primary])) if primary else float("nan"),
            "primary_mean_local": float(np.mean([b for _, b in primary])) if primary else float("nan"),
            "model_input_hw_matches_local": hw_ok,
        }
        report["variants"][v] = vr
        checks = [
            (f"{v}: all frames answered", len(entries) == len(frames)),
            (f"{v}: parse rate >= {min_parse_rate:g}", bool(vr["parse_rate"] >= min_parse_rate)),
            (f"{v}: |legacy mean - local| <= {max_legacy_mean_delta_mm:g}mm",
             bool(abs(vr["legacy_mean_colab"] - vr["legacy_mean_local"]) <= max_legacy_mean_delta_mm)),
            (f"{v}: |primary mean - local| <= {max_primary_mean_delta_mm:g}mm",
             bool(abs(vr["primary_mean_colab"] - vr["primary_mean_local"]) <= max_primary_mean_delta_mm)),
            (f"{v}: median pixel deviation vs local <= {max_median_px_dev:g}px",
             bool(vr["median_px_dev"] <= max_median_px_dev)),
            (f"{v}: model-input size equals local's", hw_ok),
        ]
        for name, ok in checks:
            report["checks"].append({"check": name, "ok": ok})
            report["passed"] = report["passed"] and ok
    return report


def print_validation_report(report: Dict[str, object]) -> None:
    for v, vr in report["variants"].items():
        print(f"[{v}] compared {vr['n_compared']} frames | parse {vr['parse_rate']:.2f} | "
              f"median px dev {vr['median_px_dev']:.1f} (within 10px: {vr['frac_within_10px']:.2f}) | "
              f"legacy mean {vr['legacy_mean_colab']:.1f} (local {vr['legacy_mean_local']:.1f}) | "
              f"corrected mean {vr['primary_mean_colab']:.1f} (local {vr['primary_mean_local']:.1f})")
    for c in report["checks"]:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['check']}")
    print("PIPELINE VALIDATION: " + ("PASSED" if report["passed"] else "FAILED"))


# ------------------------------------------------------------------------------------------
# Smoke report / env meta
# ------------------------------------------------------------------------------------------

def smoke_report(store: CheckpointStore, specs: Sequence[CandidateSpec], frames: Sequence[FrameItem],
                 variants: Sequence[str], max_new_tokens: int = 24) -> pd.DataFrame:
    """One row per candidate x variant x smoke frame with the RAW model text, so format problems
    (a model answering in prose, in another coordinate convention...) are visible immediately."""
    rows = []
    for spec in specs:
        stage = store.stages.get(spec.key, {})
        any_entry = False
        for v in spec_variants(spec, variants):
            for fr in frames:
                e = store.items.get(item_key(spec, v, fr, max_new_tokens))
                if e is None:
                    continue
                any_entry = True
                rows.append({"candidate": spec.key, "variant": v, "frame": fr.ident[-24:], "cmd": fr.instruction,
                             "parse": e.get("parse_reason", "")[:22] if e.get("parse_ok") else "FAIL",
                             "err_mm": e.get("error_mm"), "gen_s": e.get("generate_latency_s"),
                             "dtype": e.get("dtype"), "raw_text": str(e.get("raw_text"))[:90]})
        if not any_entry:
            rows.append({"candidate": spec.key, "variant": "-", "frame": "-", "cmd": "-",
                         "parse": f"[{stage.get('status', 'not run')}]", "err_mm": None, "gen_s": None,
                         "dtype": None, "raw_text": str(stage.get("detail", ""))[:90]})
    return pd.DataFrame(rows)


def render_overlay(frame: FrameItem, entry: Optional[Dict[str, object]]) -> np.ndarray:
    """RGB image (for matplotlib) of the raw frame with the TRUE target (green circle, from the
    telemetry ground truth via the per-frame homography) and the model's answer in raw-frame
    pixels (red cross; a small blue cross marks the as-parsed point when it differs, i.e. for
    model_input-space backends). For eyeballing a bad result: a consistent offset between the two
    points across frames is the signature of a wrong coordinate-space assumption."""
    import cv2

    img = frame.frame_bgr.copy()
    _a, _f, w, h = load_manifest_full(GROUND_TRUTH_MANIFEST)
    gx, gy = touch_frame_to_manifest_mm(frame.true_x_tel, frame.true_y_tel, w, h)
    gt = cv2.perspectiveTransform(np.array([[[gx, gy]]], np.float32), frame.homography.astype(np.float32))[0][0]
    cv2.circle(img, (int(gt[0]), int(gt[1])), 14, (0, 200, 0), 2)
    label = f"{frame.instruction}  true=green circle"
    if entry and entry.get("target_point_raw_px"):
        rx, ry = entry["target_point_raw_px"]
        cv2.drawMarker(img, (int(rx), int(ry)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        px, py = entry["target_point_px"]
        if abs(px - rx) + abs(py - ry) > 1.0:
            cv2.drawMarker(img, (int(px), int(py)), (255, 120, 0), cv2.MARKER_CROSS, 12, 1)
        label += f"  pred=red x  err={entry.get('error_mm', float('nan')):.0f}mm"
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img[..., ::-1]


def collect_env_meta() -> Dict[str, object]:
    meta: Dict[str, object] = {"python": sys.version.split()[0], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            meta["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception as exc:
        meta["torch_error"] = str(exc)
    for pkg in ("transformers", "accelerate", "huggingface_hub", "qwen_vl_utils", "av", "cv2", "numpy", "pandas"):
        try:
            mod = __import__(pkg)
            meta[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            meta[pkg] = None
    return meta
