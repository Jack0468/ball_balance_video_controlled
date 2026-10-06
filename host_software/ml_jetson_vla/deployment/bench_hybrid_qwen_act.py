"""Two-rate timing bench for the hybrid Arm 2 pipeline: slow Qwen2.5-VL-3B grounding + fast ACT chunks. Trains nothing.

Slow loop (background thread): Qwen grounding on real sampled Track 4 frames, through the sweep's own candidate
spec and the Arm 2 minimal-baseline policy, converted to telemetry-frame mm with the scorer's convention.
Fast loop (main thread): ACT chunk inference, RANDOM weights, every chunk_len/30 s, conditioned on the real sampled
frame, the real telemetry touch state, and the latest slow-loop target.

Custom head: ACT's observation.state is extended from 2 to 4 dims [touch_x, touch_y, target_x, target_y]. Pretrained
ACT has no target input, so this is the one non-pretrained addition, recorded in the output JSON.

Schedule model: replan k is due at t0 + k*chunk_len/30 s. Inference blocks the fast loop; a late replan starts
immediately. Frames and touch state are replayed from real sampled frames, so target age and the GPU contention
between the two loops are real, but the camera stream timing is not.

Known conflict (see the report): Qwen2.5-VL needs transformers>=5 (Dockerfile.arm2-transformers5); lerobot 0.4.4's
policies package init fails under transformers 5 (Dockerfile.arm2-lerobot pins <5). preflight() fails fast on either side.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.deployment.bench_action_models import (  # noqa: E402
    ACT_IMAGE_HW, DTYPE_NAMES, IMAGE_KEY, STATE_KEY, TARGET_NAMES, STATE_NAMES,
    build_act_policy, chunk_duration_s, implied_replan_hz, inference_fits_chunk, pixel_to_telemetry_mm,
    refuse_overwrite, summarize_latencies_ms, write_json_atomic,
)

DEFAULT_BRONZE_DIR: str = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
STALE_TARGET_S: float = 2.0
SLOW_LOOP_JOIN_TIMEOUT_S: float = 180.0


@dataclasses.dataclass(frozen=True)
class TargetUpdate:
    seq: int
    x_mm: float
    y_mm: float
    done_monotonic_s: float
    frame_ident: str
    slow_latency_s: float


class TargetStore:
    """Latest slow-loop target plus slow-loop bookkeeping, shared across the two threads under one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[TargetUpdate] = None
        self._seq = 0
        self.slow_calls = 0
        self.parse_failures = 0
        self.slow_latencies_s: list[float] = []
        self.errors: list[str] = []

    def put(self, x_mm: float, y_mm: float, frame_ident: str, slow_latency_s: float) -> None:
        with self._lock:
            self._seq += 1
            self.slow_calls += 1
            self.slow_latencies_s.append(slow_latency_s)
            self._latest = TargetUpdate(self._seq, x_mm, y_mm, time.monotonic(), frame_ident, slow_latency_s)

    def record_parse_failure(self, slow_latency_s: float) -> None:
        with self._lock:
            self.slow_calls += 1
            self.slow_latencies_s.append(slow_latency_s)
            self.parse_failures += 1

    def record_error(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)

    def latest(self) -> Optional[TargetUpdate]:
        with self._lock:
            return self._latest


def slow_loop(stop: threading.Event, frames: Sequence[Any], policy: Any, store: TargetStore) -> None:
    idx = 0
    while not stop.is_set():
        item = frames[idx % len(frames)]
        idx += 1
        t0 = time.perf_counter()
        try:
            policy.act(item.frame_bgr, item.instruction, state={})
            latency = time.perf_counter() - t0
            debug = policy.last_debug
            if not debug.get("parse_ok"):
                store.record_parse_failure(latency)
                continue
            px, py = debug["target_point_px"]
            x_mm, y_mm = pixel_to_telemetry_mm(
                item.homography, px, py, debug["coord_space"], debug.get("model_input_hw"),
                tuple(item.frame_bgr.shape[:2]),
            )
            store.put(x_mm, y_mm, item.ident, latency)
        except Exception as exc:  # a dead slow loop must fail the run, not leave stale targets in place
            store.record_error(f"{type(exc).__name__}: {exc}")
            return


def load_touch_state(bronze_dir: str, session: str, frame_index: int) -> tuple[float, float]:
    import pandas as pd

    df = pd.read_csv(os.path.join(bronze_dir, session, "telemetry.csv"))
    rows = df.loc[df["frame_index"] == frame_index]
    if len(rows) != 1:
        raise ValueError(f"{session}: expected one telemetry row for frame {frame_index}, found {len(rows)}")
    return float(rows["touch_x"].iloc[0]), float(rows["touch_y"].iloc[0])


def frame_to_tensor(frame_bgr: np.ndarray, dtype: "torch.dtype", device: str) -> "torch.Tensor":
    import torch

    if tuple(frame_bgr.shape[:2]) != ACT_IMAGE_HW:
        raise ValueError(f"frame {frame_bgr.shape[:2]} != ACT input {ACT_IMAGE_HW}")
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1]).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def check_slow_errors(store: TargetStore) -> None:
    if store.errors:
        raise RuntimeError(f"slow loop died: {store.errors[0]}")


def run_fast_loop(
    act_policy: Any, fast_inputs: Sequence[tuple[Any, tuple[float, float], str]], store: TargetStore,
    chunk_len: int, replans: int, dtype: "torch.dtype", device: str,
) -> list[dict[str, Any]]:
    import torch

    chunk_dur = chunk_duration_s(chunk_len)
    records: list[dict[str, Any]] = []
    prev_seq = 0
    t0 = time.monotonic()
    for k in range(replans):
        due = t0 + k * chunk_dur
        now = time.monotonic()
        if now < due:
            time.sleep(due - now)
        check_slow_errors(store)
        image, touch, frame_ident = fast_inputs[k % len(fast_inputs)]
        latest = store.latest()
        tx, ty = (latest.x_mm, latest.y_mm) if latest is not None else (0.0, 0.0)  # zero-fill, flagged by has_target
        state = torch.tensor([[touch[0], touch[1], tx, ty]], device=device, dtype=dtype)
        batch = {IMAGE_KEY: image, STATE_KEY: state}

        torch.cuda.synchronize()
        start = time.monotonic()
        out = act_policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        end = time.monotonic()

        seq = latest.seq if latest is not None else 0
        records.append({
            "replan": k,
            "scheduled_s": due - t0,
            "start_s": start - t0,
            "late_start_ms": (start - due) * 1000.0,
            "fast_wall_ms": (end - start) * 1000.0,
            "has_target": latest is not None,
            "target_age_ms": (start - latest.done_monotonic_s) * 1000.0 if latest is not None else None,
            "new_target_since_last_replan": seq > prev_seq,
            "target_seq": seq,
            "target_frame_ident": latest.frame_ident if latest is not None else None,
            "frame_ident": frame_ident,
            "output_finite": bool(torch.isfinite(out).all().item()),
        })
        prev_seq = max(prev_seq, seq)
    return records


def summarize_schedule(records: Sequence[dict[str, Any]], chunk_len: int,
                       stale_threshold_s: float = STALE_TARGET_S) -> dict[str, Any]:
    if len(records) == 0:
        raise ValueError("no replan records")
    ages = [float(r["target_age_ms"]) for r in records if r["has_target"]]
    n_stale = sum(1 for a in ages if a > stale_threshold_s * 1000.0)
    chunk_ms = chunk_duration_s(chunk_len) * 1000.0
    fast = [float(r["fast_wall_ms"]) for r in records]
    summary: dict[str, Any] = {
        "n_replans": len(records),
        "n_no_target": len(records) - len(ages),
        "n_new_target_since_last_replan": sum(1 for r in records if r["new_target_since_last_replan"]),
        "fraction_target_older_than_stale_threshold": n_stale / len(records),
        "stale_threshold_s": stale_threshold_s,
        "fast_wall": summarize_latencies_ms(fast),
        "fast_wall_fits_chunk_fraction": sum(1 for f in fast if f < chunk_ms) / len(fast),
        "late_start_fraction": sum(1 for r in records if r["late_start_ms"] > 1.0) / len(records),
        "implied_replan_hz_mean": implied_replan_hz(float(np.mean(fast))),
        "output_all_finite": all(bool(r["output_finite"]) for r in records),
    }
    if ages:
        summary["target_age_ms"] = summarize_latencies_ms(ages)
    return summary


def preflight() -> None:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- refusing to record non-device timings")
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        raise RuntimeError(
            f"transformers {transformers.__version__} lacks Qwen2_5_VLForConditionalGeneration. Qwen needs the "
            "arm2-transformers5 image; ACT needs the arm2-lerobot image (transformers<5). See the module docstring."
        )
    try:
        import lerobot.policies.act.modeling_act  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"lerobot policies import failed under transformers {transformers.__version__}: "
            f"{type(exc).__name__}: {exc}. This is the arm2-transformers5 image; lerobot 0.4.4 cannot import here."
        ) from exc


def run_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from importlib.metadata import version

    from ml_jetson_vla.core.minimal_vlm_policy import MinimalVLMPolicy
    from ml_jetson_vla.deployment.colab_sweep import (
        QWEN_LOCAL_COMMIT, build_backend, default_candidate_specs, prepare_frames,
    )

    spec = default_candidate_specs(max_new_tokens=args.max_new_tokens)[0]
    if spec.backend != "qwen2_5_vl_3b_instruct" or spec.kwargs.get("revision") != QWEN_LOCAL_COMMIT:
        raise RuntimeError("sweep's first candidate is not the pinned Qwen2.5-VL-3B spec")

    frames, skipped = prepare_frames(args.bronze_dir, frames_per_session=args.frames_per_session,
                                     sessions=args.sessions, verbose=True)
    usable: list[Any] = []
    touches: dict[str, tuple[float, float]] = {}
    for f in frames:
        touch = load_touch_state(args.bronze_dir, f.session, f.frame_index)
        if all(np.isfinite(touch)):
            usable.append(f)
            touches[f.ident] = touch
    if not usable:
        raise RuntimeError(f"no usable frames with finite touch state (prepare_frames skipped {len(skipped)})")

    backend = build_backend(spec, args.max_new_tokens)
    t_load = time.perf_counter()
    backend.load()  # load outside the timed schedule
    qwen_load_s = time.perf_counter() - t_load
    slow_policy = MinimalVLMPolicy(backend, prompt_variant="baseline")

    dtype = getattr(torch, DTYPE_NAMES[args.precision])
    device = "cuda"
    act_policy, act_cfg = build_act_policy(args.chunk_len, with_target=True)
    act_policy = act_policy.to(device=device, dtype=dtype).eval()
    fast_inputs = [(frame_to_tensor(f.frame_bgr, dtype, device), touches[f.ident], f.ident) for f in usable]
    warm = fast_inputs[0]
    with torch.no_grad():
        act_policy.predict_action_chunk({IMAGE_KEY: warm[0], STATE_KEY: torch.zeros(
            (1, len(STATE_NAMES) + len(TARGET_NAMES)), device=device, dtype=dtype)})

    store = TargetStore()
    stop = threading.Event()
    thread = threading.Thread(target=slow_loop, args=(stop, usable, slow_policy, store), daemon=True)
    thread.start()
    try:
        records = run_fast_loop(act_policy, fast_inputs, store, args.chunk_len, args.replans, dtype, device)
    finally:
        stop.set()
        thread.join(timeout=SLOW_LOOP_JOIN_TIMEOUT_S)
    slow_alive = thread.is_alive()
    check_slow_errors(store)

    return {
        "bench": "bench_hybrid_qwen_act",
        "precision_fast": args.precision,
        "chunk_len": args.chunk_len,
        "control_hz": 30.0,
        "chunk_duration_s": chunk_duration_s(args.chunk_len),
        "replans_requested": args.replans,
        "qwen": {
            "repo_id": spec.kwargs["model_dir"], "revision": spec.kwargs["revision"],
            "max_new_tokens": args.max_new_tokens, "dtype": backend.dtype_name,
            "load_time_s": qwen_load_s, "prompt_variant": "baseline",
            "slow_calls": store.slow_calls, "parse_failures": store.parse_failures,
            "slow_call_latency": (summarize_latencies_ms(store.slow_latencies_s)
                                  if store.slow_latencies_s else None),
            "slow_thread_still_alive_after_join": slow_alive,
        },
        "frames": {
            "sessions": sorted({f.session for f in usable}),
            "n_usable": len(usable), "n_skipped_by_prepare_frames": len(skipped),
        },
        "fast_model": {
            "policy": "act", "random_init": True, "accuracy_meaningful": False,
            "state_names": list(STATE_NAMES) + list(TARGET_NAMES),
            "custom_head": "observation.state extended 2->4 dims with the slow-loop target (not pretrained ACT)",
            "param_count": int(sum(p.numel() for p in act_policy.parameters())),
            "image_hw": list(ACT_IMAGE_HW),
        },
        "schedule": {
            "model": "replan k due at t0 + k*chunk_dur; blocking inference; late replan starts immediately",
            "frames_replayed": "real sampled frames, cycled; touch state from telemetry.csv at that frame",
            "gpu_shared_with_qwen": True,
        },
        "schedule_summary": summarize_schedule(records, args.chunk_len),
        "fast_inference_fits_chunk_p90": inference_fits_chunk(
            summarize_latencies_ms([float(r["fast_wall_ms"]) for r in records])["p90_ms"], args.chunk_len),
        "replans": records,
        "lerobot_version": version("lerobot"),
        "torch_version": torch.__version__,
    }


def base_result(args: argparse.Namespace) -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "bench": "bench_hybrid_qwen_act",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Qwen2.5-VL-3B + ACT two-rate timing (no training).")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing --out file")
    parser.add_argument("--bronze-dir", default=DEFAULT_BRONZE_DIR)
    parser.add_argument("--sessions", nargs="*", default=None, help="session dir names; default = all Track 4")
    parser.add_argument("--frames-per-session", type=int, default=3)
    parser.add_argument("--chunk-len", type=int, default=100, help="ACT chunk length (default 100)")
    parser.add_argument("--replans", type=int, default=40)
    parser.add_argument("--precision", default="bf16", choices=sorted(DTYPE_NAMES))
    parser.add_argument("--max-new-tokens", type=int, default=24, help="same as the Arm 2 sweep default")
    args = parser.parse_args(argv)
    if args.chunk_len < 1 or args.replans < 1 or args.frames_per_session < 1:
        parser.error("--chunk-len, --replans and --frames-per-session must be >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_path = os.path.abspath(args.out)
    try:
        refuse_overwrite(out_path, args.force)
    except FileExistsError as exc:
        print(f"[bench_hybrid_qwen_act] {exc}")
        return 2

    result = base_result(args)
    try:
        preflight()
        result.update(run_hybrid(args))
        result["status"] = "ok"
        code = 0
    except Exception as exc:  # recorded, never a fabricated timing
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        code = 1
    write_json_atomic(out_path, result)
    print(f"[bench_hybrid_qwen_act] status={result['status']} wrote {out_path}")
    if result["status"] == "ok":
        s = result["schedule_summary"]
        print(f"  replans={s['n_replans']} no_target={s['n_no_target']} "
              f"frac_target_older_than_{STALE_TARGET_S:.0f}s={s['fraction_target_older_than_stale_threshold']:.2f}")
    return code


if __name__ == "__main__":
    sys.exit(main())
