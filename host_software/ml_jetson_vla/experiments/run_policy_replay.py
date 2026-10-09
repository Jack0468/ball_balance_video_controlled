"""Replay driver for the real `Policy` wrappers in `core/hybrid_policy.py`
(`HybridQwenActPolicy`, `SmolVLADirectPolicy`) -- as opposed to `run_experiment.py`'s replay,
which drives a raw `FastLayer` directly. Reuses `run_experiment.py`'s session telemetry/video
loading (`load_session_telemetry`, `VideoRowReader`) rather than re-deriving it, and
`experiments.schedule.summarize_replans` for the same stats the existing replay already reports
-- no new metric invented.

Each session tick calls `policy.act(frame, instruction, {"touch_mm": (touch_x, touch_y)})` --
`core/hybrid_policy.py`'s own documented `state["touch_mm"]` contract. The replan request/landing
itself happens on a REAL background thread inside the policy (`core/hybrid_policy.py`'s
`_FastWorker`/`_Grounder`), not inline like `run_experiment.py`'s `run_replay()` -- so unlike that
script, this one steps on real wall-clock time while advancing through the session's own logged
tick grid, and its timing numbers are a hybrid of "session content" and "this machine's real
thread-scheduling/GIL behaviour", not a hardware measurement either way.

Sends nothing. `--fast-layer` (default `stub`, unchanged behaviour) picks what backs the policy:
`stub` is `StubFastLayer` (no model, constant output); `act` is `experiments/fast_layers.py`'s
`ActFastLayer` (RANDOM weights unless `--act-checkpoint` is given -- this project's established
"random-init is a legitimate state" convention, see `bench_hybrid_qwen_act.py` and
`run_experiment.py`'s own `--checkpoint`); `smolvla` is `SmolVLAFastLayer` (always the real
pretrained `lerobot/smolvla_base` at its pinned revision, never random). `act`/`smolvla` need
torch + lerobot importable, which this Windows interpreter cannot do (see
`docs/EXPERIMENT_OPTIONS_PLAN_2026_10_07.md` section 4.7) -- those variants are meant to run inside
the `arm2-lerobot:r36.4.0` Docker image on the Jetson. Regardless of `--fast-layer`, no grounding
backend is ever configured here (`HybridQwenActPolicy`'s `grounding_backend=None`), so colour
commands in the session still correctly produce `has_target=False` throughout -- that is
independent of this flag, not a bug in this driver. Each run's actual fast-layer weights state is
recorded honestly in the output JSON's `fast_layer_weights_status`/`trained_for_task` fields
(`Policy.weights_status`/`trained_for_task`, verbatim from whichever `FastLayer` was built).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.hybrid_policy import HybridQwenActPolicy, SmolVLADirectPolicy  # noqa: E402
from ml_jetson_vla.deployment.bench_action_models import refuse_overwrite, write_json_atomic  # noqa: E402
from ml_jetson_vla.experiments.fast_layers import (  # noqa: E402
    ACT_DEFAULT_CHUNK_LEN,
    ActFastLayer,
    SmolVLAFastLayer,
    StubFastLayer,
)
from ml_jetson_vla.experiments.run_experiment import (  # noqa: E402
    VideoRowReader,
    load_session_telemetry,
)
from ml_jetson_vla.experiments.schedule import (  # noqa: E402
    CONTROL_HZ,
    DEFAULT_MARGIN_MS,
    summarize_replans,
)

LOG: logging.Logger = logging.getLogger("run_policy_replay")
POLICIES: tuple[str, ...] = ("hybrid_qwen_act", "smolvla_direct")
FAST_LAYERS: tuple[str, ...] = ("stub", "act", "smolvla")
# How long to let the background worker threads catch up after the session's last tick, so a
# chunk already in flight gets a chance to land before stats are read -- not a hardware number.
DRAIN_S: float = 0.5


def build_policy(
    name: str,
    chunk_len: int,
    margin_ms: float,
    fast_layer: str = "stub",
    device: str = "cpu",
    act_checkpoint: Optional[str] = None,
) -> Any:
    if fast_layer == "act":
        layer: Any = ActFastLayer(chunk_len, device=device, checkpoint_dir=act_checkpoint)
    elif fast_layer == "smolvla":
        # SmolVLAFastLayer derives its own chunk_len from the pinned checkpoint's config, not the
        # CLI --chunk-len -- same mismatch run_experiment.py's build_fast_layer() already handles
        # by simply not passing chunk_len through for this option; followed here for consistency.
        layer = SmolVLAFastLayer(device=device)
    else:
        layer = StubFastLayer(value_deg=0.0, chunk_len=chunk_len)  # no real weights, default
    if name == "hybrid_qwen_act":
        return HybridQwenActPolicy(layer, margin_ms=margin_ms)  # grounding_backend=None: stub-only
    return SmolVLADirectPolicy(layer, margin_ms=margin_ms)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    session = os.path.abspath(args.session)
    tel = load_session_telemetry(session)
    tick_times = np.arange(0.0, args.max_sim_seconds, 1.0 / CONTROL_HZ)
    tick_rows = np.searchsorted(tel.t_s, tick_times, side="right") - 1
    if tick_rows.size == 0 or tick_rows[-1] < 1:
        raise ValueError("max-sim-seconds covers fewer than 2 telemetry rows")

    policy = build_policy(
        args.policy, args.chunk_len, args.margin_ms, args.fast_layer, args.device, args.act_checkpoint
    )
    reader = VideoRowReader(os.path.join(session, "rgb_video.mp4"))

    last_cmd = ""
    n_ticks = 0
    n_cmd_changes = 0
    n_frames_with_angles = 0
    n_frames_with_target = 0
    try:
        for tick_now, row_i in zip(tick_times, tick_rows):
            if row_i < 0:
                continue
            row = int(row_i)
            n_ticks += 1
            cmd = tel.commands[row]
            instruction: Optional[str] = None
            if cmd and cmd != last_cmd:
                last_cmd = cmd
                instruction = cmd  # raw vocabulary form -- Policy.act()'s documented convention;
                # core/hybrid_policy.py converts to natural text internally for the fast layer.
                n_cmd_changes += 1
            frame = reader.read(row)
            touch_mm = (float(tel.touch[row, 0]), float(tel.touch[row, 1]))
            command = policy.act(frame, instruction, {"touch_mm": touch_mm})
            if command.angle_targets_deg is not None:
                n_frames_with_angles += 1
            if bool(policy.last_debug.get("has_target", False)):
                n_frames_with_target += 1
            # The session clock advances far faster than CONTROL_HZ real time would -- give the
            # background worker a moment to actually run between ticks (real threads, not a
            # simulated clock like run_experiment.py's run_replay()).
            time.sleep(1.0 / CONTROL_HZ / args.speedup)
        if n_ticks > 0:
            time.sleep(DRAIN_S)
            final_command = policy.act(frame, None, {"touch_mm": touch_mm})
            if final_command.angle_targets_deg is not None:
                n_frames_with_angles += 1
    finally:
        reader.close()
        policy.close()

    records = policy.replan_records
    summary = summarize_replans(records) if records else None
    return {
        "mode": "replay",
        "sent": False,
        "status": "ok",
        "policy": args.policy,
        "session": session,
        "sim_seconds": float(args.max_sim_seconds),
        "fast_layer_weights_status": policy.weights_status,
        "trained_for_task": policy.trained_for_task,
        "timing_caveat": (
            "Session content replayed on a real background worker thread (core/hybrid_policy.py's "
            "_FastWorker), stepped at CONTROL_HZ/--speedup wall-clock sleeps between ticks -- a "
            "hybrid of session content and this machine's real thread scheduling, NOT a hardware "
            "measurement, and not comparable to run_experiment.py's inline-inference replay numbers."
        ),
        "commands": {"n_changes": n_cmd_changes},
        "n_ticks": n_ticks,
        "n_frames_with_target": n_frames_with_target,
        "n_frames_with_angle_output": n_frames_with_angles,
        "n_rejected_chunks_total": int(policy.last_debug.get("n_rejected_chunks", 0) or 0),
        "last_rejection_reason": policy.last_debug.get("last_rejection_reason"),
        "n_replans_landed": len(records),
        "summary": summary,
    }


def base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bench": "run_policy_replay",
        "policy": args.policy,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay driver for core/hybrid_policy.py's Policy wrappers (no sends).")
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--session", required=True, help="Track 4 session dir")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--force", action="store_true", help="allow overwriting --out")
    p.add_argument("--max-sim-seconds", type=float, default=20.0)
    p.add_argument("--chunk-len", type=int, default=ACT_DEFAULT_CHUNK_LEN)
    p.add_argument("--margin-ms", type=float, default=DEFAULT_MARGIN_MS)
    p.add_argument("--speedup", type=float, default=4.0, help="wall-clock sleep divisor between ticks")
    p.add_argument(
        "--fast-layer",
        choices=FAST_LAYERS,
        default="stub",
        help="stub (default, no weights) | act (ActFastLayer) | smolvla (SmolVLAFastLayer)",
    )
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--act-checkpoint", default=None, help="ACT trained checkpoint dir; default = random init")
    args = p.parse_args(argv)
    if args.max_sim_seconds <= 0 or args.chunk_len < 1 or args.speedup <= 0:
        p.error("--max-sim-seconds > 0, --chunk-len >= 1, --speedup > 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_path = os.path.abspath(args.out)
    try:
        refuse_overwrite(out_path, args.force)
    except FileExistsError as exc:
        print(f"[run_policy_replay] {exc}")
        return 2

    result = base_result(args)
    try:
        result.update(run_replay(args))
        code = 0
    except Exception as exc:  # recorded, never a fabricated number
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        code = 1
    write_json_atomic(out_path, result)
    print(f"[run_policy_replay] status={result['status']} wrote {out_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
