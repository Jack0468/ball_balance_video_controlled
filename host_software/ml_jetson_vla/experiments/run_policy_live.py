"""Live (real-serial-sends) driver for the real `Policy` wrappers in `core/hybrid_policy.py`
(`HybridQwenActPolicy`, `HybridSmallCnnActPolicy`, `SmolVLADirectPolicy`) -- the live sibling of
`run_policy_replay.py` (which sends nothing). Built 2026-10-13 to close the gap where
`run_experiment.py`'s `run_live()` was the ONLY live driver that existed, and it is Qwen-specific
and bypasses the `Policy` abstraction entirely (inline-constructs its own fast layer/grounder and
re-implements chunk scheduling/the safety gate/the serial-send loop itself). That script remains
the proven reference for every piece of real-hardware plumbing (camera capture via
`cv2.VideoCapture`, STM32 serial via `pyserial`, touch uplink parsing via
`experiments/serial_protocol.py`, the motor-consent gate, the `LEVEL_LINE`-on-exit safety
behaviour, the telemetry CSV shape) -- this module reuses all of that plumbing rather than
re-deriving it, and adds nothing new at the hardware-I/O level.

Why a new script rather than a `--mode live` on `run_policy_replay.py`: that module's docstring
and every one of its tests assert "Sends nothing" -- bolting live serial writes onto it would
either falsify that invariant for the replay path too (if shared code paths changed) or require
splitting its internals anyway. A sibling script, importing `run_policy_replay.build_policy()`
rather than re-deriving construction logic, keeps that module's "never sends" guarantee intact
and gives this one a single, obviously-named job (same reasoning `run_policy_replay.py` itself
used to justify being separate from `run_experiment.py`'s FastLayer-direct replay).

Why this needed almost none of `run_experiment.py`'s `run_live()`'s own chunk-scheduling/safety
logic re-implemented: `core/hybrid_policy.py`'s three `Policy` classes already run their own
background-threaded grounder + `_FastWorker` internally, already proven safe at the `Policy`
boundary (`PolicyCommand.angle_targets_deg` is `None` for any rejected/not-yet-landed chunk;
`_FastWorker.current_angles()` re-validates through `serial_protocol.format_angle_line()` as a
last gate -- see `test_hybrid_policy_cpu.py`'s `SafetyGateTests`). So each control tick here is
just: read camera frame, read current touch state, call `policy.act(...)`, and if
`angle_targets_deg is not None`, format+send it exactly as `run_live()` already does; otherwise
send nothing. No new safety logic was written -- it already existed at the `Policy` layer.

`--policy hybrid_qwen_act` / `smolvla_direct` here configure NO grounding backend, same as
`run_policy_replay.py`'s own `build_policy()` (Qwen's `MinimalVLMPolicy` is never constructed by
that function -- see its module docstring) -- so colour commands correctly produce no target for
those two policies on this driver too; that is `build_policy()`'s existing, already-documented
behaviour, not a new limitation introduced here. `--policy hybrid_small_cnn_act` is the one
exception: `build_policy()` constructs a real `SmallCnnGroundingPolicy` by default, so colour
commands DO produce real live targets for it (needs only `onnxruntime`+`cv2`, no Docker image).

`--homography-npy` is accepted and validated (3x3, loadable) for CLI parity with
`run_experiment.py --mode live` (same required-flag list, so this feels like a drop-in sibling,
not a different interface to relearn) -- it is currently NOT wired into any of the three
`Policy` classes via this driver, because `build_policy()` never constructs a Qwen grounding
backend here (same as replay). It is loaded and shape-checked anyway so a bad path is caught at
startup rather than silently ignored.

Written, not run on real hardware: no Jetson/serial/camera access from this environment.
Verified CPU-only via mocks (see `tests/test_run_policy_live_cpu.py`), same documented
constraint as every other piece of this task.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import sys
import threading
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

from ml_jetson_vla.deployment.bench_action_models import refuse_overwrite, write_json_atomic  # noqa: E402
from ml_jetson_vla.experiments.fast_layers import ACT_DEFAULT_CHUNK_LEN  # noqa: E402
from ml_jetson_vla.experiments.run_experiment import (  # noqa: E402
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    MOTOR_GATE_FLAG,
)
from ml_jetson_vla.experiments.run_policy_replay import (  # noqa: E402
    FAST_LAYERS,
    POLICIES,
    build_policy,
)
from ml_jetson_vla.experiments.schedule import (  # noqa: E402
    CONTROL_HZ,
    DEFAULT_MARGIN_MS,
    summarize_replans,
)
from ml_jetson_vla.experiments.serial_protocol import (  # noqa: E402
    LEVEL_LINE,
    format_angle_line,
    parse_uplink_line,
)

LOG: logging.Logger = logging.getLogger("run_policy_live")


def _describe_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    # Written, not exercised on real hardware in this task. One thread (serial uplink reader);
    # the policy itself owns its own background grounder/fast-layer threads (core/hybrid_policy.py).
    # The control-tick loop here never blocks on the policy: policy.act() itself never blocks on
    # either of its internal workers (see core/hybrid_policy.py's module docstring).
    import cv2
    import serial

    homography = np.load(args.homography_npy)
    if homography.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {homography.shape}")

    policy = build_policy(
        args.policy, args.chunk_len, args.margin_ms, args.fast_layer, args.device, args.act_checkpoint,
        args.small_cnn_grounding, args.small_cnn_mask_threshold, args.small_cnn_marker_manifest,
    )

    ser = serial.Serial(args.serial_port, args.baud, timeout=0.1)
    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        ser.close()
        policy.close()
        raise RuntimeError(f"cannot open camera index {args.camera_index}")

    lock = threading.Lock()
    touch_state: dict[str, Any] = {"xy": None, "valid_samples": 0, "samples": 0}
    stop = threading.Event()
    worker_errors: list[str] = []

    def uplink() -> None:
        while not stop.is_set():
            try:
                raw = ser.readline().decode("ascii", errors="replace")
            except Exception as exc:
                worker_errors.append(_describe_exc(exc))
                return
            sample = parse_uplink_line(raw)
            if sample is None:
                continue
            with lock:
                touch_state["samples"] += 1
                if sample.valid:
                    touch_state["valid_samples"] += 1
                    touch_state["xy"] = (sample.touch_x_mm, sample.touch_y_mm)

    uplink_thread = threading.Thread(target=uplink, daemon=True)
    uplink_thread.start()

    period = 1.0 / CONTROL_HZ
    n_ticks = 0
    n_sent = 0
    n_ticks_with_target = 0
    n_replan_submitted = 0  # diagnostic (2026-10-13): whether submit_if_due() ever actually fired is
    # otherwise invisible in the output JSON -- real ACT checkpoints showed 0 landed chunks over 60s
    # live with abundant valid target+touch ticks, and a standalone forward-pass timing test proved
    # the model itself returns in 0.34s, so the open question is specifically whether submission into
    # _FastWorker ever happened at all (0 here) vs. happened but never landed (>0 here).
    stop_reason = "max_run_s"
    last_frame: Optional[np.ndarray] = None
    last_angles = np.zeros(3)
    # args.instruction is passed on EVERY tick, not just once: core/hybrid_policy.py's
    # HybridQwenActPolicy/HybridSmallCnnActPolicy only resolve a new instruction into a target
    # when `instruction != self._last_instruction AND touch_mm is not None` at that SAME call
    # (see their act()) -- touch samples arrive asynchronously from the uplink thread and may not
    # be available yet on an early tick. Passing the instruction once and then None would let that
    # one attempt silently fail with no touch yet and never retry once touch does arrive. Repeating
    # the same string every tick is cheap (a dict lookup + string compare) and lets the policy's
    # own `_last_instruction` dedup fire exactly once, on whichever tick first has touch available.
    telemetry_rows: list[list[Any]] = []  # columns match evaluate_system_control.py's REQUIRED_COLUMNS
    t_start = time.monotonic()
    next_tick = t_start
    try:
        while True:
            now = time.monotonic()
            if now - t_start >= args.max_run_s:
                break
            if worker_errors:
                stop_reason = f"worker_error: {worker_errors[0]}"
                break
            # getattr, not a direct attribute access: `worker_errors` is not part of the formal
            # Policy protocol (core/policy_interface.py) -- only the hybrid_policy.py classes
            # define it. A minimal/mock Policy (e.g. tests' _FakePolicy) has none, and should not
            # be forced to grow one just to be driven by this script.
            policy_worker_errors = getattr(policy, "worker_errors", [])
            if policy_worker_errors:
                # Real incident, 2026-10-13: a dead _FastWorker (chunk-size mismatch between a
                # trained checkpoint and --chunk-len) went completely unnoticed here for three
                # full live-hardware runs (60s/30s/30s, zero chunks ever landing) because nothing
                # checked this before. See HybridQwenActPolicy.worker_errors's docstring.
                stop_reason = f"policy_worker_error: {policy_worker_errors[0]}"
                break

            ok, frame = cam.read()
            if ok:
                last_frame = frame
            n_ticks += 1

            with lock:
                xy = touch_state["xy"]

            tx_mm, ty_mm = float("nan"), float("nan")
            if last_frame is not None:
                command = policy.act(
                    last_frame, args.instruction, {"touch_mm": xy} if xy is not None else {}
                )
                if bool(policy.last_debug.get("has_target", False)):
                    n_ticks_with_target += 1
                if bool(policy.last_debug.get("replan_submitted", False)):
                    n_replan_submitted += 1
                tx_mm, ty_mm = command.target_x_mm, command.target_y_mm
                if command.angle_targets_deg is not None:
                    line = format_angle_line(*command.angle_targets_deg)
                    ser.write((line + "\n").encode("ascii"))
                    n_sent += 1
                    last_angles = np.asarray(command.angle_targets_deg, dtype=np.float64)

            with lock:
                t_xy = touch_state["xy"]
            tch = t_xy if t_xy is not None else (float("nan"), float("nan"))
            telemetry_rows.append([
                int(time.time() * 1000), tx_mm, ty_mm, tch[0], tch[1],
                last_angles[0], last_angles[1], last_angles[2], args.instruction,
            ])

            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        try:
            ser.write((LEVEL_LINE + "\n").encode("ascii"))  # level the plate on exit, same as run_experiment.py's run_live()
        except Exception as exc:
            LOG.error("could not send level line on exit: %s", exc)
        stop.set()
        uplink_thread.join(timeout=2.0)
        cam.release()
        ser.close()
        policy.close()

    telemetry_path = os.path.abspath(args.out) + ".telemetry.csv"
    with open(telemetry_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["host_timestamp_ms", "target_x", "target_y", "touch_x", "touch_y",
                         "theta_a", "theta_b", "theta_c", "audio_command"])
        writer.writerows(telemetry_rows)

    records = policy.replan_records
    summary = summarize_replans(records) if records else None
    with lock:
        touch_info = {"samples": touch_state["samples"], "valid_samples": touch_state["valid_samples"]}
    return {
        "mode": "live",
        "sent": True,
        "status": "ok",
        "policy": args.policy,
        "instruction": args.instruction,
        "fast_layer_weights_status": policy.weights_status,
        "trained_for_task": policy.trained_for_task,
        "stop_reason": stop_reason,
        "lines_sent": n_sent,
        "n_ticks": n_ticks,
        "n_ticks_with_target": n_ticks_with_target,
        "n_replan_submitted": n_replan_submitted,
        "n_rejected_chunks_total": int(policy.last_debug.get("n_rejected_chunks", 0) or 0),
        "last_rejection_reason": policy.last_debug.get("last_rejection_reason"),
        "touch_uplink": touch_info,
        "telemetry_csv": telemetry_path,
        "telemetry_rows": len(telemetry_rows),
        "summary": summary,
        "replans": [dataclasses.asdict(r) for r in records],
    }


def base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bench": "run_policy_live",
        "policy": args.policy,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live driver for core/hybrid_policy.py's Policy wrappers -- real serial sends, gated."
    )
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--force", action="store_true", help="allow overwriting --out")
    # policy/fast-layer construction -- same flags/defaults as run_policy_replay.py's build_policy()
    p.add_argument("--chunk-len", type=int, default=ACT_DEFAULT_CHUNK_LEN)
    p.add_argument("--margin-ms", type=float, default=DEFAULT_MARGIN_MS)
    p.add_argument(
        "--fast-layer", choices=FAST_LAYERS, default="stub",
        help="stub (default, no weights) | act (ActFastLayer) | smolvla (SmolVLAFastLayer)",
    )
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--act-checkpoint", default=None, help="ACT trained checkpoint dir; default = random init")
    p.add_argument(
        "--no-small-cnn-grounding", dest="small_cnn_grounding", action="store_false",
        help="--policy hybrid_small_cnn_act only: disable the real small-CNN grounder. On by default otherwise.",
    )
    p.add_argument(
        "--small-cnn-mask-threshold", type=float, default=0.5,
        help="--policy hybrid_small_cnn_act only: MarkerClassifier mask threshold (default 0.5).",
    )
    p.add_argument(
        "--small-cnn-marker-manifest", default=None,
        help="--policy hybrid_small_cnn_act only: path to the per-sheet marker colour/shape manifest "
             "(hardware/platform_templates/aruco_markers_0N_manifest.json) for the 2026-10-13 "
             "manifest-position colour-ID fix. Defaults to aruco_markers_01 (the sheet confirmed "
             "physically mounted as of 2026-10-13) -- override if a different sheet is mounted.",
    )
    # live (gated) -- same flags/defaults as run_experiment.py --mode live
    p.add_argument(MOTOR_GATE_FLAG, dest="i_understand_this_drives_motors", action="store_true")
    p.add_argument("--serial-port", required=True)
    p.add_argument("--baud", type=int, default=2000000)  # AngleStepControl firmware SERIAL_BAUD
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--homography-npy", required=True, help="3x3 mm->px homography, saved with numpy")
    p.add_argument("--instruction", required=True, help="one command for the whole live run, e.g. go_green")
    p.add_argument("--max-run-s", type=float, default=10.0)
    args = p.parse_args(argv)
    if args.chunk_len < 1 or args.max_run_s <= 0:
        p.error("--chunk-len >= 1, --max-run-s > 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_path = os.path.abspath(args.out)
    try:
        refuse_overwrite(out_path, args.force)
    except FileExistsError as exc:
        print(f"[run_policy_live] {exc}")
        return EXIT_USAGE
    if not args.i_understand_this_drives_motors:
        print(f"[run_policy_live] refusing: live mode drives motors; pass {MOTOR_GATE_FLAG}")
        return EXIT_USAGE

    result = base_result(args)
    try:
        result.update(run_live(args))
        code = EXIT_OK
    except Exception as exc:  # recorded, never a fabricated number
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        code = EXIT_FAILED
    write_json_atomic(out_path, result)
    print(f"[run_policy_live] status={result['status']} wrote {out_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
