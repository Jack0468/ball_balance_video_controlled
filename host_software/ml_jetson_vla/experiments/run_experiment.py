"""CLI for the Arm-2 experiment options. Plan: docs/EXPERIMENT_OPTIONS_PLAN_2026_10_07.md.

--mode replay: feeds one logged Track 4 session (telemetry.csv + rgb_video.mp4) through an option and
  SENDS NOTHING. Inference is real and measured on this machine. The timeline is the session clock,
  advanced by the measured inference time, so the schedule and staleness numbers model the async loop;
  they are not a hardware measurement. Marker targets are the logged telemetry target after a simulated
  grounding latency (no Qwen run). Use --device cuda on the Jetson only in a later, separate run.
--mode live: drives the STM32 over serial. Gated behind --i-understand-this-drives-motors. Not run in
  this task. Requires --instruction (the audio classifier is not wired in here).

Options: act_fast_statemachine (ACT_FAST + option S), act_fast_llm (ACT_FAST + option L, refuses),
smolvla_direct (SMOLVLA_DIRECT, instruction text goes to SmolVLA).
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import logging
import os
import queue
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
from ml_jetson_vla.experiments.fast_layers import (  # noqa: E402
    ACT_DEFAULT_CHUNK_LEN,
    ActFastLayer,
    FastLayer,
    SmolVLAFastLayer,
    StubFastLayer,
)
from ml_jetson_vla.experiments.orchestrators import (  # noqa: E402
    LanguageOrchestrator,
    StateMachineOrchestrator,
    TargetOrchestrator,
    TargetRequest,
)
from ml_jetson_vla.experiments.schedule import (  # noqa: E402
    CONTROL_HZ,
    DEFAULT_MARGIN_MS,
    ChunkPlan,
    ChunkScheduler,
    ReplanRecord,
    summarize_replans,
    target_age_ms,
)
from ml_jetson_vla.experiments.serial_protocol import (  # noqa: E402
    LEVEL_LINE,
    format_angle_line,
    parse_uplink_line,
)

LOG: logging.Logger = logging.getLogger("run_experiment")
OPTIONS: tuple[str, ...] = ("act_fast_statemachine", "act_fast_llm", "smolvla_direct")
MOTOR_GATE_FLAG: str = "--i-understand-this-drives-motors"
DEFAULT_GROUNDING_LATENCY_MS: float = 1700.0  # Qwen2.5-VL-3B per-call figure, MULTI_HEAD_ARCHITECTURE_SPEC risk 6
EXIT_OK: int = 0
EXIT_FAILED: int = 1
EXIT_USAGE: int = 2
EXIT_REFUSED: int = 3


def command_to_text(command: str) -> str:
    return command.replace("_", " ")


def make_orchestrator(option: str) -> Optional[TargetOrchestrator]:
    if option == "act_fast_statemachine":
        return StateMachineOrchestrator()
    if option == "act_fast_llm":
        orch = LanguageOrchestrator()
        orch.ensure_available()  # raises NotImplementedError before any model is loaded
        return orch
    return None  # smolvla_direct: no target orchestrator


def build_fast_layer(
    option: str, device: str, chunk_len: int, checkpoint_dir: Optional[str], stub_value: Optional[float] = None
) -> FastLayer:
    if stub_value is not None:
        return StubFastLayer(value_deg=stub_value, chunk_len=chunk_len)
    if option == "smolvla_direct":
        return SmolVLAFastLayer(device=device)
    return ActFastLayer(chunk_len=chunk_len, device=device, checkpoint_dir=checkpoint_dir)


class VideoRowReader:
    """Sequential reader. Rows must be requested in increasing order (replay time only moves forward)."""

    def __init__(self, path: str) -> None:
        import cv2

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self._row: int = -1
        self._frame: Optional[np.ndarray] = None

    def read(self, row: int) -> np.ndarray:
        if row < self._row:
            raise ValueError("rows must be requested in order")
        while self._row < row:
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"video ended before telemetry row {row}")
            self._row += 1
            self._frame = frame
        assert self._frame is not None
        return self._frame

    def close(self) -> None:
        self._cap.release()


@dataclasses.dataclass(frozen=True)
class SessionTelemetry:
    """One logged Track 4 session's replay-relevant columns. Factored out of `run_replay()`
    (2026-10-07) so `core/hybrid_policy.py`'s own replay driver
    (`run_policy_replay.py`) can reuse the same loading instead of re-deriving it."""

    t_s: np.ndarray
    touch: np.ndarray  # kept as logged, stale repeats included
    logged_target: np.ndarray
    commands: np.ndarray


def load_session_telemetry(session: str) -> SessionTelemetry:
    import pandas as pd

    df = pd.read_csv(os.path.join(session, "telemetry.csv"))
    t_s = (df["host_timestamp_ms"].to_numpy(dtype=np.float64) - float(df["host_timestamp_ms"].iloc[0])) / 1000.0
    touch = df[["touch_x", "touch_y"]].to_numpy(dtype=np.float64)
    logged_target = df[["target_x", "target_y"]].to_numpy(dtype=np.float64)
    commands = df["audio_command"].fillna("").astype(str).to_numpy()
    return SessionTelemetry(t_s=t_s, touch=touch, logged_target=logged_target, commands=commands)


@dataclasses.dataclass
class _PendingReplan:
    plan: ChunkPlan
    land_s: float
    target_age: Optional[float]
    new_target_arrived: bool
    row: int
    target_xy: tuple[float, float]
    has_target: bool


def _plan_detail(p: _PendingReplan, rec: ReplanRecord) -> dict[str, Any]:
    s = p.plan.safety
    return {
        **dataclasses.asdict(rec),
        "row": p.row,
        "has_target": p.has_target,
        "target_mm": [p.target_xy[0], p.target_xy[1]],
        "n_steps": s.n_steps,
        "n_violations": s.n_violations,
        "max_abs_deg": s.max_abs_deg,
        "reason": s.reason,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    session = os.path.abspath(args.session)
    tel = load_session_telemetry(session)
    t_s, touch, logged_target, commands = tel.t_s, tel.touch, tel.logged_target, tel.commands
    # Replay steps on a CONTROL_HZ grid, not on telemetry rows (rows here arrive ~24 Hz, and a
    # row-stepped check would itself create stalls wider than the margin).
    tick_times = np.arange(0.0, args.max_sim_seconds, 1.0 / CONTROL_HZ)
    tick_rows = np.searchsorted(t_s, tick_times, side="right") - 1
    if tick_rows.size == 0 or tick_rows[-1] < 1:
        raise ValueError("max-sim-seconds covers fewer than 2 telemetry rows")
    end_row = int(tick_rows[-1]) + 1

    orch = make_orchestrator(args.option)
    layer = build_fast_layer(args.option, args.device, args.chunk_len, args.checkpoint, args.stub_value)
    sched = ChunkScheduler(CONTROL_HZ, args.margin_ms)
    reader = VideoRowReader(os.path.join(session, "rgb_video.mp4"))

    grounding_arrivals: collections.deque[float] = collections.deque()
    target_xy: tuple[float, float] = (0.0, 0.0)
    has_target = False
    target_done_s = 0.0
    used_done_s = -float("inf")
    pending: Optional[_PendingReplan] = None
    last_cmd = ""
    instruction: Optional[str] = None
    n_cmd_changes = 0
    n_fixed = 0
    n_grounding_requests = 0
    skipped_triggers = 0
    details: list[dict[str, Any]] = []

    for tick_now, row_i in zip(tick_times, tick_rows):
        if row_i < 0:
            continue
        row = int(row_i)
        now = float(tick_now)
        cmd = commands[row]
        if cmd and cmd != last_cmd:
            last_cmd = cmd
            instruction = command_to_text(cmd)
            n_cmd_changes += 1
            grounding_arrivals.clear()  # a new command supersedes any grounding still in flight
            if orch is not None:
                req: TargetRequest = orch.resolve(cmd, reader.read(row), touch[row])
                if req.marker_label is not None:
                    n_grounding_requests += 1
                    grounding_arrivals.append(now + args.grounding_latency_ms / 1000.0)
                else:
                    assert req.fixed_target_mm is not None
                    n_fixed += 1
                    target_xy = req.fixed_target_mm
                    has_target = True
                    target_done_s = now

        while grounding_arrivals and grounding_arrivals[0] <= now:
            grounding_arrivals.popleft()
            target_xy = (float(logged_target[row, 0]), float(logged_target[row, 1]))
            has_target = True
            target_done_s = now

        if pending is not None and now >= pending.land_s:
            rec = sched.land(pending.land_s, pending.plan, pending.target_age, pending.new_target_arrived)
            details.append(_plan_detail(pending, rec))
            pending = None

        if pending is None and sched.should_replan(now):
            state_ok = bool(np.all(np.isfinite(touch[row])))
            if instruction is None or not state_ok:
                skipped_triggers += 1
                continue
            age = target_age_ms(target_done_s, now) if has_target else None
            new_arrived = has_target and target_done_s > used_done_s
            frame = reader.read(row)
            sched.begin_inference(now)
            plan = layer.plan(target_xy, (float(touch[row, 0]), float(touch[row, 1])), frame, instruction=instruction)
            if has_target:
                used_done_s = max(used_done_s, target_done_s)
            pending = _PendingReplan(
                plan=plan,
                land_s=now + plan.inference_ms / 1000.0,
                target_age=age,
                new_target_arrived=new_arrived,
                row=row,
                target_xy=target_xy,
                has_target=has_target,
            )
            LOG.info("replan at row %d (t=%.3fs) inference %.1f ms accepted=%s",
                     row, now, plan.inference_ms, plan.accepted)

    reader.close()
    summary = summarize_replans(sched.records) if sched.records else None
    total_violations = sum(d["n_violations"] for d in details)
    return {
        "mode": "replay",
        "sent": False,
        "status": "ok",
        "session": session,
        "rows_used": end_row,
        "sim_seconds": float(args.max_sim_seconds),
        "fast_layer": {
            "name": layer.name,
            "weights_status": layer.weights_status,
            "device": args.device,
            "chunk_len": args.chunk_len,
            "margin_ms": args.margin_ms,
        },
        "timing_caveat": (
            "Inference is measured on this machine (CPU unless --device cuda). NOT Jetson Orin numbers. "
            "Timeline is the session clock advanced by measured inference time."
        ),
        "grounding_simulation": {
            "applies_to": "marker commands only",
            "latency_ms": args.grounding_latency_ms,
            "value_source": "logged telemetry target at the arrival row (no Qwen run)",
            "n_marker_commands": n_grounding_requests,
        } if args.option == "act_fast_statemachine" else None,
        "commands": {"n_changes": n_cmd_changes, "n_fixed_actions": n_fixed},
        "skipped_triggers": skipped_triggers,
        "n_replans_landed": len(details),
        "n_violations_total": total_violations,
        "summary": summary,
        "replans": details,
    }


def _describe_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    # Written, not exercised in this task. Threads: serial uplink reader, grounding worker (marker
    # commands only), fast-layer worker. The 30 Hz send loop only writes accepted steps. It never blocks on inference.
    import serial

    import cv2

    from ml_jetson_vla.core.minimal_vlm_policy import MinimalVLMPolicy
    from ml_jetson_vla.deployment.bench_hybrid_qwen_act import TargetStore
    from ml_jetson_vla.deployment.bench_action_models import pixel_to_telemetry_mm
    from ml_jetson_vla.deployment.colab_sweep import QWEN_LOCAL_COMMIT, build_backend, default_candidate_specs

    if args.instruction is None:
        raise ValueError("live mode needs --instruction (audio classifier is not wired in this task)")
    orch = make_orchestrator(args.option)
    assert orch is not None or args.option == "smolvla_direct"
    homography = np.load(args.homography_npy)
    if homography.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {homography.shape}")

    needs_grounding = args.option == "act_fast_statemachine" and args.instruction in ("go_red", "go_green", "go_yellow", "go_black")
    grounding_policy: Optional[MinimalVLMPolicy] = None
    if needs_grounding:
        spec = default_candidate_specs(max_new_tokens=args.max_new_tokens)[0]
        if spec.kwargs.get("revision") != QWEN_LOCAL_COMMIT:
            raise RuntimeError("sweep's first candidate is not the pinned Qwen spec")
        backend = build_backend(spec, args.max_new_tokens)
        backend.load()
        grounding_policy = MinimalVLMPolicy(backend, prompt_variant="baseline")
    layer = build_fast_layer(args.option, args.device, args.chunk_len, args.checkpoint, args.stub_value)

    ser = serial.Serial(args.serial_port, args.baud, timeout=0.1)
    cam = cv2.VideoCapture(args.camera_index)
    store = TargetStore()
    lock = threading.Lock()
    touch_state: dict[str, Any] = {"xy": None, "valid_samples": 0, "samples": 0}
    stop = threading.Event()

    def uplink() -> None:
        while not stop.is_set():
            raw = ser.readline().decode("ascii", errors="replace")
            sample = parse_uplink_line(raw)
            if sample is None:
                continue
            with lock:
                touch_state["samples"] += 1
                if sample.valid:
                    touch_state["valid_samples"] += 1
                    touch_state["xy"] = (sample.touch_x_mm, sample.touch_y_mm)

    ground_q: queue.Queue[tuple[np.ndarray, str]] = queue.Queue(maxsize=1)
    fast_q: queue.Queue[tuple[np.ndarray, tuple[float, float], tuple[float, float], str, Optional[float], bool]] = queue.Queue(maxsize=1)
    fast_out: queue.Queue[tuple[ChunkPlan, Optional[float], bool]] = queue.Queue()
    worker_errors: list[str] = []

    def grounder() -> None:
        assert grounding_policy is not None
        while not stop.is_set():
            try:
                frame, text = ground_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                t0 = time.monotonic()
                grounding_policy.act(frame, text, state={})
                latency = time.monotonic() - t0
                dbg = grounding_policy.last_debug
                if not dbg.get("parse_ok"):
                    store.record_parse_failure(latency)
                    continue
                px, py = dbg["target_point_px"]
                x_mm, y_mm = pixel_to_telemetry_mm(
                    homography, px, py, dbg["coord_space"], dbg.get("model_input_hw"), frame.shape[:2]
                )
                store.put(x_mm, y_mm, "live", latency)
            except Exception as exc:
                worker_errors.append(_describe_exc(exc))
                return

    def fast_worker() -> None:
        while not stop.is_set():
            try:
                frame, target, state, text, age, new_arrived = fast_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                plan = layer.plan(target, state, frame, instruction=text)
                fast_out.put((plan, age, new_arrived))
            except Exception as exc:
                worker_errors.append(_describe_exc(exc))
                return

    threads = [threading.Thread(target=uplink, daemon=True), threading.Thread(target=fast_worker, daemon=True)]
    if grounding_policy is not None:
        threads.append(threading.Thread(target=grounder, daemon=True))
    for t in threads:
        t.start()

    sched = ChunkScheduler(CONTROL_HZ, args.margin_ms)
    period = 1.0 / CONTROL_HZ
    active_actions: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None
    fixed_target: Optional[tuple[float, float]] = None
    grounding_requested = False
    replan_pending = False
    used_done = -float("inf")
    n_sent = 0
    n_rejected = 0
    consecutive_rejects = 0
    stop_reason = "max_run_s"
    rejected_chunks: list[dict[str, Any]] = []
    last_angles = np.zeros(3)
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
            ok, frame = cam.read()
            if ok:
                last_frame = frame
            with lock:
                xy = touch_state["xy"]

            if orch is not None and fixed_target is None and not grounding_requested and xy and last_frame is not None:
                req = orch.resolve(args.instruction, last_frame, np.asarray(xy))
                if req.marker_label is not None:
                    grounding_requested = True
                    try:
                        # MinimalVLMPolicy.build_prompt() matches on the raw instruction vocabulary
                        # ("go_red"), not the grounding label ("red marker") -- queue req.instruction,
                        # matching core/hybrid_policy.py's _Grounder, which already does this correctly.
                        ground_q.put_nowait((last_frame.copy(), req.instruction))
                    except queue.Full:
                        pass
                else:
                    fixed_target = req.fixed_target_mm
            elif orch is None and last_frame is not None:
                fixed_target = (0.0, 0.0)  # smolvla_direct: no target input in the wiring

            latest = store.latest()
            if fixed_target is not None:
                target, target_done = fixed_target, now
            elif latest is not None:
                target, target_done = (latest.x_mm, latest.y_mm), latest.done_monotonic_s
            else:
                target, target_done = None, None

            while True:
                try:
                    plan, age_at_trigger, new_arr = fast_out.get_nowait()
                except queue.Empty:
                    break
                rec = sched.land(time.monotonic(), plan, age_at_trigger, new_arr)
                replan_pending = False
                if plan.accepted:
                    active_actions = plan.actions_deg
                else:
                    n_rejected += 1
                    consecutive_rejects += 1
                    active_actions = None
                    rejected_chunks.append({"record": dataclasses.asdict(rec), "reason": plan.safety.reason})
                    if consecutive_rejects >= args.max_consecutive_rejects:
                        stop_reason = "consecutive_rejects"
                        break
                if plan.accepted:
                    consecutive_rejects = 0
                    if rec.new_target_arrived and target_done is not None:
                        used_done = max(used_done, target_done)
            if stop_reason == "consecutive_rejects":
                break

            if (not replan_pending and target is not None and xy and last_frame is not None
                    and sched.should_replan(now)):
                age = target_age_ms(target_done, now) if target_done is not None else None
                new_arrived = target_done is not None and target_done > used_done
                sched.begin_inference(now)
                replan_pending = True
                try:
                    fast_q.put_nowait((last_frame.copy(), target, xy, command_to_text(args.instruction), age, new_arrived))
                except queue.Full:
                    pass

            if active_actions is not None and sched.chunk_start_s is not None:
                step = int(np.floor((now - sched.chunk_start_s) * CONTROL_HZ))
                if 0 <= step < active_actions.shape[0]:
                    line = format_angle_line(*active_actions[step])
                    ser.write((line + "\n").encode("ascii"))
                    n_sent += 1
                    last_angles = np.asarray(active_actions[step], dtype=np.float64)
            with lock:
                t_xy = touch_state["xy"]
            tx, ty = (target[0], target[1]) if target is not None else (float("nan"), float("nan"))
            tch = t_xy if t_xy is not None else (float("nan"), float("nan"))
            telemetry_rows.append([int(time.time() * 1000), tx, ty, tch[0], tch[1],
                                   last_angles[0], last_angles[1], last_angles[2], args.instruction])

            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        try:
            ser.write((LEVEL_LINE + "\n").encode("ascii"))  # level the plate on exit, same as the firmware's stale-host policy
        except Exception as exc:
            LOG.error("could not send level line on exit: %s", exc)
        stop.set()
        cam.release()
        ser.close()

    telemetry_path = os.path.abspath(args.out) + ".telemetry.csv"
    with open(telemetry_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["host_timestamp_ms", "target_x", "target_y", "touch_x", "touch_y",
                         "theta_a", "theta_b", "theta_c", "audio_command"])
        writer.writerows(telemetry_rows)
    summary = summarize_replans(sched.records) if sched.records else None
    with lock:
        touch_info = {"samples": touch_state["samples"], "valid_samples": touch_state["valid_samples"]}
    return {
        "mode": "live",
        "sent": True,
        "status": "ok",
        "option": args.option,
        "instruction": args.instruction,
        "fast_layer": {"name": layer.name, "weights_status": layer.weights_status, "device": args.device,
                       "chunk_len": args.chunk_len, "margin_ms": args.margin_ms},
        "stop_reason": stop_reason,
        "lines_sent": n_sent,
        "n_rejected_chunks": n_rejected,
        "rejected_chunks": rejected_chunks,
        "touch_uplink": touch_info,
        "telemetry_csv": telemetry_path,
        "telemetry_rows": len(telemetry_rows),
        "summary": summary,
        "replans": [dataclasses.asdict(r) for r in sched.records],
    }


def base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bench": "run_experiment",
        "option": args.option,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arm-2 experiment options: replay (no sends) or gated live.")
    p.add_argument("--option", required=True, choices=OPTIONS)
    p.add_argument("--mode", required=True, choices=["replay", "live"])
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--force", action="store_true", help="allow overwriting --out")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--chunk-len", type=int, default=ACT_DEFAULT_CHUNK_LEN)
    p.add_argument("--margin-ms", type=float, default=DEFAULT_MARGIN_MS)
    p.add_argument("--checkpoint", default=None, help="ACT trained checkpoint dir; default = random init")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--stub-value", type=float, default=None,
                   help="replay/live with NO model: constant chunk of this many degrees (schedule/safety test only)")
    # replay
    p.add_argument("--session", default=None, help="Track 4 session dir (replay)")
    p.add_argument("--max-sim-seconds", type=float, default=20.0)
    p.add_argument("--grounding-latency-ms", type=float, default=DEFAULT_GROUNDING_LATENCY_MS)
    # live (gated)
    p.add_argument(MOTOR_GATE_FLAG, dest="i_understand_this_drives_motors", action="store_true")
    p.add_argument("--serial-port", default=None)
    p.add_argument("--baud", type=int, default=2000000)  # AngleStepControl firmware SERIAL_BAUD
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--homography-npy", default=None, help="3x3 mm->px homography, saved with numpy")
    p.add_argument("--instruction", default=None, help="one command for the whole live run, e.g. go_green")
    p.add_argument("--max-run-s", type=float, default=10.0)
    p.add_argument("--max-consecutive-rejects", type=int, default=3)
    args = p.parse_args(argv)
    if args.mode == "replay" and args.session is None:
        p.error("--mode replay needs --session")
    if args.mode == "live":
        missing = [n for n in ("serial_port", "homography_npy", "instruction") if getattr(args, n) is None]
        if missing:
            p.error(f"--mode live needs {missing}")
    if args.chunk_len < 1 or args.max_sim_seconds <= 0 or args.max_run_s <= 0:
        p.error("--chunk-len >= 1, --max-sim-seconds > 0, --max-run-s > 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_path = os.path.abspath(args.out)
    try:
        refuse_overwrite(out_path, args.force)
    except FileExistsError as exc:
        print(f"[run_experiment] {exc}")
        return EXIT_USAGE
    if args.mode == "live" and not args.i_understand_this_drives_motors:
        print(f"[run_experiment] refusing: live mode drives motors; pass {MOTOR_GATE_FLAG}")
        return EXIT_USAGE

    result = base_result(args)
    try:
        result.update(run_replay(args) if args.mode == "replay" else run_live(args))
        code = EXIT_OK
    except NotImplementedError as exc:
        result["status"] = "refused"
        result["error"] = str(exc)
        code = EXIT_REFUSED
    except Exception as exc:  # recorded, never a fabricated number
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        code = EXIT_FAILED
    write_json_atomic(out_path, result)
    print(f"[run_experiment] status={result['status']} wrote {out_path}")
    if result["status"] == "refused":
        print(result["error"])
    return code


if __name__ == "__main__":
    sys.exit(main())
