"""Track 4 (`docs/LARGE_VLA_RESEARCH_SPIKE.md`) data-collection recorder, wired into
`runtime/run_jetson_standalone.py`'s Phase-A loop via `--record-track4-session`.

Writes session-structured bronze data --
`data/01_bronze/session_jetson_track4_<timestamp>/{telemetry.csv,rgb_video.mp4}` --
matching the exact on-disk convention `ml_multimodal/data_processing/generate_vla_dataset.py`
already consumes (`session_*` glob, `frame_index`-synced CSV+MP4 pair). The
`session_jetson_track4_` prefix keeps these picked up by that same glob while staying
visually distinct from the older PID-firmware-collected `session_*` dirs described in
`ml_multimodal/docs/DATA_PIPELINE.md` Phase 1 -- a different collection mechanism (Track 1
running live on the Jetson) producing the same on-disk shape.

Per `docs/LARGE_VLA_RESEARCH_SPIKE.md`'s "Training-data strategy" section, this logs
Track 1's real closed-loop operation (the expert pipeline actually balancing the ball) as
the primary training set for Track 4's eventual model -- the same expert-pipeline-as-
teacher pattern `ml_multimodal/training/train_vla.py`'s Stage 1 BC already uses, just
sourced live from the Jetson instead of a pre-recorded laptop session.

Two independent pieces, deliberately NOT sharing a thread with `TouchTelemetryLogger`'s
existing serial-I/O thread and NOT opening a second reader on the serial port -- see
`src/touch_logger.py`'s own docstring for the real ~95-107 FPS -> ~20-25 FPS regression
recorded in `docs/PROJECT_LOGBOOK.md` (19/08) the one time two threads touched one serial
handle concurrently:

  - `SessionTouchTap` SUBCLASSES `TouchTelemetryLogger` (imported, not modified -- that
    file is outside this agent's write boundary) and overrides `_handle_line()` to cache
    the latest parsed touch/motor reading before delegating to the unmodified parent
    implementation. This runs on the SAME worker thread `TouchTelemetryLogger` already
    owns -- there is still exactly one thread that ever touches `self.ser`.
  - `SessionRecorder` owns a SEPARATE dedicated worker thread + `queue.Queue` for the
    per-frame video/CSV write (`cv2.VideoWriter.write()` + one CSV row) -- CPU/IO-bound
    work with nothing to do with serial I/O, which would only compete for the GIL against
    the 25Hz telemetry stream if it ran on that thread instead.

Main-loop cost per frame is one `queue.put()` of a frame copy plus a handful of scalars --
same "cheap enqueue, no I/O on the caller's thread" contract as
`TouchTelemetryLogger.send_frame()`.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from src.touch_logger import TouchTelemetryLogger

TELEMETRY_CSV_FIELDS = [
    "frame_index",
    "host_timestamp_ms",
    "target_x",
    "target_y",
    "touch_x",
    "touch_y",
    "theta_a",
    "theta_b",
    "theta_c",
    # Extra column beyond generate_vla_dataset.py's minimum 9-field schema -- this
    # kickoff's Task 3 (convert_to_lerobot.py) reads it as the REAL per-frame language
    # instruction instead of synthesizing one from target-coordinate thresholds the way
    # generate_vla_dataset.py currently does for its own (non-Jetson) source data.
    # generate_vla_dataset.py itself only reads the 9 columns above by name (pandas), so
    # this extra column is harmless to it.
    "audio_command",
]


class SessionTouchTap(TouchTelemetryLogger):
    """Drop-in replacement for `TouchTelemetryLogger` (identical constructor/CSV/counter
    behavior) that additionally caches the latest parsed `T,...` reading for synchronous,
    best-effort, non-blocking reads from the main loop. See module docstring for why this
    is a subclass tapping the existing thread rather than a second serial reader."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._latest_lock = threading.Lock()
        self._latest: Optional[dict] = None

    def _handle_line(self, raw: bytes) -> None:
        if raw.startswith(b"T,"):
            parts = raw.decode("ascii", errors="ignore").strip().split(",")
            if len(parts) == 11:
                try:
                    touch_x = int(parts[3]) / 100.0
                    touch_y = int(parts[4]) / 100.0
                    valid = int(parts[5])
                    mot_a, mot_b, mot_c = int(parts[8]), int(parts[9]), int(parts[10])
                except ValueError:
                    pass  # malformed -- parent's own parse_errors counter records this below
                else:
                    with self._latest_lock:
                        self._latest = {
                            "touch_x": touch_x,
                            "touch_y": touch_y,
                            "valid": bool(valid),
                            "motor_a": mot_a,
                            "motor_b": mot_b,
                            "motor_c": mot_c,
                            "recv_ts": time.perf_counter(),
                        }
        # Unmodified parent behavior -- CSV row, counters, seq matching -- runs exactly
        # as it would with a plain TouchTelemetryLogger.
        super()._handle_line(raw)

    def get_latest_touch(self) -> Optional[dict]:
        """Best-effort snapshot of the most recent 'T,...' reading (touch_x/y in mm,
        valid, motor_a/b/c in raw steps), or None if nothing has arrived yet this run.
        Deliberately NOT synchronized to the calling frame -- like Phase B's
        `TelemetryReader.last_steps` in `run_jetson_standalone.py`, the uplink runs at
        its own ~25Hz cadence, independent of the vision loop's rate."""
        with self._latest_lock:
            return dict(self._latest) if self._latest is not None else None


class SessionRecorder:
    """Dedicated background worker for Track 4 session capture. Owns `telemetry.csv` and
    `rgb_video.mp4` for exactly one session; construct one per `run_jetson_standalone.py`
    process. `frame_index` is assigned by the worker thread itself (not the caller), which
    is what guarantees the CSV row count and video frame count stay in lockstep even
    though frames arrive from the main loop at a jittery rate."""

    def __init__(self, bronze_root: str, fps: float = 30.0) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(bronze_root, f"session_jetson_track4_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.csv_path = os.path.join(self.session_dir, "telemetry.csv")
        self.video_path = os.path.join(self.session_dir, "rgb_video.mp4")
        self._fps = fps

        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._writer_video: Optional[cv2.VideoWriter] = None
        self._frame_index = 0
        self.frames_written = 0
        self.rows_written = 0
        self.dropped = 0

        self._fh = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._fh, fieldnames=TELEMETRY_CSV_FIELDS)
        self._csv_writer.writeheader()

        self._thread = threading.Thread(
            target=self._run, name="track4-session-recorder", daemon=True
        )
        self._thread.start()
        print(f"[track4-session] recording to {self.session_dir}")

    def log(
        self,
        frame_bgr: np.ndarray,
        host_timestamp_ms: int,
        target_x: float,
        target_y: float,
        touch_x: Optional[float],
        touch_y: Optional[float],
        theta_a: Optional[float],
        theta_b: Optional[float],
        theta_c: Optional[float],
        audio_command: Optional[str],
    ) -> None:
        """Main-loop call: copies the frame (the camera receiver reuses its buffer, so a
        copy is required here, not optional) and enqueues everything the worker needs.
        No I/O happens on the calling thread."""
        self._queue.put(
            (
                frame_bgr.copy(),
                host_timestamp_ms,
                target_x,
                target_y,
                touch_x,
                touch_y,
                theta_a,
                theta_b,
                theta_c,
                audio_command,
            )
        )

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            (
                frame_bgr,
                host_ts,
                tx,
                ty,
                touch_x,
                touch_y,
                theta_a,
                theta_b,
                theta_c,
                cmd,
            ) = item

            if self._writer_video is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer_video = cv2.VideoWriter(
                    self.video_path, fourcc, self._fps, (w, h)
                )
                if not self._writer_video.isOpened():
                    print(
                        f"[track4-session] ERROR: could not open VideoWriter for "
                        f"{self.video_path} -- frames will be dropped, CSV rows will "
                        f"NOT match a video."
                    )

            if self._writer_video is not None and self._writer_video.isOpened():
                self._writer_video.write(frame_bgr)
            else:
                self.dropped += 1
                continue  # never let the CSV and video frame counts diverge

            self._csv_writer.writerow(
                {
                    "frame_index": self._frame_index,
                    "host_timestamp_ms": host_ts,
                    "target_x": tx,
                    "target_y": ty,
                    "touch_x": touch_x if touch_x is not None else "",
                    "touch_y": touch_y if touch_y is not None else "",
                    "theta_a": theta_a if theta_a is not None else "",
                    "theta_b": theta_b if theta_b is not None else "",
                    "theta_c": theta_c if theta_c is not None else "",
                    "audio_command": cmd or "",
                }
            )
            self._frame_index += 1
            self.frames_written += 1
            self.rows_written += 1
            if self.rows_written % 30 == 0:
                self._fh.flush()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._writer_video is not None:
            self._writer_video.release()
        self._fh.flush()
        self._fh.close()
        print(
            f"[track4-session] {self.frames_written} frames written to "
            f"{self.session_dir} ({self.dropped} dropped)"
        )
