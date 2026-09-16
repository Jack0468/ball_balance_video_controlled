"""Converts Track 4 bronze sessions (`runtime/session_recorder.py`'s
`session_jetson_track4_<timestamp>/{telemetry.csv,rgb_video.mp4}`) into a LeRobot-format
dataset for fine-tuning the Track 4 large-VLA candidate
(`docs/LARGE_VLA_RESEARCH_SPIKE.md`'s 2026-09-15 revision: SmolVLA, primary candidate).

**2026-09-17 extension (large-model fine-tuning pipeline plan, Stage 1):** also ingests the
5 usable PID/laptop-webcam sessions (`session_YYYYMMDD_HHMMSS/`, `collect_webcam_data.py`,
classical PID firmware) as a second, explicitly-tagged regime, confirmed as the schema
`qwen_multihead_policy.py` (`docs/MULTI_HEAD_ARCHITECTURE_SPEC.md` S2.1/S4) is built against
-- that module's `state_dim=2` and its whole data-flow section are written directly against
THIS script's `observation.state`/`action`/`task` feature names, not
`generate_vla_dataset.py`'s separate flat-JSON schema. Per the plan's Stage 0 finding (also
independently confirmed here by reading `session_manifest.json` and each regime's real
producing script): the PID regime has genuine per-frame `(image, state, target_x/y, action)`
supervision but **no instruction was ever issued or logged for it, full stop** -- unlike
Track 4's real `audio_command` column. Every PID frame gets an honest, distinct placeholder
task string (`NO_INSTRUCTION_PID_REGIME`, never `NO_COMMAND_YET` -- that phrase specifically
means "this Track 4 session's pre-first-command warmup window," which is not what a PID
frame is) and `has_real_language_label=0`. Two new per-frame features carry this distinction
into the LeRobot dataset itself (not just this docstring): `"regime"` (a real `dtype:
"string"` feature -- confirmed to round-trip through `add_frame()`/`save_episode()`/
`finalize()`/reload via a real, throwaway probe dataset built while writing this extension,
the same "confirm against the installed API, don't assume" standard as points 1-3 below) and
`"has_real_language_label"` (`int64`, 0/1). Regime selection for PID sessions reads
`data_processing/session_manifest.json` (5 of 6 PID sessions have a `frame_synced_csv`;
`session_20260730_174916` is excluded, matching `generate_vla_dataset.py`'s Stage-0
selection of the identical 5 sessions) rather than a hardcoded session-name list, so it stays
correct if the manifest is regenerated.

One bronze session = one LeRobot episode. Built against the REAL, installed
`lerobot==0.4.4` `LeRobotDataset` API (`C:/Users/Admin/.conda/envs/ball_balance_env`,
inspected 2026-09-15 via `inspect.signature`/`inspect.getsource` -- not guessed from the
tutorial's higher-level `record_loop()` wrapper, per this kickoff's explicit instruction).
Confirmed call shape:

    ds = LeRobotDataset.create(repo_id, fps, features, root=...)
    for each episode (bronze session):
        for each frame:
            ds.add_frame({**per-feature values, "task": <instruction str>})
        ds.save_episode()
    ds.finalize()   # REQUIRED -- without it the parquet footers never get written and
                     # the dataset can't be loaded back (confirmed via
                     # LeRobotDataset.finalize.__doc__)

Three things the kickoff prompt assumed that turned out to need adjustment once the real
API was inspected (see docs/LARGE_VLA_RESEARCH_SPIKE.md's own tentative "Confirmed schema"
note -- this is exactly the kind of gap it flagged as open) -- confirmed by actually
running this converter against a real (synthetic-input, real-code) session and reading
the resulting traceback, not by reading source alone:

  1. `add_frame()` requires a `"task"` key on EVERY frame dict (raises ValueError
     otherwise, per `lerobot.datasets.utils.validate_frame`) -- it is not optional and
     not inferred from anything else. This script supplies the real per-frame
     `audio_command` column Task 2 added to `telemetry.csv`.
  2. `LeRobotDatasetMetadata.create()` calls `root.mkdir(parents=True, exist_ok=False)` --
     the output root must NOT already exist. This script defaults to a timestamped
     output dir for exactly that reason; pass --out-root explicitly to control it, and
     don't point it at an existing dataset expecting an in-place update (not supported by
     this API -- create() is create-from-scratch only).
  3. **A per-frame `"timestamp"` override in the `add_frame()` dict is REJECTED, not
     silently accepted, in installed `lerobot==0.4.4`.** `add_frame()`'s own body does
     `frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps`,
     which reads as if a custom timestamp is supported -- but `validate_frame()` (called
     BEFORE that line, on the still-unpopped frame dict) treats `"timestamp"` as an
     unrecognized extra feature and raises `ValueError: Extra features: {'timestamp'}`,
     confirmed by actually hitting this while building this script. Net effect: real
     per-frame `host_timestamp_ms` values from `telemetry.csv` are NOT preserved in the
     converted dataset via this version's public API -- every frame's timestamp is
     silently `frame_index / fps` (uniform spacing from the `fps` this script was given),
     regardless of the real, jittery capture timing Task 2 recorded. This is a real gap
     from what the pipeline doc assumed, not a design choice made here -- flagged rather
     than worked around by hand-patching internals.

Frame image convention: LeRobotDataset image features are RGB (confirmed via the
project's own HF/PIL-based tooling elsewhere and `_save_image()`'s use of `PIL.Image`);
`cv2.VideoCapture` decodes BGR, so this script converts every frame before `add_frame()`.

Output schema (mirrors `ml_multimodal/core/dataset.py`'s `VLADataset` conceptually, per
this kickoff's Task 3 spec, even though LeRobot's on-disk format is entirely different):
  - "observation.image"  video, (H, W, 3) uint8 RGB      -- from rgb_video.mp4
  - "observation.state"  float32, (2,) [touch_x, touch_y] -- from telemetry.csv (mm)
  - "action"              float32, (3,) [theta_a, theta_b, theta_c] -- degrees
  - "task" (per-frame)    the real recognized audio command string logged by Task 2,
                           NOT synthesized from target-coordinate thresholds the way
                           generate_vla_dataset.py does for its own (different) source
                           data. Frames from before any command was heard this session
                           (state-machine's default target, no audio yet) get the literal
                           placeholder "(no command yet)" rather than a fabricated
                           "hold" -- ground truth should say a command wasn't heard, not
                           imply one was.

Rows missing touch/theta ground truth (SessionTouchTap.get_latest_touch() returned None
for that frame -- e.g. very start of a session before the first 'T,...' line arrives) are
skipped, not zero-filled, since LeRobotDataset requires every "observation.state"/"action"
value to be a real number and zero-filling would silently fabricate ground truth.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Optional

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

NO_COMMAND_YET = "(no command yet)"

# PID regime never had an instruction issued or logged at all -- a genuinely absent signal,
# not a mid-session gap the way NO_COMMAND_YET is for Track 4's pre-first-command warmup
# window. Kept as a visibly different string (never reused NO_COMMAND_YET for this regime)
# so a downstream consumer can't conflate "no command yet, one may come" with "no command
# ever existed for this whole regime." Matches generate_vla_dataset.py's Stage-0 handling
# of the identical distinction in its own (separate) flat-JSON schema.
NO_INSTRUCTION_PID_REGIME = "(no instruction: PID regime, no command was ever issued or logged)"

# Regime tag values -- deliberately matching generate_vla_dataset.py's Stage-0 "regime"
# string convention exactly ("pid_webcam" / "jetson_track4_rl"), not
# session_manifest.json's own slightly different "pid_laptop_webcam" /
# "jetson_track4_rl_teacher" spelling -- so anything downstream that filters by regime
# string (e.g. Stage 3's "grounding head trains only on Track 4" rule) can use one
# consistent value across both the flat-JSON and LeRobot dataset outputs.
TRACK4_REGIME_NAME = "jetson_track4_rl"
PID_REGIME_NAME = "pid_webcam"

REQUIRED_COLUMNS = [
    "frame_index",
    "host_timestamp_ms",
    "target_x",
    "target_y",
    "touch_x",
    "touch_y",
    "theta_a",
    "theta_b",
    "theta_c",
]


def find_sessions(bronze_dir: str, pattern: str = "session_jetson_track4_*") -> list:
    """Matches session_recorder.py's own naming convention for Track 4 sessions."""
    return sorted(glob.glob(os.path.join(bronze_dir, pattern)))


def find_pid_sessions(bronze_dir: str, manifest_path: Optional[str] = None) -> list:
    """Returns (session_dir_abs, csv_filename) pairs for every PID/laptop-webcam session
    that is actually usable at frame level, per `session_manifest.json`'s 2026-09-15 audit
    (regime == 'pid_laptop_webcam' AND frame_synced_csv is not null -- raw telemetry.csv
    has no frame_index column in this regime, only the separately-produced
    synced_telemetry.csv does; see that file's own docstring). As of the current manifest
    this resolves to exactly 5 of the 6 PID sessions (session_20260730_174916 excluded:
    never synced), matching generate_vla_dataset.py's Stage-0 selection of the identical 5
    sessions. Reads the manifest rather than hardcoding session names so this stays correct
    if the manifest is regenerated. `bronze_dir` is accepted for signature symmetry with
    find_sessions() but session paths are resolved from the manifest's own
    `session_dir` field (relative to host_software/data/), not by re-deriving from
    bronze_dir, since the manifest's convention is host_software/data/-relative, not
    01_bronze/-relative."""
    if manifest_path is None:
        manifest_path = os.path.join(_ML_JETSON_VLA_DIR, "data_processing", "session_manifest.json")
    if not os.path.exists(manifest_path):
        print(
            f"[convert_to_lerobot] WARNING: session_manifest.json not found at "
            f"{manifest_path} -- cannot include PID sessions."
        )
        return []

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pid_sessions = []
    for entry in manifest.get("sessions", []):
        if entry.get("regime") != "pid_laptop_webcam":
            continue
        csv_filename = entry.get("frame_synced_csv")
        if not csv_filename:
            print(
                f"[convert_to_lerobot] Skipping {entry.get('session_dir')}: no "
                f"frame_synced_csv in manifest (never synced -- unusable at frame level)."
            )
            continue
        session_dir_abs = os.path.join(_HOST_SOFTWARE_DIR, "data", entry["session_dir"])
        pid_sessions.append((session_dir_abs, csv_filename))
    return pid_sessions


def _build_features(image_h: int, image_w: int) -> dict:
    return {
        "observation.image": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["touch_x_mm", "touch_y_mm"],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["theta_a_deg", "theta_b_deg", "theta_c_deg"],
        },
        # 2026-09-17 extension (Stage 1): regime/language-label provenance, carried at the
        # LeRobot-schema level, not just in staging docs -- "string" dtype confirmed to
        # round-trip through add_frame()/save_episode()/finalize()/reload via a real
        # throwaway probe dataset (not assumed from reading validate_feature_string() alone).
        "regime": {
            "dtype": "string",
            "shape": (1,),
            "names": ["regime"],
        },
        "has_real_language_label": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["has_real_language_label"],
        },
    }


def _probe_first_frame_size(video_path: str) -> tuple:
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read a single frame from {video_path}")
        h, w = frame.shape[:2]
        return h, w
    finally:
        cap.release()


def _build_session_jobs(
    bronze_dir: str,
    session_pattern: str,
    include_track4: bool,
    include_pid: bool,
    pid_session_manifest: Optional[str],
) -> list:
    """Returns a list of {"session_dir", "session_name", "csv_filename", "regime"} dicts,
    Track 4 sessions first (sorted by name) then PID sessions (sorted by name) -- Track4
    ordered first specifically so the canonical-frame-size probe (first job) matches this
    script's original pre-2026-09-17 default behavior when include_pid=False."""
    jobs = []
    if include_track4:
        for d in find_sessions(bronze_dir, session_pattern):
            jobs.append({
                "session_dir": d,
                "session_name": os.path.basename(d),
                "csv_filename": "telemetry.csv",
                "regime": TRACK4_REGIME_NAME,
            })
    if include_pid:
        for d, csv_filename in find_pid_sessions(bronze_dir, pid_session_manifest):
            jobs.append({
                "session_dir": d,
                "session_name": os.path.basename(d),
                "csv_filename": csv_filename,
                "regime": PID_REGIME_NAME,
            })
    return jobs


def convert(
    bronze_dir: str,
    out_root: str,
    repo_id: str,
    fps: int = 30,
    session_pattern: str = "session_jetson_track4_*",
    include_track4_sessions: bool = True,
    include_pid_sessions: bool = True,
    pid_session_manifest: Optional[str] = None,
    max_frames_per_session: Optional[int] = None,
) -> dict:
    """max_frames_per_session: cap frames written per episode (stops early, does not
    zero-fill/pad). None (default) = full session. Intended for cheap dry-run/sample
    verification runs (data-pipeline-verification skill's "small sample before full run"
    rule) -- e.g. confirm the newly-added PID ingestion path round-trips through
    LeRobotDataset before committing to converting all 15 sessions' full frame counts."""
    jobs = _build_session_jobs(
        bronze_dir, session_pattern, include_track4_sessions, include_pid_sessions,
        pid_session_manifest,
    )
    if not jobs:
        raise FileNotFoundError(
            f"No sessions found: Track4 pattern='{session_pattern}' under {bronze_dir} "
            f"(include_track4_sessions={include_track4_sessions}), PID sessions via "
            f"session_manifest.json (include_pid_sessions={include_pid_sessions})."
        )

    # All sessions in one LeRobotDataset must share the same feature shapes -- probe the
    # first job's video for the canonical (H, W); later sessions that don't match are
    # skipped with a warning rather than silently reshaping/corrupting the dataset. (Real,
    # not assumed: PID and Track4 videos were confirmed this session to both be 640x480 via
    # cv2.VideoCapture, so this is not expected to skip anything in practice today.)
    first = jobs[0]
    first_csv = os.path.join(first["session_dir"], first["csv_filename"])
    first_video = os.path.join(first["session_dir"], "rgb_video.mp4")
    if not os.path.exists(first_csv) or not os.path.exists(first_video):
        raise FileNotFoundError(f"Missing {first['csv_filename']}/rgb_video.mp4 in {first['session_dir']}")
    canonical_h, canonical_w = _probe_first_frame_size(first_video)
    features = _build_features(canonical_h, canonical_w)

    if os.path.exists(out_root):
        raise FileExistsError(
            f"{out_root} already exists -- LeRobotDataset.create() requires a fresh "
            f"directory (root.mkdir(..., exist_ok=False)). Pass a different --out-root "
            f"or remove the existing one yourself."
        )
    os.makedirs(os.path.dirname(out_root) or ".", exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        robot_type="ball_balancer_3dof",
        use_videos=True,
    )

    stats = {
        "sessions_found": len(jobs),
        "sessions_converted": 0,
        "sessions_skipped_shape_mismatch": 0,
        "sessions_skipped_missing_files": 0,
        "frames_written": 0,
        "frames_skipped_missing_ground_truth": 0,
        "frames_skipped_video_underrun": 0,
        "by_regime": {
            TRACK4_REGIME_NAME: {"sessions_converted": 0, "frames_written": 0},
            PID_REGIME_NAME: {"sessions_converted": 0, "frames_written": 0},
        },
    }

    for job in jobs:
        session_dir = job["session_dir"]
        session_name = job["session_name"]
        regime = job["regime"]
        csv_path = os.path.join(session_dir, job["csv_filename"])
        video_path = os.path.join(session_dir, "rgb_video.mp4")

        if not os.path.exists(csv_path) or not os.path.exists(video_path):
            print(f"[convert_to_lerobot] Skipping {session_name} ({regime}): missing {job['csv_filename']} or rgb_video.mp4")
            stats["sessions_skipped_missing_files"] += 1
            continue

        df = pd.read_csv(csv_path)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"[convert_to_lerobot] Skipping {session_name} ({regime}): missing columns {missing_cols}")
            stats["sessions_skipped_missing_files"] += 1
            continue
        if regime == TRACK4_REGIME_NAME and "audio_command" not in df.columns:
            print(
                f"[convert_to_lerobot] WARNING: {session_name}'s telemetry.csv has no "
                f"'audio_command' column (older/non-Track-4 session?) -- every frame "
                f"will get the '{NO_COMMAND_YET}' placeholder instruction."
            )

        h, w = _probe_first_frame_size(video_path)
        if (h, w) != (canonical_h, canonical_w):
            print(
                f"[convert_to_lerobot] Skipping {session_name} ({regime}): frame size {(h, w)} != "
                f"canonical {(canonical_h, canonical_w)} (first session's size) -- "
                f"LeRobotDataset requires uniform feature shapes across the dataset."
            )
            stats["sessions_skipped_shape_mismatch"] += 1
            continue

        cap = cv2.VideoCapture(video_path)
        session_start_ms: Optional[float] = None
        frames_this_episode = 0
        try:
            for _, row in df.iterrows():
                if max_frames_per_session is not None and frames_this_episode >= max_frames_per_session:
                    break

                ret, frame_bgr = cap.read()
                if not ret:
                    print(
                        f"[convert_to_lerobot] {session_name} ({regime}): video ended before "
                        f"telemetry (row {int(row['frame_index'])}) -- stopping this "
                        f"episode here, matching generate_vla_dataset.py's own "
                        f"'video stream ended before telemetry' handling."
                    )
                    stats["frames_skipped_video_underrun"] += len(df) - int(row["frame_index"])
                    break

                if (
                    pd.isna(row["touch_x"]) or pd.isna(row["touch_y"])
                    or pd.isna(row["theta_a"]) or pd.isna(row["theta_b"]) or pd.isna(row["theta_c"])
                ):
                    # No ground truth for this frame (e.g. before the first 'T,...' line
                    # arrived) -- skip rather than zero-fill, see module docstring.
                    stats["frames_skipped_missing_ground_truth"] += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Real host_timestamp_ms is read (for session_start_ms bookkeeping/
                # future use) but NOT passed into add_frame() -- see module docstring
                # point 3: installed lerobot==0.4.4 rejects a "timestamp" key in the
                # frame dict (validate_frame treats it as an unrecognized extra
                # feature), confirmed by actually hitting that ValueError while
                # building this script. Every frame's on-disk timestamp ends up
                # frame_index/fps instead of the real, jittery capture time.
                host_ts = float(row["host_timestamp_ms"])
                if session_start_ms is None:
                    session_start_ms = host_ts

                if regime == TRACK4_REGIME_NAME:
                    raw_cmd = row.get("audio_command", "")
                    if pd.notna(raw_cmd) and str(raw_cmd).strip():
                        task = str(raw_cmd).strip()
                        has_real_language_label = 1
                    else:
                        task = NO_COMMAND_YET
                        has_real_language_label = 0
                else:
                    # PID regime: no instruction was ever issued or logged for this sample,
                    # full stop -- never fabricate one from target_x/y thresholds (matches
                    # generate_vla_dataset.py's Stage-0 handling of the identical regime).
                    task = NO_INSTRUCTION_PID_REGIME
                    has_real_language_label = 0

                dataset.add_frame(
                    {
                        "observation.image": frame_rgb,
                        "observation.state": np.array(
                            [row["touch_x"], row["touch_y"]], dtype=np.float32
                        ),
                        "action": np.array(
                            [row["theta_a"], row["theta_b"], row["theta_c"]], dtype=np.float32
                        ),
                        "regime": regime,
                        "has_real_language_label": np.array([has_real_language_label], dtype=np.int64),
                        "task": task,
                    }
                )
                frames_this_episode += 1
        finally:
            cap.release()

        if frames_this_episode == 0:
            print(f"[convert_to_lerobot] Skipping {session_name} ({regime}): 0 usable frames (no ground truth ever arrived)")
            dataset.clear_episode_buffer()
            stats["sessions_skipped_missing_files"] += 1
            continue

        dataset.save_episode()
        stats["sessions_converted"] += 1
        stats["frames_written"] += frames_this_episode
        stats["by_regime"][regime]["sessions_converted"] += 1
        stats["by_regime"][regime]["frames_written"] += frames_this_episode
        print(f"[convert_to_lerobot] {session_name} ({regime}): {frames_this_episode} frames -> episode saved")

    dataset.finalize()
    stats["out_root"] = out_root
    stats["repo_id"] = repo_id
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Track 4 + PID bronze sessions to a regime-tagged LeRobot-format dataset."
    )
    default_bronze = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
    parser.add_argument("--bronze-dir", type=str, default=default_bronze)
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Output dataset directory. Must not already exist (LeRobotDataset.create() "
             "requires a fresh dir). Default: "
             "data/03_gold/lerobot_combined_<timestamp>/ under host_software/.",
    )
    parser.add_argument("--repo-id", type=str, default="vri2026/combined_jetson_track4_and_pid")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--session-pattern", type=str, default="session_jetson_track4_*")
    parser.add_argument(
        "--exclude-track4", action="store_true",
        help="Skip Track 4 (session_jetson_track4_*) sessions entirely.",
    )
    parser.add_argument(
        "--exclude-pid", action="store_true",
        help="Skip the 5 usable PID/laptop-webcam sessions entirely (Track4-only, matches "
             "this script's pre-2026-09-17 default behavior).",
    )
    parser.add_argument(
        "--pid-session-manifest", type=str, default=None,
        help="Path to session_manifest.json used to select usable PID sessions. Default: "
             "ml_jetson_vla/data_processing/session_manifest.json.",
    )
    parser.add_argument(
        "--max-frames-per-session", type=int, default=None,
        help="Cap frames written per episode -- for cheap dry-run/sample verification runs, "
             "not intended for the real full-corpus conversion.",
    )
    args = parser.parse_args()

    out_root = args.out_root
    if out_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "track4" if args.exclude_pid else ("pid" if args.exclude_track4 else "combined")
        out_root = os.path.join(_HOST_SOFTWARE_DIR, "data", "03_gold", f"lerobot_{label}_{stamp}")

    stats = convert(
        args.bronze_dir, out_root, args.repo_id, fps=args.fps, session_pattern=args.session_pattern,
        include_track4_sessions=not args.exclude_track4,
        include_pid_sessions=not args.exclude_pid,
        pid_session_manifest=args.pid_session_manifest,
        max_frames_per_session=args.max_frames_per_session,
    )
    print("\n=== convert_to_lerobot.py summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
