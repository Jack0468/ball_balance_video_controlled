"""Generates `session_split.json` -- the session-level train/eval split for Arm 2 fine-tuning.

Risk #1 in `docs/MULTI_HEAD_ARCHITECTURE_SPEC.md` ("Risk register", 2026-10-06): the 60-frame eval and
the training data come from the same sessions, and adjacent frames within one session are near-duplicates,
so a frame-level split overstates accuracy. The unit here is a whole SESSION, never a frame.

Population (read from the same sources `convert_to_lerobot.py` uses, so the split can't drift from the dataset):
  - Track 4 (regime `jetson_track4_rl`): every `session_jetson_track4_*` directory under
    `host_software/data/01_bronze/` that has a `telemetry.csv` (10 sessions today).
  - PID (regime `pid_webcam`): manifest entries with regime `pid_laptop_webcam` AND a non-null
    `frame_synced_csv` (5 sessions today; `session_20260730_174916` is excluded, matching the converter).

Selection rules (deterministic; the seed only drives the one tie-break that is genuinely a free choice):
  - Track 4 eval = 3 sessions spread across the recording window: the earliest, the median, and the latest
    by timestamp in the directory name. Each is held out whole, so the grounding head (Track-4-only, real
    language labels) can be scored on sessions it never saw.
  - PID eval = 1 session drawn with `random.Random(SEED)` from the PID sessions on the day with more than one
    session (2026-08-10). Holding out a session from that day keeps the one 2026-07-28 session in train, so
    train still sees both recording days.

Usage (from host_software/, pinned interpreter):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe ml_jetson_vla/data_processing/make_session_split.py
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

SEED: int = 20261006

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
BRONZE_DIR: str = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
MANIFEST_PATH: str = os.path.join(_THIS_DIR, "session_manifest.json")
OUT_PATH: str = os.path.join(_THIS_DIR, "session_split.json")

TRACK4_REGIME: str = "jetson_track4_rl"  # matches convert_to_lerobot.TRACK4_REGIME_NAME
PID_REGIME: str = "pid_webcam"           # matches convert_to_lerobot.PID_REGIME_NAME

_TRACK4_TS_RE = re.compile(r"session_jetson_track4_(\d{8})_(\d{6})$")
_REQUIRED_CSV_COLS: Tuple[str, ...] = ("touch_x", "touch_y", "theta_a", "theta_b", "theta_c")


def track4_sessions(bronze_dir: str) -> List[str]:
    """Track 4 session directory basenames, sorted by their recording timestamp."""
    names: List[str] = []
    for path in glob.glob(os.path.join(bronze_dir, "session_jetson_track4_*")):
        if not os.path.isdir(path) or path.endswith(".dvc"):
            continue
        if not os.path.exists(os.path.join(path, "telemetry.csv")):
            continue
        names.append(os.path.basename(path))
    return sorted(names, key=_track4_timestamp)


def _track4_timestamp(name: str) -> str:
    m = _TRACK4_TS_RE.search(name)
    if m is None:
        raise ValueError(f"Track 4 session name does not carry a YYYYMMDD_HHMMSS stamp: {name}")
    return f"{m.group(1)}_{m.group(2)}"


def pid_sessions(manifest_path: str) -> List[Dict[str, str]]:
    """PID sessions usable at frame level: regime `pid_laptop_webcam` with a `frame_synced_csv`."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    out: List[Dict[str, str]] = []
    for entry in manifest.get("sessions", []):
        if entry.get("regime") != "pid_laptop_webcam" or not entry.get("frame_synced_csv"):
            continue
        out.append({
            "name": os.path.basename(entry["session_dir"].rstrip("/")),
            "csv": str(entry["frame_synced_csv"]),
        })
    return sorted(out, key=lambda d: d["name"])


def pid_session_day(name: str) -> str:
    """`session_YYYYMMDD_HHMMSS` -> `YYYYMMDD` (the recording day)."""
    return name.split("_")[1]


def count_usable_frames(csv_path: str) -> Tuple[int, int]:
    """Returns (total rows, rows with all ground-truth columns present). The converter skips rows
    missing ground truth, so the second number is what actually reaches the LeRobot dataset."""
    df = pd.read_csv(csv_path, usecols=lambda c: c in _REQUIRED_CSV_COLS)
    usable = int(df.dropna(subset=list(_REQUIRED_CSV_COLS)).shape[0])
    return int(df.shape[0]), usable


def choose_track4_eval(sessions: Sequence[str]) -> List[str]:
    """Earliest, median, latest by recording time. Index positions are fixed, not sampled."""
    n = len(sessions)
    if n < 3:
        raise ValueError(f"Need >=3 Track 4 sessions to spread an eval set, found {n}")
    picks = sorted({0, (n - 1) // 2, n - 1})
    return [sessions[i] for i in picks]


def choose_pid_eval(pids: Sequence[Dict[str, str]], seed: int) -> str:
    """Seeded draw from the PID sessions on the day that has more than one session."""
    by_day: Dict[str, List[str]] = defaultdict(list)
    for p in pids:
        by_day[pid_session_day(p["name"])].append(p["name"])
    multi_day = sorted(day for day, names in by_day.items() if len(names) > 1)
    if not multi_day:
        raise ValueError("No recording day has more than one PID session to draw a held-out session from")
    pool = sorted(by_day[multi_day[-1]])
    return random.Random(seed).choice(pool)


def build_split(bronze_dir: str, manifest_path: str, seed: int) -> Dict[str, object]:
    t4 = track4_sessions(bronze_dir)
    pid = pid_sessions(manifest_path)
    t4_eval = choose_track4_eval(t4)
    pid_eval = choose_pid_eval(pid, seed)

    t4_train = [s for s in t4 if s not in t4_eval]
    pid_names = [p["name"] for p in pid]
    pid_train = [s for s in pid_names if s != pid_eval]

    train = t4_train + pid_train
    eval_set = t4_eval + [pid_eval]
    overlap = set(train) & set(eval_set)
    if overlap:
        raise AssertionError(f"Session leakage between train and eval: {sorted(overlap)}")

    rationale = (
        "Session-level split (unit = whole session, never frame; see docs/MULTI_HEAD_ARCHITECTURE_SPEC.md "
        "risk #1). EVAL, Track 4 (3 of 10): "
        f"{t4_eval[0]} (earliest recording, 15:16), {t4_eval[1]} (median, 16:05), {t4_eval[2]} (latest, 16:47). "
        "Chosen by position in recording order so the held-out set spans the whole afternoon rather than "
        "one stretch of it. CAVEAT: all 10 Track 4 sessions were recorded on 2026-09-15 between 15:16 and 16:47, "
        "so this spreads the eval across time-of-session within one day, not across days; cross-day "
        "generalisation of the Track 4 regime is untested. These 3 hold real audio-command labels, so the "
        "grounding head can be scored on them. "
        f"EVAL, PID (1 of 5): {pid_eval}, drawn with random.Random({seed}) from the 2026-08-10 PID sessions "
        "(the day with more than one session). The only 2026-07-28 PID session stays in train, so train covers "
        "both PID recording days. PID sessions have no language labels, so this eval item tests state/action "
        "only, not grounding. "
        "TRAIN: all remaining Track 4 sessions (7) and remaining PID sessions (4). "
        "Caveat: 10 Track 4 + 5 PID sessions is a small effective N; report per-session results, not only means "
        "(risk #8). Selection is fixed-rule plus one seeded draw; re-running reproduces this file exactly."
    )
    return {"seed": seed, "train": train, "eval": eval_set, "rationale": rationale}


def regime_of(name: str, t4_names: Sequence[str]) -> str:
    return TRACK4_REGIME if name in t4_names else PID_REGIME


def print_counts(split: Dict[str, object], bronze_dir: str, pid: Sequence[Dict[str, str]]) -> None:
    t4 = track4_sessions(bronze_dir)
    pid_csv = {p["name"]: p["csv"] for p in pid}
    train: List[str] = list(split["train"])  # type: ignore[arg-type]
    eval_set: List[str] = list(split["eval"])  # type: ignore[arg-type]

    def csv_for(name: str) -> str:
        if name in pid_csv:
            return os.path.join(bronze_dir, name, pid_csv[name])
        return os.path.join(bronze_dir, name, "telemetry.csv")

    print(f"seed={split['seed']}")
    for split_name, names in (("train", train), ("eval", eval_set)):
        print(f"\n[{split_name}] {len(names)} sessions")
        per_regime: Dict[str, Dict[str, int]] = defaultdict(lambda: {"sessions": 0, "rows": 0, "usable_rows": 0})
        for name in names:
            regime = regime_of(name, t4)
            rows, usable = count_usable_frames(csv_for(name))
            per_regime[regime]["sessions"] += 1
            per_regime[regime]["rows"] += rows
            per_regime[regime]["usable_rows"] += usable
        for regime in (TRACK4_REGIME, PID_REGIME):
            c = per_regime[regime]
            print(f"  {regime:<18} sessions={c['sessions']:<3} telemetry_rows={c['rows']:<7} "
                  f"rows_with_ground_truth={c['usable_rows']}")


def main() -> int:
    seed = SEED
    split = build_split(BRONZE_DIR, MANIFEST_PATH, seed)
    with open(OUT_PATH, "w") as f:
        json.dump(split, f, indent=2)
        f.write("\n")
    print(f"Wrote {OUT_PATH}")
    print_counts(split, BRONZE_DIR, pid_sessions(MANIFEST_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
