import os
import pandas as pd
import json
import cv2
import glob


# session_jetson_track4_* sessions (RL control net, real recognized audio_command labels)
# are a genuinely different regime from every other session_* directory (PID/laptop-webcam,
# collect_webcam_data.py, no language labels at all) -- confirmed by reading each regime's
# producing script, per DATASET_CATALOG.md's 2026-09-15 audit. Never handled generically.
TRACK4_SESSION_PREFIX = "session_jetson_track4_"

# Frames the collector itself never attached a command to (Track 4's pre-first-command
# warmup window) get this explicit, honest placeholder -- matches
# ml_jetson_vla/data_processing/convert_to_lerobot.py's own NO_COMMAND_YET handling of the
# identical situation, so nothing downstream has to special-case two different spellings of
# "no command was heard yet."
NO_COMMAND_YET = "(no command yet)"


def _regime_for_session(session_name: str) -> str:
    """PID/laptop-webcam sessions (session_YYYYMMDD_HHMMSS/) never had an instruction issued
    or logged at all -- not a missing column, a genuinely absent signal (see the 2026-09-16
    correction in DATASET_CATALOG.md's Known Issues #1/#2). Track 4 sessions
    (session_jetson_track4_YYYYMMDD_HHMMSS/) carry a real audio_command column written by
    ScriptedCommandSequencer. Distinguish by the confirmed, source-verified naming
    convention session_recorder.py actually uses -- not a guess."""
    if session_name.startswith(TRACK4_SESSION_PREFIX):
        return "jetson_track4_rl"
    return "pid_webcam"


def generate_vla_dataset(bronze_dir: str, gold_dir: str) -> None:
    """
    Extracts frames from every session's RGB video in 01_bronze, pairs them with
    synchronous telemetry, and writes a unified, regime-tagged VLA dataset to
    03_gold/vla_dataset.json.

    Two regimes, each handled per its own actual, source-confirmed ground truth -- never
    handled generically, and never blended into one label scheme (DATASET_CATALOG.md's
    2026-09-15 audit, "Known Issues" #1/#2):

      - "pid_webcam" (session_YYYYMMDD_HHMMSS/, classical PID firmware via
        collect_webcam_data.py): real per-frame target_x/y and (touch_x/y, theta_a/b/c)
        state/action ground truth, but NO instruction was ever issued or logged for this
        regime -- full stop. The "audio_command" field is left `null` for every sample here.
        Do NOT synthesize a coordinate-threshold language label the way this script used to
        (that fabricated-after-the-fact behavior is exactly what produced the orphaned,
        0%-resolvable 03_gold/vla_dataset.json this function now replaces). Reads
        synced_telemetry.csv, NOT raw telemetry.csv -- raw telemetry.csv has no
        `frame_index` column in this regime (it's an independent ~25Hz serial stream, never
        joined to video frames); only the separately-produced synced_telemetry.csv carries
        the join. Sessions never run through the sync step (missing synced_telemetry.csv)
        are skipped with a clear log message, not crashed on.

      - "jetson_track4_rl" (session_jetson_track4_*/, RL control net via RLControl.cpp):
        real recognized audio_command values (go_red, hold, forward, stop, ...) read
        directly from telemetry.csv, which already carries `frame_index` natively (single
        writer thread, no separate sync step needed). Never re-derived, never overwritten by
        any synthesis path -- only the collector's own genuinely-commandless warmup frames
        (blank audio_command before the first recognized command each session) get the
        explicit NO_COMMAND_YET placeholder instead of a fabricated default like "hold".

    Every output sample carries "regime" and "has_real_language_label" so nothing downstream
    can accidentally treat a PID sample as language-supervised without checking.
    """
    images_dir = os.path.join(gold_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    vla_dataset = []
    stats = {
        "pid_webcam": {"sessions_used": 0, "sessions_skipped": 0, "samples": 0},
        "jetson_track4_rl": {
            "sessions_used": 0,
            "sessions_skipped": 0,
            "samples": 0,
            "samples_with_real_language_label": 0,
        },
    }

    session_dirs = sorted(
        d for d in glob.glob(os.path.join(bronze_dir, "session_*")) if os.path.isdir(d)
    )

    if not session_dirs:
        print(f"No session directories found in {bronze_dir}")
        return

    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        regime = _regime_for_session(session_name)
        video_path = os.path.join(session_dir, "rgb_video.mp4")

        if regime == "pid_webcam":
            csv_path = os.path.join(session_dir, "synced_telemetry.csv")
            if not os.path.exists(csv_path):
                print(
                    f"Skipping {session_name}: no synced_telemetry.csv found. The PID "
                    f"regime's raw telemetry.csv has no frame_index column (it's an "
                    f"independent, unsynced ~25Hz serial stream) -- this session was never "
                    f"run through the sync step, so there is no frame-level join available."
                )
                stats["pid_webcam"]["sessions_skipped"] += 1
                continue
        else:
            csv_path = os.path.join(session_dir, "telemetry.csv")

        if not os.path.exists(csv_path) or not os.path.exists(video_path):
            print(f"Skipping {session_name}: missing telemetry or video.")
            stats[regime]["sessions_skipped"] += 1
            continue

        print(f"Processing {session_name} (regime={regime})...")
        df = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(video_path)

        session_samples = 0
        session_real_language = 0
        try:
            for idx, row in df.iterrows():
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"Warning: Video stream ended before telemetry for {session_name} "
                        f"(row {idx}/{len(df)})"
                    )
                    break

                if (
                    pd.isna(row.get("frame_index"))
                    or pd.isna(row.get("touch_x"))
                    or pd.isna(row.get("touch_y"))
                    or pd.isna(row.get("theta_a"))
                    or pd.isna(row.get("theta_b"))
                    or pd.isna(row.get("theta_c"))
                ):
                    # No usable state/action ground truth for this frame -- skip rather than
                    # zero-fill (matches convert_to_lerobot.py's handling of the same case).
                    continue

                frame_idx = int(row["frame_index"])

                # Save frame image
                image_name = f"{session_name}_frame_{frame_idx:05d}.jpg"
                image_path = os.path.join(images_dir, image_name)
                cv2.imwrite(image_path, frame)

                if regime == "jetson_track4_rl":
                    raw_cmd = row.get("audio_command", "")
                    if pd.notna(raw_cmd) and str(raw_cmd).strip():
                        audio_command = str(raw_cmd).strip()
                        has_real_language_label = True
                        session_real_language += 1
                    else:
                        # Pre-first-command warmup window -- genuinely no instruction was
                        # heard yet. Honest placeholder, never a fabricated "hold".
                        audio_command = NO_COMMAND_YET
                        has_real_language_label = False
                else:
                    # PID regime: no instruction was ever issued or logged for this sample,
                    # full stop -- never fabricate one from target_x/y thresholds.
                    audio_command = None
                    has_real_language_label = False

                sample = {
                    "timestamp_ms": row["host_timestamp_ms"],
                    "image_path": os.path.abspath(image_path),
                    "regime": regime,
                    "has_real_language_label": has_real_language_label,
                    "audio_command": audio_command,
                    "target_x": row["target_x"],
                    "target_y": row["target_y"],
                    "state_x": row["touch_x"],
                    "state_y": row["touch_y"],
                    "action_theta_a": row["theta_a"],
                    "action_theta_b": row["theta_b"],
                    "action_theta_c": row["theta_c"],
                }
                vla_dataset.append(sample)
                session_samples += 1
        finally:
            cap.release()

        if session_samples == 0:
            print(f"Skipping {session_name}: 0 usable samples produced.")
            stats[regime]["sessions_skipped"] += 1
            continue

        stats[regime]["sessions_used"] += 1
        stats[regime]["samples"] += session_samples
        if regime == "jetson_track4_rl":
            stats[regime]["samples_with_real_language_label"] += session_real_language

    output_file = os.path.join(gold_dir, "vla_dataset.json")
    with open(output_file, "w") as f:
        json.dump(vla_dataset, f, indent=2)

    print(f"Generated VLA dataset with {len(vla_dataset)} samples at {output_file}")
    print(f"Regime breakdown: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bronze_data_dir = os.path.abspath(os.path.join(script_dir, "../../data/01_bronze"))
    gold_dir = os.path.abspath(os.path.join(script_dir, "../../data/03_gold"))

    generate_vla_dataset(bronze_data_dir, gold_dir)
