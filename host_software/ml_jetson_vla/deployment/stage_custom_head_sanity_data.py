"""Stage 1 of the approved Large-Model Fine-Tuning Pipeline plan (Data Fix -> Colab
Custom-Head Test -> 1st Fine-Tune -> Compare -> Iterate). Run on the DEV MACHINE, not in
Colab. Sibling of `stage_arm2_sweep_for_colab.py` / `stage_finetune_data_for_colab.ps1` --
same zip+Drive checkpointed-staging pattern this project already uses
(`bootstrap_model_comparison_colab.ipynb`), same self-verifying zipfile approach
`stage_arm2_sweep_for_colab.py` established (ZIP_STORED for already-compressed image bytes,
sha256 manifest, no-backslash / count assertions before declaring success).

WHAT THIS STAGES (source: `host_software/data/03_gold/vla_dataset.json`, read-only input --
NOT modified here, per the Stage-0 boundary in the fine-tuning pipeline plan)
    151,706 real, regime-tagged samples (110,993 pid_webcam + 40,713 jetson_track4_rl, of
    which 40,703 carry a real audio_command and 10 an honest "(no command yet)" placeholder),
    backed by 151,706 real JPEGs in `03_gold/images/` (~11.0 GB total, avg ~78KB/image --
    measured this session via `Get-ChildItem -File | Measure-Object -Sum Length`, NOT `du`,
    which stalled past 120s enumerating 151K files over the Windows/Cygwin path-translation
    boundary in this environment).

    --mode sample (default): a STRATIFIED subset sized for Stage 2's actual near-term need
    (a forward-pass/shape/overfit-one-batch sanity check on the custom heads -- at most a
    few hundred to low-thousands of samples, not the full corpus). Stratified so the sanity
    batch cannot accidentally be all one regime or all one audio_command:
      - pid_webcam: N_PID_PER_SESSION (default 240) drawn evenly from EACH of the 5 real
        pid_webcam sessions (7,913 - 28,794 frames each) -- an even per-session quota rather
        than size-proportional, so the smallest session (7,913 frames) isn't drowned out.
      - jetson_track4_rl: N_PER_COMMAND (default 179) drawn evenly from each of the 10 REAL
        audio_command values (excludes the "(no command yet)" placeholder from this quota --
        it is not a real language label and would just look like unexplained noise in a
        language-conditioned sanity batch).
      - All 10 "(no command yet)" placeholder records are included separately, IN ADDITION
        to the quotas above (not competing for a command slot), so their existence is
        visible to whoever runs the Stage 2 notebook rather than silently sampled out.
      Default total: 5*240 + 10*179 + 10 = 1200 + 1790 + 10 = 3000 samples, ~230MB zip --
      comfortably in the "low-thousands" range the plan calls for, trivial to upload.

    --mode full: the ENTIRE 151,706-sample / ~11GB corpus in one archive, for the eventual
    Stage 3 fine-tuning run. Not required to be uploaded tonight (see the staging script's
    own printed next-steps) -- built here because local disk (verified: ~270GB free) and
    local zip time are not the constraint, only overnight upload bandwidth is, and that is
    the user's call to make once they see the real zip size. `--shard-by-regime` splits this
    into two archives (pid_webcam / jetson_track4_rl) instead of one, for a partial-upload/
    resume story if a single ~11GB transfer proves impractical -- not built by default,
    available as one flag when needed.

WHY A NEW SCRIPT AND NOT AN EXTENSION OF `stage_finetune_data_for_colab.ps1`
    That existing script stages a DIFFERENT data shape for a DIFFERENT downstream consumer:
    raw `rgb_video.mp4` + `telemetry.csv` per session, decoded into LeRobot-format episodes
    BY `convert_to_lerobot.py` running inside Colab, for `qwen_multihead_policy.py`
    (LeRobotDataset "observation.image"/"observation.state"/"action"/"task" schema). Its own
    docstring explicitly rejected shipping "151,706 separately-extracted JPEGs" alongside
    the raw video as "pure duplication ... zero benefit" -- true in that script's world,
    where the JPEGs did not yet exist as a standalone artifact. That has changed: Stage 0
    tonight regenerated `03_gold/vla_dataset.json` + `03_gold/images/` as a real, trusted,
    already-extracted flat-JSON-plus-JPEGs artifact in its own right (9-field per-frame
    schema: image_path/regime/audio_command/target_xy/state_xy/action_theta_abc), which is
    exactly the shape a quick custom-head forward-pass/shape sanity check needs and does NOT
    need a LeRobotDataset/video-decode step to use. These are two genuinely different staged
    datasets for two different purposes (multi-head/LeRobot fine-tuning vs. a fast
    custom-head sanity check) that happen to both call themselves "Stage 1" of a fine-tuning
    pipeline -- flagged explicitly in this run's report back, not silently resolved here;
    reconciling which one is authoritative for Stage 2/3 is a decision for the user, not this
    script.

USAGE (from anywhere; requires no new pip dependencies -- json/zipfile/hashlib/random are
stdlib):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe stage_custom_head_sanity_data.py
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe stage_custom_head_sanity_data.py --mode full
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe stage_custom_head_sanity_data.py --mode full --shard-by-regime

Then upload the resulting zip(s) to Google Drive under the SAME folder the existing
notebooks already use (`MyDrive/vri2026_track4_bootstrap/`), per
`feedback_reuse_existing_export_tooling` -- see each build's printed "Next (manual)" line
for the exact target filename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_GOLD_DIR = os.path.join(_HOST_SOFTWARE_DIR, "data", "03_gold")
_GOLD_JSON_PATH = os.path.join(_GOLD_DIR, "vla_dataset.json")
_GOLD_IMAGES_DIR = os.path.join(_GOLD_DIR, "images")

TOP = "custom_head_sanity_data"
REAL_COMMANDS = [
    "left", "right", "forward", "backward", "hold", "stop",
    "go_red", "go_green", "go_yellow", "go_black",
]
PLACEHOLDER_COMMAND = "(no command yet)"
SEED = 42


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _basename_of_image(record: Dict[str, Any]) -> str:
    return os.path.basename(record["image_path"])


def load_gold_dataset() -> List[Dict[str, Any]]:
    if not os.path.exists(_GOLD_JSON_PATH):
        raise FileNotFoundError(f"{_GOLD_JSON_PATH} not found -- Stage 0 output missing")
    with open(_GOLD_JSON_PATH, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)
    return records


def build_stratified_sample(
    records: List[Dict[str, Any]],
    n_pid_per_session: int = 240,
    n_per_command: int = 179,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Even per-session (pid_webcam) / per-audio_command (jetson_track4_rl) quotas, plus all
    "(no command yet)" placeholders, so the sample can't collapse onto one session/command."""
    rng = random.Random(seed)

    pid_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    t4_by_command: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    placeholders: List[Dict[str, Any]] = []

    for rec in records:
        if rec["regime"] == "pid_webcam":
            # session id is the first "session_YYYYMMDD_HHMMSS" token in the filename.
            fname = _basename_of_image(rec)
            session_id = fname.split("_frame_")[0]
            pid_by_session[session_id].append(rec)
        elif rec["regime"] == "jetson_track4_rl":
            cmd = rec.get("audio_command")
            if cmd == PLACEHOLDER_COMMAND:
                placeholders.append(rec)
            elif cmd in REAL_COMMANDS:
                t4_by_command[cmd].append(rec)
            # else: unexpected regime/command combo -- deliberately dropped, not silently
            # merged into an existing bucket (would break the stratification guarantee).

    sample: List[Dict[str, Any]] = []
    session_report: Dict[str, int] = {}
    for session_id, recs in sorted(pid_by_session.items()):
        take = min(n_pid_per_session, len(recs))
        chosen = rng.sample(recs, take)
        sample.extend(chosen)
        session_report[session_id] = take

    command_report: Dict[str, int] = {}
    for cmd in REAL_COMMANDS:
        recs = t4_by_command.get(cmd, [])
        take = min(n_per_command, len(recs))
        chosen = rng.sample(recs, take)
        sample.extend(chosen)
        command_report[cmd] = take

    sample.extend(placeholders)

    print(f"Stratified sample: {len(sample)} total records")
    print(f"  pid_webcam per session: {session_report}")
    print(f"  jetson_track4_rl per command: {command_report}")
    print(f"  jetson_track4_rl placeholder '(no command yet)': {len(placeholders)}")
    return sample


def _write_archive(
    records: List[Dict[str, Any]],
    out_zip: str,
    manifest_json_name: str,
) -> str:
    """Writes {TOP}/<manifest_json_name> (trimmed dataset json, image_path rewritten to a
    portable POSIX-relative `images/<file>.jpg` -- the original absolute Windows path is
    meaningless once extracted in Colab/Linux) + {TOP}/images/<file>.jpg for every record.
    JPEGs are ZIP_STORED (already compressed -- deflating wastes time for ~0% gain, same
    reasoning `stage_arm2_sweep_for_colab.py` used for .mp4). Self-verifies before returning.
    """
    portable_records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for rec in records:
        fname = _basename_of_image(rec)
        src = os.path.join(_GOLD_IMAGES_DIR, fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue
        new_rec = dict(rec)
        new_rec["image_path"] = f"images/{fname}"
        portable_records.append(new_rec)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} record(s) reference images missing from {_GOLD_IMAGES_DIR}, "
            f"e.g. {missing[:5]} -- dataset.json and images/ are out of sync, stopping "
            f"rather than silently shipping a partial/mismatched archive."
        )

    if os.path.exists(out_zip):
        os.remove(out_zip)
    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)

    manifest_lines = ["sha256  size_bytes  path"]
    with zipfile.ZipFile(out_zip, "w") as zf:
        manifest_bytes = json.dumps(portable_records, indent=2).encode("utf-8")
        zf.writestr(f"{TOP}/{manifest_json_name}", manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
        for rec in portable_records:
            fname = _basename_of_image(rec)
            src = os.path.join(_GOLD_IMAGES_DIR, fname)
            zf.write(src, f"{TOP}/images/{fname}", compress_type=zipfile.ZIP_STORED)

        for rec in portable_records:
            fname = _basename_of_image(rec)
            src = os.path.join(_GOLD_IMAGES_DIR, fname)
            manifest_lines.append(f"{_sha256(src)}  {os.path.getsize(src)}  images/{fname}")
        zf.writestr(f"{TOP}/BUNDLE_MANIFEST.txt", "\n".join(manifest_lines) + "\n", compress_type=zipfile.ZIP_DEFLATED)

    # Self-verification, matching stage_arm2_sweep_for_colab.py's checks.
    with zipfile.ZipFile(out_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"zip integrity check failed at {bad}")
        names = zf.namelist()
    if any("\\" in n for n in names):
        raise RuntimeError("backslash in zip entry names")
    n_images = sum(1 for n in names if n.startswith(f"{TOP}/images/") and n.endswith(".jpg"))
    if n_images != len(portable_records):
        raise RuntimeError(f"expected {len(portable_records)} images in archive, found {n_images}")

    zip_size = os.path.getsize(out_zip)
    print(f"Archive: {out_zip}")
    print(f"  {len(names)} entries, {n_images} images, zip {zip_size / 1e6:.1f} MB ({zip_size / 1024**2:.1f} MiB)")
    return out_zip


def build_sample_archive(out_zip: str, n_pid_per_session: int, n_per_command: int) -> str:
    records = load_gold_dataset()
    sample = build_stratified_sample(records, n_pid_per_session, n_per_command)
    return _write_archive(sample, out_zip, "vla_dataset_sample.json")


def build_full_archive(out_zip: str, shard_by_regime: bool) -> List[str]:
    records = load_gold_dataset()
    if not shard_by_regime:
        return [_write_archive(records, out_zip, "vla_dataset.json")]

    base, ext = os.path.splitext(out_zip)
    outputs = []
    for regime in ("pid_webcam", "jetson_track4_rl"):
        subset = [r for r in records if r["regime"] == regime]
        shard_path = f"{base}_{regime}{ext}"
        outputs.append(_write_archive(subset, shard_path, f"vla_dataset_{regime}.json"))
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--n-pid-per-session", type=int, default=240)
    ap.add_argument("--n-per-command", type=int, default=179)
    ap.add_argument("--shard-by-regime", action="store_true",
                     help="--mode full only: split into pid_webcam / jetson_track4_rl shards")
    ap.add_argument("--out", default=None, help="output zip path (default: %%TEMP%%/<name>.zip)")
    args = ap.parse_args()

    if args.mode == "sample":
        out_zip = args.out or os.path.join(tempfile.gettempdir(), "custom_head_sanity_sample.zip")
        build_sample_archive(out_zip, args.n_pid_per_session, args.n_per_command)
        print("\nNext (manual): upload it to Google Drive at")
        print("  MyDrive/vri2026_track4_bootstrap/custom_head_sanity_sample.zip")
    else:
        out_zip = args.out or os.path.join(tempfile.gettempdir(), "custom_head_full_dataset.zip")
        outputs = build_full_archive(out_zip, args.shard_by_regime)
        print("\nNext (manual): upload the archive(s) above to Google Drive at")
        for p in outputs:
            print(f"  MyDrive/vri2026_track4_bootstrap/{os.path.basename(p)}")
        print("(Not required tonight -- this is Stage 3's eventual full-corpus input; the "
              "sample archive above is what Stage 2's custom-head sanity check needs now.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
