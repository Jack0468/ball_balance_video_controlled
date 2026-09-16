# VRI 2026 Dataset Catalog

This ledger tracks all physical data collected for the VRI 2026 project. All datasets are centralized in `host_software/data`.

> **2026-09-15 full audit.** Everything below the original two sections (which are corrected
> in place, not just appended around) was produced by a read-only, per-regime audit driven by
> `.claude/skills/dataset-integrity-check/SKILL.md`, done as Track 4 kickoff item 1
> (`docs/LARGE_VLA_RESEARCH_SPIKE.md`). Every regime's controller/sensor/schema claim below was
> confirmed by reading the actual producing script or firmware source — not inferred from a
> directory name. A companion machine-readable manifest lives at
> `host_software/ml_jetson_vla/data_processing/session_manifest.json` (see its own section
> below for the format). **This audit explicitly did not decide which regime(s) should feed
> VLA fine-tuning** — that is a follow-up decision informed by this catalog.

## Legacy iPhone Datasets — CONFIRMED, caution still accurate

*Collected using `iphone_data_logger.py`, synced into `01_bronze/video{1-5}/synced_telemetry.csv`
via `ml_vision/data_processing/sync_data.py` (not `sync_data.py` at repo top level — there is
no such top-level script; the one that actually exists is under `ml_vision/data_processing/`).*

Controller: PID firmware (`MLVisionControl`, confirmed via `synced_telemetry.csv`'s column set —
`error_x/y`, `integral_x/y`, `deriv_x/y`, `pitch`, `roll` alongside `theta_a/b/c` — the same PID
state signature as the laptop-webcam regime below, not the RL control net). Sensor: iPhone
4K/1080p, all 5 sessions physically present on disk:

| Session | `.MOV` size | `synced_telemetry.csv` | Date (file mtime) |
|---|---|---|---|
| `01_bronze/video1` | 837 MB | 6.07 MB | 2026-07-10 |
| `01_bronze/video2` | 1.37 GB | 9.15 MB | 2026-07-10 |
| `01_bronze/video3` | 1.16 GB | 9.07 MB | 2026-07-10 |
| `01_bronze/video4` | 1.74 GB | 12.88 MB | 2026-07-16 |
| `01_bronze/video5` | 1.44 GB | 11.81 MB | 2026-07-16 |

Schema (`synced_telemetry.csv`, all 5 confirmed identical): `frame_index, video_time_ms,
host_timestamp_ms, mcu_micros, target_x, target_y, touch_x, touch_y, error_x, error_y, pitch,
roll, theta_a, theta_b, theta_c, integral_x, integral_y, deriv_x, deriv_y` — **no
`audio_command`/language column anywhere**, confirmed by reading the actual header, not just
the caution note. `host_software/data/bronze/iphone_telemetry.csv` (note: `data/bronze/`, a
distinct top-level directory from `data/01_bronze/`) is a single legacy pre-split telemetry
dump superseded by the per-video `synced_telemetry.csv` files above — redundant, not a 6th
session.

> [!CAUTION]
> Confirmed still accurate 2026-09-15: the Legacy iPhone Datasets **DO NOT** contain semantic
> targets. They cannot be used to train the VLA's language bindings. **New finding**: the
> orphaned `03_gold/vla_dataset.json` described below was actually derived from this iPhone
> regime (image timestamps match `video1`'s telemetry exactly), so even that derived artifact
> never had real language labels — its `audio_command` values are 100% synthesized from
> target-coordinate thresholds, not real speech, consistent with this regime having none to
> begin with.

## `collect_vla_data.py` — CONFIRMED still unused, but the catalog's framing was misleading

`host_software/data_collection/collect_vla_data.py`'s telemetry schema (`frame_index` written
**inline** as column 0, no separate frame-timestamp file) does not match any `telemetry.csv`
found anywhere in `01_bronze/` — confirmed by diffing its `csv_writer.writerow([...])` call
against every real session's actual header. **The script itself was never run**, so the old
catalog line "No VLA data collected yet" is technically still true of *this specific script*.

**But the catalog's implication that no VLA-directed collection has happened at all was
already false before this audit**: a different, uncatalogued script —
`host_software/data_collection/collect_webcam_data.py` — has a nearly-identical purpose
(webcam + STM32 binary telemetry → `01_bronze/session_<timestamp>/`) and is the actual,
confirmed producer of all 6 PID-regime sessions below, plus `03_gold/vla_dataset.json` (now
orphaned — see Known Issues). This script was missing from the catalog entirely; it is added
as its own regime below.

## Regime: PID / Laptop-Webcam Sessions (`01_bronze/session_YYYYMMDD_HHMMSS/`)

**Producing script, confirmed by reading source**: `host_software/data_collection/collect_webcam_data.py`
(NOT `main_resnet.py`, which `ml_multimodal/docs/DATA_PIPELINE.md`'s Phase-1 diagram names —
that script only exists under the read-only `experimental_variants/resnet/` and is not this
regime's real producer; the doc's diagram is stale). Controller: PID firmware
(`MLVisionControl`) via STM32 binary telemetry struct (`<Ifffffffffffffff>`, sync header
`0xAABBCCDD`). Sensor: laptop webcam, `640x480`.

6 sessions total, all physically present:

| Session | `telemetry.csv` rows | video frames | `synced_telemetry.csv`? | images / images_cropped / masks |
|---|---|---|---|---|
| `session_20260728_102908` | 9,009 | 7,913 | yes, 7,913 rows (frame-synced) | none |
| `session_20260730_174916` | 9,425 | 9,228 | **no — never synced** | none |
| `session_20260810_104132` | 34,575 | 28,794 | yes, 28,794 rows | 28,794 / 14,834 / **0 (empty)** |
| `session_20260810_110239` | 31,058 | 20,850 | yes, 20,850 rows | 20,850 / 10,988 / 54,940 |
| `session_20260810_112047` | 39,753 | 26,677 | yes, 26,677 rows | 26,677 / 14,067 / 56,268 |
| `session_20260810_114330` | 41,357 | 26,759 | yes, 26,759 rows | 26,759 / 12,883 / 64,415 |

Raw `telemetry.csv` schema (all 6 confirmed identical, 17 columns): `host_timestamp_ms,
mcu_micros, target_x, target_y, touch_x, touch_y, error_x, error_y, pitch, roll, theta_a,
theta_b, theta_c, integral_x, integral_y, deriv_x, deriv_y`. **`frame_index` is NOT a column
in raw `telemetry.csv` for any of these 6 sessions** — telemetry (STM32 serial, ~25 Hz) and
video (`cv2.VideoWriter`, ~30 fps target) are two independent async streams; `frame_index`
only exists in the separately-produced `synced_telemetry.csv` (join key added by a sync step
against `frame_timestamps.csv`). Raw `telemetry.csv` row counts never equal video frame counts
for this reason — expected, not corruption, except for `session_20260730_174916` (see Known
Issues: never synced at all).

`images/`, `images_cropped/`, `masks/` in the 4 `session_20260810_*` directories are
`ml_vision`'s auto-labeling pipeline output written directly into the raw bronze session
directory (not a separate tier) — `masks/` count exceeding `images/` count in 3 of 4 sessions
is plausibly one mask file per detected color per frame, not verified further (`ml_vision`'s
lane, flagged rather than investigated here per this audit's scope boundary).
`02_silver/session_20260728_102908/` and `03_gold_cropped/session_20260728_102908/` are
confirmed derived from this same raw session (not independent captures) — relevant to
train/eval disjointness if any future VLA split draws on `02_silver`/`03_gold_cropped`.

**VLA-training suitability**: real closed-loop PID trajectories, real vision (webcam), but
**no language/audio labels of any kind** — `generate_vla_dataset.py`'s synthetic
coordinate-threshold labels are the only language signal available for this regime (see Known
Issues for why that script cannot currently run against it at all).

## Regime: Jetson Track 4 Sessions (`01_bronze/session_jetson_track4_YYYYMMDD_HHMMSS/`) — NEW, real, collected today

**Producing script, confirmed by reading source**: `run_jetson_standalone.py
--record-track4-session --random-sequence`, via `runtime/session_recorder.py`
(`SessionRecorder`/`SessionTouchTap`). **Controller, confirmed by reading
`run_jetson_standalone.py`'s own control-flow (not assumed from the flag name)**: Phase A only
— `JetsonExpertPolicy` (Track 1's vision+audio+state-machine pipeline) computes `target_x/y`
on the Jetson and sends it over serial; the STM32's own `RLControl.cpp` (the 9→32→32→3 RL
control net) still runs onboard and produces the actual `theta_a/b/c` actions logged here —
this is genuinely the RL-control-net regime the hypothesis table expected, not vision-only.
Sensor: Jetson USB webcam (`640x480`, confirmed via probed video frame size, matching
`docs/CAMERA_HARDWARE.md`'s documented MJPG-30fps-only finding). Audio: `--random-sequence`'s
`ScriptedCommandSequencer` — real recognized command strings, not live speech and not
synthesized from coordinates.

10 sessions, all collected 2026-09-15 between 15:16 and 17:12 (confirmed via file timestamps):

| Session | rows = video frames | null touch/theta rows | video size |
|---|---|---|---|
| `session_jetson_track4_20260915_151627` | 2,596 | 0 | 43.6 MB |
| `session_jetson_track4_20260915_153053` | 2,294 | 0 | 36.9 MB |
| `session_jetson_track4_20260915_153247` | 2,370 | 0 | 38.1 MB |
| `session_jetson_track4_20260915_160025` | 5,057 | 0 | 80.7 MB |
| `session_jetson_track4_20260915_160509` | 4,747 | 0 | 75.3 MB |
| `session_jetson_track4_20260915_161239` | 5,193 | 0 | 83.5 MB |
| `session_jetson_track4_20260915_161712` | 4,946 | 0 | 79.0 MB |
| `session_jetson_track4_20260915_163702` | 3,535 | 0 | 58.2 MB |
| `session_jetson_track4_20260915_164115` | 4,856 | 0 | 82.3 MB |
| `session_jetson_track4_20260915_164743` | 5,119 | 0 | 88.1 MB |

**Total: 40,713 frames / rows across 10 sessions, ~665.7 MB of video.** `telemetry.csv` row
count equals `rgb_video.mp4` frame count **exactly** in all 10 sessions (by construction —
`SessionRecorder`'s single worker thread assigns `frame_index` and writes both in lockstep) —
no desync found, no corrupted/truncated video detected (every video opens and its declared
frame count matches). Zero rows with missing `touch_x`/`theta_a/b/c` in any session.

Schema (all 10 confirmed identical, 10 columns): `frame_index, host_timestamp_ms, target_x,
target_y, touch_x, touch_y, theta_a, theta_b, theta_c, audio_command`. Real recognized
commands observed across the 10 sessions: `go_red, go_green, go_yellow, go_black, forward,
backward, left, right, hold, stop` (plus blank `""` for the pre-first-command warm-up window
of each session) — a 10-word active vocabulary, wider than `RT1LiteVLA`'s hardcoded 5-word
closed vocabulary noted elsewhere in this project's docs.

**VLA-training suitability**: this is the strongest regime found — real closed-loop RL-control-net
trajectories, real Jetson-sensor vision, real (if scripted, not live-speech) language labels,
zero corruption found. It is also the **only** regime whose actions come from the RL control
net rather than PID — a genuine action-distribution difference from every other regime in this
catalog, per this kickoff's own framing (do not naively merge with PID-regime sessions).

## Regime: `touch_logger.py --log-csv` output (`01_bronze/evaluation/ground_truth_*.csv`)

**Confirmed via source** (`src/touch_logger.py`, `TouchTelemetryLogger.CSV_FIELDS`): flat CSV,
NOT session-structured (no paired video, no `frame_index`/video sync), `motor_a/b/c` raw step
counts rather than `theta_a/b/c` degrees, no `audio_command` column — exactly matches the
hypothesis table's description. 15 files found, dated 2026-08-19, 7.7 KB–663 KB each. **Schema
version drift observed**: the sampled file (`ground_truth_20260819_092555.csv`) has 20
columns, missing `raw_vision_x_mm/raw_vision_y_mm/raw_err_x_mm/raw_err_y_mm/raw_err_mm` that
the currently-installed `touch_logger.py` (25 `CSV_FIELDS`) would write — this file predates
that column addition. Any future automated ingestion of this directory should check each
file's header rather than assume a fixed column count. **Not designed as, and not usable as,
VLA training data** (no video, no language) — confirmed, matches hypothesis table.

Also in `01_bronze/evaluation/`: 3 `eval_test_*` subdirectories (`control_metrics.json` +
`trajectory_plot.png` only, dated 2026-09-15), which are `evaluate_system_control.py`-style
evaluation-run artifacts, not raw session data — no images/telemetry retained, no leakage risk.

## Other / previously-unaudited directories under `host_software/data/`

- **`host_software/data_collection/data/`** — `docs/DATA_STORAGE.md` names this as
  never-audited. **Confirmed 2026-09-15: this directory does not exist.** There is nothing to
  audit; the doc's open item should be considered resolved (as "nothing here"), not still open.
- **`host_software/data/.old/`**: `YTDown_YouTube_Ball-Balancing-Demo_*.mp4` (a downloaded demo
  video, not collected data), `YT_video1/`, `images_find_plane1/` — pre-tier scratch content,
  unrelated to VLA training, no further action needed.
- **`host_software/data/bronze/`** (distinct top-level dir from `01_bronze/`): only
  `iphone_telemetry.csv`, superseded by the per-video files above — see iPhone section.
- **`host_software/data/bronze_demo.mp4`** (~100 MB): standalone demo reel, not session data.
- **`03_gold_cropped/`, `03_pose_dataset/`, `02_silver_unified_pose/`, `03_synthetic_yolo/`,
  `03_shared_vision_masks/`, `03_shared_vision_labels_00.csv/`**: confirmed `ml_vision`-owned
  pose/YOLO training-tier outputs (each has a `dataset.yaml`/YOLO-style `images/`+`labels/`
  layout), not `session_*`-shaped and not language-labeled — out of this audit's lane per this
  kickoff's explicit scope boundary; flagged rather than reached into.
- **`04_evaluation/`** (2 files) and **`04_evaluation_expert/`** (4 CSVs + `results/`): both
  use the same 19-column PID/extended telemetry schema as the bronze sessions above, but
  **`04_evaluation/202607_telemetry_eval_raw.csv` uses the column name `host_time_ms`** where
  every other file in this project (including `04_evaluation_expert`'s own files) uses
  `host_timestamp_ms` — `evaluate_system_control.py`'s `REQUIRED_COLUMNS` needs
  `host_timestamp_ms` by that exact name, so this one file would fail that check as-is. Dated
  around 2026-07-27, i.e. before any current `01_bronze/session_*` session existed — no
  filename/date overlap found with current bronze sessions, but this was not verified at the
  raw-frame level (out of this audit's time budget); flagged as an open item for whoever next
  needs a hard disjointness guarantee, not resolved here.

## Known Issues Found (contradicts a hypothesis, or is otherwise load-bearing)

1. **`03_gold/vla_dataset.json` (22.8 MB, 54,230 samples, file mtime 2026-07-27) is fully
   orphaned.** Its `image_path` entries point at `host_software/ml_vision/data/02_silver/images/`
   — confirmed that path (and `ml_vision/data/` itself) does not exist anywhere on this
   machine. Image timestamps in the JSON exactly match `01_bronze/video1/synced_telemetry.csv`'s
   `host_timestamp_ms` values, meaning this file was generated from the **iPhone** regime by
   some earlier version of the pipeline, not from any current `01_bronze/session_*` PID or
   Track4 data — it predates all 6 current PID sessions (mtime is before the earliest one).
   **0% of its referenced images resolve on disk.** This file cannot be loaded as training
   data today; anyone pointing `train_vla.py` at `03_gold/vla_dataset.json` as-is will fail
   or (worse) silently train on whatever `dataset.py`'s blank-image fallback substitutes.
2. **`generate_vla_dataset.py` cannot currently process any of the 6 PID-regime bronze
   sessions.** It hardcodes `csv_path = .../telemetry.csv` and requires `row["frame_index"]`,
   but confirmed above: raw `telemetry.csv` has no `frame_index` column in any PID session —
   only the separately-produced `synced_telemetry.csv` does. Running it today against
   `01_bronze` as-is would either crash (KeyError) on the first PID session its `session_*`
   glob reaches, or — since that same glob also matches `session_jetson_track4_*` — process
   only the Track4 sessions successfully (their `telemetry.csv` does have `frame_index`) while
   crashing on the first PID session encountered, and its synthetic coordinate-threshold
   audio-labeling logic would also silently discard Track4's real `audio_command` column since
   it never reads it. Not fixed here (read-only audit, out of scope) — flagged as a concrete,
   verified blocker for anyone about to run this script.
3. **`session_20260810_104132`'s `masks/` directory is empty** (0 files) despite `images/`
   (28,794) and `images_cropped/` (14,834) being fully populated — the other 3
   `session_20260810_*` sessions all have populated `masks/`. Looks like an incomplete
   auto-labeling run specific to this one session.
4. **`session_20260730_174916` has no `synced_telemetry.csv`** — raw `telemetry.csv` (9,425
   rows) and video (9,228 frames) were never joined, unlike every other PID session. Unusable
   for frame-level training without running the sync step first.
5. **Stray DVC pointers found on `01_bronze/session_jetson_track4_*/` (2026-09-15).** All 10
   Track4 session directories now have a matching `.dvc` file
   (`session_jetson_track4_<ts>.dvc`), dated after this audit began. `docs/DATA_STORAGE.md`
   is explicit that `01_bronze`/`02_silver` stay **outside** DVC (tar/zip-per-session is the
   documented transport, not `dvc add`). This audit did not create these — found, not caused
   — and per this audit's read-only/no-data-file-modification scope, they were left alone
   rather than removed. Flagged for whoever is running DVC commands against this repo: this
   looks like `01_bronze` being tracked in a way the project's own storage policy says it
   shouldn't be.
6. **`touch_logger.py`-produced CSVs are not all the same schema version** — see the
   `evaluation/ground_truth_*.csv` section above.

## Machine-readable manifest

`host_software/ml_jetson_vla/data_processing/session_manifest.json` — one entry per
`session_*`-shaped bronze directory (PID and Track4 regimes; the flat `touch_logger.py`/
iPhone/`.old`/vision-owned directories above are not session-shaped in the same sense and are
catalogued in prose only, not in the manifest). JSON was chosen over CSV because several
fields are naturally list-valued (`schema_columns`, `audio_commands_observed`) and because
`convert_to_lerobot.py`/any future loader is already Python reading this repo — `json.load()`
needs no extra dependency `pandas.read_csv` wouldn't also need, and nested/list fields don't
need CSV's flattening conventions. See that file's own top-level `"_format"` key for the field
definitions.
