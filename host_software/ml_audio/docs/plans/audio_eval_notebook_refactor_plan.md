# Audio Evaluation & Notebook Refactor Plan

## Status
Draft — planning only, no code changes made yet. Companion to [`.agents/agent_ml_audio.md`](../../../../.agents/agent_ml_audio.md).

## Context

`host_software/ml_audio/` was reactivated under Roadmap Phase 2 ("Audio Verification") to debug the known false-positive-on-background issue. Previously there was no persisted confusion matrix or evaluation artifact anywhere in the module — the team ran evaluation ad hoc inside a notebook and read the plot off-screen. On 2026-08-11 a run was captured (image, reproduced below as a table) that gives us the first real evidence to work from instead of impressions.

The evaluation method that produced this run lives in **`audio_command_classifier_aligned_before_deterministic_patch_final.ipynb`** — a single monolithic notebook (~3 MB, authored by a former lab partner) that combines dataset loading, augmentation, model definition, the training loop, and confusion-matrix generation (`tf.math.confusion_matrix`) in one file with no persisted output. This plan proposes breaking it apart, in the same spirit as the modular `core/` / `data_processing/` / `training/` / `evaluations/` / `tests/` / `docs/` convention already codified for `ml_vision` in `.agents/AGENTS.md`.

**Discrepancy, now resolved.** a prior read-only pass over this notebook found it hard-coded to exactly 6 classes (`["go_red","go_blue","go_green","go_yellow","hold","stop"]`, no `_background_`) and set to raise `ValueError` on any other label layout. The confusion matrix above is 12-class (includes `_background_`, `backward`, `forward`, `go_grey`, `left`, `right`). We now know why: the **12-class model and the large dataset that trained it (`data/synthetic+real_dataset_large/`) were produced by a *different, second* former lab partner's code — not the notebook above, and not the same person.** That generation code is not present in the repo and is not expected to surface.

**Decision: the dataset's on-disk file structure IS the ground truth, not a stand-in for missing code.** There is no script to locate or reconstruct — `data_processing/` will treat `synthetic+real_dataset_large/training_v2/{train,val}/<class_name>/*.wav` (12 class folders) directly as the contract to preserve and build tooling around. Do not extract/refactor code from the 6-class notebook as if it were the source of the 12-class run — the two are unrelated artifacts from two different people and both stay as read-only ground truth.

## Confusion Matrix — 2026-08-11 run (acc = 0.870, row-normalized)

Rows = true label, columns = predicted label, raw counts:

| true \ pred | _background_ | backward | forward | go_blue | go_green | go_grey | go_red | go_yellow | hold | left | right | stop | row total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **_background_** | 151 | 5 | 3 | 9 | 5 | 1 | 8 | 4 | 10 | 13 | 23 | 8 | 240 |
| **backward** | 1 | 117 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 120 |
| **forward** | 6 | 0 | 89 | 0 | 0 | 4 | 1 | 0 | 17 | 0 | 0 | 3 | 120 |
| **go_blue** | 0 | 0 | 0 | 222 | 17 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 240 |
| **go_green** | 1 | 0 | 0 | 1 | 238 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 240 |
| **go_grey** | 0 | 0 | 0 | 0 | 0 | 117 | 3 | 0 | 0 | 0 | 0 | 0 | 120 |
| **go_red** | 4 | 0 | 0 | 0 | 61 | 2 | 161 | 4 | 8 | 0 | 0 | 0 | 240 |
| **go_yellow** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 239 | 0 | 0 | 0 | 0 | 240 |
| **hold** | 1 | 0 | 35 | 0 | 0 | 3 | 0 | 0 | 195 | 0 | 0 | 0 | 234 |
| **left** | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 113 | 0 | 1 | 120 |
| **right** | 0 | 0 | 1 | 0 | 0 | 6 | 0 | 0 | 3 | 0 | 110 | 0 | 120 |
| **stop** | 11 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 227 | 240 |

## Analysis — three distinct failure clusters, do not conflate

1. **Background leaks into movement commands (the reported issue, now quantified).** True `_background_` recall is only 151/240 = **62.9%** — over a third of background clips (89/240) get classified as some command. The leakage is not evenly spread: `right` (23), `left` (13), and `hold` (10) absorb most of it, while `go_grey`, `go_green`, `go_yellow` barely attract any background at all. This is the operationally dangerous case — ambient/robot noise producing an unintended motion command — and is consistent with the earlier finding that background training data is only ~9.5% of the train set, sourced from just 4 raw recordings. It does **not** explain clusters 2–3 below.

2. **`go_red` → `go_green` confusion (61/240 = 25.4%). Root cause now confirmed: dataset corruption, not feature-space overlap or mislabeling.** Large, specific, and one-directional (green does not reciprocally get called red — only 1/240 `go_green` errors exist). A manual audit ([`docs/dataset_info_audio.md`](../dataset_info_audio.md)) found the `go_red` folder in `synthetic+real_dataset_large/` contains **truncated and empty clips still labeled `go_red`** — samples containing only partial utterances ("go", "go re", "red") or no speech at all, across both synthetic TTS voices and real recordings. The label itself isn't wrong (these clips were genuinely intended as `go_red` commands), but the audio content is broken — training on truncated/empty "go_red" clips degrades that class's learned decision boundary, which is a direct, sufficient explanation for elevated `go_red` misclassification. This is unrelated to background balance and does not need a feature-space investigation — it needs a data-quality fix. See the new section below.

3. **`forward` ↔ `hold` bidirectional confusion** (`forward`→`hold` 17/120 = 14.2%; `hold`→`forward` 35/234 = 15.0%). Also independent of background balance — these two command classes appear to sit close together in feature space.

Minor, lower-priority: `go_blue` → `go_green` (17/240 = 7.1%, one-directional).

**Conclusion:** the background-noise/matched-filter fix (more diverse background sources, rebuilt noise profile — see prior planning discussion) should measurably improve cluster 1; the `go_red`/`go_green` fix is now a dataset-cleanup task (see below), not a training or architecture change; cluster 3 (`forward`/`hold`) remains unexplained and should get the same corruption audit before assuming it's a genuine feature-space problem. Track all three separately so a fix for one isn't mistaken for a fix for all.

## Dataset Corruption: `synthetic+real_dataset_large`

[`docs/dataset_info_audio.md`](../dataset_info_audio.md) documents a manual audit finding **the dataset is corrupted**, so far confirmed in the `go_red` class:

- **Truncated commands** — clips cut off mid-word or mid-phrase but still filed under the full `go_red` label, e.g. only "go" (`en_US-lessac-medium__go_red__00071`), only "go re" (`en_US-lessac-medium__go_red__00199`, `en_US-libritts_r-medium__go_red__00089`, `en_US-ryan-medium__go_red__00021`, `en_US-ryan-medium__go_red__00063`), or only "red" (`en_US-lessac-medium__go_red__00535`, `en_US-lessac-medium__go_red__00580`).
- **Empty clips** — no speech content at all, still labeled `go_red` (`real_speaker02__go_red__speaker02__go_red__001_018`, `..._002_002`, `..._002_006`).

**Dataset composition, inferred from filenames:** the `en_US-<voice>-medium` prefixes (`lessac`, `libritts_r`, `ryan`) are Piper TTS voice model names — so the "synthetic" half of `synthetic+real_dataset_large` is machine-generated speech from at least 3 distinct TTS voices. The `real_speakerNN` prefixes are recordings from actual human speakers (at least `speaker02` confirmed so far). This matches the `synthetic+real_dataset_large` directory name and matters for remediation: synthetic clips can likely be **regenerated cleanly** from the same TTS voice/text (fixing the corruption at the source), while corrupted real-speaker clips can only be **dropped or re-recorded**, not regenerated.

**Full-dataset audit complete (2026-08-11).** Built [`data_processing/audit_dataset_corruption.py`](../../data_processing/audit_dataset_corruption.py), which reuses the exact energy-gate thresholds from the production inference path (`align_speech_to_fixed_length` in `audio_receiver_pytorch.py`: `peak < 0.03` or `rms < 0.003` ⇒ "empty") for consistency, plus a per-class statistical-outlier check on active-speech duration (robust z-score via MAD) to catch truncation candidates without needing a transcript/ASR. Ran it across all 12 classes, both `train` and `val` (19,131 clips total). Full report: [`data_processing/reports/dataset_corruption_audit.json`](../../data_processing/reports/dataset_corruption_audit.json).

**Result: the "same kind of corruption throughout the dataset" hypothesis is not confirmed — it's concentrated, not uniform.** 367/19,131 clips flagged (1.9% overall), but split very unevenly:

| Class (train) | Total | Empty | Truncated | % flagged |
|---|---|---|---|---|
| `go_red` | 2100 | 82 | 22 | **5.0%** |
| `stop` | 2100 | 50 | 31 | **3.9%** |
| `hold` | 2094 | 56 | 2 | 2.8% |
| `go_blue` | 2100 | 4 | 0 | 0.2% |
| `left` | 600 | 0 | 2 | 0.3% |
| `backward`, `forward`, `go_green`, `go_grey`, `go_yellow`, `right` | — | 0 | 0 | **0%** |

(`val` split mirrors this pattern at smaller scale: `go_red` 1 empty, `left` 4 truncated, everything else clean. Full per-class table in the JSON report.)

So: `go_red` is confirmed as the worst offender (matches the manual audit and explains the confusion-matrix finding), `stop` and `hold` carry real but smaller corruption, and **6 of the 12 command classes have zero flagged clips**. This means the `go_red`↔`go_green` cluster is explained by `go_red`'s corruption specifically — `go_green` itself is clean, consistent with the one-directional confusion pattern already observed. It also means cluster 3 (`forward`/`hold`) is only *partly* explained: `hold` does carry some corruption (2.8%) but `forward` has none at all, so that confusion is more likely a genuine feature-space issue after all, not primarily a data defect — worth retesting after `hold`'s corrupted clips are cleaned, but don't expect it to fully close.

**One important caveat on `_background_`:** it also flagged high on the "empty" check (96/1290 train, 17/240 val, ~7.4%) — but this is a different phenomenon from the `go_red`-style defect, not the same corruption. A near-silent clip labeled `_background_` isn't mislabeled or broken the way a truncated `go_red` clip is; if anything it's *too* correct — real background noise during robot operation is not silent, so a background class skewed toward near-silence undertrains the model on the actual failure condition (concurrent motor/typing/impact noise). Track this under the background-diversification thread (`Larger Background/Noise-Profile Source` section below), not as dataset corruption to remediate the same way.

**Remediation approach (data-quality fix, not a training/architecture change):**

- For `go_red`, `stop`, and `hold` flagged clips: cross-reference filename against the TTS-voice-vs-real-speaker convention noted above — regenerate synthetic (`en_US-<voice>-medium`) clips from source voice/text where feasible, drop or flag for re-recording real-speaker (`real_speakerNN`) clips.
- Leave `backward`, `forward`, `go_blue`, `go_green`, `go_grey`, `go_yellow`, `right` alone — the audit found no evidence of this defect there.
- Re-run `audit_dataset_corruption.py` after cleanup to confirm flagged counts actually drop to near-zero for the affected classes.
- Re-run evaluation after retraining to confirm the `go_red`/`go_green` cluster closes and to re-measure `forward`/`hold` with `hold`'s corrupted clips removed — don't assume either closes without re-measuring.

`audit_dataset_corruption.py` now lives in `data_processing/` as a standing QC script — wire it into the ingestion pipeline during step 4 of the refactor below so this doesn't silently recur as the dataset grows.

## Larger Background/Noise-Profile Source: 23-Minute Lab Recording

There is a longer general-lab-sounds recording available — roughly 23 minutes, currently unlabeled — that is a much richer background source than the short clips the noise profile and background training class currently draw from. **Confirmed: this is a new sample, not yet integrated into the dataset before** — distinct from `data/01_background_noise/lab_background_sound_01.wav` and the other existing files already in that directory. Those existing files were themselves recorded on different days, so `01_background_noise/` already has some session-level diversity going for it; the 23-minute recording is an additional, larger source to bring in on top of that, not a replacement or duplicate of anything already there.

It is not labeled at the granularity the classifier needs (i.e., not chopped into fixed-length clips with a class), but it can be:

- **Segment** it into fixed-length windows matching the model's input clip length (same framing the existing `_background_` clips use), producing many more background training/eval samples than the current ~1,290-clip pool.
- **Label at the segment level, not as one blanket clip.** A 23-minute "general lab sounds" recording almost certainly contains a mix of sub-conditions (silence/room tone, talking, footsteps, door/HVAC noise, equipment hum, possibly incidental typing or bench-work sounds). Spot-review segments and tag them — this doesn't require new *model* classes (per the earlier ball-drop/typing discussion, everything here still collapses to the single `_background_` class for training), but sub-tagging lets the eventual `evaluations/` tooling report background accuracy broken out by noise sub-type, which is exactly the resolution needed to tell whether the background fix is working uniformly or only on the easy cases.
- **Fold into, not replace, the existing sources** — it adds a new session/environment to the pool; the existing multi-day recordings stay in the mix too.

This work belongs in the `data_processing/` extraction (step 4 below), since it's dataset assembly, not a model or eval change.

## Proposed Modular Refactor

Target layout for `host_software/ml_audio/`, mirroring the `ml_vision` convention:

- **`core/`** (new) — deterministic DSP: matched filter, spectral-subtraction noise-profile application, energy gating, feature extraction. Currently scattered across `audio_receiver_pytorch.py` / `audio_pytorch_runtime.py`.
- **`data_processing/`** (exists, currently only `generate_noise_profile.py`) — dataset assembly and background-clip diversification; absorb the notebook's data-loading/labeling cells, add segmentation/labeling tooling for the 23-minute lab recording, and add an ingestion path that reads the existing `synthetic+real_dataset_large/training_v2/{train,val}/<class>/` layout as-is (see discrepancy note above — we're preserving compatibility with a second lab partner's dataset contract we don't have the generating source for, not re-deriving it from scratch).
- **`training/`** (new) — model definition + training loop extracted from the notebook into a plain script, e.g. `train_audio_command_classifier.py`.
- **`evaluations/`** (new) — confusion matrix + accuracy reporting extracted from the notebook, **persisted** (raw counts JSON + plot) on every run instead of ephemeral notebook output. This directly closes the reproducibility gap that made this exercise mostly guesswork.
- **`tests/`** (exists, `test_audio.py`) — unchanged, functional tests only.
- **`models/`** (exists) — unchanged.
- **`docs/`** (new, this file) — plans and pipeline docs, mirroring `ml_vision/docs/`.

**Migration principle:** lift-and-shift first, no architecture or retraining changes bundled in. Extract notebook cells into equivalent scripts with output parity, get persisted evaluation artifacts working, *then* act on the background-diversification/noise-profile plan and the red/green + forward/hold investigations as separate follow-on tasks. Keep the original notebook in place as read-only historical reference (same treatment `ml_vision` gives `experimental_variants/`) — do not delete it. Because two former lab partners' work is being consolidated here (6-class notebook author, and the separate, unlocated author of the 12-class dataset/training code), treat both existing artifacts — the notebook *and* the `synthetic+real_dataset_large/` folder contract — as read-only ground truth to preserve, not to silently pick one and discard the other.

## Proposed Order of Work

1. Extend the corruption audit beyond `go_red` — build the automated QC check (duration/VAD/energy) and run it across all 12 classes to find the true extent of the problem before deciding on remediation scope.
2. Extract evaluation logic first (lowest risk, highest immediate value) into `evaluations/evaluate_audio_classifier.py`, persisting the confusion matrix on every run — needed as the baseline to confirm the corruption fix actually closes the `go_red`/`go_green` gap once applied.
3. Extract training logic into `training/train_audio_command_classifier.py`, targeting the 12-class layout (matching the actual deployed model), not the notebook's 6-class one.
4. Extract data loading/labeling into `data_processing/`, including:
   - An ingestion path that reads the existing `synthetic+real_dataset_large/training_v2/{train,val}/<class>/` folder structure directly as the dataset contract (no generating source code to match against — the folder layout is ground truth).
   - The automated corruption-detection QC check from step 1, wired in as a standing pipeline step.
   - Segmentation + sub-labeling tooling for the new 23-minute lab recording.
5. Apply remediation (regenerate/drop/re-record corrupted clips per class), retrain, and re-run evaluation to confirm the `go_red`/`go_green` cluster actually closes.
6. Only after the above land: execute the background-source-diversification + noise-profile-rebuild plan (now including the 23-minute recording as a new source alongside the existing multi-day recordings), and revisit `forward`/`hold` — check it for the same kind of corruption before assuming it's a genuine feature-space problem.

## Open Questions

- Full extent of dataset corruption — confirmed in `go_red`, unknown elsewhere. Blocks step 1.
- Ball-drop/typing-as-distinct-label question from prior discussion remains open, deferred until `evaluations/` exists to quantify rather than guess.
