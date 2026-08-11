# Audio Evaluation & Notebook Refactor Plan

## Status
In progress. Corruption audit + quarantine, evaluation-script extraction, and the live receiver's label-order + preprocessing fixes are all done and measured. Live-stream validation shows the real bottleneck is background-training-data quality, not anything left to fix in code — the retrain track (training extraction, retrain, background diversification) is next and is now the load-bearing piece, not a follow-on. Companion to [`.agents/agent_ml_audio.md`](../../../../.agents/agent_ml_audio.md).

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

**Remediation applied (2026-08-11): quarantine, not regeneration.** Piper TTS (needed to regenerate the synthetic clips cleanly) isn't installed in the project env, and installing it plus downloading the 3 specific voice models (`lessac`, `libritts_r`, `ryan`) was judged disproportionate to fixing 1.9% of the dataset — so we quarantined instead of regenerated. [`data_processing/apply_dataset_quarantine.py`](../../data_processing/apply_dataset_quarantine.py) moved the 254 flagged non-background clips out of `training_v2/{split}/{label}/` into a sibling `data/synthetic+real_dataset_large/_quarantined_corrupt/{split}/{label}/` tree (move, not delete — reversible). `_background_` clips were explicitly excluded, per the caveat above. Manifest: [`data_processing/reports/quarantine_manifest.json`](../../data_processing/reports/quarantine_manifest.json).

| Class | Quarantined (train) | Quarantined (val) |
|---|---|---|
| `go_red` | 104 | 1 |
| `stop` | 81 | 0 |
| `hold` | 58 | 0 |
| `go_blue` | 4 | 0 |
| `left` | 2 | 4 |

**Verified clean:** re-ran `audit_dataset_corruption.py` after quarantine — every command class now scans at 0 empty / 0 truncated (18,877 clips remaining, only the untouched 113 `_background_` flags remain, tracked separately as above). Dataset is ready to retrain against once `training/train_audio_command_classifier.py` exists (step 3 below).

If 1:1 dataset-size restoration ever becomes worth it, the deferred option is: install `piper-tts`, download the matching voice checkpoints, and regenerate synthetic clips from source voice/text for the quarantined `en_US-<voice>-medium` files; quarantined `real_speakerNN` files can only be dropped or re-recorded, not regenerated. Not pursued now — re-run evaluation after retraining first to see whether it's even needed.

`audit_dataset_corruption.py` and `apply_dataset_quarantine.py` now live in `data_processing/` as standing QC/cleanup scripts — wire the audit check into the ingestion pipeline during step 4 of the refactor below so this doesn't silently recur as the dataset grows.

## Larger Background/Noise-Profile Source: 23-Minute Lab Recording

There is a longer general-lab-sounds recording available — roughly 23 minutes, currently unlabeled — that is a much richer background source than the short clips the noise profile and background training class currently draw from. **Confirmed: this is a new sample, not yet integrated into the dataset before** — distinct from `data/01_background_noise/lab_background_sound_01.wav` and the other existing files already in that directory. Those existing files were themselves recorded on different days, so `01_background_noise/` already has some session-level diversity going for it; the 23-minute recording is an additional, larger source to bring in on top of that, not a replacement or duplicate of anything already there.

It is not labeled at the granularity the classifier needs (i.e., not chopped into fixed-length clips with a class), but it can be:

- **Segment** it into fixed-length windows matching the model's input clip length (same framing the existing `_background_` clips use), producing many more background training/eval samples than the current ~1,290-clip pool.
- **Label at the segment level, not as one blanket clip.** A 23-minute "general lab sounds" recording almost certainly contains a mix of sub-conditions (silence/room tone, talking, footsteps, door/HVAC noise, equipment hum, possibly incidental typing or bench-work sounds). Spot-review segments and tag them — this doesn't require new *model* classes (per the earlier ball-drop/typing discussion, everything here still collapses to the single `_background_` class for training), but sub-tagging lets the eventual `evaluations/` tooling report background accuracy broken out by noise sub-type, which is exactly the resolution needed to tell whether the background fix is working uniformly or only on the easy cases.
- **Fold into, not replace, the existing sources** — it adds a new session/environment to the pool; the existing multi-day recordings stay in the mix too.

This work belongs in the `data_processing/` extraction (step 4 below), since it's dataset assembly, not a model or eval change.

## Evaluation Extraction — Done, and a New Finding: Production Preprocessing Doesn't Match Training/Eval

Built [`evaluations/evaluate_audio_classifier.py`](../../evaluations/evaluate_audio_classifier.py): loads `models/pytorch_v3/audio_command_classifier_state_dict_v3.pth` + `labels.json`, runs the val split, persists both raw counts (JSON) and a rendered heatmap (PNG) to `evaluations/reports/` on every run instead of vanishing off a screen. This closes the reproducibility gap that made the original confusion matrix un-followup-able.

**Getting it to reproduce the known acc=0.870 baseline surfaced a real bug, found by direct A/B testing.** The live inference path (`audio_receiver_pytorch.py`) diverges from how this model was actually trained/evaluated on four points simultaneously:

| Divergence | Effect on measured accuracy |
|---|---|
| `audio_receiver_pytorch.py` hardcodes its own 12-class label order, which is **not alphabetical** | Using it to interpret predictions collapses accuracy to ~7% |
| `align_speech_to_fixed_length`'s active-region crop (built for finding speech in a noisy rolling live buffer) | Costs ~7 points when applied to already-isolated, pre-cut dataset clips |
| Peak-renormalizing each clip to 0.95 | Costs ~2-3 points |
| Applying the spectral-subtraction noise profile at all | Costs ~30-40 points — it looks tuned for live mic/robot-noise, not clean dataset audio |

`models/pytorch_v3/labels.json` **is** alphabetical (the standard convention for scanning class folders — matches how the dataset folders are actually named/sorted), so it's the correct order; the receiver's hardcoded list is not. Reverting all four to the simple/matching form (`evaluate_audio_classifier.py`'s current defaults: alphabetical labels, direct pad/truncate, no renormalization, no noise profile) reproduces **86.3%** against the recorded 0.870 baseline — the ~0.7pt gap is fully explained by the 5 val clips removed during quarantine, not a remaining methodology error.

**This is a separate, likely-significant finding, not a data-quality issue:** if the deployed receiver really is decoding a 12-class checkpoint with the wrong label order and/or degrading input with the wrong preprocessing, that's a strong independent candidate explanation for "produces incorrect outputs during concurrent robot operation" — arguably more directly than the background-class imbalance this whole effort started from. Needs a decision on priority (see Open Questions).

**Caveat on the go_red/go_green numbers from this run:** this evaluation still used the original `v3` checkpoint, trained before the corruption quarantine — the quarantine only removed 254 clips from `training_v2` (mostly from `train`; only 1 `go_red` clip was in `val`), and training hasn't happened yet. Any change in the `go_red`/`go_green` split seen in this run's output vs. the original matrix reflects the corrected *evaluation methodology*, not the corruption fix. The corruption fix can only be measured once step 4 (training extraction + retrain) is done and this same evaluation script is re-run.

## Stage 1 Fix (Done): Label-Order Bug in the Live Receiver

Per the priority call above, tackled the smaller/unambiguous fix first, measured it, then moved to the bigger retrain track (step 4+). The label-order bug was a clean, low-risk fix (unlike the crop/renorm/noise-profile questions, which involve real design trade-offs for live streaming vs. offline clips and weren't touched here).

**Fix:** [`audio_receiver_pytorch.py`](../../audio_receiver_pytorch.py)'s `AudioCommandReceiver.__init__` now loads `labels.json` from the same directory as whatever checkpoint `model_path` points to, and uses that as the authoritative class order — matching how the model was actually trained (alphabetical, from scanning dataset class folders) instead of the hand-maintained list that had drifted out of sync for the 12-class case. Falls back to corrected (now-alphabetical) hardcoded lists only if no sibling `labels.json` exists, for older checkpoints that don't ship one. Confirmed the 7-class hardcoded fallback was already correct — only the 12-class one had drifted.

**Measured impact (same checkpoint, same otherwise-unchanged production-style preprocessing — active-region crop, peak renormalization, and spectral-subtraction noise profile all still applied, on val split):**

| Stage | Label order | Accuracy |
|---|---|---|
| Before (the bug) | receiver's hardcoded (wrong) order | 7.4% (169/2269) |
| After (this fix) | alphabetical, from `labels.json` | 47.9% (1086/2269) |

**+40.5 points from this one fix alone**, with everything else about the production pipeline untouched. Confirms this was a real, live bug — not just an artifact of the offline evaluation methodology — and that it was likely a significant contributor to "incorrect outputs during concurrent robot operation" on its own.

**Not yet closed:** 47.9% is still far below the 86.3% achieved with the idealized offline pipeline (no crop, no renorm, no noise profile). That gap is the second, harder piece — the crop/renorm/noise-profile mismatch — which is the next thing to work through, likely in tandem with the retrain track below rather than as a quick fix, since it involves an actual design decision (see Open Questions).

## Stage 2 Fix (Done, but Did Not Close the Real Gap): Simplify Live Preprocessing to Match Training

Decision: simplify the live receiver to match training (rather than retrain to match the receiver's crop/renorm/noise-profile pipeline). Implemented in [`audio_receiver_pytorch.py`](../../audio_receiver_pytorch.py): `_process_loop` now feeds the rolling audio buffer straight to `waveform_to_spectrogram` with no `align_speech_to_fixed_length` crop, no peak renormalization, and no noise-profile subtraction — the noise-profile loading code in `__init__` was removed outright rather than left dead. `min_confidence`/`min_margin` gating stays; that's a downstream accept/reject decision, not a preprocessing step, and doesn't need to match training.

**Cross-check before implementing:** `final_tester_audio.py`, a separate/legacy self-contained script, comments that its own crop+renormalize step is "same as the bronze -> silver step that built the train set" — which would have argued against this fix. Checked it before proceeding: its STFT uses n_fft=256 (129 freq bins), not the 255/128-bin convention `generate_noise_profile.py` and `audio_receiver_pytorch.py` actually use — even though its model class shapes are checkpoint-compatible with `v3.pth`. Since my empirical A/B testing already reproduced the known 0.870 baseline using the 255/128 convention (impossible if that convention were actually wrong, per how badly the wrong label order tanked accuracy), treated that comment as an unverified assumption from a disconnected/earlier script rather than counter-evidence, and proceeded.

**Offline per-clip result:** using this exact simplified pipeline is what already produced 86.3% in the evaluation-extraction section above (`evaluate_audio_classifier.py`'s defaults now match `_process_loop` exactly).

**Live continuous-stream result: only 4/11 (36%).** Built [`evaluations/evaluate_live_receiver_stream.py`](../../evaluations/evaluate_live_receiver_stream.py), which drives the actual `AudioCommandReceiver` (not a reimplementation) through `data/02_silver/master_evaluation_audio.wav` in its real file-playback mode — the same 11-command-at-10s-intervals sequence `create_master_audio.py` built, overlaid on looped background noise. Report: `evaluations/reports/live_stream_eval_20260811T033925Z.json`.

| t | expected | detected |
|---|---|---|
| 0s | go_grey | *(miss)* |
| 10s | go_blue | go_blue ✓ |
| 20s | go_green | *(miss)* |
| 30s | go_yellow | go_yellow ✓ |
| 40s | go_red | go_red ✓ |
| 50s | forward | *(miss)* |
| 60s | left | *(miss — detected go_red instead, a real misclassification)* |
| 70s | right | *(miss)* |
| 80s | backward | *(miss)* |
| 90s | hold | *(miss)* |
| 100s | stop | stop ✓ |

Looking at every command the receiver latched across the full 120s (not just the 11 target windows): it output `_background_` almost continuously — dozens of times throughout, including during windows where a command word is clearly present — with only 5 non-background detections in the entire stream. This is not a preprocessing-pipeline problem anymore; it's the original background-class issue this whole investigation started from (see "Background leaks into movement commands" in the confusion-matrix analysis above), now showing up directly: `master_evaluation_audio.wav` mixes background noise under every command, continuously, which is the realistic operating condition — and the model, trained on a background class that's only ~9.5% of the data from 4 source recordings, isn't robust to command-plus-noise mixtures at all. It defaults to background almost everywhere.

**Conclusion: Stage 1 + Stage 2 fixed two real, confirmed bugs (wrong label order, mismatched preprocessing) and both were worth fixing, but neither is the dominant lever on real-world performance.** The 86.3% clean-clip number was necessary to establish as a correct baseline, but it doesn't predict live behavior — noisy-condition performance is bottlenecked by the background-training-data problem identified back in the corruption/background-diversification sections. That makes the retrain track (steps 4-7, especially the background-diversification work with the 23-minute recording) the load-bearing fix, not an optional follow-on. Re-run `evaluate_live_receiver_stream.py` after that retrain to see if this closes.

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

1. ~~Extend the corruption audit beyond `go_red`~~ — **done.** `audit_dataset_corruption.py` scanned all 12 classes/both splits; found corruption concentrated in `go_red`/`stop`/`hold`/`go_blue`/`left`, zero elsewhere.
2. ~~Apply remediation~~ — **done.** `apply_dataset_quarantine.py` moved the 254 flagged non-background clips to `_quarantined_corrupt/`; re-audit confirms 0 empty/truncated across every command class. `_background_`'s flags were left in place (tracked separately, not corruption).
3. ~~Extract evaluation logic~~ — **done.** `evaluations/evaluate_audio_classifier.py` persists confusion matrix JSON + PNG on every run, and reproduces the known 0.870 baseline (86.3% on the post-quarantine val set).
4. ~~Fix live receiver label order (Stage 1)~~ — **done.** 7.4% → 47.9% on the production-preprocessing-otherwise-unchanged test. Real bug, real fix.
5. ~~Simplify live receiver preprocessing to match training (Stage 2)~~ — **done**, but revealed the real bottleneck isn't preprocessing: live continuous-stream test (`evaluate_live_receiver_stream.py` against `master_evaluation_audio.wav`) only detects 4/11 commands correctly, with `_background_` dominating almost the entire stream. Root cause traces back to the background-training-data problem, not anything fixable in the receiver code.
6. **Next: the retrain track, now confirmed as the load-bearing fix rather than an optional follow-on.** Extract training logic into `training/train_audio_command_classifier.py`, targeting the 12-class layout (matching the actual deployed model), not the notebook's 6-class one. Retrain against the now-quarantined dataset.
7. Re-run `evaluate_audio_classifier.py` (offline) and `evaluate_live_receiver_stream.py` (live) after retraining — offline to confirm `go_red`/`go_green` and `forward`/`hold` movement, live to check whether the 4/11 number moves at all, since that's the one that actually reflects the reported bug.
8. Extract data loading/labeling into `data_processing/`, including:
   - An ingestion path that reads the existing `synthetic+real_dataset_large/training_v2/{train,val}/<class>/` folder structure directly as the dataset contract (no generating source code to match against — the folder layout is ground truth), naturally skipping `_quarantined_corrupt/` since it sits outside `training_v2/`.
   - The corruption-detection audit wired in as a standing pipeline QC step, not a one-off.
   - Segmentation + sub-labeling tooling for the new 23-minute lab recording.
9. **Prioritize the background-source-diversification + noise-profile-rebuild work** (now including the 23-minute recording as a new source alongside the existing multi-day recordings) as part of the retrain in step 6-7, given the live-stream test shows this is the actual bottleneck, not a nice-to-have.

## Open Questions

- Whether retraining with better background diversity actually closes the 4/11 live-stream gap, or whether the model architecture itself (Conv1D×3 + Dense, ~13.5K params) is too small to separate command-plus-noise mixtures reliably — won't know until step 6-7 is done and re-measured.
- Ball-drop/typing-as-distinct-label question from prior discussion remains open, deferred until per-sub-type eval reporting exists to quantify rather than guess.
