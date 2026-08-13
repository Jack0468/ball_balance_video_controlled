# Audio Evaluation & Notebook Refactor Plan

## Status
In progress. Corruption audit + quarantine, evaluation-script extraction, the live receiver's label-order + preprocessing fixes, and a full multi-seed Colab tuning sweep (v4 -> v5 -> v6) are all done and measured. **v6** (`noise_seed0`, `models/pytorch_v6/`) is the best checkpoint produced so far by nearly every offline measure (89.6% accuracy, best `go_red`/`hold` recall yet) but ties v4's 6/11 rather than beating it on the live-stream test, with a worse failure mode (real misclassifications, not just safe misses) -- the third straight case of an offline win not translating to a live-stream win. **The live-stream ceiling has sat at 6/11 across two different mechanisms now (v4, v6)**, and the clearest remaining lever is the domain gap already identified (5 of 12 classes still 100% synthetic, and that's exactly where the misses cluster) -- recording real samples for those classes and retraining as v7 is the next step, not further tuning. Building the training script also surfaced an important correction to an assumption made earlier in this same plan: see "Correction: the conv/batchnorm layers are NOT frozen/shared across checkpoints" below. Companion to [`.agents/agent_ml_audio.md`](../../../../.agents/agent_ml_audio.md).

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

## Correction: the conv/batchnorm layers are NOT frozen/shared across checkpoints

Before building the training script, the working assumption (based on `AudioCommandClassifier` in `audio_command_classifier_pytorch.py` registering its three conv/batchnorm blocks as `register_buffer` rather than `nn.Parameter`, with `forward()` always calling `F.batch_norm(..., training=False)`) was that these layers are a **frozen feature extractor exported once from the original 6-class model and reused unchanged across every checkpoint** — i.e. that the 12-class v3 checkpoint only differs from the 6-class original in its dense head, and "training" a new checkpoint would just mean fitting a linear classifier on top of those frozen 48-dim pooled features.

**This is false, and was caught by a round-trip check before it wasted a training run.** Direct comparison of `models/pytorch_v3/audio_command_classifier_state_dict_v3.pth` against the constants baked into `audio_command_classifier_pytorch.py` shows every conv/batchnorm buffer differs substantially, not just the dense head — e.g. `norm_variance` off by 21.9, `conv1_weight` off by 0.54 max-abs, `bn3_gamma` off by 1.5. Confirmed conclusively: applying v3's own trained dense head to features extracted from the *untouched baked-in* conv/bn buffers gives ~10% accuracy, vs. v3's real 86.3%. So whatever process produced v3 trained (or fine-tuned) the entire network, not just a dense head — consistent with the plan's existing finding that the 12-class checkpoint came from a second, unlocated lab partner's code, not the 6-class notebook's.

**Practical consequence:** `AudioCommandClassifier` itself can't be trained directly either way — it has no parameters, only buffers, and its BN is hardcoded to eval-mode. [`training/train_audio_command_classifier.py`](../../training/train_audio_command_classifier.py) defines a parallel `TrainableAudioCommandClassifier` module (real `nn.Conv2d`/`nn.BatchNorm2d`/`nn.Linear` layers, identical channel counts/kernel sizes/resize-to-64x64/global-average-pool architecture), trains it with standard backprop, then exports into `AudioCommandClassifier`'s buffer-keyed state-dict format — a drop-in-compatible checkpoint requiring zero changes to `evaluate_audio_classifier.py` or `audio_receiver_pytorch.py`.

**A second bug caught the same way, before it silently produced a wrong checkpoint:** the initial trainable-model draft used the conventional Conv→BatchNorm→ReLU layer order. `AudioCommandClassifier.forward()` actually applies **Conv→ReLU→BatchNorm** (unusual, but real — verified by reading the exact line order). A round-trip test (train a few real gradient steps, export, reload into `AudioCommandClassifier`, compare outputs on the same input) passed at first only because an *untrained* model's BN is near-identity, masking the ordering bug; re-running the same check after real gradient steps exposed a ~0.05 max-abs output divergence, which went to exactly 0.0 once the layer order was corrected to match. Lesson for any future work on this file: validate round-trip export/import fidelity **after** the model has actually been trained a few steps, not just on freshly-initialized weights — identity-initialized BN hides ordering bugs that only show up once its running stats move away from the default.

### Retrain results

**v4 — quarantine-cleaned dataset, no background diversification yet.** Trained from scratch (random init, 60 epochs, Adam, plain cross-entropy, no augmentation or class weighting — deliberately kept simple per the migration principle) on the same `training_v2/{train,val}` layout used throughout this plan, i.e. the corruption-quarantined dataset. Best val accuracy during training: 83.5% (epoch 38 of 60; train accuracy saturates near 100% by ~epoch 30, val bounces 73–83% after that — some overfitting, no early-stopping/regularization tuning attempted yet, this was a first-pass "does the pipeline work end-to-end" run, not a tuned one).

| Metric | v3 (pre-quarantine, old training code) | v4 (post-quarantine, this retrain) |
|---|---|---|
| Offline val accuracy (`evaluate_audio_classifier.py`) | 86.3% | 83.5% |
| Live-stream detections (`evaluate_live_receiver_stream.py`) | 4/11 | **6/11** |
| `_background_` recall | 62.9% (151/240) | **72.5%** (174/240) |
| `forward` recall | 74.2% | **92.5%** |
| `hold` recall | 83.3% | 62.0% (regressed) |
| `go_red` recall | 67.1% | 50.2% (regressed) |
| `go_red`→`go_green` rate | 25.4% | 17.2% (improved) |

Mixed at the per-class level — `hold` and `go_red` recall both regressed despite the corruption cleanup, likely because this is a from-scratch retrain with a much simpler/less-tuned procedure than whatever produced v3, not an apples-to-apples "same training, cleaner data" comparison. But the metric the plan has been chasing since the live-stream test was built — real detections in a continuous noisy stream — improved from 4/11 to 6/11, and just as importantly the failure mode changed: v3 had one outright *misclassification* (t=60s detected `go_red` instead of `left`); v4 has none — every miss is now "no detection" (background wins), not a wrong command. Full per-clip counts: [`evaluations/reports/confusion_matrix_20260811T092401Z.json`](../../evaluations/reports/confusion_matrix_20260811T092401Z.json); live-stream detail: [`evaluations/reports/live_stream_eval_20260811T101639Z.json`](../../evaluations/reports/live_stream_eval_20260811T101639Z.json).

### Background composition, quantified

Before segmenting the 23-minute recording, checked what the existing `_background_` class actually contains by filename convention (`bgreal_<source>__<split>__NNNNNN.wav` for real recordings vs. `<piper-voice>___background___NNNNN.wav` for synthetic TTS renders). Confirms and sharpens the plan's earlier "~9.5% of train set, from 4 raw recordings" estimate:

- Of 1290 train `_background_` clips, only **90 (7%)** are real recordings, and they come from just **2** source files (`robot_background_sound.wav`, `robot_background_sound_01.wav`), not 4.
- Of 240 val `_background_` clips, **zero** are real recordings — the entire val-set measurement of background recall (used throughout this plan, e.g. the 62.9%/72.5% figures above) has never once been tested against real ambient/robot noise, only near-silent synthetic TTS renders.
- `lab_background_sound_01.wav` — confirmed 23.3 minutes at 44.1kHz stereo (resampled to 16kHz mono for training) — had zero clips derived from it anywhere in the dataset before this session, confirming the plan's "not yet integrated" note.

## Background Diversification: 23-Minute Recording Segmented and Added

[`data_processing/segment_background_recording.py`](../../data_processing/segment_background_recording.py) segments a long unlabeled recording into fixed-length (1.25s / 20000-sample, 16kHz mono) `_background_` clips matching the existing `bgreal_*` naming and format exactly, filtering out near-silent windows with the same energy gate used everywhere else in this pipeline (`align_speech_to_fixed_length`'s peak<0.03 / rms<0.003 check) so this source doesn't just add more of the near-silence filler the dataset already has too much of.

Run against `lab_background_sound_01.wav`: 1117 candidate windows → 1057 kept (60 dropped as too quiet) → split 899 train / 158 val (15% val fraction, **deliberately putting real-noise clips into val for the first time** — see the composition finding above). Manifest: [`data_processing/reports/background_segmentation_20260811T104121Z.json`](../../data_processing/reports/background_segmentation_20260811T104121Z.json). This roughly 10x's the real-recording share of the background class (90 → 989 train clips from real sources) and, for the first time, gives the val-set background measurement actual noisy-condition coverage instead of only synthetic near-silence.

**v5 — same retrain procedure as v4, on top of the background-diversified dataset.** Trained the same way as v4 (60 epochs, same hyperparameters, only the dataset changed: +899 train / +158 val real-noise background clips). Best val accuracy: 84.9% (epoch 48).

| Metric | v3 (original) | v4 (quarantine + retrain) | v5 (+ background diversification) |
|---|---|---|---|
| Offline val accuracy | 86.3% | 83.5% | **84.9%** |
| `_background_` recall | 62.9% (151/240) | 72.5% (174/240) | **88.9%** (354/398) |
| `forward` recall | 74.2% | 92.5% | 67.5% (regressed vs. v4) |
| `hold` recall | 83.3% | 62.0% | 81.2% (recovered) |
| `go_red` recall | 67.1% | 50.2% | 54.8% (still down vs. v3) |
| Live-stream detections | 4/11 | **6/11** | 5/11 |

**Background diversification is not an unambiguous win, and this needs to be reported honestly rather than as a clean success story.** Offline, it's the best result on every headline number — `_background_` recall jumped another 16.4 points to 88.9%, and overall accuracy improved over v4. But the live-stream test — the metric the plan has explicitly treated as the one that "actually reflects the reported bug" since Stage 2 — went **backward**, 6/11 (v4) → 5/11 (v5), losing the `left` detection at t=60s that v4 had. Looking at v5's live-stream detection log: it produces noticeably fewer stray `_background_` detections than v4 (12 vs. 18 across the 120s stream, consistent with the offline recall gain), but in exchange it now fires spurious high-confidence `go_red` detections in three windows where `go_red` isn't the right answer (t≈0.5s during the `go_grey` window, t≈51.6s during `forward`, t≈71.9s during `right`) — `go_red` recall was already the weakest command class in both v4 and v5, and this looks like the same weakness manifesting as false positives instead of just false negatives now that background is a less likely competing answer. Full detail: [`evaluations/reports/confusion_matrix_20260811T111853Z.json`](../../evaluations/reports/confusion_matrix_20260811T111853Z.json), [`evaluations/reports/live_stream_eval_20260811T113143Z.json`](../../evaluations/reports/live_stream_eval_20260811T113143Z.json).

**Read with real caveats, not as a final verdict:** this is one training run per stage (no seed averaging), with a deliberately simple/untuned procedure (no augmentation, no class weighting, no regularization or early-stopping beyond keeping the best-val-accuracy epoch, fixed 60 epochs) — both v4 and v5 show the same overfitting pattern (train accuracy saturating near 100% by epoch ~30 while val bounces within a ~5-10 point band for the rest of training), so some of the v4→v5 live-stream difference could plausibly be training-run noise rather than a genuine causal effect of background diversification. The right next step before drawing a firm conclusion is a tuning pass (early stopping, class weighting — `go_red` in particular — mild augmentation) and/or multiple seeds per configuration, not simply picking v4 or v5 as "the" retrain result.

## Tuning Pass, Staged Through Colab

Per step 10 above, moved from single-shot local retrains to a proper tuning pass: augmentation, class weighting, weight decay, early stopping, multiple seeds. Local CPU can't afford the full sweep, so this is staged as local ablation (cheap, fast iteration, CPU) -> Colab GPU sweep (the real multi-seed run).

**Local augmentation ablation** ([`training/experiment_augmentations.py`](../../training/experiment_augmentations.py), [`training/audio_augmentations.py`](../../training/audio_augmentations.py)): pure numpy/scipy waveform augmentations (noise-mixing, speed perturbation, synthetic reverb, gain jitter, time-shift) -- no torchaudio/librosa available in the env, and none of these need more than numpy/scipy. Ran 25-epoch, single-seed, 30%-train-subset comparisons:

| Config | Best val acc |
|---|---|
| `light` (gain + time-shift) | 79.4% |
| `speed_perturb` | 77.5% |
| `combined_no_reverb` (follow-up) | 77.1% |
| `combined` (all 5, incl. reverb) | 76.5% |
| `baseline` (no augmentation) | 76.5% |
| `noise_mix` | 75.6% |
| `reverb` | **57.3%** |

**One clear, confident conclusion: drop reverb.** Consistently ~20 points behind everything else at every epoch, not just noise -- the synthetic room-impulse-response approximation (convolution with an exponentially-decayed noise kernel, since no real RIR dataset is available) is too aggressive or the wrong kind of distortion for this problem. Excluded from `PRESETS` in `audio_augmentations.py`.

**Everything else is genuinely inconclusive at this budget**, including a targeted follow-up (`combined_no_reverb`, i.e. combined minus just reverb: 77.1%, barely above baseline's 76.5%) -- differences of 1-4 points are within the noise already seen between epochs in the full v4/v5 runs. Notably `noise_mix` -- the augmentation most directly relevant to the actual live-stream problem -- underperformed baseline here but was still visibly climbing at epoch 25 with no plateau, the signature of an augmentation that needs more epochs to pay off, not one that doesn't work. This is exactly why the next step is a proper multi-seed sweep at full epoch budget, not another quick local guess.

**Portability refactor for Colab:** split pure DSP code (constants + the STFT transform) out of `audio_receiver_pytorch.py` into a new [`audio_dsp.py`](../../audio_dsp.py), since the training scripts were transitively importing `sounddevice` (needed only for live mic input, and requiring system PortAudio libs) just to get a sample-rate constant. Training now depends on nothing beyond torch/numpy/soundfile/scipy. Also vectorized the STFT computation (`batch_spectrograms` now runs one batched `torch.stft` call instead of looping per-sample) -- this model is small enough (~13.5K trainable params) that the per-sample Python loop was the actual bottleneck, not model compute; vectorizing dropped local per-epoch spectrogram cost by roughly 2.5x and is what makes a GPU worth using at all here (verified numerically identical output before/after, just faster).

**Production training script upgraded** (`training/train_audio_command_classifier.py`, still the single source of truth used both locally and in Colab): `--augmentation {none,light,speed,noise,combined}`, `--class-weights` (sklearn-style balanced weighting -- motivated by the real 3.3x count gap between the original 6 classes and the 5 movement classes added later, see "Correction" section above), `--weight-decay`, `--patience` (early stopping), `--device` (auto-detects CUDA). `--augmentation none` with no other flags reproduces the exact v4/v5 training behavior.

**Colab package**: [`data_processing/prepare_colab_package.py`](../../data_processing/prepare_colab_package.py) zips the training code + `training_v2/` dataset preserving the local `ml_audio/...` directory structure, so the notebook's imports and `DEFAULT_DATASET_ROOT` resolution work unchanged after extraction -- verified end-to-end (extract, import, discover 17507 train files) before treating it as ready. Output: 600.8MB / 19939 files, gitignored (`models/` and `*.zip` already excluded).

[`training/colab_augmentation_sweep.ipynb`](../../training/colab_augmentation_sweep.ipynb): loads the dataset once, then sweeps all 5 augmentation presets x 3 seeds (150 epochs, patience 20, class weighting + weight decay on throughout -- those aren't part of the ablation, the count-imbalance fix is a settled decision, not an open question) -- 15 runs total, reusing `train()` from the production script directly rather than reimplementing anything. Reports mean +/- std val accuracy per config (the number the local single-seed tests couldn't give), exports the single best checkpoint in the same `AudioCommandClassifier`-compatible format as v4/v5.

**Colab sweep run complete (2026-08-12), overnight, ~3.6 hours of GPU compute across all 15 runs.** Crash-resilience wasn't needed in the end (it completed in one session), but was built and verified before the run per the plan above.

| Config | Mean val acc (3 seeds) | Std | Best single seed | Avg s/epoch |
|---|---|---|---|---|
| `noise` | **88.52%** | 1.46% | 89.62% | 4.3s |
| `combined` | 87.75% | 0.39% | 88.05% | 21.8s |
| `speed` | 86.84% | 1.72% | 88.67% | 35.7s |
| `light` | 86.36% | 1.02% | 87.31% | 2.8s |
| `none` (baseline) | 85.32% | 1.01% | 86.20% | 2.1s |

**Noise-mixing wins clearly, exactly as the local ablation's climbing-not-plateaued val curve predicted.** At the 25-epoch local test budget, `noise` underperformed baseline (75.6% vs 76.5%) with a val curve still visibly rising -- the writeup at the time flagged this as "needs more epochs to pay off, not evidence it doesn't work," and a full budget with 3 seeds confirms that: `noise` beats every other config's *mean* by at least 0.8 points and beats baseline by 3.2 points, with the best individual run (`noise_seed0`, 89.6%) exceeding even the original pre-quarantine v3 baseline (86.3%). `speed` is a clear resource outlier -- 10-15x more expensive per epoch than everything else (likely `scipy.signal.resample`'s cost in an unvectorized per-sample augmentation loop) for a worse mean than `noise` -- not worth it as currently implemented; deprioritize or optimize before using it again.

**v6 = `noise_seed0` (89.62% offline), exported to `models/pytorch_v6/`.** Full v3-v6 comparison:

| Metric | v3 (original) | v4 (quarantine) | v5 (+ bg diversification) | v6 (+ noise-mix tuning) |
|---|---|---|---|---|
| Offline accuracy | 86.3% | 83.5% | 84.9% | **89.6%** |
| `_background_` recall | 62.9% | 72.5% | **88.9%** | 79.4% |
| `go_red` recall | 67.1% | 50.2% | 54.8% | **72.4%** |
| `hold` recall | 83.3% | 62.0% | 81.2% | **88.5%** |
| `forward` recall | 74.2% | **92.5%** | 67.5% | 88.3% |
| Live-stream detections | 4/11 | **6/11** | 5/11 | **6/11** |

**v6 is the best checkpoint produced so far by almost every offline measure** -- best overall accuracy, best `go_red` and `hold` recall of any checkpoint yet (both finally past v3's original numbers), second-best `forward` and background recall. **But it does not improve on the live-stream number** -- ties v4 at 6/11 rather than beating it, and the *way* it gets there is a step backward from v4's pattern: v4's 5 misses were all "no detection" (background wins, the safe failure mode); v6 has two real misclassifications instead (`go_red` detected during the `left` window, `stop` detected during the `backward` window), the same kind of regression flagged for v5. v6's live-stream detection log also shows background firing far more often (~70+ times across 120s) than v4's ~18 -- consistent with its lower offline background recall (79.4% vs v5's 88.9%), plausibly because training-time noise-mixing teaches the model that noise-plus-command should resolve to the *command*, making it less willing to default to background broadly, including on some genuinely-background val/live audio. Reports: [`evaluations/reports/confusion_matrix_20260812T205731Z.json`](../../evaluations/reports/confusion_matrix_20260812T205731Z.json), [`evaluations/reports/live_stream_eval_20260812T210001Z.json`](../../evaluations/reports/live_stream_eval_20260812T210001Z.json).

**Third data point for the same pattern now:** v5 and v6 each improved offline metrics substantially over their predecessor while *not* improving (v5: regressing; v6: flat, with a worse failure mode) the live-stream number. Offline val accuracy on isolated, pre-cut clips is clearly not a reliable proxy for continuous-noisy-stream performance for this model/dataset -- any future change should be judged primarily on `evaluate_live_receiver_stream.py`, not `evaluate_audio_classifier.py`, even though the latter is far cheaper to run. The live-stream ceiling has now sat at 6/11 across two different checkpoints (v4, v6) produced by two different mechanisms (label-order/preprocessing fixes; full retrain + noise-mix tuning) -- worth treating as a real signal that something structural is capping it there, not just an artifact of either specific run.

**Not yet done:** the real human recordings for the 5 currently-100%-synthetic movement classes (`forward`/`backward`/`left`/`right`/`go_grey`) -- still the most likely lever left for the live-stream ceiling specifically, since 4 of those 5 classes are exactly where the remaining misses cluster (`forward`, `left`, `right`, `backward` all miss in both v4 and v6; only `go_grey` is caught, and only in v6). Recording those and retraining as v7 (on top of v6's winning `noise` augmentation config) is the clear next step, per the earlier discussion of this being a separate, additive measured stage rather than something to bundle into the augmentation-tuning result.

**Also pending:** the user is recording additional real human samples for the 5 currently-100%-synthetic movement classes (`forward`/`backward`/`left`/`right`/`go_grey`) separately. Once available, fold them into `training_v2/` (matching the existing `real_<speaker>__<label>__...` naming convention already used for the original 6 classes) and repackage/retrain -- this directly targets the domain-gap finding, likely a bigger lever than any augmentation choice, and should be tracked as its own measured stage rather than bundled into the same retrain as the augmentation sweep result.

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
6. ~~The retrain track~~ — **done for a first pass.** `training/train_audio_command_classifier.py` exists, targets the 12-class layout, and trains the full network end-to-end (see "Correction" section above for why a linear-probe-on-frozen-features approach, tried first, didn't work). **v4** (quarantine-cleaned, pre-diversification) trained and measured: 83.5% offline, 6/11 live (up from 4/11), mixed per-class results.
7. ~~Re-run both eval scripts after retraining~~ — **done for v4**, see results table above. Live-stream number *did* move (4/11 → 6/11) and the failure mode improved (no more wrong-command misfires, only misses). Re-run again for v5 once that finishes (in progress).
8. Extract data loading/labeling into `data_processing/` as a general ingestion module — **partially done**: the corruption audit (`audit_dataset_corruption.py`) and now background segmentation (`segment_background_recording.py`) both exist as standing scripts reading the `training_v2/{train,val}/<class>/` layout directly. Not yet done: wiring the corruption audit into a single ingestion pipeline entry point, and sub-labeling tooling for background noise sub-types (see Open Questions).
9. ~~Prioritize background-source-diversification~~ — **done.** `segment_background_recording.py` added 899 train / 158 val real-noise clips from the 23-minute recording (see section above). **v5** retrained on top of this and both eval scripts re-run: offline metrics and background recall both improved over v4, but live-stream regressed (6/11 → 5/11) — see the v3/v4/v5 table above. Diversification is not yet a clean, unambiguous win on the metric that matters most.
10. **Next: a tuning pass, not another single-shot retrain.** Both v4 and v5 show the same overfitting signature (train accuracy → ~100% by epoch 30, val accuracy noisy thereafter) from a deliberately minimal first-pass procedure (no augmentation, no class weighting, no regularization/early-stopping beyond keeping the best-val epoch). Before deciding whether background diversification actually helps live performance, or picking either v4 or v5 as "the" retrain result: add early stopping, class-weight `go_red` specifically (weakest class in both retrains, and the one now producing false positives in v5's live-stream log), and ideally run 2-3 seeds per configuration to separate real effects from training-run noise.

## Open Questions

- **Does background diversification actually help live performance, net?** v5 improved every offline metric (accuracy, background recall +16.4 points) but regressed the live-stream number (6/11 → 5/11) versus v4, with a new failure mode (spurious `go_red` false positives in windows where `go_red` isn't correct, replacing what used to be safer "no detection" misses). This is exactly the kind of gap between offline and live metrics the plan has been trying to close since Stage 2 — worth resolving with a tuning/multi-seed pass (see step 10) before concluding either way.
- `go_red` recall is the one command class that hasn't recovered even after quarantine removed its worst corruption (67.1% v3 → 50.2% v4 → 54.8% v5, all below the original) and is now implicated in v5's new live-stream false positives — deserves targeted attention (class weighting, or a closer look at whether quarantine actually removed enough of the bad clips) rather than assuming the corruption fix alone would resolve it.
- Whether the model architecture itself (~13.5K params, unusual Conv→ReLU→BatchNorm ordering — see "Correction" section above) is fundamentally limited for separating command-plus-noise mixtures, independent of data/training quality — not yet answerable; requires the tuning pass above to rule out "just needs better training" first.
- Ball-drop/typing-as-distinct-label question from prior discussion remains open, deferred until per-sub-type eval reporting exists to quantify rather than guess.
