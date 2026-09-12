---
name: dataset-integrity-check
description: Use to audit an existing vision/audio/multimodal dataset at rest for corruption, leakage, duplication, or coverage problems — before training against it or reporting comparative results from it. Distinct from data-pipeline-verification, which checks a processing script's output before trusting it.
---

# Dataset integrity check

Run this before training against a dataset you didn't just personally verify, or before reporting any comparative number derived from one.

## 1. Train/eval isolation — check physical disjointness, not just filename disjointness

Confirm the evaluation set is disjoint from anything the training mix touches at the level of underlying session/frame identity, not just "different CSV file." When both training and eval derive from overlapping raw sessions via a temporal split, confirm the split boundary lands on the *exact same rows* in both — a shared session-based split key (not independently-computed percentages) is what actually guarantees this, not merely avoiding duplicate filenames.

## 2. Know whether you have same-sheet or truly-unseen-sheet evaluation

Evaluating on held-out frames from configurations that also contributed to training is real evidence but strictly weaker than evaluating on a genuinely unseen configuration. State which one you have; don't imply the stronger claim by omission.

## 3. Duplicate/collision check

Verify no filename collisions across merged sources, and that per-session prefixing (or an equivalent disambiguation scheme) actually prevented cross-session numbering collisions when raw per-session numbering resets at zero.

## 4. Corrupted-file audit, per class/label

Scan for truncated, empty, or zero-length clips/frames — check per-class, not just in aggregate. A corrupted-clip problem concentrated in one label class in this project was originally misread as a model/architecture confusion problem; it was actually a data-quality issue that only surfaced under a per-class audit.

## 5. Frozen/duplicate-sample filtering

Discard samples where the underlying physical signal is flat within its own noise floor for an implausibly long stretch — sensor debounce/hold artifacts (e.g. a resistive touchpad's release debounce) produce phantom static labels that a model can memorize as real signal.

## 6. Spatial/class coverage check, full distribution not a single threshold

Report the full percentile spread (mean/median/p95/max) rather than a pass/fail against one fixed floor — a fixed floor can saturate to ~100% as a dataset grows and stop being a discriminating check.

## 7. Label sanity spot-check, even on a dataset believed already correct

Visually confirm a random sample of labels lands on the correct target using the pipeline's overlay verification tool (see `data-pipeline-verification`). Multiple real label bugs in this project's history survived one or more prior "verified" passes — a dataset being previously checked is not a reason to skip checking it again after any upstream change.

## 8. Report exact counts, not impressions

"Row counts identical, only derived values changed" is a strong, checkable claim to make when true. "Should be about the same" is not — always state the actual before/after counts (total, per-class, per-session) when claiming a dataset did or didn't change.
