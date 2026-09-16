---
name: model-iteration-constraints
description: Use when designing a new model architecture, making a structural change to an existing one, or deciding between candidate architectures/data sources on this project. Surfaces the resource/architecture constraints and comparison methodology required before any resulting numbers can be trusted.
---

# Model iteration: constraints and methodology

## 1. Read the locked constraints and decisions FIRST

Before sketching an architecture: the hard resource budget (`CLAUDE.md`'s FPGA BRAM/DSP limits, if this model may ever target hardware), the locked architecture-decision table (don't silently re-litigate anything marked LOCKED), and the relevant module's parameter-budget doc (e.g. `docs/plans/ml_system_parameter_budget.md` for vision). A model designed without checking these first has, in this project's history, needed a full retrain-and-relabel cycle to fix.

## 2. Check hardware-deployment compatibility before investing training time

If any candidate layer type isn't yet confirmed compatible with the eventual deployment target (e.g. hls4ml-native support, if this model may go to FPGA — see the `fpga-pipeline` skill), check that *before* training, not after. An unsupported layer needs a scoped custom-kernel effort budgeted in up front, not discovered as a surprise blocker post-training with a checkpoint that now can't ship as designed.

## 3. Isolate the variable you're actually testing

If comparing two architectures (or two model classes), confirm they were trained on identical data before attributing any accuracy difference to architecture. "Different architecture" and "different training data" are two separate variables — conflating them produces a result that *looks* like an architecture finding but is actually a data-quality finding. This project currently has an open, deliberately-undecided small-vs-medium vision model comparison for exactly this reason (the two classes were trained on different-but-similar data) — don't resolve an analogous situation by assumption; a clean ablation requires holding data constant.

## 4. Never rank architectures/hyperparameters from a small seed count

A 3-seed ranking in this project was overturned by a 5-seed re-run — twice, in two different ways: once it reversed which of two candidate architectures looked more stable, and once it revealed an apparent "architecture instability" was actually a symmetric effect of an unrelated shared hyperparameter (loss weighting), affecting both candidates equally. Use ≥5 seeds before treating any comparative result as real, and explicitly check whether an observed instability is specific to your candidate or shared across all candidates before attributing it to the thing you're actually trying to test.

## 5. Evaluation isolation

Only evaluate on genuinely held-out data — never on data the training mix has seen, including indirectly via synthetic augmentation derived from the same base images/sessions. See `dataset-integrity-check` for how to confirm this concretely rather than assuming it from directory naming.

## 6. Numerical dtype/precision hygiene

Verify tensors feeding a loss function are the intended dtype before a first real GPU/mixed-precision run. A numpy scalar silently promoted to `float64` inside an otherwise-`float32` pipeline can pass unnoticed on CPU (autocast is a no-op there) and only surface as a hard crash later on GPU — check dtypes explicitly at the point a new derived-label computation is added, don't rely on "it worked before" from a different execution path.

## 7. Before declaring a checkpoint ready to hand off

Confirm the exported artifact's (ONNX/etc.) parameter count matches the intended architecture exactly. Wrap export in try/except so a training run's real, successful checkpoint and metrics aren't obscured or made to look failed by an unrelated, separate export-step failure — training/checkpointing/eval succeeding is what actually matters; export is best-effort on top of that.

## 8. Version, don't overwrite

Bump a version identifier for a structural change rather than overwriting the previous checkpoint in place — this preserves the prior result as a real comparison point rather than silently losing the reference the new result is supposed to be measured against.

## 9. Every cloud/Colab training notebook needs reconnect support, not just crash-resilient saving

Saving checkpoints to Drive/cloud storage during training is not the same thing as being able to resume from one — `colab_nemo_finetune.ipynb` did the former but not the latter for a long stretch of this project (`trainer.fit(model)` called with no `ckpt_path`), so a session dropping mid-run meant the next attempt silently re-trained from the original pretrained checkpoint instead of continuing, with no error or warning that this had happened. Two real checkpoints in this project's audio track (`NOISE_MIX` seed0/seed1) were permanently stranded half-trained this way before it was caught. When wiring up a new training notebook (or auditing an existing one) on any environment prone to disconnects (Colab, spot instances, etc.):
- Persist enough state to actually resume, not just the best-scoring weights — with PyTorch Lightning this means `ModelCheckpoint(..., save_last=True)`, which restores epoch count, optimizer/scheduler state, and other callbacks' state (e.g. `EarlyStopping`'s own patience counter) on top of a freshly-constructed model object, not only the best-by-metric checkpoint saved via `save_top_k`.
- Check for that "last" checkpoint before starting and pass it to the fit call (`trainer.fit(model, ckpt_path=...)`), gated behind an explicit toggle (e.g. `RESUME_IF_AVAILABLE`) that can be forced off — resuming into a run whose hyperparameters have since changed (a different seed, architecture variant, or augmentation config) would silently mix two experiments' worth of training into one checkpoint.
- Verify the resume path actually works — locally, on a couple of quick epochs, with a fresh model object rebuilt the same way the real notebook does it — before trusting it on a real multi-hour run. Confirmed via exactly this kind of test: Lightning logs "Restoring states from the checkpoint path" / "Restored all states from the checkpoint" and continues the epoch counter forward rather than restarting it.
