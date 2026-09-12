---
name: data-pipeline-verification
description: Use before trusting the output of a new or modified data-processing script — labeling, sync, merge, augmentation, or compositing — on this project's vision/audio/multimodal datasets. Verifies correctness on a small sample before committing to a full-dataset run. Distinct from dataset-integrity-check, which audits a dataset already at rest.
---

# Data pipeline verification

This project has shipped multiple labeling/sync bugs that survived a plausible-looking accuracy number and were only caught by a subsequent, deliberate small-sample check. Run this sequence every time before trusting a changed pipeline's full output.

## 1. Dry-run on a small sample first

Never point a new or modified pipeline at the full dataset first. Use (or add) a `--limit`-style flag for cheap dry-run testing before a full run that can take minutes to tens of minutes over tens of thousands of frames.

## 2. Build or reuse a visual verification tool

Overlay the derived label directly on the source image/frame — no trained model required. This is the fastest way to catch a systematic bug (sign error, wrong axis, wrong scale, wrong offset): seeing a label land visibly off the true target is immediate and unambiguous, whereas a trained model's accuracy number alone can look "plausible" even when trained against a subtly wrong label (a real point-reflection sign bug in this project reported a deceptively small 4.4px error against the wrong target). If a tool like this already exists for the pipeline in question, reuse it rather than building a new one.

## 3. Spot-check across every distinct session/sheet/condition, not just one

A bug can be symmetric in a way that only shows up on certain configurations — a real marker-mask sizing bug in this project affected every shape but was only caught by checking non-circle shapes (triangle/square/hexagon) individually, not by checking circles alone.

## 4. Verify row/frame counts are IDENTICAL when a fix shouldn't change which rows are included

If a fix is supposed to only change *derived values* (not which frames/rows exist), confirm the row count before and after is exactly the same. An unexpected count change means the fix touched more than intended; an unchanged count is real, checkable evidence the fix was surgical — not an assumption to state without checking.

## 5. Check for stale intermediate files contaminating the new output

A previous script version's leftover derived files (e.g. old per-feature mask blobs from an earlier, differently-shaped pipeline) can silently get pulled into a new merge/combine step by a generic filename-matching pattern, even after the generating code itself has been fixed. Clear derived-output directories when the generating logic's *shape* changes — don't assume overwrite-in-place is complete.

## 6. Confirm synchronization is timestamp-based, not index/frame-count-based

Whenever merging two independently-sampled streams (camera fps and telemetry Hz are essentially never equal or perfectly stable in this project), sync by timestamp. Index/row-count alignment silently drifts.

## 7. Re-run coverage/spatial-density diagnostics after the change, not just accuracy metrics

A labeling bug can leave coverage numbers completely untouched (they're often computed from the same telemetry that was never wrong) while the *derived* pixel/mm projection is wrong. Use coverage as one signal among several, never the only one. When reporting coverage, use the full percentile spread (mean/median/p95/max), not a single fixed threshold — a fixed floor can saturate to ~100% as a dataset grows and stop being discriminating.

## 8. Only commit to the full-scale re-run once 1-7 pass on the small sample

Full reprocessing in this project has historically taken 10+ minutes over tens of thousands of frames across multiple sessions — expensive to redo if an earlier step here was skipped and a bug is found afterward.
