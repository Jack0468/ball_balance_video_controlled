---
name: data-processing-pipeline
description: Use when building or extending a raw-capture-to-labeled-training-data pipeline for vision, audio, or multimodal/VLA data on this project. Covers the shared Medallion-tier contract every module follows, plus the pipeline-specific ground-truth method for each.
---

# Data processing pipeline

## Shared contract across every pipeline (Medallion architecture)

`data/01_bronze/` = raw recorded sessions (video/audio + telemetry, untouched) → `data/02_silver/` = processed/labeled pairs → `data/03_gold/` = final curated train/eval splits. Keep these physically separate — never let a later-tier or training-only artifact leak backward into what an evaluation script reads from an earlier tier.

**Ground truth should always come from a deterministic, physically-derived source — never from the same kind of model output you're trying to train or evaluate.** Derive labels from known procedural geometry (e.g. a manifest of hardcoded physical marker coordinates) plus a coordinate transform, or from telemetry backward-mapped through that transform — not from running a CV/ML heuristic (like color tracking) that could itself drift and silently become the thing you're "evaluating against."

**Sync independently-sampled streams by timestamp, never by index or row count.** Camera fps and telemetry Hz are essentially never equal or perfectly stable in this project's hardware.

Before trusting the output of any pipeline you build or change here, run the `data-pipeline-verification` skill's sequence; before trusting a dataset already produced, run `dataset-integrity-check`.

## Vision pipeline specifics

- Ball position ground truth = telemetry (`{pitch, roll}` or `{touch_x, touch_y}`) backward-mapped through the ArUco homography to pixel space (`auto_label_shared_vision.py`) — never color-tracking.
- Marker ground truth = the procedurally-known manifest's physical coordinates, projected through the same homography — never derived by detecting the marker itself (that's circular).
- Synthetic compositing exists to correct real-sheet combinatorial gaps (which shape/color pairs are underrepresented on the physical printed sheets) — compute the real coverage numerically first, then bias synthetic sampling toward the actual gap; don't guess the gap or assume round numbers.
- Marker/shape rendering (both real-mask generation and synthetic compositing) must use a real physical mm→px scale derived from the platform's actual dimensions, not an arbitrary tuned constant — and must render true per-shape polygon geometry (triangle/square/hexagon/circle), not a circle fallback for everything.

## Audio pipeline specifics

- Prefer deterministic, explainable feature-extraction transforms (STFT/spectral profiling/matched filtering) over learned features wherever accuracy allows, mirroring the vision side's classical-first preference.
- Background/silence-class diversity is a first-class training requirement, not an afterthought — a background class that's a small fraction of the training mix is a confirmed root cause of real-world misclassification in this project, independent of any preprocessing bugs.
- **Live production preprocessing must be verified byte-for-byte consistent with what training used** (label ordering, normalization, cropping) before trusting any live-stream evaluation number. The two silently diverging is a confirmed, severe (multi-tens-of-points accuracy loss) failure mode here, not a hypothetical risk — check this explicitly any time the production receiver or the training preprocessing changes independently.

## Multimodal / VLA pipeline specifics

- Behavioral-cloning training data should come from real closed-loop runs of the actual control policy being imitated or improved — including its real noise, corrections, and recovery behavior — not scripted straight-to-target trajectories. Scripted-only data has already been flagged in this project as insufficiently diverse (no disturbance/recovery coverage), and real closed-loop logging infrastructure already exists for this — use it rather than generating synthetic-only trajectories.
- If any evaluation or comparison script in this pipeline uses mocked/stubbed inputs anywhere (dummy images, mock coordinates, a stubbed training stage), identify and explicitly disclose exactly which parts are real vs. stubbed before trusting or reporting any numbers it produces. This pipeline has real, currently-stubbed pieces (a training stage that only prints instead of running, mock-coordinate eval scripts) — assuming the scaffold is production-ready by default has been wrong before.

## Whichever pipeline

Keep any synthetic/real mixing ratio or spatial/class-density rebalancing as an explicit, checkable parameter in the generation script rather than a hardcoded implicit assumption, and verify the *actual achieved* ratio/coverage after generation — don't assume the requested ratio was hit exactly just because it was requested.
