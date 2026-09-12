# Data & Model Storage / Portability

This document covers **where the bytes physically live and how they move between
machines** — a different concern from the Medallion tier *contract*
(`01_bronze` → `02_silver` → `03_gold`), which is defined in
`.claude/skills/data-processing-pipeline/SKILL.md` and governs directory
semantics, not transport.

## Current state (audited 2026-09-12)

Everything under `host_software/data/`, `host_software/ml_audio/data/`, and
model weight directories (`*.pt`, `*.pth`, `*.onnx`, `models/`) is
`.gitignore`'d. There is **no version control, checksum manifest, or
automated fetch path** for any of it today — moving to a new machine means a
manual copy, and nothing records which weight file matches which training
run. Scale, at time of writing:

| Tier / location | Size | File count |
|---|---|---|
| `host_software/data/01_bronze` | ~59 GB | ~334K |
| `host_software/data/02_silver` (+ `_unified_pose`) | ~17 GB | ~284K |
| `host_software/data/03_gold` (+ variants) | ~9 GB | ~419K |
| `host_software/ml_audio/data` | ~1 GB | ~20K |
| `host_software/ml_vision/models` | ~2.5 GB | 381 |

**~85 GB and ~1.1M files total.** This file count is the binding constraint
on every option below — it rules out naive "just sync the folder" approaches
regardless of which cloud/server backend is chosen.

## Rules by tier

- **`01_bronze` / `02_silver` (raw + intermediate, ~1M+ small files):**
  Never transport as loose files. Tar/zip each session directory
  (`session_XYZ/`) before it leaves the collecting machine. One archive per
  session turns "hundreds of thousands of files" into "one file per
  session," which is the only way any transport below stays reliable at
  this scale. These tiers are also the least reusable across machines —
  prefer regenerating `02_silver` from `01_bronze` archives on the target
  machine over syncing it directly, per the existing pipeline scripts.
- **`03_gold` (curated train/eval splits) and model weights
  (`ml_vision/models/`, `ml_audio/models/`, `ml_multimodal` /
  `ml_jetson_vla` checkpoints):** these are the artifacts that actually need
  reproducible versioning tied to a specific training run or reported
  result. This is where **DVC** (Data Version Control) is the right tool —
  lightweight `.dvc` pointer files get committed to Git while the real bytes
  live in a configurable remote (S3-compatible, SSH, WebDAV, a local/NAS
  mount). Not yet adopted; see "Planned" below.

## Storage backend guidance

- **Home cloud server (in progress):** the best long-term fit once it's
  reachable. Plan is a DVC remote over SSH or an S3-compatible store
  (e.g. MinIO) pointed at it, replacing manual copies for `03_gold` and
  model weights. Also a valid plain rsync/robocopy target for bronze/silver
  session archives.
- **OneDrive / Google Drive:** use only as an **occasional off-site mirror
  of already-archived, gold-tier artifacts** — a handful of large `.zip`/
  model-checkpoint files, not the live working tree. Both consumer sync
  clients are documented to corrupt Git-internal files (`.git/`, lock
  files) when they resync mid-operation, and both degrade badly once
  directory file counts climb into the hundreds of thousands — exactly the
  regime `01_bronze`/`03_gold` sit in today. Concretely: **never point a
  OneDrive/Google Drive sync folder at this repo's working tree, a DVC
  cache, or any bronze/silver directory.**
- **Git-LFS:** installed on this machine (`git lfs env` succeeds) but not
  configured (`.gitattributes` has no LFS patterns) and not recommended here
  — GitHub's free LFS tier (1 GB storage / 1 GB bandwidth per month) is far
  below the ~85 GB in play, and per-file pointer overhead makes it a poor
  fit for a dataset with ~1.1M individual files regardless of remote size.
  DVC (above) supersedes it for this project's scale.

## Planned, not yet implemented

- `dvc init` + tracking `03_gold/` and the model weight directories, remote
  pointed at the home server once available.
- A checksummed manifest (or DVC's own hash tracking) so a given weight file
  can be tied back to the training run/dataset version that produced it.
- A "Fetching data & model weights" step in the root `README.md` (currently
  absent — see `README.md`'s Getting Started section).

Until the above lands, treat any data/model transfer as a manual,
undocumented step — verify checksums or at least file counts/sizes against
this table after copying, since nothing else will catch a partial transfer.
