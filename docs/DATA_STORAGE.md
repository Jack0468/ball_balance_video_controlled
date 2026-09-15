# Data & Model Storage / Portability

This document covers **where the bytes physically live and how they move between
machines** — a different concern from the Medallion tier *contract*
(`01_bronze` → `02_silver` → `03_gold`), which is defined in
`.claude/skills/data-processing-pipeline/SKILL.md` and governs directory
semantics, not transport.

## Current state (DVC wired up 2026-09-13, superseding the "planned" section this doc used to end with)

The home server (`https://github.com/Jack0468/home_server`, private repo) is live and
runs a MinIO (S3-compatible) container as a DVC remote, reachable over Tailscale — no
public exposure. **This project is now wired into it.** `03_gold/`, `03_synthetic_yolo/`,
`yolo_raw_dataset/`, `ml_vision/models/`, `ml_audio/data/`, `ml_audio/models/`,
`ml_multimodal/models/`, and `ml_jetson_vla/models/` (the `Qwen2.5-VL-3B-Instruct`
checkpoint) are `dvc add`-ed; the `.dvc` pointer files are committed to git, the actual
bytes live only in MinIO (or a machine's local `.dvc/cache`). `01_bronze/` and `02_silver/`
deliberately stay **outside** DVC by default (still tar/zip-per-session, per the rules
below) — only the reproducible-artifact tiers are versioned this way as a rule.

**Scoped exception (2026-09-15):** `host_software/data/01_bronze/session_jetson_track4_*`
(the Track 4 Jetson standalone-run sessions, `telemetry.csv` + `rgb_video.mp4` each) ARE
individually `dvc add`-ed and pushed, per explicit user instruction. Unlike the bulk legacy
bronze data (hundreds of thousands of small frame-capture files, the actual reason for the
`01_bronze` exclusion), each Track 4 session is exactly 2 files — small enough that DVC's
per-file overhead isn't a problem, and this data is valuable enough (real Jetson-collected
runs for VLA fine-tuning) to warrant real versioning rather than manual copies. This is a
per-session opt-in, not a reversal of the general `01_bronze` rule — new bronze data outside
this specific naming pattern is not automatically covered.

**Connection details, credentials, and the Colab-specific Tailscale bootstrap live in the
`home_server` repo, not here** (`docs/DVC_SETUP.md`, `docs/COLAB_SETUP.md`,
`docs/CONNECTING_A_NEW_DEVICE.md`) — deliberately not duplicated into this file, since
that's the repo that actually owns the server and would drift if this file also tried to
track its IP/endpoint/credential-handling details. Quick reference for day-to-day use in
*this* repo:
```bash
dvc pull   # fetch tracked datasets/weights (needs Tailscale connected first)
dvc push   # after adding/changing a tracked dataset or weight file
```

## Original audit (2026-09-12, kept for the scale numbers)

Everything under `host_software/data/`, `host_software/ml_audio/data/`, and
model weight directories (`*.pt`, `*.pth`, `*.onnx`, `models/`) was
`.gitignore`'d with no version control, checksum manifest, or automated fetch
path — moving to a new machine meant a manual copy, and nothing recorded which
weight file matched which training run. Scale at the time of the audit
(the DVC-tracked tiers above are this same data, now versioned):

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
- **`03_gold` (curated train/eval splits), `ml_audio/data/`, and model weights
  (`ml_vision/models/`, `ml_audio/models/`, `ml_multimodal/models/`;
  `ml_jetson_vla` has no checkpoints of its own yet — see its `ARCHITECTURE.md`):**
  these are the artifacts that need reproducible versioning tied to a specific
  training run or reported result. **DVC is now adopted for these** — lightweight
  `.dvc` pointer files are committed to Git while the real bytes live in the home
  server's MinIO remote. See "Current state" above.

## Storage backend guidance

- **Home cloud server (live):** the DVC remote for `03_gold`, `ml_audio/data/`,
  and all model weight directories — a MinIO (S3-compatible) container reachable
  only over Tailscale (no public exposure). See "Current state" above for the
  quick-reference commands and `home_server/docs/DVC_SETUP.md` /
  `COLAB_SETUP.md` for full setup. Also a valid plain rsync/robocopy target for
  bronze/silver session archives (outside DVC, per the rule above).
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

## Still open / not yet done

- `01_bronze`/`02_silver` still have no automated per-session archival step —
  tarring/zipping before a transfer is a manual discipline, not yet scripted.
- No backup of the home server's MinIO data itself yet — see `home_server`'s
  own `BLUEPRINT.md` §5 ("Backups") for that project's open item; this repo's
  data is only as durable as that server's storage until it's resolved there.

DVC's own hash tracking (each `.dvc` file's `md5`) already ties a tracked
directory's contents to the exact bytes referenced — that's the "which weight
file matches which training run" manifest the original version of this
section asked for, no separate tool needed.
