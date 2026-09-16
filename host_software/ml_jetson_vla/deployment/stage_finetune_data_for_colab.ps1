<#
.SYNOPSIS
  Stage 1 of the large-model fine-tuning pipeline plan (docs: greedy-sleeping-fairy).
  Zips the raw source data Stage 2/3's Colab notebooks need to build the LeRobot-format
  fine-tuning dataset for host_software/ml_jetson_vla/core/qwen_multihead_policy.py, and
  uploads that single archive to Google Drive -- extending the exact zip+Drive pattern
  bootstrap_model_comparison_colab.ipynb (Item 5) already built and validated, per that
  plan's explicit "extend, don't reinvent" instruction. Run on the DEV MACHINE, not in Colab.

.WHY THIS SHIPS RAW VIDEO, NOT A PRE-BUILT LEROBOT DATASET OR EXTRACTED JPEGS
  qwen_multihead_policy.py's state_dim=2 and its whole data-flow section (S2.1/S4 of
  docs/MULTI_HEAD_ARCHITECTURE_SPEC.md) are written directly against
  data_processing/convert_to_lerobot.py's "observation.image"/"observation.state"/"action"/
  "task" LeRobotDataset feature schema -- confirmed by reading qwen_multihead_policy.py's own
  code comments this session, not generate_vla_dataset.py's separate flat-JSON schema. That
  means Stage 2/3 need a real LeRobotDataset, which convert_to_lerobot.py (extended this
  session to also ingest the PID regime, regime-tagged) builds by decoding each session's
  rgb_video.mp4 frame-by-frame and re-encoding it into LeRobot's own per-episode video
  format. Three staging options were considered:
    1. Ship raw session videos, run convert_to_lerobot.py in Colab.      <- CHOSEN
    2. Ship raw session videos AND 151,706 separately-extracted JPEGs.   <- rejected: pure
       duplication, ~2-3x the data for zero benefit (the JPEGs would be re-derived from the
       exact same video bytes already being shipped).
    3. Run convert_to_lerobot.py locally first, ship the already-built LeRobotDataset.     <-
       rejected: this machine has no CUDA GPU and Compress-Archive/video re-encoding of the
       full ~151K-frame corpus is a genuinely long-running CPU job better spent in Colab
       (which needs to run the conversion once anyway for its own on-disk dataset layout);
       shipping raw compressed mp4 is also smaller than LeRobot's re-encoded per-episode
       video in practice for this checkpoint's SVT-AV1 encoder settings (confirmed via this
       session's own verification run: 40-frame/session samples re-encoded larger per-frame
       than the source mp4's compression ratio).
  So: ship compressed source video (small, already-compact) + the CSVs + the extended
  converter script itself; run the actual LeRobot conversion in Colab, where the real
  training also happens.

.WHAT GETS STAGED
  - session_manifest.json, DATASET_CATALOG.md (context/audit trail)
  - data_processing/convert_to_lerobot.py (vendored, read-only copy -- run this exact script
    in Colab against the extracted sessions/ directory to build the real LeRobotDataset)
  - All 10 session_jetson_track4_*/  { telemetry.csv, rgb_video.mp4 }
  - The 5 usable PID sessions (session_manifest.json: regime=pid_laptop_webcam AND
    frame_synced_csv is not null) -- { synced_telemetry.csv, rgb_video.mp4 } ONLY. Raw
    telemetry.csv (unsynced, no frame_index) is NOT shipped -- convert_to_lerobot.py's PID
    path reads synced_telemetry.csv exclusively, per find_pid_sessions()'s own docstring.
    images/, images_cropped/, masks/, labels_*.csv (ml_vision's auto-labeling outputs living
    inside these same raw bronze dirs) are NOT shipped -- irrelevant to VLA fine-tuning,
    would bloat the archive for no benefit (confirmed by checking those subdirs' purpose
    against DATASET_CATALOG.md, not assumed from directory presence alone).
    session_20260730_174916 is excluded (never synced -- session_manifest.json's own
    "never_synced" corruption_flag), matching generate_vla_dataset.py's Stage-0 selection.

.USAGE
  From the repo root (C:\Users\Admin\Documents\Windows_codespace\VRI_2026):
    pwsh -File host_software\ml_jetson_vla\deployment\stage_finetune_data_for_colab.ps1
  Then upload the resulting zip to Google Drive at:
    MyDrive/vri2026_track4_bootstrap/finetune_pipeline_data.zip
  (same GDRIVE_ROOT bootstrap_model_comparison_colab.ipynb already uses, per
  feedback_reuse_existing_export_tooling -- a new Colab notebook for Stage 2/3 should read
  from this same Drive folder rather than a new one.)

  Note the 01_bronze DVC scoped exception (docs/DATA_STORAGE.md): these session directories
  may need `dvc pull` on the dev machine first if not already materialized locally -- this
  script does not do that itself, matching Item 5's own note on the same point.
#>

$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Item "$PSScriptRoot\..\..\..").FullName
$HostSoftwareDir = Join-Path $RepoRoot "host_software"
$BronzeDir = Join-Path $HostSoftwareDir "data\01_bronze"
$ManifestPath = Join-Path $HostSoftwareDir "ml_jetson_vla\data_processing\session_manifest.json"
$CatalogPath = Join-Path $HostSoftwareDir "data_collection\DATASET_CATALOG.md"
$ConverterScriptPath = Join-Path $HostSoftwareDir "ml_jetson_vla\data_processing\convert_to_lerobot.py"

$Staging = Join-Path $env:TEMP "vla_finetune_pipeline_staging"
$SessionsOut = Join-Path $Staging "sessions"
$ZipPath = Join-Path $env:TEMP "finetune_pipeline_data.zip"

if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
New-Item -ItemType Directory -Force -Path $SessionsOut | Out-Null

Copy-Item $ManifestPath (Join-Path $Staging "session_manifest.json")
Copy-Item $CatalogPath (Join-Path $Staging "DATASET_CATALOG.md")
Copy-Item $ConverterScriptPath (Join-Path $Staging "convert_to_lerobot.py")

$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json

$stagedSessions = @()

# --- Track 4 sessions: all 10, telemetry.csv + rgb_video.mp4 ---
$track4Sessions = $manifest.sessions | Where-Object { $_.regime -eq "jetson_track4_rl_teacher" }
foreach ($s in $track4Sessions) {
    $shortName = Split-Path $s.session_dir -Leaf
    # manifest's own "_format.session_dir_path_convention": relative to host_software/data/
    $srcDir = Join-Path $HostSoftwareDir ("data\" + ($s.session_dir -replace '/', '\'))
    $destDir = Join-Path $SessionsOut $shortName
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    Copy-Item (Join-Path $srcDir "telemetry.csv") (Join-Path $destDir "telemetry.csv")
    Copy-Item (Join-Path $srcDir "rgb_video.mp4") (Join-Path $destDir "rgb_video.mp4")
    $stagedSessions += [PSCustomObject]@{ session = $shortName; regime = "jetson_track4_rl"; csv = "telemetry.csv" }
}

# --- PID sessions: the 5 usable ones (frame_synced_csv not null), synced_telemetry.csv + rgb_video.mp4 ---
$pidSessions = $manifest.sessions | Where-Object { $_.regime -eq "pid_laptop_webcam" -and $null -ne $_.frame_synced_csv }
foreach ($s in $pidSessions) {
    $shortName = Split-Path $s.session_dir -Leaf
    $srcDir = Join-Path $HostSoftwareDir ("data\" + ($s.session_dir -replace '/', '\'))
    $destDir = Join-Path $SessionsOut $shortName
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    Copy-Item (Join-Path $srcDir $s.frame_synced_csv) (Join-Path $destDir $s.frame_synced_csv)
    Copy-Item (Join-Path $srcDir "rgb_video.mp4") (Join-Path $destDir "rgb_video.mp4")
    $stagedSessions += [PSCustomObject]@{ session = $shortName; regime = "pid_webcam"; csv = $s.frame_synced_csv }
}

Write-Host "Staged $($track4Sessions.Count) Track4 sessions + $($pidSessions.Count) PID sessions."
$stagedSessions | Format-Table -AutoSize

if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path "$Staging\*" -DestinationPath $ZipPath -CompressionLevel Optimal

$zipSizeBytes = (Get-Item $ZipPath).Length
$stagingSizeBytes = (Get-ChildItem $Staging -Recurse -File | Measure-Object -Property Length -Sum).Sum

Write-Host ""
Write-Host "Staging dir (pre-zip): $Staging -- $([math]::Round($stagingSizeBytes / 1GB, 2)) GB"
Write-Host "Archive: $ZipPath -- $([math]::Round($zipSizeBytes / 1GB, 2)) GB"
Write-Host ""
Write-Host "Next step (manual, per the plan's staging-only scope): upload $ZipPath to Google Drive at"
Write-Host "  MyDrive/vri2026_track4_bootstrap/finetune_pipeline_data.zip"
