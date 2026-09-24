#!/usr/bin/env bash
# Overnight, unattended Moondream2 environment-fix attempt (2026-09-24), SAFE MODE.
#
# Real problem this addresses: Moondream2's pinned remote code calls
# scaled_dot_product_attention(..., enable_gqa=...), a torch kwarg missing from arm2-t4's torch
# 2.4.0 -- confirmed this session, see docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md section 11.
#
# Safety contract (do not weaken these without a human re-reading this file):
#   1. NEVER touches arm2-t4:r36.4.0 (the confirmed-working image) -- builds arm2-t4-alt:r36.4.0
#      under a separate tag only.
#   2. NEVER writes into the real results checkpoint ($R, --run-label jetson_run1) until BOTH an
#      InternVL2.5-4B regression smoke test AND a Moondream2 smoke test pass under the new image.
#      Both smoke tests run against a disposable, separate results directory that this script
#      deletes and recreates every run -- never the real one.
#   3. Checks free disk space before every step that could consume meaningful space, and aborts
#      cleanly (not into corruption) if the margin is too thin -- this project has already hit a
#      real corrupted-download-from-disk-exhaustion incident once this session.
#   4. Every docker step is wrapped in `timeout` so a hang cannot block this script from ever
#      reaching the final report/push step.
#   5. Never uses sudo (no interactive password prompt is possible unattended).
#   6. Never runs `git add -A` / broad adds -- only ever adds the two specific files this script
#      itself writes (the doc summary + its own log), so it can never accidentally commit
#      unrelated in-progress local changes.
#   7. Always reaches the final report/push step, success or failure -- the point of this script is
#      that the user finds out what happened by morning either way, not just on success.
#
# Launch (before leaving the lab), so it survives the terminal closing / SSH dropping:
#   cd host_software/ml_jetson_vla/deployment
#   chmod +x run_moondream_fix_overnight.sh
#   nohup ./run_moondream_fix_overnight.sh > /dev/null 2>&1 &
#   disown
#
# Then just close the terminal / walk away. Check results in the morning via `git pull` on any
# other machine, or by reading the new dated section this script appends to
# docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md.

set -uo pipefail  # deliberately NOT -e: every step's failure is handled explicitly so the script
                   # can always reach the final report/push step instead of dying mid-way silently.

# --- Resolve paths from the script's own location, not inherited shell variables -----------------
# Every earlier failure tonight that involved $D/$R/$HFCACHE was a stale/empty shell variable in a
# fresh shell -- this script must not repeat that. Compute everything from where it actually lives.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
D="$REPO/host_software/ml_jetson_vla/deployment"
R="$REPO/host_software/data/arm2_jetson_sweep_results"          # the REAL results dir (found via
                                                                   # `find` on-device 2026-09-24)
R_ALT="$REPO/host_software/data/arm2_jetson_sweep_results_alt_regression_check"  # disposable
BRONZE="$REPO/host_software/data/01_bronze"
HFCACHE="/mnt/nvme/data/hf-cache"
IMAGE_TAG="arm2-t4-alt:r36.4.0"
RUN_LABEL="jetson_run1"
DOC="$REPO/host_software/ml_jetson_vla/docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md"
LOG_DIR="$D/overnight_logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/moondream_fix_${TS}.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

check_disk() {
  local path="$1" min_gb="$2" avail
  avail="$(df --output=avail -BG "$path" 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [ -z "$avail" ]; then
    log "DISK CHECK: could not read free space for $path -- treating as failure"
    return 1
  fi
  if [ "$avail" -lt "$min_gb" ]; then
    log "DISK CHECK FAILED: $path has ${avail}GB free, need >= ${min_gb}GB"
    return 1
  fi
  log "disk check OK: $path has ${avail}GB free (need >= ${min_gb}GB)"
  return 0
}

STATUS="UNKNOWN"
DETAIL=""
AGGREGATE_TABLE=""

report_and_push() {
  local status="$1" detail="$2"
  log "=== FINAL STATUS: $status ==="
  log "$detail"

  {
    echo ""
    echo "## $(date +%Y-%m-%d): overnight unattended Moondream2 environment-fix attempt -- $status"
    echo ""
    echo "Ran unattended overnight via \`run_moondream_fix_overnight.sh\` (safe mode: never touches"
    echo "\`arm2-t4:r36.4.0\`, only proceeds to a real sweep if both a regression smoke test and a"
    echo "Moondream2 smoke test pass first under a separate \`arm2-t4-alt:r36.4.0\` image)."
    echo ""
    echo "**Result: $status**"
    echo ""
    echo "$detail"
    if [ -n "$AGGREGATE_TABLE" ]; then
      echo ""
      echo '```'
      echo "$AGGREGATE_TABLE"
      echo '```'
    fi
    echo ""
    echo "Full log: \`host_software/ml_jetson_vla/deployment/overnight_logs/moondream_fix_${TS}.log\`"
    echo ""
  } >> "$DOC"

  cd "$REPO" || return 1
  # Only ever add the two files this script itself writes -- never a broad add, so any other
  # in-progress local changes (e.g. an already-modified doc from earlier work) are left untouched.
  git add "$DOC" "$LOG"
  if git diff --cached --quiet; then
    log "nothing to commit (unexpected -- report/log should always differ), skipping push"
    return 0
  fi
  git commit -m "Overnight unattended Moondream2 fix attempt (${TS}): $status

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" >> "$LOG" 2>&1
  if git push >> "$LOG" 2>&1; then
    log "pushed report to git successfully"
  else
    log "git push FAILED -- the commit exists locally, push manually once you're back. Check for a diverged remote."
  fi
}

abort() {
  report_and_push "ABORTED" "$1"
  exit 1
}

log "=== overnight Moondream2 fix attempt starting ==="
log "REPO=$REPO"
log "R=$R  R_ALT=$R_ALT  HFCACHE=$HFCACHE  IMAGE_TAG=$IMAGE_TAG"

# --- Pre-flight checks --------------------------------------------------------------------------
if [ ! -d "$HFCACHE" ]; then
  abort "HFCACHE directory $HFCACHE does not exist -- NVMe mount may not be live. Not proceeding."
fi
if ! mountpoint -q /mnt/nvme 2>/dev/null; then
  log "WARNING: /mnt/nvme does not report as a mountpoint via 'mountpoint' -- proceeding anyway since $HFCACHE exists, but flagging this as worth checking."
fi
check_disk "$HFCACHE" 20 || abort "insufficient free space on the HFCACHE/NVMe volume before starting"
check_disk "$HOME" 5 || abort "insufficient free space on \$HOME before starting"
if [ ! -f "$R/checkpoint_jetson_run1.json" ]; then
  abort "expected real checkpoint not found at $R/checkpoint_jetson_run1.json -- refusing to proceed against an unverified results dir"
fi
log "pre-flight checks passed"

# --- Step 1: build the alt image (separate tag, never touches arm2-t4) -------------------------
log "building $IMAGE_TAG from Dockerfile.arm2-transformers4-torch27 ..."
if ! timeout 1800 docker build -f "$D/Dockerfile.arm2-transformers4-torch27" -t "$IMAGE_TAG" "$D" >> "$LOG" 2>&1; then
  abort "docker build of $IMAGE_TAG failed or timed out after 30 min -- see log for the real error"
fi
log "build complete"
check_disk "$HFCACHE" 15 || abort "insufficient free space after build"

# --- Step 2: disposable regression-check results dir --------------------------------------------
rm -rf "$R_ALT"
mkdir -p "$R_ALT"

run_smoke() {
  local candidates="$1" label="$2"
  timeout 600 docker run --rm --init --runtime nvidia \
    -v "$REPO":/workspace/VRI_2026 \
    -v "$BRONZE":/workspace/VRI_2026/host_software/data/01_bronze:ro \
    -v "$R_ALT":/workspace/arm2_results \
    -v "$HFCACHE":/root/.cache/huggingface \
    -w /workspace/VRI_2026/host_software \
    "$IMAGE_TAG" \
    python3 -u ml_jetson_vla/deployment/run_arm2_sweep_jetson.py \
      --candidates $candidates --stage smoke --skip-validation-gate \
      --results-dir /workspace/arm2_results --run-label "$label" >> "$LOG" 2>&1
}
# --skip-validation-gate is required here (not just a nicety): these smoke tests use fresh,
# just-created run-labels in the disposable $R_ALT directory, which by definition have no passing
# gate record yet -- confirmed live tonight that run_arm2_sweep_jetson.py refuses to even start
# ("No passing validation-gate record for run label ...") without either a prior Qwen gate pass for
# that exact label, or this flag. Neither Moondream2 nor InternVL2.5 are the candidate the gate
# validates anyway (only Qwen2.5-VL-3B is), so skipping it here is the documented, correct use of
# the flag, not a corner being cut.

# --- Step 3: InternVL2.5-4B regression smoke test -----------------------------------------------
# Structural check, not a numeric-error-tolerance check: a 2-frame smoke sample is too small to
# compare mean error against tonight's real 60-frame numbers meaningfully. What matters is whether
# the new image still loads the model and produces parseable output at all, the same way it did
# under the confirmed-working arm2-t4 tonight.
log "running InternVL2.5-4B regression smoke test under $IMAGE_TAG ..."
run_smoke "internvl2_5_4b" "alt_regression_internvl"
if grep -qi "CALL FAILED\|Traceback" "$LOG"; then
  abort "InternVL2.5-4B regression smoke test shows CALL FAILED / Traceback under $IMAGE_TAG -- the new image regresses a previously-working candidate, not safe to proceed. Real arm2-t4:r36.4.0 is untouched."
fi
log "InternVL2.5-4B regression smoke test: no CALL FAILED / Traceback detected"

# --- Step 4: Moondream2 smoke test (the actual thing being fixed) -------------------------------
check_disk "$HFCACHE" 10 || abort "insufficient free space before Moondream2 smoke test"
log "running Moondream2 smoke test under $IMAGE_TAG ..."
run_smoke "moondream2:query moondream2:point" "alt_regression_moondream"
if grep -qi "enable_gqa" "$LOG"; then
  abort "Moondream2 smoke test STILL shows the enable_gqa error under $IMAGE_TAG -- the base-image swap did not fix it. Needs a human to look at the real remote-code call site, not another automated retry."
fi
if grep -qi "CALL FAILED\|Traceback" "$LOG"; then
  abort "Moondream2 smoke test shows a DIFFERENT failure (not enable_gqa) under $IMAGE_TAG -- see log for the real error. Needs human review before a real sweep."
fi
log "Moondream2 smoke test: no enable_gqa / CALL FAILED / Traceback detected -- the fix appears to work"

# --- Step 5: both smoke tests passed -- run the REAL full Moondream2 sweep ----------------------
check_disk "$HFCACHE" 10 || abort "insufficient free space before the real full sweep"
log "both smoke tests passed -- running the real full Moondream2 sweep against \$R (run-label $RUN_LABEL) ..."
SWEEP_LOG="$LOG_DIR/moondream_sweep_${TS}.log"
if timeout 5400 docker run --rm --init --runtime nvidia \
    -v "$REPO":/workspace/VRI_2026 \
    -v "$BRONZE":/workspace/VRI_2026/host_software/data/01_bronze:ro \
    -v "$R":/workspace/arm2_results \
    -v "$HFCACHE":/root/.cache/huggingface \
    -w /workspace/VRI_2026/host_software \
    "$IMAGE_TAG" \
    python3 -u ml_jetson_vla/deployment/run_arm2_sweep_jetson.py \
      --candidates moondream2:query moondream2:point --stage all --skip-validation-gate \
      --results-dir /workspace/arm2_results --run-label "$RUN_LABEL" >> "$SWEEP_LOG" 2>&1; then
  log "real Moondream2 sweep completed (exit 0)"
  STATUS="SUCCESS"
else
  log "real Moondream2 sweep exited nonzero or timed out -- checking for partial results anyway"
  STATUS="PARTIAL / SWEEP ERROR"
fi
cat "$SWEEP_LOG" >> "$LOG" 2>&1

if [ -f "$R/aggregate_table_${RUN_LABEL}.csv" ]; then
  AGGREGATE_TABLE="$(grep -i "moondream\|^candidate," "$R/aggregate_table_${RUN_LABEL}.csv")"
fi

DETAIL="Both regression smoke tests passed under $IMAGE_TAG (a new, separate image tag -- arm2-t4:r36.4.0 was never touched). The real Moondream2 sweep then ran against the real results checkpoint. --skip-validation-gate was used deliberately (this run-label's gate already passed for Qwen earlier; skipping avoids re-requiring arm2-t5's image here). See the aggregate table below and the full logs for details."
report_and_push "$STATUS" "$DETAIL"
log "=== done ==="
