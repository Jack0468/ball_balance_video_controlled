# Project TODO Ledger

Significant open items only — architecture decisions, real unresolved bugs, and genuine
blockers, not minor cleanup notes. `docs/PROJECT_LOGBOOK.md` holds the full history/rationale
behind every entry here; this file tracks current actionable status only, and should stay
short enough to actually be read before starting new work.

**Workflow (see `CLAUDE.md`)**: at the end of a work session, review what was attempted versus
actually completed. Anything flagged, started-but-unfinished, or explicitly deferred must be
added here before the session is considered wrapped up — don't wait to be asked.

**Format**: `- [ ] **Title** (flagged <date>, owner: <agent/person>) — one-line description.
Source: <doc/entry reference>.` Check the box and move the line to a dated "Resolved" note (or
just delete it, logbook already has the history) once genuinely done.

## Vision (`ml-vision`)

- [ ] **Marker colour classification needs a robust design, not another hardcoded fix**
  (flagged 2026-10-13, owner: ml-vision) — Today's live Jetson test revealed the physical test
  sheet in use (`aruco_markers_01`, plain circles) doesn't match the sheet every HSV bin has
  ever been calibrated against (`aruco_markers_03`, triangles/squares/hexagons). Static
  per-colour HSV bins are fundamentally fragile to sheet/ink/lighting changes -- this exact
  class of failure has now happened twice (15/09/2026, 13/10/2026), each time "fixed" by
  re-tuning bins against one specific historical dataset. **The next fix should be a genuinely
  robust approach** (e.g. per-session/per-sheet auto-calibration, relative/contrastive colour
  matching instead of absolute bins, or explicit sheet-identity detection with sheet-specific
  calibration loaded at runtime) rather than a third round of hardcoded bin tuning that will
  just as easily break on the next sheet or lighting change. Source: `docs/PROJECT_LOGBOOK.md`
  13/10/2026 entries (`f88877e`, `9c76748`).

## Multimodal / VLA (`ml-multimodal`)

- [ ] **ACT_FAST `run_2` checkpoint's action-chunk head trained on a safety-incompatible action
  distribution; decide retraining approach** (flagged 2026-10-13) — the gold dataset blends
  `pid_webcam` (58.7% of samples exceed the firmware's +/-11.025 deg limit, up to -22.3/+29.7 deg)
  and `jetson_track4_rl` (0% exceed it) regimes for action-chunk supervision. Live test with
  normalisation correctly applied got 100% chunk rejection (111/111) — not a bug, the model
  legitimately learned to predict values the safety gate must refuse. Open decision: retrain on
  `jetson_track4_rl` only (40,713 samples, 0% violations, matches real deployment regime) vs.
  filtering `pid_webcam` to its compliant 41.3% vs. other approach. Not an agent call — reverses a
  considered Stage-0 data-plan choice. Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Real live
  re-test with normalisation applied: 100% chunk rejection."
- [ ] **Eval cell (h) torchcodec crash in `act_train_colab.ipynb`** (flagged 2026-10-13) —
  diagnosed as likely a Colab-side decoder-backend issue (torchcodec vs. the software decoder
  used to verify the dataset's own video files are clean), not source-data corruption. Not
  fixed. Next step: force `torchcodec` to CPU/software decode on the specific crashing episode
  to confirm. Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Eval-only torchcodec video-decode
  crash."
- [ ] **Qwen grounding full Colab run still interim** (flagged 2026-10-13) — only 5 of 10
  sessions scored as of last check (93.3% ball-xy hit@20mm, 59.9% marker-grounding -- both
  interim). Re-score once all 10 sessions finish; the marker-grounding discrepancy vs. the
  validated 68.3% sweep figure is unexplained and needs the full run to resolve. Source:
  `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Real interim Qwen2.5-VL-3B grounding results."
- [x] **Live hardware test of a real trained ACT checkpoint (checklist item 9)** (flagged
  2026-10-13, resolved same day) — confirmed live: 205 lines sent, 3/3 chunks accepted, real
  touch+grounding. Root cause of 3 earlier failed attempts (looked like a hang, was actually a
  silently-swallowed chunk-size-mismatch crash) fixed in `fast_layers.py`/`core/hybrid_policy.py`.
  Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Checklist item 9... root-caused a 3-run hang."
- [ ] **Touch-sensor single-position anomaly, unexplained** (flagged 2026-10-13) — one specific
  ball placement (overlapping an ArUco marker directly) produced 0/136 valid touch samples,
  while all four platform corners tested normal read fine. Localized, not systemic, not chased
  further. Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, H1 corner-testing entries.

## FPGA

- [ ] **Arm 3 framing decision** (flagged 2026-10-13) — is Arm 3 a real fabricated photonic
  chip, a simulation/co-design-only arm, or a commercial product? Everything in the first-draft
  interface spec is written conditional on "real chip" since that's the only scenario with
  concrete numbers available -- this is a one-line question that unblocks or reframes the whole
  spec, not yet answered. Source: `fpga/docs/optical_bridge_interface_spec_draft.md` section 1.
- [ ] **Camera/UDP video streaming** (status per `CLAUDE.md`'s Current State table: "separately
  unresolved") -- not touched this session, carried over from project state.

## Hardware (human/EE decisions — not an agent's call)

- [ ] **Power connector, FPGA power delivery, microphone integration, PCB consolidation** — all
  four listed as open in `CLAUDE.md`'s "Hardware Decisions (OPEN)" table, requiring human/EE
  sign-off. No agent should choose or design around these without that sign-off.

## Process / Meta

- [ ] **Run the full thorough TODO-mining pass across all project docs** (flagged 2026-10-13,
  deferred to tonight -- not run during today's lab session to keep tokens free for hardware
  work) — systematically mine `docs/PROJECT_LOGBOOK.md` (full history, not just today) plus
  `docs/EVALUATION_STRATEGY.md`, `host_software/ml_jetson_vla/docs/*`, and any other project doc
  for "not yet done"/"flagged"/"open" items, **verify each one against the current codebase**
  before adding it here (the logbook is append-only history, not a live-status doc -- many
  older flagged items may already be resolved by later work), and add only genuinely still-open,
  significant items. This is itself the next thing to do with this file.
