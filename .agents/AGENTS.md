# Moved

This file's content has moved to keep it from drifting out of sync with itself (it had — see `docs/PROJECT_LOGBOOK.md`, 2026-08-20 entry, for the consolidation rationale).

- Project constraints, locked decisions, current state, coding conventions, repo structure: **`CLAUDE.md`** (repo root).
- Per-domain agent context (FPGA, vision, audio, multimodal/VLA): **`.claude/agents/`**.
- Reusable procedures (FPGA pipeline, data verification, dataset integrity, model iteration, data processing): **`.claude/skills/`**.
- Full project history and rationale: **`docs/PROJECT_LOGBOOK.md`** (unchanged).

This directory (`.agents/`) still hosts `codev_mcp.py`, referenced by `.mcp.json` — that's unaffected.
