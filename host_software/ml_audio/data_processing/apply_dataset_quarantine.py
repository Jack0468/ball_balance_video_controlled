"""Move flagged corrupt clips out of the active training_v2 folders.

Reads data_processing/reports/dataset_corruption_audit.json (produced by
audit_dataset_corruption.py) and relocates flagged command-class clips into a
sibling _quarantined_corrupt/ tree, mirroring the original split/label
structure. This is a move, not a delete -- fully reversible.

_background_ clips are intentionally excluded here: near-silent background
clips aren't the same defect as a truncated command mislabeled as complete
(see docs/plans/audio_eval_notebook_refactor_plan.md), so they're left alone
and tracked separately under the background-diversification effort.
"""

import json
import os
import shutil
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
REPORT_PATH = os.path.join(SCRIPT_DIR, "reports", "dataset_corruption_audit.json")
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "reports", "quarantine_manifest.json")

DATASET_ROOT = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "training_v2"
)
QUARANTINE_ROOT = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "_quarantined_corrupt"
)

EXCLUDED_LABELS = {"_background_"}


def main() -> None:
    with open(REPORT_PATH) as f:
        report = json.load(f)

    moved: list[dict] = []
    skipped_missing: list[str] = []

    for clip in report["flagged_clips"]:
        if clip["label"] in EXCLUDED_LABELS:
            continue

        src = os.path.join(ML_AUDIO_DIR, clip["path"])
        if not os.path.exists(src):
            skipped_missing.append(clip["path"])
            continue

        dest_dir = os.path.join(QUARANTINE_ROOT, clip["split"], clip["label"])
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))

        shutil.move(src, dest)
        moved.append(
            {
                "label": clip["label"],
                "split": clip["split"],
                "reason": "empty" if clip["is_empty"] else "truncated",
                "from": os.path.relpath(src, ML_AUDIO_DIR),
                "to": os.path.relpath(dest, ML_AUDIO_DIR),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": os.path.relpath(REPORT_PATH, ML_AUDIO_DIR),
        "excluded_labels": sorted(EXCLUDED_LABELS),
        "moved_count": len(moved),
        "skipped_missing_count": len(skipped_missing),
        "moved": moved,
        "skipped_missing": skipped_missing,
    }

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    per_class: dict[str, int] = {}
    for m in moved:
        key = f"{m['split']}/{m['label']}"
        per_class[key] = per_class.get(key, 0) + 1

    print(f"Moved {len(moved)} clips to {QUARANTINE_ROOT}")
    if skipped_missing:
        print(f"Skipped {len(skipped_missing)} clips (already missing on disk)")
    print("\nPer-class quarantine counts:")
    for key in sorted(per_class):
        print(f"  {key:24s} {per_class[key]}")
    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
