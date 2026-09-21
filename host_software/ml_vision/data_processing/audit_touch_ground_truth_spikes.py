"""Audits the raw touch-plate telemetry behind Dataset 8's ball-position labels
for the same single-frame touch-plate ground-truth artifact found in the Track 1
live-evaluation telemetry (see host_software/src/touch_ground_truth_filter.py's
module docstring for the full root-cause investigation, 2026-09-18).

shared_vision_labels.csv already carries touch_x/touch_y alongside the derived
per-frame training label (ball_x_px/ball_y_px, image_file), synced 1:1 to each
labeled frame -- no join against telemetry.csv/synced_telemetry.csv is needed,
this script flags directly against the labels file each session already has.

For each of Dataset 8's 4 raw sessions, this:
  1. Runs the shared isolated-spike/physical-bounds detector against touch_x/
     touch_y in shared_vision_labels.csv.
  2. Saves a per-session trajectory plot (touch_x/touch_y vs. frame_index) with
     flagged frames highlighted.
  3. Writes a combined summary JSON listing every flagged frame (session,
     frame_index, image_file) for ml-vision to act on -- this script only
     quantifies and flags the finding, it does not drop/relabel/retrain
     anything.

Deliberately does NOT cross-reference against
host_software/ml_vision/evaluations/reports/shared_vision_v2_inference_predictions.csv's
pred_x/pred_y: TouchProbe.cpp's own "FRAME MATCH" comment documents that the
vision-space mm frame and the touchscreen-space mm frame are different,
uncalibrated-against-each-other rectangles (vision maps the plate to
+/-70.0x+/-55.0mm, the touchscreen maps it to +/-93.75x+/-70.5mm). Comparing
pred_x/pred_y directly against touch_x/touch_y would conflate that unresolved,
pre-existing fixed affine mismatch with genuine per-frame residuals, producing
a number that looks like a real cross-check but isn't -- not attempted here.

Usage:
    python audit_touch_ground_truth_spikes.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_VISION_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_VISION_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR,):
    if _p not in sys.path:
        sys.path.append(_p)

from src.touch_ground_truth_filter import flag_touch_position_outliers  # noqa: E402

DATASET_8_SESSIONS = [
    "session_20260810_104132",
    "session_20260810_110239",
    "session_20260810_112047",
    "session_20260810_114330",
]

BRONZE_DIR = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
SUMMARY_JSON_PATH = os.path.join(BRONZE_DIR, "touch_ground_truth_spike_audit_summary.json")


def plot_session_spikes(df, flagged, session, output_path):
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_x.plot(df["frame_index"], df["touch_x"], "r-", linewidth=0.6, label="touch_x (mm)")
    ax_x.scatter(df.loc[flagged, "frame_index"], df.loc[flagged, "touch_x"], color="black", s=25, zorder=5, label="flagged spike")
    ax_x.set_ylabel("touch_x (mm)")
    ax_x.set_title(f"Touch-Plate Ground Truth Spike Audit: {session}")
    ax_x.legend(loc="upper right", fontsize=8)

    ax_y.plot(df["frame_index"], df["touch_y"], "b-", linewidth=0.6, label="touch_y (mm)")
    ax_y.scatter(df.loc[flagged, "frame_index"], df.loc[flagged, "touch_y"], color="black", s=25, zorder=5, label="flagged spike")
    ax_y.set_ylabel("touch_y (mm)")
    ax_y.set_xlabel("frame_index")
    ax_y.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def audit_session(session):
    labels_path = os.path.join(BRONZE_DIR, session, "shared_vision_labels.csv")
    df = pd.read_csv(labels_path)
    df = df.sort_values("frame_index").reset_index(drop=True)

    flagged = flag_touch_position_outliers(df["touch_x"], df["touch_y"]).to_numpy()
    n_flagged = int(flagged.sum())

    plot_path = os.path.join(BRONZE_DIR, session, "touch_ground_truth_spike_audit.png")
    plot_session_spikes(df, flagged, session, plot_path)

    flagged_frames = df.loc[flagged, ["frame_index", "image_file", "touch_x", "touch_y", "ball_x_px", "ball_y_px"]]
    return {
        "session": session,
        "total_labeled_frames": int(len(df)),
        "n_flagged": n_flagged,
        "flagged_percent": (100.0 * n_flagged / len(df)) if len(df) else 0.0,
        "plot": os.path.relpath(plot_path, _HOST_SOFTWARE_DIR),
        "flagged_frames": flagged_frames.to_dict(orient="records"),
    }


def main():
    all_results = []
    total_labeled = 0
    total_flagged = 0

    for session in DATASET_8_SESSIONS:
        result = audit_session(session)
        all_results.append(result)
        total_labeled += result["total_labeled_frames"]
        total_flagged += result["n_flagged"]
        print(f"{session}: {result['n_flagged']} / {result['total_labeled_frames']} labeled frames flagged "
              f"({result['flagged_percent']:.3f}%) -- plot saved to {result['plot']}")

    summary = {
        "total_labeled_frames_dataset8": total_labeled,
        "total_flagged_frames": total_flagged,
        "total_flagged_percent": (100.0 * total_flagged / total_labeled) if total_labeled else 0.0,
        "per_session": all_results,
        "note": (
            "These frames are flagged, not corrected. Deciding whether to drop or "
            "re-derive them and whether to retrain shared_vision_backbone_v2 is an "
            "ml-vision decision, out of scope for this audit."
        ),
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTotal: {total_flagged} / {total_labeled} labeled frames flagged "
          f"({summary['total_flagged_percent']:.3f}%). Summary saved to {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
