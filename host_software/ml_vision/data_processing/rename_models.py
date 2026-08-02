import os
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ml_vision_dir = os.path.join(base_dir, "ml_vision")

renames = {
    "models/yolov8_platform_markers_v1": "models/yolov8_platform_markers_v1",
    "models/yolov8_platform_markers_v2": "models/yolov8_platform_markers_v2",
    "models/yolov8_marker_ball_v1": "models/yolov8_marker_ball_v1",
    "models/yolov8_platform_bbox_v1": "models/yolov8_platform_bbox_v1",
    "models/yolov8_platform_bbox_v1-2": "models/yolov8_platform_bbox_v2",
    "models/yolov8_platform_bbox_v1-3": "models/yolov8_platform_bbox_v3",
    "models/yolov8_platform_bbox_v1-4": "models/yolov8_platform_bbox_v4",
    "models/yolov8_platform_pose_iphone_v1": "models/yolov8_platform_pose_iphone_v1",
    "models/resnet18_expert_tracker_iphone_v1": "models/resnet18_expert_tracker_iphone_v1_v1",
    "models/resnet18_expert_tracker_iphone_v1_A": "models/resnet18_expert_tracker_iphone_v1_v2",
    "models/resnet18_expert_tracker_iphone_v1_B": "models/resnet18_expert_tracker_iphone_v1_v3",
    "models/resnet18_expert_tracker_iphone_v1_C": "models/resnet18_expert_tracker_iphone_v1_v4",
    "models/resnet18_expert_tracker_iphone_v1_D": "models/resnet18_expert_tracker_iphone_v1_v5",
    "models/resnet18_expert_tracker_iphone_v1_subset": "models/resnet18_expert_subset_v1",
    "models/resnet50_expert_tracker_iphone_v1": "models/resnet50_expert_tracker_iphone_v1_v1",
    "models/resnet18_temporal_tracker_v1": "models/resnet18_temporal_tracker_v1",
    "models/mlp_corrector_iphone_v1": "models/mlp_mlp_corrector_iphone_v1_v1",
    "runs/models/yolov8_unified_pose_v1": "models/yolov8_unified_pose_v1",
    "runs/models/yolov8_marker_ball_v1_A": "models/yolov8_marker_ball_v2",
    "runs/pose/yolov8_platform_pose_v2": "models/yolov8_platform_pose_v2",
}

# 1. Rename and move folders
for old_rel, new_rel in renames.items():
    old_path = os.path.join(ml_vision_dir, old_rel)
    new_path = os.path.join(ml_vision_dir, new_rel)

    if os.path.exists(old_path):
        print(f"Renaming {old_rel} -> {new_rel}")
        shutil.move(old_path, new_path)

# Delete empty runs directory if it exists
runs_dir = os.path.join(ml_vision_dir, "runs")
if os.path.exists(runs_dir):
    try:
        shutil.rmtree(runs_dir)
        print("Deleted runs directory.")
    except Exception as e:
        print(f"Could not delete runs directory: {e}")

# 2. Find and replace in codebase
# Map old base names to new base names
str_replacements = {
    old.split("/")[-1]: new.split("/")[-1] for old, new in renames.items()
}
# Also replace the 'runs/...' paths explicitly
str_replacements["runs/models/yolov8_unified_pose_v1"] = "models/yolov8_unified_pose_v1"
str_replacements["runs/models/yolov8_marker_ball_v1_A"] = "models/yolov8_marker_ball_v2"
str_replacements["runs/pose/yolov8_platform_pose_v2"] = "models/yolov8_platform_pose_v2"

# Iterate over .py and .md files
for root, _, files in os.walk(base_dir):
    # skip hidden and venv dirs
    if ".git" in root or ".venv" in root or "__pycache__" in root:
        continue
    for f in files:
        if f.endswith(".py") or f.endswith(".md"):
            filepath = os.path.join(root, f)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()

            original_content = content
            for old_str, new_str in str_replacements.items():
                content = content.replace(old_str, new_str)

            if content != original_content:
                with open(filepath, "w", encoding="utf-8") as file:
                    file.write(content)
                print(f"Updated references in {filepath}")

print("Migration complete.")
