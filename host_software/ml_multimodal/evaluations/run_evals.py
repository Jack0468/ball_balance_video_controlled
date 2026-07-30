import os
import subprocess
import glob
import time

base_dir = "c:/Users/Admin/Documents/Windows_codespace/VRI_2026/host_software/ml_vision"
eval_dir = os.path.join(base_dir, "evaluations")
models_dir = os.path.join(base_dir, "models")

# We will skip bbox models
skip_models = ["yolov8_platform_bbox_v1", "yolov8_platform_bbox_v2", "yolov8_platform_bbox_v3", "yolov8_platform_bbox_v4"]

# Find all YOLO models (they have weights/best.pt)
yolo_models = []
for p in glob.glob(os.path.join(models_dir, "yolov8_*/weights/best.pt")):
    model_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    if model_name not in skip_models:
        yolo_models.append(model_name)

# Find all ResNet models (they have .pth)
resnet_models = []
for p in glob.glob(os.path.join(models_dir, "resnet*/expert_*.pth")):
    model_name = os.path.basename(os.path.dirname(p))
    resnet_models.append(model_name)
for p in glob.glob(os.path.join(models_dir, "resnet*/best_*.pth")):
    model_name = os.path.basename(os.path.dirname(p))
    resnet_models.append(model_name)
resnet_models = list(set(resnet_models))

print("Found YOLO models to evaluate:", yolo_models)
print("Found ResNet models to evaluate:", resnet_models)

def run_cmd(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, cwd=eval_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating: {e}")

# Evaluate YOLO
for model in yolo_models:
    # Most YOLO models here track the platform and use homography
    if "marker_ball" in model:
        # If it doesn't do homography, maybe evaluate_yolo.py is better, but let's try homography first or evaluate_yolo.py
        cmd = f"python evaluate_yolo.py --model_path ../models/{model}/weights/best.pt"
    else:
        cmd = f"python evaluate_yolo_homography.py --model_path ../models/{model}/weights/best.pt"
    run_cmd(cmd)

# Evaluate ResNet
for model in resnet_models:
    arch = "resnet50" if "resnet50" in model else "resnet18"
    pth_file = glob.glob(os.path.join(models_dir, model, "*.pth"))[0]
    pth_name = os.path.basename(pth_file)
    cmd = f"python evaluate_expert_tracker.py --model_path ../models/{model}/{pth_name} --arch {arch}"
    run_cmd(cmd)

# Evaluate Corrector
corrector_path = os.path.join(models_dir, "mlp_corrector_v1")
if os.path.exists(corrector_path):
    pth_file = glob.glob(os.path.join(corrector_path, "*.pth"))[0]
    pth_name = os.path.basename(pth_file)
    cmd = f"python evaluate_corrector.py --mlp_corrector_v1_path ../models/mlp_corrector_v1/{pth_name}"
    run_cmd(cmd)

print("All evaluations complete.")
