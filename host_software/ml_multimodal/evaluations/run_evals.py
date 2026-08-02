import os
import subprocess
import glob

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml_vision")
)
eval_dir = os.path.join(base_dir, "evaluations")
models_dir = os.path.join(base_dir, "models")

# Define Dataset mappings
DATASETS = {
    "_iphone_": {
        "dir": "../../data/02_silver/images_iphone",
        "csv": "labels_sequential.csv"
    },
    "_0728_": {
        "dir": "../../data/02_silver/session_20260728_102908",
        "csv": "labels_sequential.csv"
    },
    "_0730_": {
        "dir": "../../data/02_silver/session_20260730_174916",
        "csv": "cnn_sequential_features.csv"
    }
}

def get_dataset_args(model_name):
    for tag, config in DATASETS.items():
        if tag in model_name:
            return config["dir"], config["csv"]
    return None, None

# We will skip bbox models
skip_models = [
    "yolov8_platform_bbox_v1",
    "yolov8_platform_bbox_v2",
    "yolov8_platform_bbox_v3",
    "yolov8_platform_bbox_v4",
]

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

# Find all CNN models
cnn_models = []
for p in glob.glob(os.path.join(models_dir, "cnn_*/*.pth")):
    model_name = os.path.basename(os.path.dirname(p))
    cnn_models.append(model_name)
cnn_models = list(set(cnn_models))

print("Found YOLO models to evaluate:", yolo_models)
print("Found ResNet models to evaluate:", resnet_models)
print("Found CNN models to evaluate:", cnn_models)

def run_cmd(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, cwd=eval_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating: {e}")

# Evaluate YOLO
for model in yolo_models:
    data_dir, csv_name = get_dataset_args(model)
    if not data_dir:
        print(f"Skipping {model}: Dataset not identified.")
        continue
    
    if "marker_ball" in model:
        cmd = f"python evaluate_yolo.py --model_path ../models/{model}/weights/best.pt"
    else:
        cmd = f"python evaluate_yolo_homography.py --model_path ../models/{model}/weights/best.pt --data_dir {data_dir} --csv_name {csv_name}"
    run_cmd(cmd)

# Evaluate ResNet
for model in resnet_models:
    data_dir, csv_name = get_dataset_args(model)
    if not data_dir:
        print(f"Skipping {model}: Dataset not identified.")
        continue

    arch = "resnet50" if "resnet50" in model else "resnet18"
    pth_file = glob.glob(os.path.join(models_dir, model, "*.pth"))[0]
    pth_name = os.path.basename(pth_file)
    cmd = f"python evaluate_resnet_expert_tracker.py --model_path ../models/{model}/{pth_name} --arch {arch} --data_dir {data_dir} --csv_name {csv_name}"
    run_cmd(cmd)

# Evaluate CNN
for model in cnn_models:
    data_dir, csv_name = get_dataset_args(model)
    if not data_dir:
        print(f"Skipping {model}: Dataset not identified.")
        continue

    pth_file = glob.glob(os.path.join(models_dir, model, "*.pth"))[0]
    pth_name = os.path.basename(pth_file)
    cmd = f"python evaluate_cnn_2d_tracker.py --model_path ../models/{model}/{pth_name} --data_dir {data_dir} --csv_name {csv_name}"
    run_cmd(cmd)

print("All evaluations complete.")