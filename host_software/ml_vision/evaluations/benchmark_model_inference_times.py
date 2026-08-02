import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
vision_dir = os.path.abspath(os.path.join(script_dir, ".."))
if vision_dir not in __import__("sys").path:
    __import__("sys").path.insert(0, vision_dir)
models_dir_default = os.path.abspath(os.path.join(script_dir, "../models"))
data_dir_default = os.path.abspath(os.path.join(script_dir, "../../data/02_silver"))


class BallDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]["image_file"])
        image = Image.open(img_name).convert("RGB")
        touch_x = self.labels_df.iloc[idx]["touch_x"]
        touch_y = self.labels_df.iloc[idx]["touch_y"]
        MAX_X_BOUND, MAX_Y_BOUND = 200.0, 200.0
        target = torch.tensor(
            [touch_x / MAX_X_BOUND, touch_y / MAX_Y_BOUND], dtype=torch.float32
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def is_model_dir(name: str) -> bool:
    lower = name.lower()
    return (
        lower != "archive"
        and "temporal" not in lower
        and ("yolo" in lower or "resnet" in lower or "mlp" in lower or "cnn" in lower)
    )


def find_yolo_model_path(model_root: str, ext=".pt") -> Optional[str]:
    weights_dir = os.path.join(model_root, "weights")
    candidates = []
    if os.path.isdir(weights_dir):
        candidates.extend(
            [
                os.path.join(weights_dir, fn)
                for fn in os.listdir(weights_dir)
                if fn.endswith(ext)
            ]
        )
    candidates.extend(
        [
            os.path.join(model_root, fn)
            for fn in os.listdir(model_root)
            if fn.endswith(ext)
        ]
    )
    for name in [f"best{ext}", f"last{ext}"]:
        path = os.path.join(weights_dir, name)
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def find_resnet_checkpoint(model_root: str, ext=".pth") -> Optional[str]:
    for name in [
        f"expert_tracker_best{ext}",
        f"expert_tracker_latest{ext}",
        f"model_best{ext}",
        f"best{ext}",
    ]:
        candidate = os.path.join(model_root, name)
        if os.path.exists(candidate):
            return candidate
    pths = [
        os.path.join(model_root, fn)
        for fn in os.listdir(model_root)
        if fn.endswith(ext)
    ]
    return pths[0] if pths else None


def find_mlp_checkpoint(model_root: str, ext=".pth") -> Optional[str]:
    for name in [f"best_corrector{ext}", "best_corrector.bin", f"best{ext}"]:
        candidate = os.path.join(model_root, name)
        if os.path.exists(candidate):
            return candidate
    pths = [
        os.path.join(model_root, fn)
        for fn in os.listdir(model_root)
        if fn.endswith(ext) or (ext == ".pth" and fn.endswith(".bin"))
    ]
    return pths[0] if pths else None


def read_dataset_csv(data_dir: str, csv_name: str = "labels_sequential.csv") -> pd.DataFrame:
    csv_path = os.path.join(data_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def select_sample_rows(
    df: pd.DataFrame, fraction: float, max_samples: int, seed: int
) -> pd.DataFrame:
    test_df = df.iloc[int(0.8 * len(df)) :]
    sample_count = min(max(1, int(len(test_df) * fraction)), max_samples)
    return test_df.sample(n=sample_count, random_state=seed).reset_index(drop=True)


def sample_dataset_indices(
    dataset_length: int, fraction: float, max_samples: int, seed: int
) -> List[int]:
    test_indices = list(range(int(0.8 * dataset_length), dataset_length))
    sample_count = min(max(1, int(len(test_indices) * fraction)), max_samples)
    random.seed(seed)
    return random.sample(test_indices, sample_count)


def update_evaluation_metrics(model_root: str, metrics: Dict[str, float]) -> None:
    metrics_path = os.path.join(model_root, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(metrics)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=4)


def get_dataset_args_for_model(model_name: str) -> str:
    if "_0730_" in model_name:
        return "cnn_sequential_features.csv"
    return "labels_sequential.csv"


def benchmark_yolo_model(
    model_name: str, model_path: str, sample_df: pd.DataFrame, images_dir: str
) -> Dict[str, float]:
    model = YOLO(model_path)
    
    disk_times = []
    prep_times = []
    net_times = []
    post_times = []
    
    warmup_n = min(2, len(sample_df))
    for idx in range(warmup_n):
        row = sample_df.iloc[idx]
        img_name = row.get("image_file", f"frame_{row.name:04d}.jpg")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, row.get("image_file"))
        img = cv2.imread(img_path)
        if img is not None:
            _ = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)

    for _, row in sample_df.iterrows():
        img_name = row.get("image_file", f"frame_{row.name:04d}.jpg")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, row.get("image_file"))
        
        t0 = time.perf_counter()
        img = cv2.imread(img_path)
        t1 = time.perf_counter()
        
        if img is None:
            continue
            
        results = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        res = results[0]
        
        disk_time = (t1 - t0) * 1000.0
        prep_time = res.speed.get('preprocess', 0.0)
        net_time = res.speed.get('inference', 0.0)
        post_time = res.speed.get('postprocess', 0.0)
        
        disk_times.append(disk_time)
        prep_times.append(prep_time)
        net_times.append(net_time)
        post_times.append(post_time)

    return summarize_granular_timing(disk_times, prep_times, net_times, post_times)


def benchmark_resnet_model(
    model_name: str,
    checkpoint_path: str,
    sample_indices: List[int],
    dataset: Dataset,
    device: torch.device,
) -> Dict[str, float]:
    if "cnn" in model_name.lower():
        from training.basic_cnn import BasicCNN
        model = BasicCNN()
    else:
        arch = "resnet50" if "resnet50" in model_name.lower() else "resnet18"
        if arch == "resnet50":
            model = models.resnet50(weights=None)
        else:
            model = models.resnet18(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    disk_times = []
    prep_times = []
    net_times = []
    post_times = []

    with torch.no_grad():
        for idx in sample_indices:
            row = dataset.labels_df.iloc[idx]
            img_name = row.get("image_file", f"frame_{idx:04d}.jpg")
            img_path = os.path.join(dataset.root_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(dataset.root_dir, row.get("image_file"))
            
            t0 = time.perf_counter()
            img = Image.open(img_path).convert("RGB")
            t1 = time.perf_counter()
            
            if dataset.transform:
                inputs = dataset.transform(img)
            else:
                inputs = transforms.ToTensor()(img)
            inputs = inputs.unsqueeze(0).to(device)
            t2 = time.perf_counter()
            
            outputs = model(inputs)
            t3 = time.perf_counter()
            
            _ = outputs.cpu().numpy()
            t4 = time.perf_counter()
            
            disk_times.append((t1 - t0) * 1000.0)
            prep_times.append((t2 - t1) * 1000.0)
            net_times.append((t3 - t2) * 1000.0)
            post_times.append((t4 - t3) * 1000.0)

    return summarize_granular_timing(disk_times, prep_times, net_times, post_times)


def benchmark_onnx_model(
    model_name: str,
    checkpoint_path: str,
    sample_indices: List[int],
    dataset: Dataset,
) -> Dict[str, float]:
    import onnxruntime as ort
    
    session = ort.InferenceSession(checkpoint_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(2):
        if "resnet18" in model_name.lower() or "cnn" in model_name.lower():
            dummy = np.random.randn(1, 3, 240, 320).astype(np.float32)
        elif "resnet50" in model_name.lower():
            dummy = np.random.randn(1, 3, 480, 640).astype(np.float32)
        else:
            dummy = np.random.randn(1, 5).astype(np.float32) # MLP
            
        session.run(None, {input_name: dummy})
        
    disk_times = []
    prep_times = []
    net_times = []
    post_times = []
    
    for idx in sample_indices:
        row = dataset.labels_df.iloc[idx]
        img_name = row.get("image_file", f"frame_{idx:04d}.jpg")
        img_path = os.path.join(dataset.root_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset.root_dir, row.get("image_file"))
        
        t0 = time.perf_counter()
        img = Image.open(img_path).convert("RGB")
        t1 = time.perf_counter()
        
        if dataset.transform:
            inputs = dataset.transform(img)
        else:
            inputs = transforms.ToTensor()(img)
        inputs_np = inputs.unsqueeze(0).numpy().astype(np.float32)
        t2 = time.perf_counter()
        
        _ = session.run(None, {input_name: inputs_np})
        t3 = time.perf_counter()
        
        disk_times.append((t1 - t0) * 1000.0)
        prep_times.append((t2 - t1) * 1000.0)
        net_times.append((t3 - t2) * 1000.0)
        post_times.append(0.0)

    return summarize_granular_timing(disk_times, prep_times, net_times, post_times)


def benchmark_mlp_model(
    model_name: str,
    yolo_path: str,
    mlp_path: str,
    sample_df: pd.DataFrame,
    images_dir: str,
    device: torch.device,
    is_onnx: bool = False
) -> Dict[str, float]:
    from core.corrector_mlp import CorrectorMLP
    from core.coordinate_math import HomographyProjector

    yolo_model = YOLO(yolo_path)
    
    if is_onnx:
        import onnxruntime as ort
        session = ort.InferenceSession(mlp_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
    else:
        mlp_model = CorrectorMLP().to(device)
        checkpoint = torch.load(mlp_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            mlp_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            mlp_model.load_state_dict(checkpoint)
        mlp_model.eval()

    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    
    disk_times = []
    prep_times = []
    net_times = []
    post_times = []

    warmup_n = min(2, len(sample_df))
    for idx in range(warmup_n):
        row = sample_df.iloc[idx]
        img_name = row.get("image_file", f"frame_{row.name:04d}.jpg")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, row.get("image_file"))
        img = cv2.imread(img_path)
        if img is not None:
            _ = yolo_model.predict(source=img, imgsz=640, conf=0.5, verbose=False)

    for _, row in sample_df.iterrows():
        img_name = row.get("image_file", f"frame_{row.name:04d}.jpg")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, row.get("image_file"))

        t0 = time.perf_counter()
        img = cv2.imread(img_path)
        t1 = time.perf_counter()
        
        if img is None:
            continue
            
        results = yolo_model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        res = results[0] if len(results) > 0 else None
        yolo_total = res.speed.get('preprocess',0.0) + res.speed.get('inference',0.0) + res.speed.get('postprocess',0.0) if res else 0.0
        
        t2 = time.perf_counter()
        corners = None
        ball_center = None
        if res is not None and res.boxes is not None:
            classes = res.boxes.cls.cpu().numpy()
            boxes = res.boxes.xywh.cpu().numpy()
            for i, cls in enumerate(classes):
                if int(cls) == 0 and res.keypoints is not None and len(res.keypoints.xy) > i:
                    kpts = res.keypoints.xy[i].cpu().numpy()
                    if len(kpts) == 4: corners = kpts
                elif int(cls) == 1:
                    ball_center = (boxes[i][0], boxes[i][1])

        if corners is None or ball_center is None:
            disk_times.append((t1 - t0) * 1000.0)
            prep_times.append(yolo_total)
            net_times.append(0.0)
            post_times.append(0.0)
            continue

        features_tensor = None
        if projector.update_homography(corners):
            features = np.array([
                ball_center[0], ball_center[1], 0.0, 0.0,
                corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                corners[2][0], corners[2][1], corners[3][0], corners[3][1],
                0.0, 0.0
            ], dtype=np.float32)
            features[0:12:2] /= 640.0
            features[1:12:2] /= 480.0
            features[12:] /= 100.0
            features_tensor = torch.tensor(features).unsqueeze(0)
            
        t3 = time.perf_counter()
        
        t4 = time.perf_counter()
        if features_tensor is not None:
            if is_onnx:
                _ = session.run(None, {input_name: features_tensor.numpy()})
            else:
                with torch.no_grad():
                    _ = mlp_model(features_tensor.to(device))
        t5 = time.perf_counter()
        
        disk_times.append((t1 - t0) * 1000.0)
        prep_times.append(yolo_total + (t3 - t2) * 1000.0)
        net_times.append((t5 - t4) * 1000.0)
        post_times.append(0.0)

    return summarize_granular_timing(disk_times, prep_times, net_times, post_times)


def summarize_granular_timing(disk: List[float], prep: List[float], net: List[float], post: List[float]) -> Dict[str, float]:
    d = np.array(disk, dtype=np.float32) if disk else np.array([0.0])
    pr = np.array(prep, dtype=np.float32) if prep else np.array([0.0])
    n = np.array(net, dtype=np.float32) if net else np.array([0.0])
    po = np.array(post, dtype=np.float32) if post else np.array([0.0])
    total = d + pr + n + po
    
    return {
        "Mean_Disk_IO_Time_ms": float(np.mean(d)),
        "Mean_Preprocess_Time_ms": float(np.mean(pr)),
        "Mean_Network_Time_ms": float(np.mean(n)),
        "Mean_Postprocess_Time_ms": float(np.mean(po)),
        "Mean_Inference_Time_ms": float(np.mean(total)),
        "Max_Inference_Time_ms": float(np.max(total)),
        "FPS_Estimate": float(1000.0 / np.mean(total)) if np.mean(total) > 0 else 0.0,
        "Sampled_Frames": len(total),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Granular inference benchmark for YOLO, ResNet, MLP and ONNX models"
    )
    parser.add_argument(
        "--models_dir",
        default=models_dir_default,
        help="Path to ml_vision/models directory",
    )
    parser.add_argument(
        "--data_dir",
        default=data_dir_default,
        help="Path to the evaluation data directory",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.1,
        help="Fraction of the test split to benchmark",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of frames to benchmark per model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output_json",
        default=os.path.join(script_dir, "inference_time_summary.json"),
        help="Summary JSON output",
    )
    parser.add_argument(
        "--update_metrics",
        action="store_true",
        help="Update evaluation_metrics.json in each model folder with the new inference time values",
    )
    parser.add_argument(
        "--model-types",
        type=str,
        default="yolo,resnet,mlp,cnn,onnx",
        help="Comma-separated model types to benchmark: yolo,resnet,mlp,cnn,onnx",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default="",
        help="Comma-separated exact model directory names to benchmark",
    )
    args = parser.parse_args()

    allowed_model_types = {
        t.strip().lower() for t in args.model_types.split(",") if t.strip()
    }
    allowed_model_names = {n.strip() for n in args.model_names.split(",") if n.strip()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_data(model_name):
        csv_name = get_dataset_args_for_model(model_name)
        df = None
        images_dir = None
        
        if "_0730_" in model_name:
            data_sub = os.path.join(args.data_dir, "session_20260730_174916")
            if os.path.exists(data_sub):
                df = read_dataset_csv(data_sub, csv_name)
                images_dir = os.path.join(data_sub, "images")
        elif "_0728_" in model_name:
            data_sub = os.path.join(args.data_dir, "session_20260728_102908")
            if os.path.exists(data_sub):
                df = read_dataset_csv(data_sub, csv_name)
                images_dir = os.path.join(data_sub, "images")
        elif "_iphone_" in model_name:
            data_sub = os.path.join(args.data_dir, "images_iphone")
            if os.path.exists(data_sub):
                df = read_dataset_csv(data_sub, csv_name)
                images_dir = os.path.join(data_sub, "images")

        if df is None:
            df = read_dataset_csv(args.data_dir, csv_name)
            images_dir = os.path.join(args.data_dir, "images")

        return df, images_dir

    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            try:
                summary = json.load(f)
            except:
                summary = {"models": [], "sample_fraction": args.sample_fraction, "max_samples": args.max_samples}
    else:
        summary = {
            "models": [],
            "sample_fraction": args.sample_fraction,
            "max_samples": args.max_samples,
        }

    for model_name in sorted(os.listdir(args.models_dir)):
        if not is_model_dir(model_name):
            continue

        model_root = os.path.join(args.models_dir, model_name)
        if not os.path.isdir(model_root):
            continue

        if allowed_model_names and model_name not in allowed_model_names:
            continue
            
        df, images_dir = get_data(model_name)

        model_type = "unknown"
        metrics = None
        try:
            if "yolo" in model_name.lower() and "yolo" in allowed_model_types:
                model_type = "yolo"
                model_path = find_yolo_model_path(model_root, ext=".pt")
                if model_path:
                    sample_df = select_sample_rows(df, args.sample_fraction, args.max_samples, args.seed)
                    metrics = benchmark_yolo_model(model_name, model_path, sample_df, images_dir)
                    
            elif ("resnet" in model_name.lower() or "cnn" in model_name.lower()) and ("resnet" in allowed_model_types or "cnn" in allowed_model_types):
                model_type = "cnn" if "cnn" in model_name.lower() else "resnet"
                checkpoint_path = find_resnet_checkpoint(model_root, ext=".pth")
                if checkpoint_path:
                    test_dataset = BallDataset(
                        os.path.join(images_dir, "../", get_dataset_args_for_model(model_name)),
                        images_dir,
                        transform=transforms.Compose([
                            (transforms.Resize((240, 320)) if ("resnet18" in model_name.lower() or "cnn" in model_name.lower()) else transforms.Resize((480, 640))),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                    )
                    sample_indices = sample_dataset_indices(len(test_dataset), args.sample_fraction, args.max_samples, args.seed)
                    metrics = benchmark_resnet_model(model_name, checkpoint_path, sample_indices, test_dataset, device)
                    
            elif "mlp" in model_name.lower() and "mlp" in allowed_model_types:
                model_type = "mlp"
                mlp_path = find_mlp_checkpoint(model_root, ext=".pth")
                if mlp_path:
                    yolo_model_path = os.path.join(args.models_dir, "yolov8_platform_pose_markers_iphone_v1", "weights", "best.pt")
                    sample_df = select_sample_rows(df, args.sample_fraction, args.max_samples, args.seed)
                    metrics = benchmark_mlp_model(model_name, yolo_model_path, mlp_path, sample_df, images_dir, device)
                    
            if metrics:
                model_summary = {"model_name": model_name, "model_type": model_type, "model_root": model_root, **metrics}
                print(f"{model_name}: {metrics['Mean_Inference_Time_ms']:.2f} ms avg (Net: {metrics['Mean_Network_Time_ms']:.2f} ms)")
                if args.update_metrics:
                    update_evaluation_metrics(model_root, metrics)
                
                # Update existing or append
                found = False
                for m in summary["models"]:
                    if m["model_name"] == model_name:
                        m.update(model_summary)
                        found = True
                        break
                if not found:
                    summary["models"].append(model_summary)
                
        except Exception as exc:
            print(f"Error benchmarking PyTorch {model_name}: {exc}")

        if "onnx" in allowed_model_types:
            try:
                if model_type == "yolo":
                    onnx_path = find_yolo_model_path(model_root, ext=".onnx")
                    if onnx_path:
                        sample_df = select_sample_rows(df, args.sample_fraction, args.max_samples, args.seed)
                        metrics_onnx = benchmark_yolo_model(model_name, onnx_path, sample_df, images_dir)
                        onnx_summary = {"model_name": f"{model_name}_ONNX", "model_type": "onnx", "model_root": model_root, **metrics_onnx}
                        found = False
                        for m in summary["models"]:
                            if m["model_name"] == onnx_summary["model_name"]:
                                m.update(onnx_summary)
                                found = True
                                break
                        if not found:
                            summary["models"].append(onnx_summary)
                        print(f"{model_name}_ONNX: {metrics_onnx['Mean_Inference_Time_ms']:.2f} ms avg (Net: {metrics_onnx['Mean_Network_Time_ms']:.2f} ms)")
                
                elif model_type in ["resnet", "cnn"]:
                    onnx_path = find_resnet_checkpoint(model_root, ext=".onnx")
                    if onnx_path:
                        test_dataset = BallDataset(
                            os.path.join(images_dir, "../", get_dataset_args_for_model(model_name)),
                            images_dir,
                            transform=transforms.Compose([
                                (transforms.Resize((240, 320)) if ("resnet18" in model_name.lower() or "cnn" in model_name.lower()) else transforms.Resize((480, 640))),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
                        )
                        sample_indices = sample_dataset_indices(len(test_dataset), args.sample_fraction, args.max_samples, args.seed)
                        metrics_onnx = benchmark_onnx_model(model_name, onnx_path, sample_indices, test_dataset)
                        onnx_summary = {"model_name": f"{model_name}_ONNX", "model_type": "onnx", "model_root": model_root, **metrics_onnx}
                        found = False
                        for m in summary["models"]:
                            if m["model_name"] == onnx_summary["model_name"]:
                                m.update(onnx_summary)
                                found = True
                                break
                        if not found:
                            summary["models"].append(onnx_summary)
                        print(f"{model_name}_ONNX: {metrics_onnx['Mean_Inference_Time_ms']:.2f} ms avg (Net: {metrics_onnx['Mean_Network_Time_ms']:.2f} ms)")
                        
                elif model_type == "mlp":
                    onnx_path = find_mlp_checkpoint(model_root, ext=".onnx")
                    if onnx_path:
                        yolo_model_path = os.path.join(args.models_dir, "yolov8_platform_pose_markers_iphone_v1", "weights", "best.pt")
                        sample_df = select_sample_rows(df, args.sample_fraction, args.max_samples, args.seed)
                        onnx_summary = {"model_name": f"{model_name}_ONNX", "model_type": "onnx", "model_root": model_root, **metrics_onnx}
                        
                        found = False
                        for m in summary["models"]:
                            if m["model_name"] == onnx_summary["model_name"]:
                                m.update(onnx_summary)
                                found = True
                                break
                        if not found:
                            summary["models"].append(onnx_summary)
                            
                        print(f"{model_name}_ONNX: {metrics_onnx['Mean_Inference_Time_ms']:.2f} ms avg (Net: {metrics_onnx['Mean_Network_Time_ms']:.2f} ms)")
                        
            except Exception as exc:
                print(f"Error benchmarking ONNX {model_name}: {exc}")

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved benchmark summary to {args.output_json}")


if __name__ == "__main__":
    main()
