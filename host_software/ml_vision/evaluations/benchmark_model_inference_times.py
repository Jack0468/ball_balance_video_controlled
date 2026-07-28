import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
vision_dir = os.path.abspath(os.path.join(script_dir, '..'))
if vision_dir not in __import__('sys').path:
    __import__('sys').path.insert(0, vision_dir)
models_dir_default = os.path.abspath(os.path.join(script_dir, '../models'))
data_dir_default = os.path.abspath(os.path.join(script_dir, '../../data/02_silver'))


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

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]['image_file'])
        image = Image.open(img_name).convert('RGB')
        touch_x = self.labels_df.iloc[idx]['touch_x']
        touch_y = self.labels_df.iloc[idx]['touch_y']
        MAX_BOUND = 200.0
        target = torch.tensor([touch_x / MAX_BOUND, touch_y / MAX_BOUND], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def is_model_dir(name: str) -> bool:
    lower = name.lower()
    return (
        lower != 'archive'
        and 'temporal' not in lower
        and ('yolo' in lower or 'resnet' in lower or 'mlp' in lower or 'cnn' in lower)
    )


def find_yolo_model_path(model_root: str) -> Optional[str]:
    weights_dir = os.path.join(model_root, 'weights')
    candidates = []
    if os.path.isdir(weights_dir):
        candidates.extend([os.path.join(weights_dir, fn) for fn in os.listdir(weights_dir) if fn.endswith('.pt')])
    candidates.extend([os.path.join(model_root, fn) for fn in os.listdir(model_root) if fn.endswith('.pt')])
    for name in ['best.pt', 'last.pt']:
        path = os.path.join(weights_dir, name)
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def find_resnet_checkpoint(model_root: str) -> Optional[str]:
    for name in ['expert_tracker_best.pth', 'expert_tracker_latest.pth', 'model_best.pth', 'best.pth']:
        candidate = os.path.join(model_root, name)
        if os.path.exists(candidate):
            return candidate
    pths = [os.path.join(model_root, fn) for fn in os.listdir(model_root) if fn.endswith('.pth')]
    return pths[0] if pths else None


def find_mlp_checkpoint(model_root: str) -> Optional[str]:
    for name in ['best_corrector.pth', 'best_corrector.bin', 'best.pth']:
        candidate = os.path.join(model_root, name)
        if os.path.exists(candidate):
            return candidate
    pths = [os.path.join(model_root, fn) for fn in os.listdir(model_root) if fn.endswith('.pth') or fn.endswith('.bin')]
    return pths[0] if pths else None


def read_dataset_csv(data_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, 'labels_sequential.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def select_sample_rows(df: pd.DataFrame, fraction: float, max_samples: int, seed: int) -> pd.DataFrame:
    test_df = df.iloc[int(0.8 * len(df)):]
    sample_count = min(max(1, int(len(test_df) * fraction)), max_samples)
    return test_df.sample(n=sample_count, random_state=seed).reset_index(drop=True)


def sample_dataset_indices(dataset_length: int, fraction: float, max_samples: int, seed: int) -> List[int]:
    test_indices = list(range(int(0.8 * dataset_length), dataset_length))
    sample_count = min(max(1, int(len(test_indices) * fraction)), max_samples)
    random.seed(seed)
    return random.sample(test_indices, sample_count)


def update_evaluation_metrics(model_root: str, metrics: Dict[str, float]) -> None:
    metrics_path = os.path.join(model_root, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(metrics)
    with open(metrics_path, 'w') as f:
        json.dump(existing, f, indent=4)


def benchmark_yolo_model(model_name: str, model_path: str, sample_df: pd.DataFrame, images_dir: str) -> Dict[str, float]:
    model = YOLO(model_path)
    inference_times = []
    warmup_n = min(2, len(sample_df))

    for idx in range(warmup_n):
        row = sample_df.iloc[idx]
        img_path = os.path.join(images_dir, row['image_file'])
        _ = model.predict(source=img_path, imgsz=640, conf=0.5, verbose=False)

    for _, row in sample_df.iterrows():
        img_path = os.path.join(images_dir, row['image_file'])
        t0 = time.perf_counter()
        _ = model.predict(source=img_path, imgsz=640, conf=0.5, verbose=False)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000.0)

    return summarize_timing(inference_times)


def benchmark_resnet_model(model_name: str, checkpoint_path: str, sample_indices: List[int], dataset: Dataset, device: torch.device) -> Dict[str, float]:
    if 'cnn' in model_name.lower():
        from training.basic_cnn import BasicCNN
        model = BasicCNN()
    else:
        arch = 'resnet50' if 'resnet50' in model_name.lower() else 'resnet18'
        if arch == 'resnet50':
            model = models.resnet50(weights=None)
            img_size = (480, 640)
        else:
            model = models.resnet18(weights=None)
            img_size = (240, 320)

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    subset = Subset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
    inference_times = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            t0 = time.perf_counter()
            outputs = model(inputs)
            t1 = time.perf_counter()
            batch_time = (t1 - t0) * 1000.0
            inference_times.extend([batch_time / inputs.size(0)] * inputs.size(0))

    return summarize_timing(inference_times)


def benchmark_mlp_model(model_name: str, yolo_path: str, mlp_path: str, sample_df: pd.DataFrame, images_dir: str, device: torch.device) -> Dict[str, float]:
    from core.corrector_mlp import CorrectorMLP
    from core.coordinate_math import HomographyProjector

    yolo_model = YOLO(yolo_path)
    mlp_model = CorrectorMLP().to(device)
    checkpoint = torch.load(mlp_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        mlp_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        mlp_model.load_state_dict(checkpoint)
    mlp_model.eval()

    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    inference_times = []

    warmup_n = min(2, len(sample_df))
    for idx in range(warmup_n):
        row = sample_df.iloc[idx]
        img_path = os.path.join(images_dir, row['image_file'])
        _ = yolo_model.predict(source=img_path, imgsz=640, conf=0.5, verbose=False)

    for _, row in sample_df.iterrows():
        img_path = os.path.join(images_dir, row['image_file'])
        img = Image.open(img_path).convert('RGB')

        t0 = time.perf_counter()
        results = yolo_model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        res = results[0] if len(results) > 0 else None
        corners = None
        ball_center = None
        if res is not None and res.boxes is not None:
            classes = res.boxes.cls.cpu().numpy()
            boxes = res.boxes.xywh.cpu().numpy()
            for i, cls in enumerate(classes):
                if int(cls) == 0 and res.keypoints is not None and len(res.keypoints.xy) > i:
                    kpts = res.keypoints.xy[i].cpu().numpy()
                    if len(kpts) == 4:
                        corners = kpts
                elif int(cls) == 1:
                    ball_center = (boxes[i][0], boxes[i][1])

        if corners is None or ball_center is None:
            # If the example is invalid, we still want to record the inference cost as YOLO+MLP execution.
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000.0)
            continue

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
            features_tensor = torch.tensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = mlp_model(features_tensor)

        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000.0)

    return summarize_timing(inference_times)


def summarize_timing(times: List[float]) -> Dict[str, float]:
    times_arr = np.array(times, dtype=np.float32)
    return {
        'Mean_Inference_Time_ms': float(np.mean(times_arr)),
        'Max_Inference_Time_ms': float(np.max(times_arr)),
        'FPS_Estimate': float(1000.0 / np.mean(times_arr)) if np.mean(times_arr) > 0 else 0.0,
        'Sampled_Frames': int(len(times_arr))
    }


def main():
    parser = argparse.ArgumentParser(description='Quick inference benchmark for YOLO, ResNet, and MLP models')
    parser.add_argument('--models_dir', default=models_dir_default, help='Path to ml_vision/models directory')
    parser.add_argument('--data_dir', default=data_dir_default, help='Path to the evaluation data directory')
    parser.add_argument('--sample_fraction', type=float, default=0.1, help='Fraction of the test split to benchmark')
    parser.add_argument('--max_samples', type=int, default=200, help='Maximum number of frames to benchmark per model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--output_json', default=os.path.join(script_dir, 'inference_time_summary.json'), help='Summary JSON output')
    parser.add_argument('--update_metrics', action='store_true', help='Update evaluation_metrics.json in each model folder with the new inference time values')
    parser.add_argument('--model-types', type=str, default='yolo,resnet,mlp,cnn', help='Comma-separated model types to benchmark: yolo,resnet,mlp,cnn')
    parser.add_argument('--model-names', type=str, default='', help='Comma-separated exact model directory names to benchmark')
    args = parser.parse_args()

    allowed_model_types = {t.strip().lower() for t in args.model_types.split(',') if t.strip()}
    allowed_model_names = {n.strip() for n in args.model_names.split(',') if n.strip()}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    df = read_dataset_csv(args.data_dir)
    images_dir = os.path.join(args.data_dir, 'images')

    summary = {'models': [], 'data_dir': args.data_dir, 'sample_fraction': args.sample_fraction, 'max_samples': args.max_samples}

    for model_name in sorted(os.listdir(args.models_dir)):
        if not is_model_dir(model_name):
            continue

        model_root = os.path.join(args.models_dir, model_name)
        if not os.path.isdir(model_root):
            continue

        if allowed_model_names and model_name not in allowed_model_names:
            continue

        model_type = 'unknown'
        metrics = None
        try:
            if 'yolo' in model_name.lower() and 'yolo' in allowed_model_types:
                model_type = 'yolo'
                model_path = find_yolo_model_path(model_root)
                if model_path is None:
                    print(f"Skipping {model_name}: no YOLO checkpoint found.")
                    continue
                sample_df = select_sample_rows(df, args.sample_fraction, args.max_samples, args.seed)
                metrics = benchmark_yolo_model(model_name, model_path, sample_df, images_dir)
            elif ('resnet' in model_name.lower() or 'cnn' in model_name.lower()) and ('resnet' in allowed_model_types or 'cnn' in allowed_model_types):
                model_type = 'cnn' if 'cnn' in model_name.lower() else 'resnet'
                checkpoint_path = find_resnet_checkpoint(model_root)
                if checkpoint_path is None:
                    print(f"Skipping {model_name}: no CNN/ResNet checkpoint found.")
                    continue
                test_dataset = BallDataset(os.path.join(args.data_dir, 'labels_sequential.csv'), images_dir, transform=transforms.Compose([
                    transforms.Resize((240, 320)) if ('resnet18' in model_name.lower() or 'cnn' in model_name.lower()) else transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
                sample_indices = sample_dataset_indices(len(test_dataset), args.sample_fraction, args.max_samples, args.seed)
                metrics = benchmark_resnet_model(model_name, checkpoint_path, sample_indices, test_dataset, device)
            elif 'mlp' in model_name.lower() and 'mlp' in allowed_model_types:
                model_type = 'mlp'
                mlp_path = find_mlp_checkpoint(model_root)
                if mlp_path is None:
                    print(f"Skipping {model_name}: no MLP checkpoint found.")
                    continue
                yolo_model_path = os.path.join(args.models_dir, 'yolov8_platform_pose_markers_v1', 'weights', 'best.pt')
                if not os.path.exists(yolo_model_path):
                    raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}. Please ensure yolov8_platform_pose_markers_v1/weights/best.pt exists.")
                sample_df = select_sample_rows(df, args.sample_fraction, args.max_samples, args.seed)
                metrics = benchmark_mlp_model(model_name, yolo_model_path, mlp_path, sample_df, images_dir, device)
            else:
                continue
        except Exception as exc:
            print(f"Error benchmarking {model_name}: {exc}")
            continue

        model_summary = {
            'model_name': model_name,
            'model_type': model_type,
            'model_root': model_root,
            **metrics
        }
        print(f"{model_name}: {metrics['Mean_Inference_Time_ms']:.2f} ms avg over {metrics['Sampled_Frames']} frames")

        if args.update_metrics:
            update_evaluation_metrics(model_root, metrics)

        summary['models'].append(model_summary)

    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Saved benchmark summary to {args.output_json}")


if __name__ == '__main__':
    main()
