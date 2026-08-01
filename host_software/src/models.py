import os
import torch
import torch.nn as nn
from torchvision import models


def load_yolo_model(model_path, device):
    from ultralytics import YOLO

    print(f"Loading YOLO Model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return None
    model = YOLO(model_path)
    model.to(device)
    return model


def load_mlp_corrector_v1_model(
    model_path, device, input_dim=14, hidden_dim=128, output_dim=2
):
    """Load an MLP corrector model.

    This is a generic loader that will instantiate `CorrectorMLP` with the
    provided dimensions and attempt to load `model_path`. If loading fails,
    the model is returned with random weights.

    Kept the original function name for compatibility with existing callers.
    """
    from ml_vision.core.corrector_mlp import CorrectorMLP

    print("Loading MLP Corrector Model...")
    model = CorrectorMLP(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded weights from {model_path}")
        except Exception as e:
            print(
                f"WARNING: Could not load weights from {model_path}: {e}. Using random weights."
            )
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")

    model = model.to(device)
    model.eval()
    return model


def load_expert_model(model_path, device):
    print("Loading PyTorch ResNet18 Expert Model...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Training script saves a full checkpoint dict; handle both formats.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "?")
            print(f"Successfully loaded checkpoint (epoch {epoch}) from {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")

    model = model.to(device)
    model.eval()
    return model


def load_cnn_expert_model(model_path, device):
    from ml_vision.training.basic_cnn import BasicCNN

    print("Loading Custom BasicCNN Expert Model...")
    model = BasicCNN()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Training script saves a full checkpoint dict; handle both formats.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "?")
            print(f"Successfully loaded checkpoint (epoch {epoch}) from {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")

    model = model.to(device)
    model.eval()
    return model


def process_vision_frame(frame, yolo_model, mlp_corrector_v1_model, projector, device):
    import numpy as np

    results = yolo_model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
    if not results or len(results) == 0 or results[0].boxes is None:
        return None, None, None

    res = results[0]
    classes = res.boxes.cls.cpu().numpy()
    boxes = res.boxes.xywh.cpu().numpy()

    corners = None
    ball_box = None
    detected_markers = {}

    for i, cls in enumerate(classes):
        c = int(cls)
        if c == 0:
            if res.keypoints is not None and len(res.keypoints.xy) > i:
                kpts = res.keypoints.xy[i].cpu().numpy()
                if len(kpts) == 4:
                    corners = kpts
        elif c == 1:
            ball_box = boxes[i]
        elif c >= 2:
            name = yolo_model.names[c].replace("_marker", "").replace("_target", "")
            detected_markers[name] = boxes[i]

    if corners is None or ball_box is None:
        return None, None, None

    homography_x, homography_y = 0.0, 0.0
    marker_coords = {}
    if projector.update_homography(corners):
        hx, hy = projector.project_point(ball_box[0], ball_box[1])
        if hx is not None and hy is not None:
            homography_x, homography_y = hx, hy
        for name, box in detected_markers.items():
            mx, my = projector.project_point(box[0], box[1])
            if mx is not None and my is not None:
                marker_coords[name] = (mx, my)

    features = np.array(
        [
            ball_box[0],
            ball_box[1],
            ball_box[2],
            ball_box[3],
            corners[0][0],
            corners[0][1],
            corners[1][0],
            corners[1][1],
            corners[2][0],
            corners[2][1],
            corners[3][0],
            corners[3][1],
            homography_x,
            homography_y,
        ],
        dtype=np.float32,
    )
    features[0:12:2] /= 640.0
    features[1:12:2] /= 480.0
    features[12:] /= 100.0
    input_tensor = torch.tensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        output = mlp_corrector_v1_model(input_tensor)
    cam_x, cam_y = output[0].cpu().numpy()

    return cam_x, cam_y, marker_coords
