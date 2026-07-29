import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1. The MLP Corrector Architecture
# ---------------------------------------------------------------------------
class CorrectorMLP(nn.Module):
    """
    Translates YOLO bounding boxes and homography features into physical 
    (cam_x, cam_y) coordinates.
    """
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=2):
        super(CorrectorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------------------
# 2. Model Loaders
# ---------------------------------------------------------------------------
def load_yolo_model(model_path, device):
    print(f"Loading YOLO Model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return None
    model = YOLO(model_path)
    model.to(device)
    return model

def load_mlp_corrector_v1_model(model_path, device):
    print("Loading MLP Corrector Model...")
    model = CorrectorMLP(input_dim=14, hidden_dim=128, output_dim=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")
    
    model = model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 3. The Vision Forward Pass
# ---------------------------------------------------------------------------
def process_vision_frame(frame, yolo_model, mlp_corrector_v1_model, projector, device):
    """
    Runs a single frame through YOLO, extracts features, projects homography, 
    and outputs the final physical coordinates via the MLP corrector.
    """
    results = yolo_model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
    if not results or len(results) == 0 or results[0].boxes is None:
        return None, None, None
        
    res = results[0]
    classes = res.boxes.cls.cpu().numpy()
    boxes = res.boxes.xywh.cpu().numpy()
    
    corners = None
    ball_box = None
    detected_markers = {}
    
    # Parse YOLO outputs
    for i, cls in enumerate(classes):
        c = int(cls)
        if c == 0:
            if res.keypoints is not None and len(res.keypoints.xy) > i:
                kpts = res.keypoints.xy[i].cpu().numpy()
                if len(kpts) == 4: corners = kpts
        elif c == 1:
            ball_box = boxes[i]
        elif c >= 2:
            name = yolo_model.names[c].replace('_marker', '').replace('_target', '')
            detected_markers[name] = boxes[i]
            
    if corners is None or ball_box is None:
        return None, None, None
        
    # Project through homography
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
                
    # Normalize features for the MLP
    features = np.array([
        ball_box[0], ball_box[1], ball_box[2], ball_box[3],
        corners[0][0], corners[0][1], corners[1][0], corners[1][1],
        corners[2][0], corners[2][1], corners[3][0], corners[3][1],
        homography_x, homography_y
    ], dtype=np.float32)
    
    features[0:12:2] /= 640.0
    features[1:12:2] /= 480.0
    features[12:] /= 100.0
    
    input_tensor = torch.tensor(features).unsqueeze(0).to(device)
    
    # Run through the corrector
    with torch.no_grad():
        output = mlp_corrector_v1_model(input_tensor)
        
    cam_x, cam_y = output[0].cpu().numpy()
    
    return cam_x, cam_y, marker_coords