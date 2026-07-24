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

def load_corrector_model(model_path, device):
    from ml_vision.core.corrector_mlp import CorrectorMLP
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

def load_expert_model(model_path, device):
    print("Loading PyTorch ResNet18 Expert Model...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")
        
    model = model.to(device)
    model.eval()
    return model
