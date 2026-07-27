import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dataset import VLADataset
from core.vla_architecture import RT1LiteVLA

def evaluate_vla():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(script_dir, "../../data/03_gold/vla_dataset.json"))
    model_path = os.path.abspath(os.path.join(script_dir, "../models/vla_v1/best_vla.pth"))
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VLADataset(dataset_path, transform=transform)
    
    # Let's just evaluate on the whole dataset for simplicity in this script
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RT1LiteVLA().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0
    all_errors = []
    
    print(f"Evaluating VLA model on {len(dataset)} samples...")
    
    with torch.no_grad():
        for imgs, cmds, states, actions in test_loader:
            imgs, cmds, states, actions = imgs.to(device), cmds.to(device), states.to(device), actions.to(device)
            preds = model(imgs, cmds, states)
            loss = criterion(preds, actions)
            total_loss += loss.item() * imgs.size(0)
            
            # Compute Euclidean error in action space
            errors = torch.norm(preds - actions, dim=1).cpu().numpy()
            all_errors.extend(errors)
            
    mse = total_loss / len(dataset)
    mean_euclidean_error = float(np.mean(all_errors))
    
    print(f"Evaluation Complete!")
    print(f"Mean Squared Error (Action Space): {mse:.4f}")
    print(f"Mean Euclidean Action Error: {mean_euclidean_error:.4f}")
    
    # Save metrics
    metrics = {
        "MSE_Action": mse,
        "Mean_Euclidean_Action_Error": mean_euclidean_error
    }
    
    metrics_path = os.path.join(os.path.dirname(model_path), "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    evaluate_vla()
