import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.corrector_mlp import CorrectorMLP

class YoloFeatureDataset(Dataset):
    def __init__(self, csv_file, split='train'):
        df = pd.read_csv(csv_file)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # 14 features: ball_x, ball_y, ball_w, ball_h, kpt0_x... kpt3_y, homography_x, homography_y
        self.X = self.df[[
            'ball_x', 'ball_y', 'ball_w', 'ball_h',
            'kpt0_x', 'kpt0_y', 'kpt1_x', 'kpt1_y',
            'kpt2_x', 'kpt2_y', 'kpt3_x', 'kpt3_y',
            'homography_x', 'homography_y'
        ]].values.astype('float32')
        
        # Targets: touch_x, touch_y
        self.y = self.df[['touch_x', 'touch_y']].values.astype('float32')
        
        # We should normalize the inputs to help the MLP learn faster
        # The pixel coordinates are bounded to 640x480 max from YOLO
        self.X[:, 0:12:2] /= 640.0 # x coords
        self.X[:, 1:12:2] /= 480.0 # y coords
        # Homography features are in mm, usually bounded by roughly [-100, 100]
        self.X[:, 12:] /= 100.0
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="../data/02_silver/yolo_features.csv", help="Path to yolo features CSV")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    csv_path = os.path.abspath(os.path.join(script_dir, args.data_csv))
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Please run extract_yolo_features.py first!")
        return
        
    train_dataset = YoloFeatureDataset(csv_path, split='train')
    test_dataset = YoloFeatureDataset(csv_path, split='test')
    
    print(f"Loaded {len(train_dataset)} training and {len(test_dataset)} testing samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CorrectorMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    train_losses = []
    test_losses = []
    
    best_test_loss = float('inf')
    save_dir = os.path.abspath(os.path.join(script_dir, '../models/corrector'))
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'best_corrector.pth')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # Jitter Augmentation: Add small Gaussian noise to input features during training
            noise = torch.randn_like(X) * 0.005
            X_noisy = X + noise
            
            optimizer.zero_grad()
            out = model(X_noisy)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                total_test_loss += loss.item()
                
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_path)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss (MSE mm^2): {avg_train_loss:.2f} | Test Loss: {avg_test_loss:.2f}")
            
    print(f"Training complete! Best model saved to {model_save_path}")
    print(f"Best Test Loss (MSE): {best_test_loss:.2f} (Approx RMSE = {best_test_loss**0.5:.2f} mm)")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Corrector MLP Training Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    print("Saved training_curve.png")

if __name__ == '__main__':
    train()
