import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from temporal_ball_dataset import SequenceBallDataset, TemporalTrainTransform, TemporalTestTransform
from resnet18_lstm import TemporalExpertTracker

def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '../data/02_silver'))
    
    csv_path = os.path.join(data_dir, 'labels_sequential.csv')
    images_dir = os.path.join(data_dir, 'images')
    project_dir = os.path.abspath(os.path.join(script_dir, '../models/temporal_expert_tracker'))
    os.makedirs(project_dir, exist_ok=True)
    
    # 1. Datasets
    seq_len = 10
    train_transform = TemporalTrainTransform()
    test_transform = TemporalTestTransform()
    
    full_dataset_train = SequenceBallDataset(csv_file=csv_path, root_dir=images_dir, seq_len=seq_len, transform=train_transform)
    full_dataset_test = SequenceBallDataset(csv_file=csv_path, root_dir=images_dir, seq_len=seq_len, transform=test_transform)
    
    # Strictly sequential split to avoid data leakage
    indices = list(range(len(full_dataset_train)))
    train_size = int(0.8 * len(indices))
    
    train_dataset = Subset(full_dataset_train, indices[:train_size])
    test_dataset = Subset(full_dataset_test, indices[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Total Sequences: {len(full_dataset_train)}. Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    # 2. Model
    model = TemporalExpertTracker(hidden_size=256, num_layers=1)
    model = model.to(device)
    
    # 3. Training setup
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    num_epochs = 30
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # Targets are (Batch, SeqLen, 2). We only need the target for the LAST frame in the sequence
            # because the model only predicts the ball position at the end of the sequence.
            final_targets = targets[:, -1, :].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, final_targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
                
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                final_targets = targets[:, -1, :].to(device)
                outputs = model(images)
                loss = criterion(outputs, final_targets)
                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader)
        scheduler.step(avg_test_loss)
        
        print(f"=== Epoch {epoch+1} Summary ===")
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = os.path.join(project_dir, 'best_temporal_tracker.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

if __name__ == '__main__':
    train_model()
