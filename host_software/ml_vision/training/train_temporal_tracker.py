import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from temporal_ball_dataset import SequenceBallDataset, TemporalTrainTransform, TemporalTestTransform
import argparse
from resnet18_lstm import TemporalExpertTracker

def train_model():
    parser = argparse.ArgumentParser(description="Train Temporal Expert Tracker")
    parser.add_argument("--save_dir", default="../models/temporal_expert_tracker", help="Directory to save the trained models")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '../data/02_silver'))
    
    csv_path = os.path.join(data_dir, 'labels_sequential.csv')
    images_dir = os.path.join(data_dir, 'images')

    if os.path.isabs(args.save_dir):
        project_dir = args.save_dir
    else:
        project_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
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
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Total Sequences: {len(full_dataset_train)}. Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    # 2. Model
    model = TemporalExpertTracker(hidden_size=256, num_layers=1)
    model = model.to(device)
    
    # 3. Training setup
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    num_epochs = 30
    start_epoch = 0
    best_test_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint.get('best_test_loss', float('inf'))
        else:
            model.load_state_dict(checkpoint)
            
    for epoch in range(start_epoch, num_epochs):
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
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_loss': best_test_loss if avg_test_loss >= best_test_loss else avg_test_loss
        }
        
        latest_path = os.path.join(project_dir, 'latest_temporal_tracker.pth')
        torch.save(checkpoint, latest_path)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = os.path.join(project_dir, 'best_temporal_tracker.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved new best model to {save_path}")

if __name__ == '__main__':
    train_model()
