import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import argparse
from ball_dataset import BallDataset
from basic_cnn import BasicCNN

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def main():
    parser = argparse.ArgumentParser(description="Train Basic CNN Expert Tracker")
    parser.add_argument("--data_dir", default="../../data/02_silver/session_20260728_102908", help="Path to session data directory")
    parser.add_argument("--csv_name", default="labels_normalized.csv", help="Name of the labels CSV file")
    parser.add_argument("--save_dir", default="../models", help="Directory to save the trained models")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    args = parser.parse_args()

    print("Initializing PyTorch Custom Basic CNN Expert Tracker Model...")
    model = BasicCNN()
    img_size = (240, 320)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # Speeds up fixed-size batch training
    model = model.to(device)
    
    # 2. Set absolute paths for dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle absolute vs relative data_dir
    data_dir = os.path.abspath(args.data_dir)
        
    # Resolve CSV path: check if the argument directly exists as a path,
    # otherwise treat it as relative to the data_dir.
    if os.path.exists(args.csv_name):
        csv_path = os.path.abspath(args.csv_name)
    else:
        csv_path = os.path.join(data_dir, args.csv_name)
        
    images_dir = os.path.join(data_dir, 'images')
    
    # Handle absolute vs relative save_dir
    # Dynamically update the save_dir to ensure models are kept organized by architecture
    if os.path.basename(args.save_dir) == "models" or os.path.basename(args.save_dir) == "models/":
        args.save_dir = os.path.join(args.save_dir, "cnn_expert_tracker")
    project_dir = os.path.abspath(args.save_dir)
    
    # Ensure models directory exists
    os.makedirs(project_dir, exist_ok=True)
    
    print(f"Loading dataset from: {csv_path}")
    
    # Define Transforms
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Create Dataset and DataLoader
    full_dataset_train = BallDataset(csv_file=csv_path, root_dir=images_dir, transform=train_transform)
    full_dataset_test = BallDataset(csv_file=csv_path, root_dir=images_dir, transform=test_transform)
    
    # Split strictly sequentially: Train on first 80%, Test on strictly subsequent 20%
    # This prevents temporal data leakage across video frames.
    indices = list(range(len(full_dataset_train)))
    train_size = int(0.8 * len(indices))
    
    train_dataset = Subset(full_dataset_train, indices[:train_size])
    test_dataset = Subset(full_dataset_test, indices[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Found {len(full_dataset_train)} total images -> {len(train_dataset)} Train | {len(test_dataset)} Test.")
    
    # 4. Training loop setup
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    num_epochs = 50
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n[DIAGNOSTIC] Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"[DIAGNOSTIC] Successfully loaded state! Resuming from Epoch {start_epoch + 1}")
        else:
            model.load_state_dict(checkpoint)
            print("[DIAGNOSTIC] Loaded bare model weights. Resuming from Epoch 1")
    else:
        if args.resume:
            print(f"\n[DIAGNOSTIC] WARNING: Checkpoint '{args.resume}' not found. Starting from SCRATCH!")
        else:
            print("\n[DIAGNOSTIC] No resume checkpoint provided. Starting from SCRATCH!")
            
    save_path = os.path.join(project_dir, 'cnn_expert_tracker_v1/expert_tracker_best.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Starting training on {device}...")
    
    # Initialize Mixed Precision Scaler for faster training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    import csv
    log_path = os.path.join(project_dir, 'cnn_expert_tracker_v1/training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss'])
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with AMP
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward and optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / len(train_dataset)
        
        # --- TEST PHASE ---
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                running_test_loss += test_loss.item() * inputs.size(0)
                
                # DIAGNOSTIC: Check if model is stuck in mean prediction trap
                if i == 0:
                    print(f"\n[DIAGNOSTIC] Epoch {epoch+1} - First batch predictions vs targets:")
                    for j in range(min(5, outputs.size(0))):
                        t_np = targets[j].cpu().numpy()
                        p_np = outputs[j].cpu().numpy()
                        print(f"  Target: [{t_np[0]:.4f}, {t_np[1]:.4f}] | Pred: [{p_np[0]:.4f}, {p_np[1]:.4f}]")
                
        epoch_test_loss = running_test_loss / len(test_dataset)
        
        # Step the scheduler
        scheduler.step(epoch_test_loss)
        
        print(f"--- Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} ---")
        
        # Append to log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, epoch_train_loss, epoch_test_loss])
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss if epoch_test_loss >= best_loss else epoch_test_loss
        }
        
        # Save the best model based on TEST loss
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(checkpoint, save_path)
            print(f"Saved new best model to {save_path}")
            
        # Save the latest model at the end of every epoch just in case training is interrupted
        latest_path = os.path.join(project_dir, 'cnn_expert_tracker_v1/expert_tracker_latest.pth')
        torch.save(checkpoint, latest_path)

    print("Training complete!")

if __name__ == '__main__':
    # Required for Windows multiprocessing (num_workers > 0)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
