import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

import argparse
from ball_dataset import BallDataset

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 Expert Tracker")
    parser.add_argument("--data_dir", default="../data/02_silver", help="Path to data directory")
    parser.add_argument("--csv_name", default="labels_normalized.csv", help="Name of the labels CSV file")
    parser.add_argument("--save_dir", default="../models", help="Directory to save the trained models")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    args = parser.parse_args()

    print("Initializing PyTorch Expert Tracker Model (ResNet18)...")
    
    # 1. Initialize pre-trained ResNet18
    # We use a standard CNN backbone which will easily allow us to add
    # multi-task heads in the future (e.g., finding coloured markers, or predicting control signals)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the classification head with a regression head for (x, y)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 2. Set absolute paths for dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle absolute vs relative data_dir
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
        
    csv_path = os.path.join(data_dir, args.csv_name)
    images_dir = os.path.join(data_dir, 'images')
    
    # Handle absolute vs relative save_dir
    if os.path.isabs(args.save_dir):
        project_dir = args.save_dir
    else:
        project_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))
    
    # Ensure models directory exists
    os.makedirs(project_dir, exist_ok=True)
    
    print(f"Loading dataset from: {csv_path}")
    
    # Define Transforms
    # We apply RandomAffine (Shift, Scale, Rotate) and RandomPerspective (Tilt) to simulate camera movement.
    # Because the model predicts the ball's *intrinsic* physical coordinate on the board (touch_x, touch_y),
    # moving the camera does NOT change the physical label! This brilliantly forces the model to learn 
    # camera-invariance by locating the purple shape of the platform's outline rather than memorizing absolute pixel locations.
    # Note: RandomHorizontalFlip is strictly forbidden as it creates a physically impossible mirrored board.
    train_transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((240, 320)),
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Found {len(full_dataset_train)} total images -> {len(train_dataset)} Train | {len(test_dataset)} Test.")
    
    # 4. Training loop setup
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    num_epochs = 10
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
        else:
            model.load_state_dict(checkpoint)
            
    save_path = os.path.join(project_dir, 'resnet18_expert_tracker/expert_tracker_best.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                running_test_loss += test_loss.item() * inputs.size(0)
                
        epoch_test_loss = running_test_loss / len(test_dataset)
        
        # Step the scheduler
        scheduler.step(epoch_test_loss)
        
        print(f"--- Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} ---")
        
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
            
        # Save the latest model at the end of every epoch just in case Colab crashes!
        latest_path = os.path.join(project_dir, 'resnet18_expert_tracker/expert_tracker_latest.pth')
        torch.save(checkpoint, latest_path)

    print("Training complete!")

if __name__ == '__main__':
    # Required for Windows multiprocessing (num_workers > 0)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
