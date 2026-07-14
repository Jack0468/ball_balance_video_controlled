import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models

import argparse
from ball_dataset import BallDataset

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 Expert Tracker (Subset)")
    parser.add_argument("--data_dir", default="../data/02_silver", help="Path to data directory")
    args = parser.parse_args()

    print("Initializing PyTorch Expert Tracker Model (ResNet18) [SUBSET MODE]...")
    
    # 1. Initialize pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the classification head with a regression head for (x, y)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 2. Set absolute paths for dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
        
    csv_path = os.path.join(data_dir, 'labels.csv')
    images_dir = os.path.join(data_dir, 'images')
    project_dir = os.path.abspath(os.path.join(script_dir, '../models'))
    
    os.makedirs(project_dir, exist_ok=True)
    
    print(f"Loading dataset from: {csv_path}")
    full_dataset = BallDataset(csv_file=csv_path, root_dir=images_dir)
    
    # --- SUBSET LOGIC ---
    # We extract a CONTIGUOUS block of 1000 frames to prevent temporal data leakage.
    # The first 800 frames will be the training set, and the strictly subsequent 
    # 200 frames will be the test set, simulating a "future trajectory" test.
    START_INDEX = 15000  # Jump somewhere into the middle of the dataset
    SUBSET_SIZE = 1000 # 33 seconds worth of data
    
    if len(full_dataset) >= START_INDEX + SUBSET_SIZE:
        indices = list(range(START_INDEX, START_INDEX + SUBSET_SIZE))
    else:
        indices = list(range(len(full_dataset)))
        
    # Split strictly sequentially: Train first, Test next
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Note: num_workers=0 is heavily recommended on Windows CPU to prevent multiprocessing hangs
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Created contiguous subset of {len(indices)} images -> {len(train_dataset)} Train | {len(test_dataset)} Test.")
    
    # 4. Training loop setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    best_loss = float('inf')
    save_path = os.path.join(project_dir, 'resnet18_expert_tracker_subset/expert_tracker_subset_best.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        # --- TRAIN PHASE ---
        model.train()
        running_train_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_train_loss / len(train_dataset)
        
        # --- TEST PHASE ---
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)
                
        epoch_test_loss = running_test_loss / len(test_dataset)
        
        print(f"--- Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} ---")
        
        # Save the best model based on TEST loss
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
            
        # Save the latest model at the end of every epoch just in case Colab crashes
        latest_path = os.path.join(project_dir, 'resnet18_expert_tracker_subset/expert_tracker_subset_latest.pth')
        torch.save(model.state_dict(), latest_path)

    print("Training complete!")

if __name__ == '__main__':
    # Required for Windows multiprocessing, though we set num_workers=0 here for safety
    import multiprocessing
    multiprocessing.freeze_support()
    main()
