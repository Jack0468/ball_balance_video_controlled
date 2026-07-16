import os
import torch
from temporal_ball_dataset import SequenceBallDataset, TemporalTrainTransform
from resnet18_lstm import TemporalExpertTracker
from torch.utils.data import DataLoader

def verify():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '../data/02_silver'))
    csv_path = os.path.join(data_dir, 'labels_sequential.csv')
    images_dir = os.path.join(data_dir, 'images')
    
    print("1. Instantiating Dataset...")
    transform = TemporalTrainTransform()
    dataset = SequenceBallDataset(csv_file=csv_path, root_dir=images_dir, seq_len=10, transform=transform)
    
    print(f"Dataset length: {len(dataset)}")
    
    print("2. Instantiating DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print("3. Fetching one batch...")
    images, targets = next(iter(loader))
    print(f"Images shape: {images.shape} (Expected: 4, 10, 3, 240, 320)")
    print(f"Targets shape: {targets.shape} (Expected: 4, 10, 2)")
    
    print("4. Instantiating Model...")
    model = TemporalExpertTracker(hidden_size=256, num_layers=1)
    
    print("5. Forward Pass...")
    out = model(images)
    print(f"Output shape: {out.shape} (Expected: 4, 2)")
    
    print("Verification Successful!")

if __name__ == '__main__':
    verify()
