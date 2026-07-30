import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class UnifiedDataset(Dataset):
    """
    Dataset for the Unified CNN Tracker (33 outputs)
    Outputs: [kpts (8), ball (2), markers (22), ball_present (1)]
    All coordinates should be normalized between [0, 1] for training stability.
    """
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        
        csv_path = os.path.join(data_dir, 'yolo_features.csv')
        df = pd.read_csv(csv_path)
        
        # Filter by split
        self.labels_df = df[df['split'] == split].reset_index(drop=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        img_path = os.path.join(self.images_dir, row['image_file'])
        try:
            image = Image.open(img_path).convert("RGB")
            # Get original image dimensions before resize
            orig_w, orig_h = image.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and target if there's a file error
            return torch.zeros((3, 240, 320)), torch.zeros(33)

        image = self.transform(image)
        
        # Build 33-element target tensor
        target = np.zeros(33, dtype=np.float32)
        
        # 0-7: Platform Keypoints (normalized)
        target[0] = row['kpt0_x'] / orig_w
        target[1] = row['kpt0_y'] / orig_h
        target[2] = row['kpt1_x'] / orig_w
        target[3] = row['kpt1_y'] / orig_h
        target[4] = row['kpt2_x'] / orig_w
        target[5] = row['kpt2_y'] / orig_h
        target[6] = row['kpt3_x'] / orig_w
        target[7] = row['kpt3_y'] / orig_h
        
        # 8-9: Ball Center (normalized)
        if row['ball_present'] == 1.0:
            target[8] = row['ball_x'] / orig_w
            target[9] = row['ball_y'] / orig_h
        else:
            target[8] = -1.0
            target[9] = -1.0
            
        # 10-31: Markers 2 to 12
        idx_offset = 10
        for c in range(2, 13):
            mx = row[f'marker{c}_x']
            my = row[f'marker{c}_y']
            if mx >= 0 and my >= 0:
                target[idx_offset] = mx / orig_w
                target[idx_offset + 1] = my / orig_h
            else:
                target[idx_offset] = -1.0
                target[idx_offset + 1] = -1.0
            idx_offset += 2
            
        # 32: Ball Presence Logit
        target[32] = row['ball_present']
        
        return image, torch.tensor(target, dtype=torch.float32)
