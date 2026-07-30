import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BallPixelDataset(Dataset):
    """
    Dataset for loading unwarped camera images and targeting the raw PIXEL coordinates
    of the ball (ball_x, ball_y), instead of the physical touch_x, touch_y.
    Used for the hybrid ArUco + CNN architecture where the CNN only tracks the ball in 2D.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # We only want to train on images where the ball is actually present
        if 'ball_present' in self.labels_df.columns:
            self.labels_df = self.labels_df[self.labels_df['ball_present'] == 1.0]
            
        # Drop any frames where the target label is NaN (e.g. from mixed datasets)
        if 'touch_x' in self.labels_df.columns:
            self.labels_df = self.labels_df.dropna(subset=['touch_x', 'touch_y']).reset_index(drop=True)
        else:
            self.labels_df = self.labels_df.dropna(subset=['ball_x', 'ball_y']).reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]['image_file'])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {img_name} not found. Check if the silver dataset is fully extracted.")

        # Original image dimensions from the webcam are usually 640x480
        # The BasicCNN Spatial Softmax outputs in the range [-1.0, 1.0].
        if 'touch_x' in self.labels_df.columns:
            # Train directly on physical touch pad telemetry!
            # Platform is 187.5mm x 142.0mm
            touch_x = self.labels_df.iloc[idx]['touch_x']
            touch_y = self.labels_df.iloc[idx]['touch_y']
            
            # Map [0, 187.5] -> [-1, 1]
            norm_x = (touch_x / (187.5 / 2.0)) - 1.0
            norm_y = (touch_y / (142.0 / 2.0)) - 1.0
        else:
            # Train on raw image pixels (old method)
            ball_x = self.labels_df.iloc[idx]['ball_x']
            ball_y = self.labels_df.iloc[idx]['ball_y']
            
            norm_x = (ball_x / 320.0) - 1.0
            norm_y = (ball_y / 240.0) - 1.0
        
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
