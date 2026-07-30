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

        # Original image dimensions vary depending on the dynamic ArUco crop!
        # We must normalize relative to the *actual* crop dimensions before it gets resized to 320x240.
        # The BasicCNN Spatial Softmax outputs in the range [-1.0, 1.0].
        ball_x = self.labels_df.iloc[idx]['ball_x']
        ball_y = self.labels_df.iloc[idx]['ball_y']
        
        orig_w, orig_h = image.size
        
        # Normalize to [-1.0, 1.0]
        norm_x = (ball_x / (orig_w / 2.0)) - 1.0
        norm_y = (ball_y / (orig_h / 2.0)) - 1.0
        
        # Account for when the ball has fallen off the platform (or is out of bounds)
        # by clamping the coordinates strictly to the [-1.0, 1.0] range.
        # If we don't clamp, the SpatialSoftmax loss will cause the model to collapse to the center.
        norm_x = max(-1.0, min(1.0, norm_x))
        norm_y = max(-1.0, min(1.0, norm_y))
        
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
