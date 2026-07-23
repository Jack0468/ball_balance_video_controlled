import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

class TemporalTrainTransform:
    def __init__(self):
        # Photometric transforms can be applied independently to simulate per-frame noise
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.erasing = transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))

    def __call__(self, images):
        # 1. Resize all images
        images = [TF.resize(img, (240, 320)) for img in images]
        
        # 2. Determine ONE random geometric warp for the ENTIRE sequence
        # This prevents the camera from "violently shaking" frame-to-frame, which would confuse the LSTM
        angle = random.uniform(-10, 10)
        translate = (random.randint(int(-0.1 * 320), int(0.1 * 320)), random.randint(int(-0.1 * 240), int(0.1 * 240)))
        scale = random.uniform(0.9, 1.1)
        shear = [0.0, 0.0]
        
        do_perspective = random.random() < 0.5
        if do_perspective:
            startpoints, endpoints = transforms.RandomPerspective.get_params(320, 240, 0.2)
            
        transformed_tensors = []
        for img in images:
            # Apply identical geometric warp
            img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear)
            if do_perspective:
                img = TF.perspective(img, startpoints, endpoints)
                
            # Apply per-frame photometric noise
            img = self.color_jitter(img)
            img = self.blur(img)
            
            # To Tensor
            tensor = self.to_tensor(img)
            tensor = self.erasing(tensor)
            tensor = self.normalize(tensor)
            
            transformed_tensors.append(tensor)
            
        return torch.stack(transformed_tensors) # Shape: (SeqLen, C, H, W)

class TemporalTestTransform:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, images):
        transformed_tensors = []
        for img in images:
            img = TF.resize(img, (240, 320))
            tensor = self.to_tensor(img)
            tensor = self.normalize(tensor)
            transformed_tensors.append(tensor)
        return torch.stack(transformed_tensors)


class SequenceBallDataset(Dataset):
    """
    Custom Dataset for loading continuous sequences of video frames
    to train an LSTM (TemporalTracker).
    """
    def __init__(self, csv_file, root_dir, seq_len=10, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # We need to make sure we don't grab a sequence that crosses a file boundary.
        # But since our data was generated sequentially, we will assume continuous flow.
        # A more robust implementation would check if host_timestamp_ms jumps by more than ~100ms.
        # For this implementation, we just return slices of size `seq_len`.
        
    def __len__(self):
        return len(self.labels_df) - self.seq_len + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = []
        targets = []
        
        for i in range(self.seq_len):
            row = self.labels_df.iloc[idx + i]
            img_name = os.path.join(self.root_dir, row['image_file'])
            
            try:
                image = Image.open(img_name).convert('RGB')
            except FileNotFoundError:
                raise FileNotFoundError(f"Image {img_name} not found.")
                
            images.append(image)
            
            touch_x = row['touch_x']
            touch_y = row['touch_y']
            MAX_BOUND = 200.0
            targets.append(torch.tensor([touch_x / MAX_BOUND, touch_y / MAX_BOUND], dtype=torch.float32))

        # Apply transformations
        if self.transform:
            image_sequence = self.transform(images) # returns (SeqLen, C, H, W) tensor
        else:
            raise ValueError("A TemporalTransform must be provided to SequenceBallDataset")
            
        target_sequence = torch.stack(targets) # (SeqLen, 2)

        return image_sequence, target_sequence
