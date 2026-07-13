import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class BallDataset(Dataset):
    """
    Custom Dataset for loading pre-processed ball balance images
    and their corresponding touch_x, touch_y coordinates.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        if transform:
            self.transform = transform
        else:
            # ResNet/MobileNet transforms using 4:3 aspect ratio (320x240)
            self.transform = transforms.Compose([
                transforms.Resize((240, 320)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]['image_file'])
        
        # Some rows might reference images that don't exist if data is partially synced,
        # but we assume the data pipeline guarantees existence or handles it.
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # If an image is missing, we could throw an error or handle it. 
            # Throwing error is safer to catch data issues early.
            raise FileNotFoundError(f"Image {img_name} not found. Check if the silver dataset is fully extracted.")

        # Extract touch_x and touch_y
        touch_x = self.labels_df.iloc[idx]['touch_x']
        touch_y = self.labels_df.iloc[idx]['touch_y']
        
        # Normalize coordinates to approximately [-1, 1] assuming a max 200mm bound
        MAX_BOUND = 200.0
        target = torch.tensor([touch_x / MAX_BOUND, touch_y / MAX_BOUND], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
