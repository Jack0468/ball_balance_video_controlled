import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class VLADataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform

        # Simple vocabulary for audio commands
        self.vocab = {
            "hold": 0,
            "go red": 1,
            "go blue": 2,
            "go green": 3,
            "go yellow": 4,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image (Vision Token)
        img_path = item["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            # Fallback for missing images in a mock dataset
            img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        # Audio Token (Word index)
        cmd_idx = self.vocab.get(item.get("audio_command", "hold"), 0)

        # State Token
        state = torch.tensor(
            [item.get("state_x", 0), item.get("state_y", 0)], dtype=torch.float32
        )

        # Action Targets (Expert PID output to imitate)
        action = torch.tensor(
            [
                item.get("action_theta_a", 0),
                item.get("action_theta_b", 0),
                item.get("action_theta_c", 0),
            ],
            dtype=torch.float32,
        )

        return img, cmd_idx, state, action
