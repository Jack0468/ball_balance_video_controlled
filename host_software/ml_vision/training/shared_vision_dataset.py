import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SharedVisionDataset(Dataset):
    """Load paired image, ball coordinate, and marker mask targets for the shared backbone training task."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        mask_dir: Optional[str] = None,
        input_size: Tuple[int, int] = (128, 128),
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mask_dir = mask_dir or os.path.join(root_dir, "masks")
        self.input_size = input_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels_df)

    def _make_gaussian_heatmap(self, cx: float, cy: float, sigma: float = 5.0) -> torch.Tensor:
        """Generate a single-channel Gaussian heatmap with peak at (cx, cy) in pixel coords."""
        h, w = self.input_size
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        heatmap = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))
        return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.labels_df.iloc[idx]
        image_path = os.path.join(self.root_dir, row["image_file"])
        mask_path = os.path.join(self.mask_dir, row["image_file"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize mask to target resolution (nearest-neighbour preserves binary values)
        mask = mask.resize((self.input_size[1], self.input_size[0]), Image.NEAREST)

        # Apply the full transform pipeline (Resize + ToTensor handled by transform)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.Compose([
                transforms.Resize((self.input_size[0], self.input_size[1])),
                transforms.ToTensor(),
            ])(image)

        mask_tensor = transforms.ToTensor()(mask)  # → (1, H, W), values in [0,1]
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[0:1, :, :]  # ensure single channel

        # Ball pixel coordinates in the *original* warped frame, mapped to CNN input resolution
        # input_size is (H, W), ball_x is horizontal → divide by W (input_size[1])
        # ball_y is vertical → divide by H (input_size[0])
        ball_x_orig = float(row.get("ball_x_px", 0.0))
        ball_y_orig = float(row.get("ball_y_px", 0.0))

        # Scale pixel coords from original warp resolution to CNN input resolution
        orig_w = float(row.get("warp_w", self.input_size[1]))
        orig_h = float(row.get("warp_h", self.input_size[0]))
        ball_x_scaled = ball_x_orig * (self.input_size[1] / orig_w)
        ball_y_scaled = ball_y_orig * (self.input_size[0] / orig_h)

        # Normalise to [0, 1] — x divided by W, y divided by H
        ball_xy = torch.tensor(
            [ball_x_scaled / self.input_size[1], ball_y_scaled / self.input_size[0]],
            dtype=torch.float32,
        )

        # Generate Gaussian heatmap target for the marker heatmap head
        # Uses the mask centroid as the heatmap centre if available
        # (falls back to image centre if mask is empty)
        mask_np = np.array(mask_tensor[0].numpy())
        ys_nonzero, xs_nonzero = np.where(mask_np > 0.5)
        if len(xs_nonzero) > 0:
            heatmap_cx = float(xs_nonzero.mean())
            heatmap_cy = float(ys_nonzero.mean())
        else:
            heatmap_cx = self.input_size[1] / 2.0
            heatmap_cy = self.input_size[0] / 2.0
        heatmap_target = self._make_gaussian_heatmap(heatmap_cx, heatmap_cy)

        return image_tensor, ball_xy, mask_tensor, heatmap_target
