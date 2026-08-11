import os
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from host_software.ml_vision.training.augmentations import build_eval_transform


class SharedVisionDataset(Dataset):
    """Load paired image, ball coordinate, and marker mask targets for the shared backbone training task.

    Applies its transform jointly to (image, mask, ball keypoint) via albumentations,
    so geometric augmentation (translation/rotation/perspective) can't desync the mask
    or ball-position label from what the augmented image actually shows.
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        mask_dir: Optional[str] = None,
        input_size: Tuple[int, int] = (128, 128),
        transform: Optional[A.Compose] = None,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.labels_df = labels_df.reset_index(drop=True) if labels_df is not None else pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mask_dir = mask_dir or os.path.join(root_dir, "masks")
        self.input_size = input_size
        self.transform = transform or build_eval_transform(input_size)

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

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        ball_x = float(row.get("ball_x_px", 0.0))
        ball_y = float(row.get("ball_y_px", 0.0))

        # A.Resize (always the first step in the transform) rescales the keypoint along
        # with the image, so ball_x/ball_y just need to be in the *source* image's pixel
        # space -- no separate warp_w/warp_h rescaling step needed.
        transformed = self.transform(image=image, mask=mask, keypoints=[(ball_x, ball_y)])
        t_image = transformed["image"]
        t_mask = transformed["mask"]
        kp_x, kp_y = transformed["keypoints"][0]

        h, w = self.input_size
        image_tensor = torch.from_numpy(t_image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(t_mask).float().unsqueeze(0) / 255.0

        # Normalise to [0, 1] -- x divided by W, y divided by H
        ball_xy = torch.tensor([kp_x / w, kp_y / h], dtype=torch.float32)

        # Generate Gaussian heatmap target for the marker heatmap head, centred on the
        # (post-augmentation) mask centroid -- falls back to image centre if mask is empty.
        mask_np = mask_tensor[0].numpy()
        ys_nonzero, xs_nonzero = np.where(mask_np > 0.5)
        if len(xs_nonzero) > 0:
            heatmap_cx = float(xs_nonzero.mean())
            heatmap_cy = float(ys_nonzero.mean())
        else:
            heatmap_cx = w / 2.0
            heatmap_cy = h / 2.0
        heatmap_target = self._make_gaussian_heatmap(heatmap_cx, heatmap_cy)

        return image_tensor, ball_xy, mask_tensor, heatmap_target
