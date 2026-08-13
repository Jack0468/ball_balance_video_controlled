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

    def _make_gaussian_heatmap(self, cx: float, cy: float, sigma: float = 5.0) -> np.ndarray:
        """Single Gaussian peak at (cx, cy) in pixel coords, as a plain (H, W) array
        so callers can np.maximum-combine several peaks before converting to a tensor once."""
        h, w = self.input_size
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        heatmap = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))
        return heatmap.astype(np.float32)

    def _make_multi_peak_heatmap(self, mask_np: np.ndarray, sigma: float = 5.0) -> torch.Tensor:
        """One Gaussian peak per individual marker (connected component of the
        combined mask), max-combined -- NOT a single peak at the mean position of
        every marker's pixels merged together. The old single-centroid approach
        degenerated on real data: sheets place markers in a symmetric quincunx
        layout, so the mean of all visible markers' pixels lands at ~the same fixed
        point (image centre) on virtually every frame regardless of which markers
        are present -- verified empirically (std < 0.15px across sampled frames from
        all 3 marker sheets). See docs/PROJECT_LOGBOOK.md, 2026-08-12 ("heatmap
        target degeneracy"). Frames with no markers get an all-zero target --
        correctly representing "no marker anywhere" rather than a fake peak at centre.
        """
        h, w = self.input_size
        mask_uint8 = (mask_np > 0.5).astype(np.uint8)
        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        if num_labels <= 1:  # label 0 is background only -- no foreground components
            return torch.zeros((1, h, w), dtype=torch.float32)

        combined = np.zeros((h, w), dtype=np.float32)
        for label in range(1, num_labels):
            # cv2.connectedComponentsWithStats always returns centroids as float64 --
            # cast to plain Python float (not np.float32(), which is still a numpy
            # scalar) so numpy's type promotion doesn't silently upcast the whole
            # Gaussian computation to float64 when subtracted from the float32 grid.
            # That upcast previously produced a torch.float64 heatmap_target, which
            # trained fine on CPU (autocast disabled there) but crashed in backward()
            # on GPU, where autocast/GradScaler enforce float32 -- see
            # docs/PROJECT_LOGBOOK.md, 2026-08-13.
            cx, cy = float(centroids[label][0]), float(centroids[label][1])
            combined = np.maximum(combined, self._make_gaussian_heatmap(cx, cy, sigma))

        return torch.from_numpy(combined.astype(np.float32)).unsqueeze(0)

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

        # Multi-peak heatmap target: one Gaussian per marker, not one Gaussian at the
        # mean of every marker's pixels combined -- see _make_multi_peak_heatmap().
        mask_np = mask_tensor[0].numpy()
        heatmap_target = self._make_multi_peak_heatmap(mask_np)

        return image_tensor, ball_xy, mask_tensor, heatmap_target
