import csv
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset
from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone


def test_dataset_and_model_shapes(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / "sample.png"
    mask_path = mask_dir / "sample.png"
    Image.new("RGB", (64, 64), color="black").save(image_path)
    Image.new("L", (64, 64), color=255).save(mask_path)

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_file", "ball_x_px", "ball_y_px"])
        writer.writerow(["sample.png", 32, 16])

    dataset = SharedVisionDataset(
        csv_file=str(csv_path),
        root_dir=str(images_dir),
        mask_dir=str(mask_dir),
        input_size=(64, 64),
    )
    image, ball_xy, mask = dataset[0]

    assert image.shape == (3, 64, 64)
    assert ball_xy.shape == (2,)
    assert mask.shape == (1, 64, 64)

    model = SharedVisionBackbone()
    batch = torch.randn(2, 3, 64, 64)
    ball_xy_pred, tracker_features, mask_logits, heatmap_logits = model(batch)

    assert ball_xy_pred.shape == (2, 2)
    assert tracker_features.shape == (2, 64)
    assert mask_logits.shape == (2, 1, 64, 64)
    assert heatmap_logits.shape == (2, 1, 64, 64)
