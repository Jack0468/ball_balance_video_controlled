import csv
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset
from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone


def _write_sample(images_dir: Path, mask_dir: Path, name: str, mask_fill: int) -> None:
    Image.new("RGB", (64, 64), color="black").save(images_dir / name)
    Image.new("L", (64, 64), color=mask_fill).save(mask_dir / name)


def test_dataset_item_matches_real_4tuple_contract(tmp_path):
    """SharedVisionDataset.__getitem__ returns a 4-tuple
    (image, ball_xy, mask, heatmap_target) -- NOT the 3-tuple this test used to
    assert (image, ball_xy, mask), which predates the heatmap head
    (shared_vision_dataset.py's _make_multi_peak_heatmap(), added per
    docs/PROJECT_LOGBOOK.md 2026-08-12 "heatmap target degeneracy"). Also checks
    the heatmap target's actual semantics -- a per-marker Gaussian peak at the
    mask's connected-component centroid, not an all-zero or degenerate target --
    which is exactly the bug class that motivated the heatmap head's redesign in
    the first place."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # A fully-lit (255) mask gives _make_multi_peak_heatmap() one real connected
    # component spanning the whole frame -- enough to confirm the heatmap target
    # is non-degenerate (has real signal, isn't all-zero) without needing to draw
    # a specific blob shape.
    _write_sample(images_dir, mask_dir, "sample.png", mask_fill=255)

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
    assert len(dataset) == 1

    image, ball_xy, mask, heatmap_target = dataset[0]

    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert ball_xy.shape == (2,)
    # ball_xy is normalised to [0, 1] (x/W, y/H) -- see shared_vision_dataset.py:108.
    assert torch.all((ball_xy >= 0.0) & (ball_xy <= 1.0))
    assert mask.shape == (1, 64, 64)
    assert heatmap_target.shape == (1, 64, 64)
    # Non-degenerate: a real mask must produce a real (non-all-zero) heatmap peak.
    assert heatmap_target.max() > 0.5


def test_dataset_item_no_markers_gives_allzero_heatmap(tmp_path):
    """A frame with no visible marker (all-black mask) must produce an all-zero
    heatmap target -- correctly representing "no marker anywhere" -- rather than
    a fake peak at the image centre, which was the exact degenerate-target bug
    _make_multi_peak_heatmap() was written to fix (see its own docstring and
    docs/PROJECT_LOGBOOK.md, 2026-08-12, "heatmap target degeneracy")."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    _write_sample(images_dir, mask_dir, "sample.png", mask_fill=0)

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
    _, _, mask, heatmap_target = dataset[0]

    assert float(mask.sum()) == 0.0
    assert float(heatmap_target.sum()) == 0.0


def test_model_forward_matches_real_3tuple_contract():
    """SharedVisionBackbone.forward() returns a 3-tuple
    (ball_xy, mask_logits, heatmap_logits) -- NOT the 4-tuple with a
    `tracker_features` output this test used to assert. `feature_head` was
    deliberately removed (train_cnn_2d_tracker_marker.py:50, "it had no semantic
    target and wasted ~5K params") -- there is no 4th output any more."""
    model = SharedVisionBackbone(input_size=(64, 64))
    model.eval()
    batch = torch.randn(2, 3, 64, 64)

    output = model(batch)
    assert len(output) == 3
    ball_xy_pred, mask_logits, heatmap_logits = output

    assert ball_xy_pred.shape == (2, 2)
    assert mask_logits.shape == (2, 1, 64, 64)
    assert heatmap_logits.shape == (2, 1, 64, 64)
    # Heads produce raw logits, not probabilities -- downstream callers
    # (evaluate_shared_vision_backbone.py, marker_classifier.py) apply sigmoid
    # themselves. Checked structurally (the last layer of each decoder head is a
    # plain nn.Conv2d, not a Sigmoid) rather than by testing whether the *values*
    # happen to fall outside [0, 1]: this tiny randomly-initialized conv net's
    # pre-sigmoid activations are tightly clustered near 0 by construction
    # (BatchNorm + small random weights), so a value-range assertion was flaky --
    # it failed/passed depending on unrelated global torch RNG state from
    # whichever other test module happened to run earlier in the same session.
    assert isinstance(model.mask_head[-1], torch.nn.Conv2d)
    assert isinstance(model.heatmap_head[-1], torch.nn.Conv2d)


def test_dataset_mask_read_survives_ultralytics_cv2_imread_monkeypatch(tmp_path, monkeypatch):
    """Regression test for a real, newly-discovered bug: `ultralytics` (imported
    anywhere earlier in the SAME process -- e.g. a notebook/script that also
    trains or evaluates a YOLO model, which the still-open Arm 1 small-vs-medium
    comparison routinely does side by side with this exact dataset class)
    globally monkeypatches cv2.imread at import time
    (ultralytics/utils/__init__.py: "cv2.imread, cv2.imwrite, cv2.imshow =
    imread, imwrite, imshow") to a replacement that unconditionally appends a
    trailing channel axis to every 2D (grayscale) result
    (ultralytics/utils/patches.py's imread(), "Always ensure 3 dimensions").

    Confirmed directly while fixing this test file: a plain
    cv2.imread(path, cv2.IMREAD_GRAYSCALE) returns (H, W) before
    `from ultralytics import YOLO` runs anywhere in the process, and (H, W, 1)
    after -- silently, for the rest of that process's life. Reproduced here
    without actually importing the real (slow) ultralytics package, by
    monkeypatching cv2.imread the same way it does, to pin down
    SharedVisionDataset's own defensive squeeze
    (`if mask.ndim == 3: mask = mask[..., 0]`, shared_vision_dataset.py) rather
    than relying on it being exercised only by accident of import order."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    _write_sample(images_dir, mask_dir, "sample.png", mask_fill=255)

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_file", "ball_x_px", "ball_y_px"])
        writer.writerow(["sample.png", 32, 16])

    real_imread = cv2.imread

    def _ultralytics_style_imread(filename, flags=cv2.IMREAD_COLOR):
        im = real_imread(filename, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im

    monkeypatch.setattr(cv2, "imread", _ultralytics_style_imread)

    dataset = SharedVisionDataset(
        csv_file=str(csv_path),
        root_dir=str(images_dir),
        mask_dir=str(mask_dir),
        input_size=(64, 64),
    )
    _, _, mask, heatmap_target = dataset[0]

    assert mask.shape == (1, 64, 64)
    assert heatmap_target.shape == (1, 64, 64)
