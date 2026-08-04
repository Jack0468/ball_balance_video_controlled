import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "data_processing" / "auto_label_shared_vision.py"
)
SPEC = importlib.util.spec_from_file_location("auto_label_shared_vision", MODULE_PATH)
AUTO_LABEL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUTO_LABEL)


def test_manifest_projection_and_mask_render(tmp_path):
    manifest = {
        "platform_width_mm": 100.0,
        "platform_height_mm": 100.0,
        "markers": [
            {
                "id": 0,
                "name": "demo_circle",
                "shape": "circle",
                "center_mm": [10.0, 10.0],
                "size_mm": 6.0,
            }
        ],
    }
    manifest_path = tmp_path / "ground_truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest = AUTO_LABEL.load_manifest(manifest_path)
    assert loaded_manifest[0]["id"] == 0

    homography = np.eye(3, dtype=np.float32)
    projected = AUTO_LABEL.project_points(
        np.array([[10.0, 10.0]], dtype=np.float32),
        homography,
    )
    assert np.allclose(projected[0], [10.0, 10.0])

    mask = AUTO_LABEL.render_marker_mask((64, 64), (10, 10), loaded_manifest[0])
    assert mask.sum() > 0


def test_manifest_resolution_prefers_sheet_specific_manifest(tmp_path):
    template_dir = tmp_path / "hardware" / "platform_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = template_dir / "aruco_markers_02_manifest.json"
    manifest_path.write_text(json.dumps({"markers": []}), encoding="utf-8")

    resolved = AUTO_LABEL.resolve_manifest_path(
        Path("host_software/data/02_silver/aruco_markers_02"),
        template_dir=template_dir,
    )

    assert resolved == manifest_path
