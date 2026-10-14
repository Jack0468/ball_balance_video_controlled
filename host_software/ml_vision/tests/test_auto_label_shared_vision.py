import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "data_processing" / "auto_label_shared_vision.py"
)
SPEC = importlib.util.spec_from_file_location("auto_label_shared_vision", MODULE_PATH)
AUTO_LABEL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUTO_LABEL)


def _real_schema_manifest() -> dict:
    """Matches load_manifest_full()'s actual required schema -- aruco_markers +
    features + platform_width_mm/platform_height_mm -- NOT the old "markers"-key
    shape this test used to use (which no real manifest, and no validation in
    load_manifest_full(), has ever accepted)."""
    return {
        "platform_width_mm": 100.0,
        "platform_height_mm": 100.0,
        "aruco_markers": [
            {"id": 0, "name": "corner_tl", "role": "corner", "center_mm": [0.0, 0.0], "size_mm": 15.0},
            {"id": 1, "name": "corner_tr", "role": "corner", "center_mm": [100.0, 0.0], "size_mm": 15.0},
        ],
        "features": [
            {
                "id": 0,
                "name": "demo_circle",
                "shape": "circle",
                "center_mm": [10.0, 10.0],
                "size_mm": 6.0,
                "color": "blue",
            },
            {
                "id": 1,
                "name": "demo_triangle",
                "shape": "triangle",
                "center_mm": [50.0, 50.0],
                "size_mm": 8.0,
                "color": "red",
            },
        ],
    }


def test_load_manifest_full_parses_real_schema(tmp_path):
    """load_manifest_full() is the real loader (there is no load_manifest()) --
    confirms it returns the documented 4-tuple and that each features/aruco_markers
    entry survives the round trip, so a future schema regression here (e.g. a key
    renamed without updating every call site) is caught by a test, not discovered
    the way the point-reflection and mask-undersizing bugs were (see
    docs/PROJECT_LOGBOOK.md, 2026-08-12)."""
    manifest = _real_schema_manifest()
    manifest_path = tmp_path / "ground_truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    aruco_markers, features, platform_w_mm, platform_h_mm = AUTO_LABEL.load_manifest_full(manifest_path)

    assert platform_w_mm == 100.0
    assert platform_h_mm == 100.0
    assert len(aruco_markers) == 2
    assert {m["id"] for m in aruco_markers} == {0, 1}
    assert len(features) == 2
    assert features[0]["shape"] == "circle"
    assert features[1]["shape"] == "triangle"
    assert features[1]["color"] == "red"


def test_load_manifest_full_rejects_missing_aruco_markers(tmp_path):
    """load_manifest_full() explicitly raises ValueError when 'aruco_markers' is
    missing/empty (auto_label_shared_vision.py's own validation) -- pins down that
    a manifest using the wrong/old schema fails loudly instead of silently
    producing an empty homography basis."""
    manifest = {"platform_width_mm": 100.0, "platform_height_mm": 100.0, "features": []}
    manifest_path = tmp_path / "bad_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="aruco_markers"):
        AUTO_LABEL.load_manifest_full(manifest_path)


def test_project_points_applies_nontrivial_homography():
    """Uses a real (non-identity) similarity homography -- a plain identity
    matrix (the old version of this test) would pass even if project_points()
    ignored its `homography` argument entirely."""
    # Scale mm by 2x and translate by (5, 5): [x, y] -> [2x + 5, 2y + 5].
    homography = np.array(
        [[2.0, 0.0, 5.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    projected = AUTO_LABEL.project_points(
        np.array([[10.0, 20.0], [0.0, 0.0]], dtype=np.float32),
        homography,
    )
    assert np.allclose(projected[0], [25.0, 45.0])
    assert np.allclose(projected[1], [5.0, 5.0])

    # Empty input must return an empty (0, 2) array, not raise.
    empty = AUTO_LABEL.project_points(np.empty((0, 2), dtype=np.float32), homography)
    assert empty.shape == (0, 2)


@pytest.mark.parametrize("shape,min_area_fraction_of_circle", [
    ("circle", 1.0),
    ("square", 1.2),  # a square circumscribing the same radius has a visibly larger area than the circle
    ("triangle", 0.5),  # the real equilateral-triangle geometry is visibly smaller than the circle
    ("hexagon", 0.7),
])
def test_render_marker_mask_uses_real_polygon_geometry_per_shape(shape, min_area_fraction_of_circle):
    """render_marker_mask() must render the marker's ACTUAL shape (circle / square
    / triangle / hexagon via real polygon vertices), not silently fall back to a
    circle for every non-triangle shape -- which was the real, previously-shipped
    bug this file is meant to catch (docs/PROJECT_LOGBOOK.md, 2026-08-12, "marker
    mask undersizing" / shape fallback). Verified by checking each shape's real
    rendered area against the reference circle's area at the same size_mm, rather
    than just asserting `mask.sum() > 0` (which a wrong-shape circle fallback would
    also satisfy)."""
    size_mm = 8.0
    marker = {"shape": shape, "size_mm": size_mm, "color": "blue"}
    circle_marker = {"shape": "circle", "size_mm": size_mm, "color": "blue"}

    out_shape = (128, 128)
    center = (64.0, 64.0)

    mask = AUTO_LABEL.render_marker_mask(out_shape, center, marker)
    circle_mask = AUTO_LABEL.render_marker_mask(out_shape, center, circle_marker)

    assert mask.sum() > 0, f"{shape} mask rendered empty"
    area = int(np.count_nonzero(mask))
    circle_area = int(np.count_nonzero(circle_mask))
    assert circle_area > 0

    # Sanity bound: real shape geometry must be in a plausible range relative to
    # the reference circle of the same size_mm, not just "nonzero" -- a buggy
    # circle-fallback for square/triangle/hexagon would make `area` exactly equal
    # `circle_area`, which none of the non-circle assertions below allow.
    if shape == "circle":
        assert area == circle_area
    else:
        assert area != circle_area
        assert area >= min_area_fraction_of_circle * circle_area * 0.5  # generous floor, not a tight fit
        assert area <= circle_area * 2.5  # generous ceiling -- catches a gross size regression


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
