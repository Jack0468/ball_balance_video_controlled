import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


TOUCHPAD_W = 187.5
TOUCHPAD_H = 142.0

# Extra margin (mm) added outward beyond the true paper edge when building the
# warp-destination corners. Corner ArUco markers sit only ~0.75mm inside the
# paper edge on the current templates, so a zero margin leaves no tolerance
# for homography/detection jitter -- a few mm of margin (pulled from whatever
# background surrounds the printed sheet) keeps markers fully in frame.
PAPER_MARGIN_MM = 6.0


def load_manifest_full(manifest_path: Path) -> Tuple[List[Dict], List[Dict], float, float]:
    """Load a platform manifest and return (aruco_markers, features, platform_width_mm, platform_height_mm).

    Schema (all manifests must follow this):
      aruco_markers    — ArUco fiducial markers used for homography computation.
                         Each entry: {id, name, role, center_mm, size_mm}
      features         — Colored target markers that the CNN must detect.
                         Each entry: {id, name, shape, center_mm, size_mm, color}
      platform_width_mm, platform_height_mm — true physical paper/platform
                         boundary, used as the warp-destination corners.

    Returns
    -------
    aruco_markers      : list of dicts, one per ArUco fiducial (IDs 0-5)
    features           : list of dicts, one per colored target marker
    platform_width_mm  : physical width of the paper/platform
    platform_height_mm : physical height of the paper/platform
    """
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    aruco_markers = payload.get("aruco_markers", [])
    features = payload.get("features", [])

    if not aruco_markers:
        raise ValueError(
            f"Manifest {manifest_path} has no 'aruco_markers' key.\n"
            "Expected schema: {{\"aruco_markers\": [...], \"features\": [...]}}"
        )

    if "platform_width_mm" not in payload or "platform_height_mm" not in payload:
        raise ValueError(
            f"Manifest {manifest_path} is missing 'platform_width_mm'/'platform_height_mm'."
        )

    return aruco_markers, features, float(payload["platform_width_mm"]), float(payload["platform_height_mm"])


def build_aruco_lookup(aruco_markers: List[Dict]) -> Dict[int, List[float]]:
    """Build an {id: [x_mm, y_mm]} lookup from the manifest's aruco_markers list."""
    return {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}


def build_paper_corners(
    platform_width_mm: float,
    platform_height_mm: float,
    margin_mm: float = PAPER_MARGIN_MM,
) -> np.ndarray:
    """Return the 4 warp-destination corners (TL, TR, BR, BL) in mm.

    Based on the true physical paper edge, not the ArUco marker positions --
    using marker centers as the warp-destination corners clips roughly half
    of every corner marker out of the warped image, since corner markers are
    inset from the edge (e.g. 12mm on the current templates). Matches the
    PLATFORM_CORNERS_MM convention used by generate_aruco_cropped_dataset.py
    and extract_cnn_sequential_features.py.

    `margin_mm` expands the corners outward beyond the true paper edge, so
    small homography/detection errors don't clip markers that sit close to
    the edge. The extra margin is pulled from whatever background surrounds
    the printed sheet in the raw camera frame.
    """
    return np.array(
        [
            [-margin_mm, -margin_mm],
            [platform_width_mm + margin_mm, -margin_mm],
            [platform_width_mm + margin_mm, platform_height_mm + margin_mm],
            [-margin_mm, platform_height_mm + margin_mm],
        ],
        dtype=np.float32,
    )


def resolve_manifest_path(
    session_dir: Optional[Path],
    manifest_hint: Optional[Path] = None,
    template_dir: Optional[Path] = None,
) -> Path:
    base_dir = template_dir or Path("hardware/platform_templates")

    if manifest_hint is not None:
        return manifest_hint

    session_name = (session_dir.name if session_dir is not None else "").lower()
    candidate_names = []

    if session_name:
        candidate_names.append(f"{session_name}_manifest.json")
        if session_name.startswith("aruco_markers_"):
            candidate_names.append(f"{session_name}.json")
            candidate_names.append("ground_truth_manifest.json")

    if not candidate_names:
        candidate_names.append("ground_truth_manifest.json")

    for candidate_name in candidate_names:
        candidate_path = base_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find a manifest for {session_dir or 'the current session'} in {base_dir}"
    )


def project_points(points_mm: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if points_mm.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array([points_mm], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, homography)[0]
    return projected.astype(np.float32)


def render_marker_mask(shape: Tuple[int, int], center_xy: Tuple[float, float], marker: Dict) -> np.ndarray:
    """Render a marker's mask at its true printed size and shape.

    size_mm is the shape's RADIUS, not diameter -- confirmed against the TikZ
    source (e.g. aruco_markers_03.tex's `circle (8mm)`, and its "~8mm effective
    radius" comment shared by every feature on that sheet). radius_px converts
    it via the platform's actual mm-to-pixel scale (the warp always maps the
    fixed mm rectangle [-margin, W+margin] x [-margin, H+margin] onto the full
    output frame, so this is an exact scale, not an approximation) -- NOT the
    previous `size_mm * 0.35` constant, which covered only ~34% of the marker's
    true visible area (verified against real photos -- see docs/PROJECT_LOGBOOK.md,
    2026-08-12, "marker mask undersizing").

    square and hexagon now get real polygon geometry, matching the vertex
    convention in generate_synthetic_marker_composites.py's polygon_vertices()
    -- previously fell back to a circle (same as anything other than "triangle"),
    so real-sheet masks now match printed marker shape too, not just corrected size.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    size_mm = float(marker.get("size_mm", 6.0))
    cx, cy = center_xy
    px_per_mm = ((w / (TOUCHPAD_W + 2 * PAPER_MARGIN_MM)) + (h / (TOUCHPAD_H + 2 * PAPER_MARGIN_MM))) / 2.0
    radius_px = max(2, int(round(size_mm * px_per_mm)))

    marker_shape = marker.get("shape", "circle")
    if marker_shape == "triangle":
        # Equilateral, matching aruco_markers_03.tex's actual vertices (apex offset
        # 9.24mm, base offset 4.62mm, half-base-width 8mm for size_mm=8.0) -- NOT a
        # symmetric apex/base offset. radius_px matches the half-base-width exactly;
        # apex/base offsets follow the standard equilateral-triangle centroid split
        # (apex is 2x as far from the centroid as the base is, i.e. 2/sqrt(3) and
        # 1/sqrt(3) of the half-base-width respectively). The old symmetric formula
        # put the base ~73% too far from centre, visibly oversizing the mask (see
        # docs/PROJECT_LOGBOOK.md, 2026-08-12, "triangle mask still oversized").
        apex_offset = radius_px * 2.0 / np.sqrt(3.0)
        base_offset = radius_px * 1.0 / np.sqrt(3.0)
        points = np.array(
            [
                [cx, cy - apex_offset],
                [cx + radius_px, cy + base_offset],
                [cx - radius_px, cy + base_offset],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
    elif marker_shape == "square":
        points = np.array(
            [
                [cx - radius_px, cy - radius_px],
                [cx + radius_px, cy - radius_px],
                [cx + radius_px, cy + radius_px],
                [cx - radius_px, cy + radius_px],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
    elif marker_shape == "hexagon":
        angles = np.deg2rad(np.arange(6) * 60.0)
        points = np.stack(
            [cx + radius_px * np.cos(angles), cy + radius_px * np.sin(angles)], axis=1
        ).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), radius_px, 255, -1)

    return mask


INTERMEDIATE_WARP_SIZE = 500  # Warp to this resolution first to preserve sub-pixel quality
WARP_BORDER_PAD = 20  # Pixel padding around frame edges before warping (matches legacy CROP_PAD=20)


def warp_to_platform(
    frame_bgr: np.ndarray,
    aruco_homography: np.ndarray,
    output_size: Tuple[int, int] = (128, 128),
    paper_corners: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the raw camera frame to a flat top-down view of the platform.

    Two-stage process (as per implementation_plan_shared_backbone_cnn.md):
      1. Perspective warp to 500x500 -- preserves sub-pixel accuracy during the
         perspective correction interpolation.
      2. Downsample to output_size (128x128) using INTER_AREA for quality.

    The source frame is padded by WARP_BORDER_PAD pixels (BORDER_REPLICATE) on
    all sides before warping. This prevents silent black fill if any platform
    corner lands near the raw camera frame boundary (replicates the legacy
    CROP_PAD=20 behaviour from generate_aruco_cropped_dataset.py).

    Parameters
    ----------
    frame_bgr      : raw camera frame (BGR)
    aruco_homography: mm->px homography from estimate_homography_from_aruco()
    output_size    : final output resolution (W, H), default 128x128
    paper_corners  : (4,2) float32 array of [TL, TR, BR, BL] paper corner
                     positions in mm. Derived from manifest via
                     build_paper_corners(). If None, raises.

    Returns
    -------
    warped      : output_size BGR image of the top-down platform view
    warp_matrix : 3x3 matrix mapping original camera pixels -> output_size pixels
    """
    if paper_corners is None:
        raise ValueError("paper_corners must be provided (use build_paper_corners())")

    out_w, out_h = output_size
    intermed = INTERMEDIATE_WARP_SIZE
    pad = WARP_BORDER_PAD

    # Pad the frame on all sides so corners near the boundary don't go black.
    # BORDER_REPLICATE extends edge pixels, giving the warp real texture to
    # interpolate from rather than black fill.
    frame_padded = cv2.copyMakeBorder(
        frame_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )

    # Project paper corners mm -> camera pixels, then shift by the pad offset
    # so coordinates align with the padded frame.
    src_px = project_points(paper_corners, aruco_homography)  # shape (4, 2)
    src_px_padded = src_px + np.array([pad, pad], dtype=np.float32)

    # Stage 1: Perspective warp padded frame to 500x500.
    dst_intermed = np.array(
        [[0, 0], [intermed, 0], [intermed, intermed], [0, intermed]],
        dtype=np.float32,
    )
    warp_500 = cv2.getPerspectiveTransform(src_px_padded.astype(np.float32), dst_intermed)
    warped_500 = cv2.warpPerspective(frame_padded, warp_500, (intermed, intermed))

    # Stage 2: Downsample to output_size using area averaging.
    warped = cv2.resize(warped_500, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # Compose a single warp matrix: original cam_px -> output_size pixels.
    # Full chain: cam_px -> shift(+pad) -> warp_500 -> scale(128/500)
    # M_final = M_scale @ M_warp500 @ M_shift
    m_shift = np.array(
        [[1.0, 0.0, float(pad)],
         [0.0, 1.0, float(pad)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scale_x = out_w / intermed
    scale_y = out_h / intermed
    m_scale = np.array(
        [[scale_x, 0.0, 0.0],
         [0.0, scale_y, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warp_matrix = m_scale @ warp_500.astype(np.float64) @ m_shift

    return warped, warp_matrix



def apply_warp_to_point(pt_px: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    """Apply a perspective warp matrix to a single (x, y) pixel coordinate."""
    pts = pt_px.reshape(1, 1, 2).astype(np.float32)
    warped_pt = cv2.perspectiveTransform(pts, warp_matrix)[0][0]
    return warped_pt


def estimate_homography_from_aruco(
    frame_bgr: np.ndarray,
    aruco_lookup: Dict[int, List[float]],
) -> Optional[np.ndarray]:
    """Detect ArUco markers and compute the mm→camera-pixel homography.

    Parameters
    ----------
    frame_bgr   : raw camera frame (BGR)
    aruco_lookup: {id: [x_mm, y_mm]} built from the manifest's aruco_markers list

    Returns None if fewer than 4 marker correspondences are found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten()
    image_points = []
    physical_points = []

    for marker_id, corner in zip(ids, corners):
        # Use manifest-derived lookup -- accepts any ID present in the manifest.
        mid = int(marker_id)
        if mid not in aruco_lookup:
            continue
        center = corner[0].mean(axis=0)
        image_points.append([center[0], center[1]])
        physical_points.append(aruco_lookup[mid])

    if len(image_points) < 4:
        return None

    homography, _ = cv2.findHomography(
        np.array(physical_points, dtype=np.float32),
        np.array(image_points, dtype=np.float32),
    )
    return homography


def auto_label_session(
    session_dir: Path,
    manifest_path: Path,
    output_csv_path: Path,
    output_mask_dir: Path,
    output_cropped_dir: Path,
    image_dir: Optional[Path] = None,
    output_size: Tuple[int, int] = (128, 128),
    limit: Optional[int] = None,
) -> int:
    # Load manifest: ArUco fiducials for homography, features for CNN mask targets
    aruco_markers, features, platform_width_mm, platform_height_mm = load_manifest_full(manifest_path)
    aruco_lookup = build_aruco_lookup(aruco_markers)
    paper_corners = build_paper_corners(platform_width_mm, platform_height_mm)

    labels_path = session_dir / "labels_normalized.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    df = pd.read_csv(labels_path)
    if limit is not None:
        df = df.head(limit)
    if image_dir is None:
        image_dir = session_dir / "images"

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_cropped_dir.mkdir(parents=True, exist_ok=True)

    out_w, out_h = output_size
    results = []
    for _, row in df.iterrows():
        image_name = row.get("image_file")
        if not image_name:
            continue

        frame_path = image_dir / str(image_name)
        if not frame_path.exists():
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        aruco_homography = estimate_homography_from_aruco(frame, aruco_lookup)
        if aruco_homography is None:
            continue

        # Warp frame to flat top-down platform view and save to images_cropped/
        warped_frame, warp_matrix = warp_to_platform(frame, aruco_homography, output_size, paper_corners)
        cropped_path = output_cropped_dir / Path(image_name).name
        cv2.imwrite(str(cropped_path), warped_frame)

        # Project ball position from mm → camera px → warped px
        touch_x = float(row.get("touch_x", 0.0))
        touch_y = float(row.get("touch_y", 0.0))
        # touch_x/touch_y's sign convention (MCU/PID telemetry frame) is inverted on
        # BOTH axes relative to the mm frame build_paper_corners()/features use --
        # confirmed empirically: with the old (touch_x + W/2, H/2 - touch_y) formula,
        # every derived ball_x_px/ball_y_px landed at (128-true_x, 128-true_y), a full
        # point reflection, while feature markers (which don't go through touch_x/y at
        # all) landed correctly. See docs/PROJECT_LOGBOOK.md (2026-08-12, "ball label
        # point-reflection bug").
        ball_x_mm = (TOUCHPAD_W / 2.0) - touch_x
        ball_y_mm = (TOUCHPAD_H / 2.0) + touch_y
        ball_pt_mm = np.array([[ball_x_mm, ball_y_mm]], dtype=np.float32)
        ball_pt_cam_px = project_points(ball_pt_mm, aruco_homography)[0]
        ball_pt_warped = apply_warp_to_point(ball_pt_cam_px, warp_matrix)

        # Generate masks for colored TARGET features only (not ArUco fiducials)
        # All coordinates are in warped 128x128 space
        for feature in features:
            feature_center_mm = np.array(feature.get("center_mm", [0.0, 0.0]), dtype=np.float32)
            feature_cam_px = project_points(feature_center_mm.reshape(1, 2), aruco_homography)[0]
            feature_warped_px = apply_warp_to_point(feature_cam_px, warp_matrix)
            mask = render_marker_mask(
                (out_h, out_w),
                (feature_warped_px[0], feature_warped_px[1]),
                feature,
            )
            mask_path = output_mask_dir / f"{Path(image_name).stem}_{feature['id']}_mask.png"
            cv2.imwrite(str(mask_path), mask)

        row_dict = row.to_dict()
        # Coordinates in warped 128x128 space
        row_dict["ball_x_px"] = float(ball_pt_warped[0])
        row_dict["ball_y_px"] = float(ball_pt_warped[1])
        row_dict["ball_visible"] = 1
        row_dict["image_file"] = str(Path(image_name).name)  # filename only, cropped image
        results.append(row_dict)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv_path, index=False)
    else:
        pd.DataFrame(columns=["image_file"]).to_csv(output_csv_path, index=False)

    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared-vision auto labels from telemetry + marker manifest")
    parser.add_argument("--session-dir", required=True, help="Path to session directory (must contain labels_normalized.csv and images/)")
    parser.add_argument("--manifest", help="Path to marker manifest JSON (auto-resolved if omitted)")
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: <session-dir>/shared_vision_labels.csv)")
    parser.add_argument("--output-mask-dir", default=None, help="Output mask dir (default: <session-dir>/masks/)")
    parser.add_argument("--output-cropped-dir", default=None, help="Output cropped images dir (default: <session-dir>/images_cropped/)")
    parser.add_argument("--images-dir", default=None, help="Override source image directory")
    parser.add_argument("--output-size", default="128x128", help="Warped output resolution WxH (default: 128x128)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (dry runs)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    manifest_path = resolve_manifest_path(
        session_dir=session_dir,
        manifest_hint=Path(args.manifest) if args.manifest else None,
    )

    # Default all outputs to be co-located with the session directory
    output_csv_path = Path(args.output_csv) if args.output_csv else session_dir / "shared_vision_labels.csv"
    output_mask_dir = Path(args.output_mask_dir) if args.output_mask_dir else session_dir / "masks"
    output_cropped_dir = Path(args.output_cropped_dir) if args.output_cropped_dir else session_dir / "images_cropped"
    images_dir = Path(args.images_dir) if args.images_dir else None

    w_str, h_str = args.output_size.lower().split("x")
    output_size = (int(w_str), int(h_str))

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    count = auto_label_session(
        session_dir=session_dir,
        manifest_path=manifest_path,
        output_csv_path=output_csv_path,
        output_mask_dir=output_mask_dir,
        output_cropped_dir=output_cropped_dir,
        image_dir=images_dir,
        output_size=output_size,
        limit=args.limit,
    )
    print(f"Wrote {count} labeled rows to {output_csv_path}")
    print(f"Cropped images saved to {output_cropped_dir}")
    print(f"Masks saved to {output_mask_dir}")


if __name__ == "__main__":
    main()
