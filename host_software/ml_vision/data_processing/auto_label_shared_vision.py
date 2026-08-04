import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


TOUCHPAD_W = 187.5
TOUCHPAD_H = 142.0
PAPER_W = 164.0
PAPER_H = 124.0
OFFSET_X = (TOUCHPAD_W - PAPER_W) / 2.0
OFFSET_Y = (TOUCHPAD_H - PAPER_H) / 2.0

PAPER_CORNERS_MM = np.array(
    [
        [OFFSET_X, TOUCHPAD_H - OFFSET_Y],
        [OFFSET_X + PAPER_W, TOUCHPAD_H - OFFSET_Y],
        [OFFSET_X + PAPER_W, TOUCHPAD_H - (OFFSET_Y + PAPER_H)],
        [OFFSET_X, TOUCHPAD_H - (OFFSET_Y + PAPER_H)],
    ],
    dtype=np.float32,
)


def load_manifest(manifest_path: Path) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = []
    markers = payload.get("markers", [])
    if markers:
        entries.extend(markers)

    features = payload.get("features", [])
    if features:
        entries.extend(features)

    if not entries:
        raise ValueError(f"No markers or features found in manifest: {manifest_path}")

    return entries


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
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    size_mm = float(marker.get("size_mm", 6.0))
    cx, cy = center_xy
    radius_px = max(2, int(round(size_mm * 0.35)))

    if marker.get("shape", "circle") == "triangle":
        points = np.array(
            [
                [cx, cy - radius_px],
                [cx + radius_px, cy + radius_px],
                [cx - radius_px, cy + radius_px],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
    else:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), radius_px, 255, -1)

    return mask


def estimate_homography_from_aruco(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(frame_bgr)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)

    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten()
    image_points = []
    physical_points = []

    for marker_id, corner in zip(ids, corners):
        if marker_id not in {0, 1, 2, 3}:
            continue
        center = corner[0].mean(axis=0)
        image_points.append([center[0], center[1]])
        physical_points.append([PAPER_CORNERS_MM[marker_id][0], PAPER_CORNERS_MM[marker_id][1]])

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
    image_dir: Optional[Path] = None,
) -> int:
    markers = load_manifest(manifest_path)
    labels_path = session_dir / "labels_normalized.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    df = pd.read_csv(labels_path)
    if image_dir is None:
        image_dir = session_dir / "images"

    output_mask_dir.mkdir(parents=True, exist_ok=True)

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

        homography = estimate_homography_from_aruco(frame)
        if homography is None:
            continue

        touch_x = float(row.get("touch_x", 0.0))
        touch_y = float(row.get("touch_y", 0.0))
        ball_x_mm = touch_x + (TOUCHPAD_W / 2.0)
        ball_y_mm = (TOUCHPAD_H / 2.0) - touch_y
        ball_pt_mm = np.array([[ball_x_mm, ball_y_mm]], dtype=np.float32)
        ball_pt_px = project_points(ball_pt_mm, homography)[0]

        for marker in markers:
            marker_center_mm = np.array(marker.get("center_mm", [0.0, 0.0]), dtype=np.float32)
            marker_center_px = project_points(marker_center_mm.reshape(1, 2), homography)[0]
            mask = render_marker_mask(frame.shape[:2][::-1], (marker_center_px[0], marker_center_px[1]), marker)
            mask_path = output_mask_dir / f"{Path(image_name).stem}_{marker['id']}_mask.png"
            cv2.imwrite(str(mask_path), mask)

        row_dict = row.to_dict()
        row_dict["ball_x_px"] = float(ball_pt_px[0])
        row_dict["ball_y_px"] = float(ball_pt_px[1])
        row_dict["ball_visible"] = 1
        results.append(row_dict)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv_path, index=False)
    else:
        pd.DataFrame(columns=["image_file"]).to_csv(output_csv_path, index=False)

    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared-vision auto labels from telemetry + marker manifest")
    parser.add_argument("--session-dir", default="host_software/data/02_silver/aruco_markers_01")
    parser.add_argument("--manifest")
    parser.add_argument("--output-csv", default="host_software/data/03_shared_vision_labels.csv")
    parser.add_argument("--output-mask-dir", default="host_software/data/03_shared_vision_masks")
    parser.add_argument("--images-dir")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    manifest_path = resolve_manifest_path(
        session_dir=session_dir,
        manifest_hint=Path(args.manifest) if args.manifest else None,
    )
    output_csv_path = Path(args.output_csv)
    output_mask_dir = Path(args.output_mask_dir)
    images_dir = Path(args.images_dir) if args.images_dir else None

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    count = auto_label_session(
        session_dir=session_dir,
        manifest_path=manifest_path,
        output_csv_path=output_csv_path,
        output_mask_dir=output_mask_dir,
        image_dir=images_dir,
    )
    print(f"Wrote {count} labeled rows to {output_csv_path}")


if __name__ == "__main__":
    main()
