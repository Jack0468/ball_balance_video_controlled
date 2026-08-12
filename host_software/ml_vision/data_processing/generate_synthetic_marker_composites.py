"""Synthetically composite marker shapes onto Dataset 4's blank-platform frames.

Implements Component 3 of docs/plans/implementation_plan_shared_backbone_cnn.md
("For the blank platform session, we will synthetically generate and composite
markers at random positions. This prevents the model from memorizing the
hard-coded physical positions on the real printed sheets.") extended to also vary
shape/color: real sheets (aruco_markers_01/02/03) only cover 11 of the 20 possible
(shape, color) combinations -- see MISSING_COMBOS below -- biasing synthetic
generation toward the missing 9 teaches the mask/heatmap heads to detect markers by
appearance, not just recall a fixed catalogue.

Base frames are Dataset 4 (host_software/data/01_bronze/session_20260810_104132),
already warped to 128x128 with correct ball_x_px/ball_y_px labels (post the
2026-08-12 ball-label sign fix -- see docs/PROJECT_LOGBOOK.md). Placement avoids
three things per frame: the ball + its shadow (detected via darkness thresholding,
not just a fixed-radius circle around the label point, since a shadow can extend
asymmetrically), the 6 ArUco fiducials (fixed pixel positions for a given manifest,
computed once), and already-placed synthetic markers (via incremental footprint
accumulation, not just centre-to-centre distance).

Run as a module from the repo root:

    python -m host_software.ml_vision.data_processing.generate_synthetic_marker_composites \
        --output-dir host_software/data/03_gold/shared_vision_synthetic \
        --limit 200 --num-variants-per-image 1
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from host_software.ml_vision.data_processing.auto_label_shared_vision import (
    PAPER_MARGIN_MM,
    TOUCHPAD_H,
    TOUCHPAD_W,
    load_manifest_full,
)

SHAPES = ["circle", "triangle", "square", "hexagon"]
COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "blue": (200, 30, 30),
    "black": (25, 25, 25),
    "red": (30, 30, 200),
    "green": (30, 160, 30),
    "yellow": (30, 220, 220),
}
# The 9 (shape, color) combinations absent from all 3 real sheets (verified against
# hardware/platform_templates/aruco_markers_{01,02,03}_manifest.json -- real coverage
# is 11/20: circle in all 5 colors, triangle in {blue,red}, square in {blue,yellow},
# hexagon in {blue,green}).
MISSING_COMBOS: List[Tuple[str, str]] = [
    ("triangle", "black"), ("triangle", "green"), ("triangle", "yellow"),
    ("square", "black"), ("square", "red"), ("square", "green"),
    ("hexagon", "black"), ("hexagon", "red"), ("hexagon", "yellow"),
]
ALL_COMBOS: List[Tuple[str, str]] = [(s, c) for s in SHAPES for c in COLORS_BGR]


def sample_shape_color(rng: random.Random, missing_combo_bias: float) -> Tuple[str, str]:
    if rng.random() < missing_combo_bias:
        return rng.choice(MISSING_COMBOS)
    return rng.choice(ALL_COMBOS)


def mm_to_px(x_mm: float, y_mm: float, output_size: Tuple[int, int]) -> Tuple[float, float]:
    """Same fixed-scale mm->px conversion used in evaluate_shared_vision_backbone.py:
    the warp always maps the mm rectangle [-margin, W+margin] x [-margin, H+margin]
    onto the full output_size pixel frame, so this is an exact affine scale, not an
    approximation, independent of any single frame's homography."""
    out_w, out_h = output_size
    px_x = (x_mm + PAPER_MARGIN_MM) * out_w / (TOUCHPAD_W + 2 * PAPER_MARGIN_MM)
    px_y = (y_mm + PAPER_MARGIN_MM) * out_h / (TOUCHPAD_H + 2 * PAPER_MARGIN_MM)
    return px_x, px_y


def build_aruco_exclusion_zones(
    aruco_markers: List[Dict], output_size: Tuple[int, int], safety_margin_px: float = 3.0
) -> List[Tuple[int, int, int]]:
    """Fixed (px_x, px_y, radius_px) circles for the 6 ArUco fiducials -- constant
    across every frame for a given manifest, so computed once and reused."""
    out_w, _ = output_size
    px_per_mm_x = out_w / (TOUCHPAD_W + 2 * PAPER_MARGIN_MM)
    zones = []
    for marker in aruco_markers:
        cx_mm, cy_mm = marker["center_mm"]
        px_x, px_y = mm_to_px(cx_mm, cy_mm, output_size)
        radius_px = (float(marker.get("size_mm", 22.5)) / 2.0) * px_per_mm_x + safety_margin_px
        zones.append((int(round(px_x)), int(round(px_y)), int(round(radius_px))))
    return zones


def build_exclusion_mask(
    output_size: Tuple[int, int],
    gray_img: np.ndarray,
    ball_xy: Tuple[float, float],
    aruco_zones: List[Tuple[int, int, int]],
    ball_dark_threshold: int = 100,
    ball_roi_half: int = 25,
    ball_dilate_px: int = 4,
) -> np.ndarray:
    """Binary (0/255) forbidden-to-draw-on mask: ArUco fiducials + ball/shadow.
    Already-placed synthetic markers get folded in incrementally by the caller."""
    out_w, out_h = output_size
    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    for px, py, radius in aruco_zones:
        cv2.circle(mask, (px, py), radius, 255, -1)

    bx, by = int(round(ball_xy[0])), int(round(ball_xy[1]))
    x0, x1 = max(0, bx - ball_roi_half), min(out_w, bx + ball_roi_half)
    y0, y1 = max(0, by - ball_roi_half), min(out_h, by + ball_roi_half)
    if x1 > x0 and y1 > y0:
        roi = gray_img[y0:y1, x0:x1]
        _, dark = cv2.threshold(roi, ball_dark_threshold, 255, cv2.THRESH_BINARY_INV)
        if dark.any():
            kernel = np.ones((ball_dilate_px, ball_dilate_px), np.uint8)
            dark = cv2.dilate(dark, kernel)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], dark)
        else:
            # Fallback: thresholding found nothing (unusual) -- fixed-radius circle
            # around the label point rather than leaving the ball undefended.
            cv2.circle(mask, (bx, by), ball_roi_half // 2, 255, -1)

    return mask


def polygon_vertices(shape: str, size_px: int, rotation_deg: float) -> Optional[np.ndarray]:
    """Local (0,0)-centred polygon vertices for a given shape; None for circle
    (handled separately via cv2.circle). Mirrors render_marker_mask()'s triangle
    geometry (auto_label_shared_vision.py) and extends it to square/hexagon --
    real polygon masks for shapes that today only ever get a circle-fallback mask
    on real data. This function is new, local to this script; render_marker_mask()
    itself is untouched so real-sheet masks stay reproducible."""
    if shape == "triangle":
        # Equilateral, matching render_marker_mask()'s corrected geometry: apex is
        # 2x as far from centre as the base (standard equilateral centroid split),
        # not a symmetric apex/base offset -- see docs/PROJECT_LOGBOOK.md, 2026-08-12.
        apex_offset = size_px * 2.0 / np.sqrt(3.0)
        base_offset = size_px * 1.0 / np.sqrt(3.0)
        pts = np.array([[0, -apex_offset], [size_px, base_offset], [-size_px, base_offset]], dtype=np.float32)
    elif shape == "square":
        h = size_px
        pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float32)
    elif shape == "hexagon":
        angles = np.deg2rad(np.arange(6) * 60.0)
        pts = np.stack([size_px * np.cos(angles), size_px * np.sin(angles)], axis=1)
    else:
        return None

    theta = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    return pts @ rot.T


def local_shape_footprint(shape: str, size_px: int, rotation_deg: float) -> Tuple[np.ndarray, int]:
    """Small local binary patch (side = 2*pad+1) with the shape's silhouette centred
    in it, used both for placement-collision testing and mask rendering without
    re-running fillPoly for every rejection-sampling attempt."""
    pad = size_px + 4
    side = 2 * pad + 1
    patch = np.zeros((side, side), dtype=np.uint8)
    center = (pad, pad)
    if shape == "circle":
        cv2.circle(patch, center, size_px, 255, -1)
    else:
        verts = polygon_vertices(shape, size_px, rotation_deg) + np.array(center)
        cv2.fillPoly(patch, [verts.astype(np.int32)], 255)
    return patch, pad


def place_markers(
    output_size: Tuple[int, int],
    n: int,
    rng: random.Random,
    exclusion_mask: np.ndarray,
    missing_combo_bias: float,
    min_size_px: int,
    max_size_px: int,
    footprint_gap_px: int,
    max_attempts: int = 100,
) -> List[Dict]:
    """Rejection-samples each marker's actual footprint (not just its centre point)
    against the shared exclusion mask, which is updated in-place as markers are
    placed so later markers can't overlap earlier ones either."""
    out_w, out_h = output_size
    placed: List[Dict] = []
    mask = exclusion_mask.copy()

    for _ in range(n):
        shape, color = sample_shape_color(rng, missing_combo_bias)
        size_px = rng.randint(min_size_px, max_size_px)
        rotation_deg = rng.uniform(0, 360) if shape != "circle" else 0.0
        footprint, pad = local_shape_footprint(shape, size_px, rotation_deg)
        # Dilate the footprint used for collision-checking (not the one drawn later)
        # by footprint_gap_px so placed markers keep a visible gap, not just zero overlap.
        gap_kernel = np.ones((footprint_gap_px, footprint_gap_px), np.uint8)
        footprint_padded = cv2.dilate(footprint, gap_kernel)

        placed_ok = False
        for _ in range(max_attempts):
            cx = rng.randint(pad, out_w - pad - 1)
            cy = rng.randint(pad, out_h - pad - 1)
            y0, y1 = cy - pad, cy + pad + 1
            x0, x1 = cx - pad, cx + pad + 1
            region = mask[y0:y1, x0:x1]
            if not np.any(cv2.bitwise_and(region, footprint_padded)):
                mask[y0:y1, x0:x1] = np.maximum(region, footprint)
                placed.append({
                    "shape": shape, "color": color, "center_px": [cx, cy],
                    "size_px": size_px, "rotation_deg": rotation_deg,
                })
                placed_ok = True
                break
        if not placed_ok:
            continue  # skip this marker slot -- no valid spot found, logged by caller via count

    return placed


def draw_marker(img: np.ndarray, marker: Dict, rng: random.Random) -> np.ndarray:
    """Alpha-blended fill + simulated glare, generalized from
    generate_synthetic_yolo_dataset.py's draw_marker_with_glare() to arbitrary
    polygon shapes (that function only ever draws circles)."""
    cx, cy = marker["center_px"]
    color_bgr = COLORS_BGR[marker["color"]]
    size_px = marker["size_px"]

    overlay = img.copy()
    if marker["shape"] == "circle":
        cv2.circle(overlay, (cx, cy), size_px, color_bgr, -1)
    else:
        verts = polygon_vertices(marker["shape"], size_px, marker["rotation_deg"]) + np.array([cx, cy])
        cv2.fillPoly(overlay, [verts.astype(np.int32)], color_bgr)

    alpha = rng.uniform(0.6, 0.9)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    if rng.random() > 0.3:
        glare_overlay = img.copy()
        glare_r = int(size_px * rng.uniform(0.4, 0.7))
        gx = cx + rng.randint(-size_px // 2, size_px // 2)
        gy = cy + rng.randint(-size_px // 2, size_px // 2)
        cv2.circle(glare_overlay, (gx, gy), glare_r, (255, 255, 255), -1)
        glare_alpha = rng.uniform(0.4, 0.8)
        img = cv2.addWeighted(glare_overlay, glare_alpha, img, 1 - glare_alpha, 0)

        r = size_px * 2
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 0)

    return img


def composite_one_variant(
    base_img: np.ndarray,
    ball_xy: Tuple[float, float],
    aruco_zones: List[Tuple[int, int, int]],
    output_size: Tuple[int, int],
    rng: random.Random,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    exclusion_mask = build_exclusion_mask(output_size, gray, ball_xy, aruco_zones)

    n = rng.randint(args.min_markers, args.max_markers)
    placements = place_markers(
        output_size, n, rng, exclusion_mask, args.missing_combo_bias,
        args.min_marker_size_px, args.max_marker_size_px, args.min_marker_dist_px,
    )

    img = base_img.copy()
    out_w, out_h = output_size
    combined_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    for marker in placements:
        img = draw_marker(img, marker, rng)
        if args.circle_only_masks:
            single_mask = np.zeros((out_h, out_w), dtype=np.uint8)
            cv2.circle(single_mask, tuple(marker["center_px"]), marker["size_px"], 255, -1)
        else:
            footprint, pad = local_shape_footprint(marker["shape"], marker["size_px"], marker["rotation_deg"])
            single_mask = np.zeros((out_h, out_w), dtype=np.uint8)
            cx, cy = marker["center_px"]
            y0, y1 = max(0, cy - pad), min(out_h, cy + pad + 1)
            x0, x1 = max(0, cx - pad), min(out_w, cx + pad + 1)
            fy0, fx0 = y0 - (cy - pad), x0 - (cx - pad)
            single_mask[y0:y1, x0:x1] = footprint[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0)]
        combined_mask = np.maximum(combined_mask, single_mask)

    return img, combined_mask, placements


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite synthetic markers onto Dataset 4's blank-platform frames")
    parser.add_argument("--base-session-dir", type=Path, default=Path("host_software/data/01_bronze/session_20260810_104132"))
    parser.add_argument("--base-csv", type=Path, default=None, help="Defaults to <base-session-dir>/shared_vision_labels.csv")
    parser.add_argument("--base-images-dir", type=Path, default=None, help="Defaults to <base-session-dir>/images_cropped")
    parser.add_argument("--manifest", type=Path, default=Path("hardware/platform_templates/ground_truth_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-size", default="128x128")
    parser.add_argument("--num-variants-per-image", type=int, default=4)
    parser.add_argument("--min-markers", type=int, default=1)
    parser.add_argument("--max-markers", type=int, default=5)
    parser.add_argument(
        "--min-marker-size-px", type=int, default=4,
        help="Real markers are size_mm=8.0 (radius) on every sheet, which converts to "
        "~5.1px radius at 128x128 (8.0 * 128/(187.5+12)) -- was previously 8, ~1.6x too "
        "large, before render_marker_mask()'s own size_mm*0.35 formula was corrected to "
        "use this same real mm/px scale (see docs/PROJECT_LOGBOOK.md, 2026-08-12).",
    )
    parser.add_argument("--max-marker-size-px", type=int, default=7)
    parser.add_argument("--min-marker-dist-px", type=int, default=14, help="Extra gap enforced between marker footprints")
    parser.add_argument("--missing-combo-bias", type=float, default=0.75)
    parser.add_argument(
        "--circle-only-masks", action="store_true",
        help="Render synthetic masks as circles (matching real-sheet square/hexagon mask "
        "fallback) instead of true polygons -- fallback toggle in case polygon masks turn "
        "out to create a 'polygon=synthetic, circle=real' shortcut during training",
    )
    parser.add_argument("--session-name", default="synthetic_composite_00")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N base rows (dry runs)")
    args = parser.parse_args()

    base_csv = args.base_csv or args.base_session_dir / "shared_vision_labels.csv"
    base_images_dir = args.base_images_dir or args.base_session_dir / "images_cropped"
    w_str, h_str = args.output_size.lower().split("x")
    output_size = (int(w_str), int(h_str))

    aruco_markers, _, _, _ = load_manifest_full(args.manifest)
    aruco_zones = build_aruco_exclusion_zones(aruco_markers, output_size)

    base_df = pd.read_csv(base_csv)
    if args.limit is not None:
        base_df = base_df.head(args.limit)

    images_out = args.output_dir / "images"
    masks_out = args.output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = []
    combo_counts: Dict[str, int] = {}
    n_skipped_frames = 0
    frame_index = 0

    for _, row in base_df.iterrows():
        img_path = base_images_dir / str(row["image_file"])
        base_img = cv2.imread(str(img_path))
        if base_img is None:
            n_skipped_frames += 1
            continue
        ball_xy = (float(row["ball_x_px"]), float(row["ball_y_px"]))

        for variant in range(args.num_variants_per_image):
            img, mask, placements = composite_one_variant(base_img, ball_xy, aruco_zones, output_size, rng, args)
            if not placements:
                continue

            out_name = f"{args.session_name}__{Path(str(row['image_file'])).stem}_v{variant:02d}.png"
            cv2.imwrite(str(images_out / out_name), img)
            cv2.imwrite(str(masks_out / out_name), mask)

            for m in placements:
                key = f"{m['shape']}-{m['color']}"
                combo_counts[key] = combo_counts.get(key, 0) + 1

            rows.append({
                "image_file": out_name,
                "ball_x_px": ball_xy[0],
                "ball_y_px": ball_xy[1],
                "ball_visible": row.get("ball_visible", 1),
                "session": args.session_name,
                "frame_index": frame_index,
                "touch_x": row.get("touch_x", 0.0),
                "touch_y": row.get("touch_y", 0.0),
                "synthetic_markers_json": json.dumps(placements),
            })
            frame_index += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_dir / "labels.csv", index=False)

    print(f"Generated {len(out_df)} synthetic rows from {len(base_df)} base frames "
          f"({n_skipped_frames} base frames skipped -- image not found).")
    print("Per-combo counts:")
    for combo, count in sorted(combo_counts.items()):
        flag = " (was missing from real data)" if tuple(combo.split("-")) in MISSING_COMBOS else ""
        print(f"  {combo}: {count}{flag}")
    missing_covered = sum(1 for s, c in MISSING_COMBOS if f"{s}-{c}" in combo_counts)
    print(f"Missing-combo coverage: {missing_covered}/{len(MISSING_COMBOS)}")
    print(f"Wrote {args.output_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()
