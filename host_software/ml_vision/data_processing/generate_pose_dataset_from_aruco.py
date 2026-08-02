import cv2
import cv2.aruco as aruco
import numpy as np
import os
import argparse

# Physical Platform Dimensions (Touchpad)
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# The physical marker locations relative to the top-left of the touchpad
MARKER_PHYSICAL_MM = {
    0: [12.0, 130.0],
    1: [175.5, 130.0],
    2: [175.5, 12.0],
    3: [12.0, 12.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0],
}

# The user noted the printed paper is exactly 164.0 x 124.0 mm.
# Centered on the 187.5 x 142.0 touchpad, the paper corners are:
# Offset X: (187.5 - 164.0) / 2 = 11.75
# Offset Y: (142.0 - 124.0) / 2 = 9.0
PAPER_CORNERS_MM = np.array(
    [
        [11.75, 9.0],  # Top-Left
        [11.75 + 164.0, 9.0],  # Top-Right
        [11.75 + 164.0, 9.0 + 124.0],  # Bottom-Right
        [11.75, 9.0 + 124.0],  # Bottom-Left
    ],
    dtype=np.float32,
)


def get_aruco_homography(img):
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(img)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(
            img, dictionary, parameters=parameters
        )

    if ids is None:
        return None

    ids = ids.flatten()

    img_pts = []
    phys_pts = []

    for i, marker_id in enumerate(ids):
        if marker_id in MARKER_PHYSICAL_MM:
            # We must apply the Y-flip that we discovered physically in the ArUco session
            phys_x = MARKER_PHYSICAL_MM[marker_id][0]
            phys_y = PLATFORM_H - MARKER_PHYSICAL_MM[marker_id][1]

            c = corners[i][0]
            cx = c[:, 0].mean()
            cy = c[:, 1].mean()

            img_pts.append([cx, cy])
            phys_pts.append([phys_x, phys_y])

    if len(img_pts) < 4:
        return None

    M, _ = cv2.findHomography(
        np.array(phys_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)
    )
    return M


def format_pose_label(class_id, keypoints, img_w, img_h):
    # Determine bounding box from keypoints
    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    cx_norm = max(0.0, min(1.0, cx / img_w))
    cy_norm = max(0.0, min(1.0, cy / img_h))
    w_norm = max(0.0, min(1.0, w / img_w))
    h_norm = max(0.0, min(1.0, h / img_h))

    label = f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

    for kx, ky in keypoints:
        kx_n = max(0.0, min(1.0, kx / img_w))
        ky_n = max(0.0, min(1.0, ky / img_h))
        label += f" {kx_n:.6f} {ky_n:.6f} 2"  # visibility=2

    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="host_software/data/01_bronze/session_20260730_174916/rgb_video.mp4",
    )
    parser.add_argument("--out_dir", default="host_software/data/03_pose_dataset")
    args = parser.parse_args()

    out_images = os.path.join(args.out_dir, "images")
    out_labels = os.path.join(args.out_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        return

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting ArUco corners from {total_frames} frames...")

    success_count = 0

    # We will sample 1 every 5 frames to reduce redundancy (ArUco is static)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 5 != 0:
            continue

        M = get_aruco_homography(frame)
        if M is None:
            continue

        # Project paper corners into pixel space
        h, w = frame.shape[:2]

        # Note: Apply the same Y-flip to the PAPER_CORNERS_MM to match the physical orientation!
        flipped_paper = PAPER_CORNERS_MM.copy()
        flipped_paper[:, 1] = PLATFORM_H - flipped_paper[:, 1]

        pts = np.array([flipped_paper], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(pts, M)[0]

        # Check if it goes totally out of bounds (sanity check)
        if (
            np.any(warped_corners < -100)
            or np.any(warped_corners[:, 0] > w + 100)
            or np.any(warped_corners[:, 1] > h + 100)
        ):
            continue

        label = format_pose_label(0, warped_corners, w, h)

        # The user requested filenames to match the telemetry sync.
        # Assuming timestamps or just matching video frame indexing.
        # To make it perfectly sync-able, if we use frame_{frame_idx} they map exactly.
        img_name = f"frame_{frame_idx:06d}.jpg"
        txt_name = f"frame_{frame_idx:06d}.txt"

        cv2.imwrite(os.path.join(out_images, img_name), frame)
        with open(os.path.join(out_labels, txt_name), "w") as f:
            f.write(label + "\n")

        success_count += 1
        if success_count % 100 == 0:
            print(f"Generated {success_count} YOLO-Pose labeled frames...")

    cap.release()
    print(f"Successfully generated {success_count} YOLO-Pose labeled frames!")

    # Generate the yaml file pointing to the dataset using POSIX-style paths for YOLO
    # Ensure no backslashes in YOLO yaml
    out_dir_posix = os.path.abspath(args.out_dir).replace("\\", "/")
    yaml_content = f"""path: {out_dir_posix}
train: images
val: images

kpt_shape: [4, 3] # 4 keypoints (corners), 3 values (x, y, vis)

names:
  0: paper_platform
"""
    with open(os.path.join(args.out_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    main()
