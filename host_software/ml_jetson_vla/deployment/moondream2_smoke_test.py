"""Smoke test for Moondream2 (`vikhyatk/moondream2`), one of the Arm 2 minimal-baseline's
general-purpose VLM candidates (`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`). Mirrors
`qwen_vl_smoke_test.py`'s CLI shape, but calls the model's own native `.point()` API rather
than a generic `generate()` + text prompt (see module docstring below for why).

**Config/README verified for real, 2026-09-18** (not the full ~4GB weights -- see
`core/vlm_backends.py`'s module docstring for exactly what was fetched and why): this repo's
`config.json` declares `"architectures": ["HfMoondream"]` and `"auto_map": {"AutoModelForCausalLM":
"hf_moondream.HfMoondream"}` -- i.e. it is loaded via `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)`, not a stock `transformers` VLM class the way the other three
candidates are. Its `README.md` confirms the real native call shape used below:
`model.point(image, "<label>")["points"]` (returns points normalized [0,1]x[0,1], see
`vlm_backends.py`'s `Moondream2Backend` docstring for the WebSearch-confirmed coordinate
convention), `model.detect(image, "<label>")["objects"]`, `model.query(image,
"<question>")["answer"]`. **Not executed against the full weights in this session.**
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM

DEFAULT_REPO_ID = "vikhyatk/moondream2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for Moondream2 (native .point() API)")
    parser.add_argument("--image", type=str, required=False, help="Path to a test image")
    parser.add_argument("--camera_id", type=int, default=-1, help="USB camera ID to capture from")
    parser.add_argument("--target_label", type=str, default="green marker",
                         help="Short noun phrase for .point() -- moondream2's real API takes a "
                              "label directly, not a free-form question (see module docstring).")
    parser.add_argument("--model_dir", type=str, default=None,
                         help=f"Local checkpoint dir, or leave unset to pull {DEFAULT_REPO_ID} from the Hub")
    args = parser.parse_args()

    use_camera = args.image is None or args.camera_id >= 0
    camera_id = args.camera_id if args.camera_id >= 0 else 0
    image_path = args.image

    if use_camera:
        import cv2
        print(f"Capturing frame from camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Could not capture frame from camera.")
            return
        image_path = "live_capture.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Saved captured frame to {image_path}")
    elif not os.path.exists(image_path):
        print(f"Error: Provided image path {image_path} does not exist.")
        return

    model_path = args.model_dir or DEFAULT_REPO_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Moondream2 (trust_remote_code) from {model_path} in BF16 on {device}...")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    from PIL import Image
    pil_image = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_image.size

    print(f"Running model.point(image, {args.target_label!r})...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_gen = time.time()
    result = model.point(pil_image, args.target_label)
    gen_time = time.time() - start_gen

    print("\n" + "=" * 40)
    print("--- Generation Results ---")
    print(f"Wall-clock Latency: {gen_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print("\nRaw result:", result)
    points = result.get("points", [])
    if points:
        px, py = points[0]["x"] * img_w, points[0]["y"] * img_h
        print(f"First point, denormalized to this image ({img_w}x{img_h}): ({px:.1f}, {py:.1f}) px")
    else:
        print("No points returned.")
    print("=" * 40)


if __name__ == "__main__":
    main()
