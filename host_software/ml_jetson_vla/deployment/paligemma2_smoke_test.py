"""Smoke test for PaliGemma2-3B-mix (`google/paligemma2-3b-mix-448`), one of the Arm 2
minimal-baseline's general-purpose VLM candidates (`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`).
Mirrors `qwen_vl_smoke_test.py`'s CLI shape.

**Not executed against real downloaded weights in this session** -- same reasoning as
`internvl25_smoke_test.py`'s docstring (no local checkpoint, multi-GB download not attempted).
What WAS verified for real, 2026-09-18: `hasattr(transformers, "PaliGemmaForConditionalGeneration")`
is `True` on this machine's installed `transformers==5.17.0`. **Important prompting caveat,
confirmed via WebSearch 2026-09-18, not assumed**: unlike the other three candidates,
PaliGemma2's "mix" checkpoint is instruction-tuned for a mix of specific task prefixes (e.g.
`"detect {object}"`), not open natural-language chat the way Qwen/InternVL/Moondream2 are --
its own documented output convention for detection is NOT JSON, it is `<locY1><locX1><locY2>
<locX2>` special location tokens normalized to a 1024-cell grid. This script defaults `--prompt`
to that native `detect` convention rather than the free-form question the other smoke tests use,
since asking PaliGemma2 an open question the way the other candidates are asked would not be a
fair/representative test of this model's actual real-world usage pattern.
"""

import argparse
import os
import re
import time

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

DEFAULT_REPO_ID = "google/paligemma2-3b-mix-448"
LOC_TOKEN_RE = re.compile(r"<loc(\d{4})>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for PaliGemma2-3B-mix")
    parser.add_argument("--image", type=str, required=False, help="Path to a test image")
    parser.add_argument("--camera_id", type=int, default=-1, help="USB camera ID to capture from")
    parser.add_argument("--prompt", type=str, default="detect green marker",
                         help="PaliGemma2-mix's native task-prefix prompt convention, e.g. "
                              "'detect {object}' -- NOT a free-form question, see module docstring.")
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

    print(f"Loading PaliGemmaForConditionalGeneration from {model_path} in BF16 on {device}...")
    start_load = time.time()
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    from PIL import Image
    pil_image = Image.open(image_path).convert("RGB")
    inputs = processor(text=args.prompt, images=pil_image, return_tensors="pt").to(device)

    print("Running inference pass...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_gen = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_time = time.time() - start_gen
    trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    output_text = processor.decode(trimmed, skip_special_tokens=True)

    print("\n" + "=" * 40)
    print("--- Generation Results ---")
    print(f"Wall-clock Latency: {gen_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print("\nGenerated Text (raw <loc> tokens):")
    print(output_text)
    loc_tokens = LOC_TOKEN_RE.findall(output_text)
    if len(loc_tokens) == 4:
        y1, x1, y2, x2 = (int(t) / 1024.0 for t in loc_tokens)
        print(f"Parsed bbox (normalized 0-1, Y1X1Y2X2 order): ({y1:.3f},{x1:.3f})-({y2:.3f},{x2:.3f})")
    else:
        print(f"Could not parse exactly 4 <loc> tokens (found {len(loc_tokens)}) -- model may "
              f"have answered outside its native detect convention for this prompt.")
    print("=" * 40)


if __name__ == "__main__":
    main()
