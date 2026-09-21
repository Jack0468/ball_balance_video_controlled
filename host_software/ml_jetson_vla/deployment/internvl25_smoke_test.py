"""**WARNING (2026-09-19): this standalone script's load path is WRONG for `OpenGVLab/InternVL2_5-4B`
and should not be used for that checkpoint.** The paragraph below claims native `transformers`
support; that only shows the class exists in the library. The 4B checkpoint is a
`trust_remote_code` repo with its own `model.chat()` API -- there is no official HF-native
conversion of it (official ones: InternVL2_5-2B-MPO-hf / 8B-MPO-hf, InternVL3-*-hf,
InternVL3_5-*-HF). Use `core/vlm_backends.py`'s `InternVLBackend` (rewritten against the model
card) or the Colab sweep notebook. This file is left as originally written for the record; see
`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` §2 correction and §8.4.

Smoke test for InternVL2.5-4B (`OpenGVLab/InternVL2_5-4B`), one of the Arm 2 minimal-
baseline's general-purpose VLM candidates (`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`).
Mirrors `qwen_vl_smoke_test.py`'s CLI shape (`--image`/`--camera_id`/`--prompt`/`--model_dir`)
deliberately, so both scripts are run/read the same way.

**Not executed against real downloaded weights in this session** -- no local checkpoint under
`models/`, and this candidate's ~4B-param checkpoint is a multi-GB download not attempted here
(see the multi-candidate doc's verification section for exactly what was and wasn't run).
What WAS verified for real, 2026-09-18: `hasattr(transformers, "InternVLForConditionalGeneration")`
is `True` on this machine's installed `transformers==5.17.0` -- HF Transformers ships native
InternVL support directly (no `trust_remote_code` needed with this version), confirmed by
import, not assumed from an older InternVL-loading tutorial that predates native support. The
`from_pretrained`/`apply_chat_template`/`generate` call shape below follows that native
integration's documented pattern (`AutoModelForImageTextToText`-family models all share this
shape in current `transformers`) -- run this script on a machine with the checkpoint (or
internet access to download it) to get a real number; nothing here is fabricated to look like
one.
"""

import argparse
import os
import time

import torch
from transformers import AutoProcessor, InternVLForConditionalGeneration

DEFAULT_REPO_ID = "OpenGVLab/InternVL2_5-4B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for InternVL2.5-4B")
    parser.add_argument("--image", type=str, required=False, help="Path to a test image")
    parser.add_argument("--camera_id", type=int, default=-1, help="USB camera ID to capture from")
    parser.add_argument("--prompt", type=str, default="Identify the ball's position or the nearest colored marker in this image.")
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

    print(f"Loading InternVLForConditionalGeneration from {model_path} in BF16 on {device}...")
    start_load = time.time()
    model = InternVLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    from PIL import Image
    pil_image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=pil_image, return_tensors="pt").to(device)

    print("Running inference pass...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_gen = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_time = time.time() - start_gen
    trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    output_text = processor.decode(trimmed, skip_special_tokens=True)

    print("\n" + "=" * 40)
    print("--- Generation Results ---")
    print(f"Wall-clock Latency: {gen_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print("\nGenerated Text:")
    print(output_text)
    print("=" * 40)


if __name__ == "__main__":
    main()
