"""Smoke test for Qwen2.5-VL-3B-Instruct on Jetson (or, for CPU-only feasibility checks,
any machine with the checkpoint on disk -- see `load_qwen_vl_model()`'s dtype note below).

**2026-09-18 extension:** `load_qwen_vl_model()` and `generate_qwen_vl()` below are now
importable functions, not just CLI-internal logic -- factored out so
`core/vlm_backends.py`'s `QwenVLBackend` (the Arm 2 minimal-baseline multi-candidate wrapper,
`docs/ARM2_MINIMAL_BASELINE_SCOPE.md`) reuses this exact, already-working model-loading path
instead of duplicating it, per this project's standing "reuse existing tooling" convention.
`main()`'s CLI behavior is unchanged -- it now just calls these two functions internally.
"""

import os
import re
import time
import argparse
from typing import Optional

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen2_5_vl_3b_instruct")
)


_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def load_qwen_vl_model(
    model_path: str = DEFAULT_MODEL_DIR,
    device: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    dtype: Optional["torch.dtype"] = None,
    revision: Optional[str] = None,
):
    """Loads the Qwen2.5-VL-3B-Instruct checkpoint + processor. Returns (model, processor, device).

    **Dtype fix, 2026-09-18**: this used to hardcode `torch_dtype=torch.float16` regardless of
    device -- `docs/MULTI_HEAD_ARCHITECTURE_SPEC.md` S6 already flagged this as a real,
    findable mismatch (the checkpoint's own `config.json` declares `torch_dtype: bfloat16`,
    and bf16/fp16 have different exponent/mantissa splits -- forcing a bf16-trained checkpoint
    into fp16 is a plausible source of activation overflow) but left it unfixed since that doc
    was design-only. Fixed here: bfloat16 is used on both CUDA and CPU (matches the native
    checkpoint dtype in both cases; CPU fp16 also has weaker/less-tested op coverage than bf16
    in PyTorch, an independent reason to prefer bf16 off-GPU). `device` is auto-detected
    (`cuda` if available, else `cpu`) unless explicitly overridden.

    **2026-09-19 (Colab support), defaults unchanged**: `dtype` (default None -> bfloat16, the
    behavior above) lets a T4 (no native bf16) request fp16; `model_path` may now also be a
    Hugging Face repo id (`"Qwen/Qwen2.5-VL-3B-Instruct"`) when no such local path exists, and
    `revision` pins a commit for that case (ignored for local paths). The local checkpoint
    `models/qwen2_5_vl_3b_instruct` was downloaded at commit
    66285546d2b821cf421d4f5eb2576359d3770cd3 (from its `.cache/huggingface/download/*.metadata`),
    which is what the Colab sweep pins so weights match the local run.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_local = os.path.exists(model_path)
    if not is_local and not _HF_REPO_ID_RE.match(model_path):
        raise FileNotFoundError(
            f"Model path {model_path} does not exist and is not a Hugging Face repo id. Ensure "
            f"the checkpoint has been downloaded or pass an explicit model_path."
        )
    hub_kwargs = {"revision": revision} if (revision and not is_local) else {}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype if dtype is not None else torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        **hub_kwargs,
    )
    processor_kwargs = dict(hub_kwargs)
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    return model, processor, device


def generate_qwen_vl(
    model,
    processor,
    device: str,
    image,
    prompt: str,
    max_new_tokens: int = 128,
) -> dict:
    """Runs one prompt+image generate() call. `image` is a path (str) OR an in-memory
    RGB `np.ndarray` (qwen_vl_utils.process_vision_info accepts either via a temp-file
    round trip for the ndarray case, since its documented input contract is path/URL/base64/
    PIL, not raw ndarrays -- see the ndarray branch below, verified against the installed
    `qwen_vl_utils.process_vision_info` signature, not assumed). Returns a dict with
    `output_text`, `latency_s` (generate() wall-clock only, not preprocessing), and
    `peak_vram_gb` (None on CPU)."""
    image_arg = image
    _tmp_path = None
    if isinstance(image, np.ndarray):
        import cv2
        import tempfile

        fd, _tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(_tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_arg = _tmp_path

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_arg, "max_pixels": 1280 * 28 * 28},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(device)

        # 2026-09-19: the size of the image the vision tower ACTUALLY sees. Qwen2.5-VL emits
        # absolute coordinates in THIS space, not in the original frame's pixels (the processor
        # downsizes a 640x480 frame to 504x364 under this project's max_pixels; the smart-resize
        # is measured from the real processor output here, never assumed). Ignoring it made the
        # 2026-09-18 scorer read every answer ~27%/32% too large in x/y.
        model_input_hw = None
        grid = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
        if grid is not None and len(grid):
            patch = int(getattr(processor.image_processor, "patch_size", 14))
            model_input_hw = (int(grid[0][1]) * patch, int(grid[0][2]) * patch)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start_gen = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latency_s = time.time() - start_gen

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        peak_vram_gb = None
        if device == "cuda":
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return {"output_text": output_text, "latency_s": latency_s, "peak_vram_gb": peak_vram_gb,
                "model_input_hw": model_input_hw}
    finally:
        if _tmp_path is not None and os.path.exists(_tmp_path):
            os.remove(_tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Qwen2.5-VL-3B-Instruct on Jetson")
    parser.add_argument("--image", type=str, required=False, help="Path to a test image (e.g., from Track 4 session)")
    parser.add_argument("--camera_id", type=int, default=-1, help="USB camera ID to capture from (default: -1, meaning none unless --image is missing)")
    parser.add_argument("--prompt", type=str, default="Identify the ball's position or the nearest colored marker in this image.", help="Text prompt")
    parser.add_argument("--model_dir", type=str, default=None, help="Explicit path to the model directory (overrides relative path)")
    args = parser.parse_args()

    # Determine if we need to capture from a camera
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

    if args.model_dir:
        model_path = os.path.abspath(args.model_dir)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # If the script is accidentally inside qwen_venv, we need to go up one more level
        if os.path.basename(script_dir) == "qwen_venv":
            model_path = os.path.abspath(os.path.join(script_dir, "..", "..", "models", "qwen2_5_vl_3b_instruct"))
        else:
            model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "qwen2_5_vl_3b_instruct"))

    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        print("Please ensure the checkpoint has been downloaded or provide --model_dir.")
        return

    print(f"Loading Qwen2_5_VLForConditionalGeneration from {model_path} in BF16 (native checkpoint dtype)...")
    start_load = time.time()
    model, processor, device = load_qwen_vl_model(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds on device={device}.")

    print(f"Processing inputs (Image: {image_path})...")
    print("Running inference pass...")
    result = generate_qwen_vl(model, processor, device, image_path, args.prompt, max_new_tokens=128)

    print("\n" + "="*40)
    print("--- Generation Results ---")
    print(f"Wall-clock Latency: {result['latency_s']:.2f} seconds")

    if result["peak_vram_gb"] is not None:
        print(f"Peak VRAM allocated: {result['peak_vram_gb']:.2f} GB")

    print("\nGenerated Text:")
    print(result["output_text"])
    print("="*40)

if __name__ == "__main__":
    main()
