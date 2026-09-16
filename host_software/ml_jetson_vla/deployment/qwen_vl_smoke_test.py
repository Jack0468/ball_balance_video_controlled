import os
import time
import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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

    print(f"Loading Qwen2_5_VLForConditionalGeneration from {model_path} in FP16...")
    start_load = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")

    print(f"Processing inputs (Image: {image_path})...")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image_path,
                    "max_pixels": 1280 * 28 * 28, # Cap resolution to ~1MP to prevent massive hangs
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    print("Running inference pass...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_gen = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
    gen_time = time.time() - start_gen
    
    # Trim the prompt from the generated IDs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*40)
    print("--- Generation Results ---")
    print(f"Wall-clock Latency: {gen_time:.2f} seconds")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM allocated: {peak_mem:.2f} GB")
        
    print("\nGenerated Text:")
    print(output_text[0])
    print("="*40)

if __name__ == "__main__":
    main()
