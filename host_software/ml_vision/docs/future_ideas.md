# ML Vision: Future Ideas & Optimizations

This document serves as a repository for future concepts, optimizations, and experimental features for the VRI_2026 computer vision pipeline. 

## Synthetic Data V2: Advanced Background Randomization

Currently, generating synthetic data from a limited number of base images can cause machine learning models (like YOLO-Pose) to overfit to the background texture (e.g., the wood-grain table) rather than learning the actual shape of the balancing platform.

To create a more robust "Version 2" synthetic dataset, we can implement aggressive background replacement during the data augmentation pipeline:

1. **Polygon Extraction**: Use the known corner coordinates in the base images to create a mathematical polygon mask and extract *only* the platform pixels.
2. **Background Substitution**: Paste the extracted platform onto thousands of randomly selected background images (e.g., from the COCO dataset, floor textures, or heavily randomized noise).
3. **Aggressive Color & Lighting**: Apply severe shadow, brightness, and contrast augmentations directly to the platform to simulate different room environments.

**Why we aren't doing this yet:**
While this provides an incredibly robust, software-only solution, it requires a significant development effort to write the background substitution pipeline, sourcing varied background datasets, and requires another 16+ hour training cycle for the models. We have opted for a more lightweight Two-Stage Bounding Box approach for our current iteration.

## Full YOLO Bounding Box Training

Our current Two-Stage Bounding Box pipeline relies on a `YOLOv8n` model to identify the platform area before classical CV processing. Currently, the model in production (`platform_bbox_model-4`) was fine-tuned extremely quickly (1 epoch on 32 images) as a rapid proof of concept. 

To achieve maximum robustness under different lighting conditions, room environments, and partial occlusions, we should perform a **Full YOLO Train**:
1. Run `augment_dataset.py` with `total_images=2000` to generate a comprehensive bounding box dataset.
2. Edit `train_platform_detector.py` to use `epochs=50` and `imgsz=640`.
3. Train the model using a GPU instance for maximum accuracy.
4. Export the resulting model to `ONNX` or `OpenVINO` to achieve higher FPS.
