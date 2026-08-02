import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to YOLO model")
    args = parser.parse_args()

    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the best trained YOLO model weights
    model_path = os.path.abspath(os.path.join(script_dir, args.model_path))

    # Path to the dataset configuration file we want to evaluate on
    data_path = os.path.abspath(
        os.path.join(script_dir, "../data_processing/raw_dataset.yaml")
    )

    print(f"Evaluating model: {model_path}")
    print(f"On dataset: {data_path}")

    # Load model
    model = YOLO(model_path)

    # Run evaluation
    # This will automatically print the precision/recall tables to standard out,
    # and save detailed plots, confusion matrices, and metrics to yolo_eval_results/<model_basename>
    eval_project_dir = os.path.abspath(os.path.join(script_dir, "yolo_eval_results"))

    # Extract the base model name (e.g. 'yolov8_marker_ball_v1') from the model path
    # model_path is typically something like models/yolov8_marker_ball_v1/weights/best.pt
    model_basename = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    if not model_basename or model_basename == ".":
        model_basename = "val"

    metrics = model.val(
        data=data_path, imgsz=640, project=eval_project_dir, name=model_basename
    )


if __name__ == "__main__":
    main()
