from ultralytics import YOLO
import os
import torch


def export_model(model_path, is_local_path=True):
    print(f"Loading YOLO model: {model_path}...")

    if is_local_path:
        resolved_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), model_path)
        )
    else:
        resolved_path = model_path

    model = YOLO(resolved_path)

    print(f"Exporting YOLO to ONNX format...")
    output_path = model.export(format="onnx", opset=12, dynamic=False)
    print(f"Success! YOLO ONNX model saved to: {output_path}\n")


def export_cnn_tracker(model_path):
    print(f"Loading CNN Tracker model: {model_path}...")
    import sys

    training_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../training")
    )
    if training_dir not in sys.path:
        sys.path.append(training_dir)
    from basic_cnn import BasicCNN

    resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    model = BasicCNN(num_outputs=2)
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 3, 240, 320)
    output_path = resolved_path.replace(".pth", ".onnx")

    print(f"Exporting CNN Tracker to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Success! CNN ONNX model saved to: {output_path}\n")


def export_resnet_tracker(model_path, arch="resnet18"):
    print(f"Loading ResNet Tracker model: {model_path}...")
    import sys
    import torch.nn as nn
    from torchvision import models

    resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))

    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 3, 240, 320)
    output_path = resolved_path.replace(".pth", ".onnx")

    print(f"Exporting ResNet Tracker to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Success! ResNet ONNX model saved to: {output_path}\n")

def export_mlp_corrector(model_path):
    print(f"Loading MLP Time Corrector model: {model_path}...")
    import sys

    training_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../training")
    )
    if training_dir not in sys.path:
        sys.path.append(training_dir)
    from train_mlp_corrector_time import MLPCorrectorTime

    resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    model = MLPCorrectorTime(window_size=1)
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 1 frame * 5 features [cnn_x, cnn_y, target_x, target_y, dt]
    dummy_input = torch.randn(1, 5)
    output_path = resolved_path.replace(".pth", ".onnx")

    print(f"Exporting MLP Corrector to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Success! MLP ONNX model saved to: {output_path}\n")


def main():
    print("--- Model to ONNX Exporter ---\n")

    # 1. Export the standard YOLOv8n Ball Tracker
    # export_model("../models/base_models/yolov8n/weights/yolov8n.pt", is_local_path=True)

    # 2. Export our custom YOLO-Pose model
    export_model("../models/yolov8_platform_pose_markers_0728_v4/weights/best.pt", is_local_path=True)

    # 3. Export ResNet Expert Tracker
    export_resnet_tracker("../models/resnet18_expert_tracker_0728_v6/expert_tracker_best.pth", arch="resnet18")

    # 4. Export CNN Tracker v3
    export_cnn_tracker("../models/cnn_2d_tracker_0730_v3/expert_tracker_best.pth")

    # 5. Export MLP Corrector Time varuco_v1
    export_mlp_corrector(
        "../models/mlp_corrector_time_varuco_0730_v1/mlp_corrector_best.pth"
    )

    print(
        "All models exported successfully! They are ready to be used with onnxruntime or TensorRT."
    )


if __name__ == "__main__":
    main()
