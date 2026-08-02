import os
import sys

# Ensure we can import our local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import tensorflow as tf
import openvino as ov
from ultralytics import YOLO


def export_yolo():
    print("Exporting YOLO model to OpenVINO...")
    yolo_path = os.path.join(
        "..", "models", "yolov8_platform_markers_v2", "weights", "best.pt"
    )
    if not os.path.exists(yolo_path):
        print(f"YOLO model not found at {yolo_path}")
        return
    model = YOLO(yolo_path)
    model.export(format="openvino")
    print("YOLO export complete.")


def export_mlp_corrector_iphone_v1():
    print("Exporting Corrector MLP to OpenVINO...")
    mlp_corrector_iphone_v1_path = os.path.join(
        "..", "models", "mlp_corrector_iphone_v1", "best_mlp_corrector_iphone_v1.pth"
    )
    if not os.path.exists(mlp_corrector_iphone_v1_path):
        print(f"Corrector model not found at {mlp_corrector_iphone_v1_path}")
        return

    # We must import CorrectorMLP dynamically if src.models isn't enough, but it should be
    # Actually wait, src.models loads from ml_vision.core.mlp_corrector_iphone_v1_mlp
    from ml_vision.core.mlp_corrector_iphone_v1_mlp import CorrectorMLP as CMLP

    model = CMLP(input_dim=14, hidden_dim=128, output_dim=2)
    model.load_state_dict(torch.load(mlp_corrector_iphone_v1_path, map_location="cpu"))
    model.eval()

    # Convert to OpenVINO
    example_input = torch.randn(1, 14)
    ov_model = ov.convert_model(model, example_input=example_input)

    output_path = os.path.join(
        "..", "models", "mlp_corrector_iphone_v1", "best_mlp_corrector_iphone_v1.xml"
    )
    ov.save_model(ov_model, output_path)
    print(f"Corrector export complete: {output_path}")


def export_audio():
    print("Exporting Audio Classifier to OpenVINO...")
    audio_path = os.path.join(
        "..",
        "..",
        "ml_audio",
        "models",
        "audio_command_classifier",
        "best_classifier.keras",
    )
    if not os.path.exists(audio_path):
        print(f"Audio model not found at {audio_path}")
        return

    # Load Keras model
    tf_model = tf.keras.models.load_model(audio_path, compile=False)

    # Convert to OpenVINO with input shape
    ov_model = ov.convert_model(tf_model, input=[1, 155, 129, 1])

    output_path = os.path.join(
        "..",
        "..",
        "ml_audio",
        "models",
        "audio_command_classifier",
        "best_classifier.xml",
    )
    ov.save_model(ov_model, output_path)
    print(f"Audio export complete: {output_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    export_yolo()
    export_mlp_corrector_iphone_v1()
    export_audio()
    print("All exports finished successfully!")
