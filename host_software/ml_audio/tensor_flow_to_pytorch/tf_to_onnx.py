from pathlib import Path

import tensorflow as tf
import tf2onnx


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (SCRIPT_DIR.parent / "models" / "audio_command_classifier" / "best_classifier.keras").resolve()
OUTPUT_PATH = (SCRIPT_DIR / "audio_command_classifier.onnx").resolve()


def main():
    keras_model = tf.keras.models.load_model(MODEL_PATH)

    tf2onnx.convert.from_keras(
        keras_model,
        output_path=str(OUTPUT_PATH),
    )

    print(f"Loaded Keras model from: {MODEL_PATH}")
    print(f"Saved ONNX model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()