from pathlib import Path

import torch
from onnx2torch import convert


SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = (SCRIPT_DIR / "audio_command_classifier.onnx").resolve()
STATE_DICT_PATH = (SCRIPT_DIR / "audio_command_classifier_from_onnx.pth").resolve()


def main():
	pytorch_model = convert(str(ONNX_MODEL_PATH))
	pytorch_model.eval()

	torch.save(pytorch_model.state_dict(), STATE_DICT_PATH)
	print(f"Loaded ONNX model from: {ONNX_MODEL_PATH}")
	print(f"Saved PyTorch state_dict to: {STATE_DICT_PATH}")

	# Keras input shape is (1, 155, 129, 1) before any internal transposition.
	dummy_input = torch.randn(1, 155, 129, 1)
	output = pytorch_model(dummy_input)
	print("Test inference successful. Output shape:", tuple(output.shape))


if __name__ == "__main__":
	main()