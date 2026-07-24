import argparse
from pathlib import Path

import torch

from audio_command_classifier_pytorch import load_audio_command_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Export the embedded audio PyTorch model to checkpoint and TorchScript files."
    )
    parser.add_argument(
        "--output-dir",
        default=r"c:\Users\aritr\Downloads\ball_balance_video_controlled\host_software\ml_audio\models\audio_command_classifier\pytorch",
        help="Directory where exported PyTorch files will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_audio_command_classifier()
    model.eval()

    state_dict_path = output_dir / "audio_command_classifier_state_dict.pth"
    torchscript_path = output_dir / "audio_command_classifier.pt"

    torch.save(model.state_dict(), state_dict_path)

    scripted_model = torch.jit.script(model)
    scripted_model.save(str(torchscript_path))

    print(f"Saved state_dict checkpoint: {state_dict_path}")
    print(f"Saved TorchScript model: {torchscript_path}")


if __name__ == "__main__":
    main()