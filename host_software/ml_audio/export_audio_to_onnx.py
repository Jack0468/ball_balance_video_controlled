import os
import sys
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
# Add ml_audio to path so we can import AudioCommandClassifier
if script_dir not in sys.path:
    sys.path.append(script_dir)

from audio_command_classifier_pytorch import AudioCommandClassifier

def export_audio_model():
    model_path = os.path.join(script_dir, "models", "pytorch_v3", "audio_command_classifier_state_dict_v3.pth")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading {model_path}...")
    # It has 12 classes
    model = AudioCommandClassifier(num_classes=12)
    device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input of shape [1, 155, 128]
    # batch=1, frames=155, freqs=128
    dummy_input = torch.randn(1, 155, 128)

    output_path = os.path.join(script_dir, "models", "audio_command_classifier_v3.onnx")
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Audio model exported successfully!")

if __name__ == "__main__":
    export_audio_model()
