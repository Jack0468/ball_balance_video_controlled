import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

try:
    from .audio_command_classifier_pytorch import load_audio_command_classifier
except ImportError:
    from audio_command_classifier_pytorch import load_audio_command_classifier


def to_tensor(array_like):
    return torch.as_tensor(np.asarray(array_like), dtype=torch.float32)


def build_state_dict_from_keras(keras_model):
    norm_layer = keras_model.get_layer("normalization")
    conv1 = keras_model.get_layer("conv2d")
    bn1 = keras_model.get_layer("batch_normalization")
    conv2 = keras_model.get_layer("conv2d_1")
    bn2 = keras_model.get_layer("batch_normalization_1")
    conv3 = keras_model.get_layer("conv2d_2")
    bn3 = keras_model.get_layer("batch_normalization_2")
    dense = keras_model.get_layer("dense")

    norm_mean, norm_variance, _ = norm_layer.get_weights()
    conv1_kernel, conv1_bias = conv1.get_weights()
    bn1_gamma, bn1_beta, bn1_mean, bn1_var = bn1.get_weights()
    conv2_kernel, conv2_bias = conv2.get_weights()
    bn2_gamma, bn2_beta, bn2_mean, bn2_var = bn2.get_weights()
    conv3_kernel, conv3_bias = conv3.get_weights()
    bn3_gamma, bn3_beta, bn3_mean, bn3_var = bn3.get_weights()
    dense_kernel, dense_bias = dense.get_weights()

    return OrderedDict(
        norm_mean=to_tensor(norm_mean),
        norm_variance=to_tensor(norm_variance),
        conv1_weight=to_tensor(np.transpose(conv1_kernel, (3, 2, 0, 1))),
        conv1_bias=to_tensor(conv1_bias),
        bn1_gamma=to_tensor(bn1_gamma),
        bn1_beta=to_tensor(bn1_beta),
        bn1_mean=to_tensor(bn1_mean),
        bn1_var=to_tensor(bn1_var),
        conv2_weight=to_tensor(np.transpose(conv2_kernel, (3, 2, 0, 1))),
        conv2_bias=to_tensor(conv2_bias),
        bn2_gamma=to_tensor(bn2_gamma),
        bn2_beta=to_tensor(bn2_beta),
        bn2_mean=to_tensor(bn2_mean),
        bn2_var=to_tensor(bn2_var),
        conv3_weight=to_tensor(np.transpose(conv3_kernel, (3, 2, 0, 1))),
        conv3_bias=to_tensor(conv3_bias),
        bn3_gamma=to_tensor(bn3_gamma),
        bn3_beta=to_tensor(bn3_beta),
        bn3_mean=to_tensor(bn3_mean),
        bn3_var=to_tensor(bn3_var),
        dense_weight=to_tensor(np.transpose(dense_kernel, (1, 0))),
        dense_bias=to_tensor(dense_bias),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export the trained Keras audio model to PyTorch checkpoint and TorchScript files."
    )
    parser.add_argument(
        "--keras-model",
        default=r"c:\Users\aritr\Downloads\ball_balance_video_controlled\host_software\ml_audio\models\audio_command_classifier\best_classifier.keras",
        help="Path to the trained Keras model file.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"c:\Users\aritr\Downloads\ball_balance_video_controlled\host_software\ml_audio\models\audio_command_classifier\pytorch",
        help="Directory where exported PyTorch files will be written.",
    )
    args = parser.parse_args()

    keras_model_path = Path(args.keras_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keras_model = tf.keras.models.load_model(keras_model_path)
    state_dict = build_state_dict_from_keras(keras_model)

    model = load_audio_command_classifier()
    model.load_state_dict(state_dict)
    model.eval()

    state_dict_path = output_dir / "audio_command_classifier_state_dict.pth"
    torchscript_path = output_dir / "audio_command_classifier.pt"

    torch.save(state_dict, state_dict_path)

    scripted_model = torch.jit.script(model)
    scripted_model.save(str(torchscript_path))

    print(f"Loaded Keras model: {keras_model_path}")
    print(f"Saved state_dict checkpoint: {state_dict_path}")
    print(f"Saved TorchScript model: {torchscript_path}")


if __name__ == "__main__":
    main()