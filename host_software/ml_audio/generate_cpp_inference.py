# generate_cpp_inference_full.py

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

def write_array(f, name, arr):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    f.write(f"static const float {name}[] = {{\n")
    for i in range(0, len(arr), 8):
        f.write("    " + ", ".join(f"{x:.9g}f" for x in arr[i:i+8]) + ",\n")
    f.write("};\n\n")

def write_conv(f, prefix, layer):
    kernel, bias = layer.get_weights()
    f.write(f"// {prefix} kernel shape {kernel.shape}\n")
    write_array(f, f"{prefix}_KERNEL", kernel)
    write_array(f, f"{prefix}_BIASES", bias)
    f.write(f"static const int {prefix}_KERNEL_K = {kernel.shape[0]};\n")
    f.write(f"static const int {prefix}_KERNEL_W = {kernel.shape[1]};\n")
    f.write(f"static const int {prefix}_IN_CH = {kernel.shape[2]};\n")
    f.write(f"static const int {prefix}_OUT_CH = {kernel.shape[3]};\n\n")

def write_bn(f, prefix, layer):
    gamma, beta, mean, var = layer.get_weights()
    write_array(f, f"{prefix}_GAMMA", gamma)
    write_array(f, f"{prefix}_BETA", beta)
    write_array(f, f"{prefix}_MOVING_MEAN", mean)
    write_array(f, f"{prefix}_MOVING_VARIANCE", var)
    f.write(f"static const float {prefix}_EPSILON = {layer.epsilon:.9g}f;\n\n")

def write_norm(f, prefix, layer):
    weights = layer.get_weights()
    mean = weights[0]
    var = weights[1]
    write_array(f, f"{prefix}_MEAN", mean)
    write_array(f, f"{prefix}_VARIANCE", var)
    f.write(f"static const float {prefix}_EPSILON = {layer.epsilon:.9g}f;\n\n")

def write_dense(f, prefix, layer):
    kernel, bias = layer.get_weights()
    write_array(f, f"{prefix}_KERNEL", kernel)
    write_array(f, f"{prefix}_BIASES", bias)
    f.write(f"static const int {prefix}_IN = {kernel.shape[0]};\n")
    f.write(f"static const int {prefix}_OUT = {kernel.shape[1]};\n\n")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

with open(outdir / "audio_model_weights.h", "w") as f:
    f.write("// Auto-generated weights header\n")
    f.write("#pragma once\n#include <cstddef>\n\n")

    conv_id = 0
    dense_id = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            write_norm(f, layer.name.upper(), layer)

        elif isinstance(layer, tf.keras.layers.Conv2D):
            name = "CONV2D" if conv_id == 0 else f"CONV2D_{conv_id}"
            write_conv(f, name, layer)
            conv_id += 1

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            write_bn(f, layer.name.upper(), layer)

        elif isinstance(layer, tf.keras.layers.Dense):
            name = "DENSE" if dense_id == 0 else f"DENSE_{dense_id}"
            write_dense(f, name, layer)
            dense_id += 1

print("Updated audio_model_weights.h generated.")