import argparse
import subprocess
import re
import numpy as np
import tensorflow as tf

LABELS = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--binary", required=True)
parser.add_argument("--input", default="test_input_64x64.txt")
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)

x64 = np.loadtxt(args.input, dtype=np.float32).reshape(1, 64, 64, 1)

# C++ starts from the already-resized 64x64 input.
# So Python must skip the Keras resizing layer and start from normalization.
y = x64
for layer in model.layers[1:]:
    y = layer(y, training=False)

python_logits = np.asarray(y).reshape(-1)

cpp_result = subprocess.run(
    [args.binary],
    capture_output=True,
    text=True,
    check=True
)

print(cpp_result.stdout)

numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", cpp_result.stdout)
cpp_logits = np.array([float(v) for v in numbers[-6:]], dtype=np.float32)

print("Python logits:", python_logits)
print("C++ logits:", cpp_logits)
print("Absolute diff:", np.abs(python_logits - cpp_logits))
print("L2 diff:", np.linalg.norm(python_logits - cpp_logits))
print("Python predicted:", LABELS[int(np.argmax(python_logits))])
print("C++ predicted:", LABELS[int(np.argmax(cpp_logits))])
print("Same prediction:", int(np.argmax(python_logits)) == int(np.argmax(cpp_logits)))