import re
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# PyTorch Definition of your STM32 Actor Network
# ---------------------------------------------------------------------------
class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = torch.clamp(x, -1.0, 1.0)  # Matches actor action clipping [-1, 1]
        return x


def parse_c_array(header_content, array_name):
    """Extracts floating point values from a C array definition in weights.h"""
    pattern = rf"{array_name}\[\]\s*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, header_content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array {array_name} in weights.h")
    
    # Extract numbers (handles scientific notation like -1.29903719e-04f)
    raw_str = match.group(1)
    cleaned_str = re.sub(r'[fF]', '', raw_str) # strip C float 'f' suffixes
    values = [float(val) for val in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', cleaned_str)]
    return np.array(values, dtype=np.float32)


def convert_h_to_pth(header_path="weights.h", output_path="control_model.pth"):
    with open(header_path, "r") as f:
        content = f.read()

    # Parse raw 1D float arrays from C header
    w1_raw = parse_c_array(content, "NN_W1") # Shape in C: (9, 32)
    b1_raw = parse_c_array(content, "NN_B1") # Shape in C: (32,)
    w2_raw = parse_c_array(content, "NN_W2") # Shape in C: (32, 32)
    b2_raw = parse_c_array(content, "NN_B2") # Shape in C: (32,)
    w3_raw = parse_c_array(content, "NN_W3") # Shape in C: (32, 3)
    b3_raw = parse_c_array(content, "NN_B3") # Shape in C: (3,)

    # C++ loops perform: acc += obs[i] * NN_W1[i * 32 + j]
    # In PyTorch, linear layer weight is expected in shape (out_features, in_features).
    # Therefore, C++ shape (in, out) must be transposed to (out, in).
    w1 = w1_raw.reshape(9, 32).T
    w2 = w2_raw.reshape(32, 32).T
    w3 = w3_raw.reshape(32, 3).T

    # Instantiate model and load parameters
    model = ControlNet()
    state_dict = {
        "fc1.weight": torch.tensor(w1, dtype=torch.float32),
        "fc1.bias": torch.tensor(b1_raw, dtype=torch.float32),
        "fc2.weight": torch.tensor(w2, dtype=torch.float32),
        "fc2.bias": torch.tensor(b2_raw, dtype=torch.float32),
        "fc3.weight": torch.tensor(w3, dtype=torch.float32),
        "fc3.bias": torch.tensor(b3_raw, dtype=torch.float32),
    }

    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), output_path)
    print(f"Successfully converted '{header_path}' -> '{output_path}'")


if __name__ == "__main__":
    convert_h_to_pth()