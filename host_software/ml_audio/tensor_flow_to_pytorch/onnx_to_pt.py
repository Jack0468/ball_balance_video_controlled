import torch
from onnx2torch import convert

# 1. Convert the ONNX model to a PyTorch nn.Module
onnx_model_path = 'converted_model.onnx'
pytorch_model = convert(onnx_model_path)

# 2. Put the model in evaluation mode
pytorch_model.eval()

# 3. Save the PyTorch model (saving the state_dict is best practice)
torch.save(pytorch_model.state_dict(), 'pytorch_weights.pth')
print("ONNX model successfully converted to PyTorch!")

# --- Optional: Test the conversion ---
# Create a dummy tensor matching your model's expected input shape
# Note: PyTorch expects channels-first format (N, C, H, W) 
# whereas Keras usually uses channels-last (N, H, W, C). 
# The conversion process usually handles this transposition internally, 
# but pass your dummy input in the format Keras originally expected.
dummy_input = torch.randn(1, 224, 224, 3) 
output = pytorch_model(dummy_input)
print("Test inference successful. Output shape:", output.shape)