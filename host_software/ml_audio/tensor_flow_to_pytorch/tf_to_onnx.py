import tensorflow as tf
import tf2onnx

# 1. Load your existing Keras model
keras_model_path = 'your_model.keras' # or 'your_model.h5'
keras_model = tf.keras.models.load_model(keras_model_path)

# 2. Convert the model to ONNX
# tf2onnx can usually infer the input signature directly from the Keras model
onnx_model, _ = tf2onnx.convert.from_keras(
    keras_model, 
    output_path="converted_model.onnx"
)

print("Keras model successfully saved as converted_model.onnx")