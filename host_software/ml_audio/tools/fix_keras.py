import zipfile
import json
import os
import tempfile
import shutil

keras_file = r'ml_audio\models\audio_command_classifier\best_classifier.keras'

# Extract everything to a temp dir
temp_dir = tempfile.mkdtemp()
with zipfile.ZipFile(keras_file, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Read config.json
config_path = os.path.join(temp_dir, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config_data = json.load(f)

# Recursively remove input_axes and output_axes
def remove_bad_keys(d):
    if isinstance(d, dict):
        d.pop('input_axes', None)
        d.pop('output_axes', None)
        d.pop('renorm', None)
        d.pop('renorm_clipping', None)
        d.pop('renorm_momentum', None)
        d.pop('quantization_config', None)
        for k, v in d.items():
            remove_bad_keys(v)
    elif isinstance(d, list):
        for item in d:
            remove_bad_keys(item)

remove_bad_keys(config_data)

# Write it back
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_data, f)

# Re-zip
new_keras = keras_file + '.fixed'
with zipfile.ZipFile(new_keras, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir)
            zip_ref.write(file_path, arcname)

shutil.rmtree(temp_dir)

# Overwrite old file
os.replace(new_keras, keras_file)
print("Successfully fixed best_classifier.keras!")
