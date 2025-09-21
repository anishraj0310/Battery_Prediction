"""
Test ONNX inference with the quantized model.
"""

import onnxruntime as ort
import numpy as np
import json

# Load model and preprocessing
ort_session = ort.InferenceSession("runs/gru/model.int8.onnx")
with open("runs/gru/preproc_export.json", "r") as f:
    preproc = json.load(f)

# Sample features
features = {
    "battery_pct": 60.0,
    "screen_on_hours": 2.0,
    "brightness": 0.8,
    "temperature_C": 25.0,
    "capacity_health": 0.95,
    "location_lat": 28.61,
    "location_lon": 77.21,
    "app_active": "YouTube",  # This will be encoded
    "network_active": 1,
    "fast_charge": 0,
    "is_weekend": 0
}

# Preprocess input (similar to training)
input_vec = []
# Numeric scaling
for i, feat in enumerate(preproc["num_features"]):
    scale = preproc["num_scales"][i]
    min_val = preproc["num_mins"][i]
    val = (features[feat] - min_val) / scale if scale != 0 else 0
    input_vec.append(val)

# Categorical encoding (single int)
for feat in preproc["cat_features"]:
    categories = preproc["cat_categories"][preproc["cat_features"].index(feat)]
    if feat == "app_active":
        # find index
        idx = categories.index(features[feat]) if features[feat] in categories else 0
        input_vec.append(idx)
    else:
        input_vec.append(features[feat])

# Create sequence (repeat for 6 timesteps)
sequence = [input_vec] * 6
input_array = np.array(sequence, dtype=np.float32).reshape(1, 6, -1)

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: input_array}
ort_outs = ort_session.run(None, ort_inputs)

delta_pct, tte_hours = ort_outs[0][0]
print(f"Predicted battery drop: {delta_pct:.2f}%, Time to 5%: {tte_hours:.1f} hours")
