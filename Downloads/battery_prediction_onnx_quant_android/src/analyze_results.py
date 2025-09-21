"""
Analyze battery prediction model results for research paper.
"""
import pandas as pd
import numpy as np
import json
import onnxruntime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_preprocess(csv_path, preproc_path, seq_len=6):
    # Load preprocessing params
    with open(preproc_path) as f:
        preproc = json.load(f)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Preprocess features
    for feat, min_val, scale in zip(preproc["num_features"], preproc["num_mins"], preproc["num_scales"]):
        df[feat] = (df[feat] - min_val) / scale
    
    # Convert categorical
    for feat, cats in zip(preproc["cat_features"], preproc["cat_categories"]):
        df[feat] = pd.Categorical(df[feat], categories=cats).codes
    
    return df

def get_predictions(model_path, df, preproc_path, seq_len=6):
    # Load preprocessing params
    with open(preproc_path) as f:
        preproc = json.load(f)
    
    # Initialize ONNX Runtime
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Create sequences
    features = preproc["num_features"] + preproc["cat_features"]
    sequences = []
    actuals = []
    
    for i in range(len(df) - seq_len):
        seq = []
        for t in range(seq_len):
            row = df.iloc[i + t]
            vec = [row[f] for f in features]
            seq.append(vec)
        actuals.append([
            df.iloc[i + seq_len]["delta_next_hour_pct"],
            df.iloc[i + seq_len]["tte_5pct_hours"]
        ])
        sequences.append(seq)
    
    # Get predictions
    inputs = np.array(sequences, dtype=np.float32)
    predictions = session.run(None, {"input": inputs})[0]
    
    return np.array(predictions), np.array(actuals)

# Load data and get predictions
df = load_and_preprocess("data/synthetic_sessions.csv", "runs/gru/preproc_export.json")
predictions, actuals = get_predictions("runs/gru/model.onnx", df, "runs/gru/preproc_export.json")

# Calculate metrics
metrics = {
    "Battery Drop Rate (Next Hour)": {
        "MSE": mean_squared_error(actuals[:, 0], predictions[:, 0]),
        "MAE": mean_absolute_error(actuals[:, 0], predictions[:, 0]),
        "R2": r2_score(actuals[:, 0], predictions[:, 0])
    },
    "Time to 5% Battery": {
        "MSE": mean_squared_error(actuals[:, 1], predictions[:, 1]),
        "MAE": mean_absolute_error(actuals[:, 1], predictions[:, 1]),
        "R2": r2_score(actuals[:, 1], predictions[:, 1])
    }
}

# Print metrics table
print("\nModel Performance Metrics:")
print("-" * 50)
for target, metric_dict in metrics.items():
    print(f"\n{target}:")
    for metric_name, value in metric_dict.items():
        print(f"{metric_name}: {value:.4f}")

# Print sample predictions
print("\nSample Predictions (First 5 records):")
print("-" * 50)
print("\nBattery Drop Rate Predictions (% per hour):")
for i in range(5):
    print(f"Actual: {actuals[i][0]:6.2f} | Predicted: {predictions[i][0]:6.2f}")

print("\nTime to 5% Battery Predictions (hours):")
for i in range(5):
    print(f"Actual: {actuals[i][1]:6.2f} | Predicted: {predictions[i][1]:6.2f}")
