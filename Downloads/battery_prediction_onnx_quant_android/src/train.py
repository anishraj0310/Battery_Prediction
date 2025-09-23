"""
Train a GRU model for battery prediction.
Saves model to runs/gru/latest.pt and preprocessing params to runs/gru/preproc_export.json.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler

# Model definition
class BatteryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# Data preprocessing
def preprocess_data(csv_path, seq_len=6):
    df = pd.read_csv(csv_path)

    # Numeric features
    num_features = ["battery_pct", "screen_on_hours", "brightness", "temperature_C", "capacity_health", "location_lat", "location_lon"]
    scalers = {}
    for feat in num_features:
        scaler = MinMaxScaler()
        df[feat] = scaler.fit_transform(df[[feat]])
        scalers[feat] = {"min": scaler.min_[0], "scale": scaler.scale_[0]}

    # Categorical features
    cat_features = ["app_active", "network_active", "fast_charge", "is_weekend"]
    cat_categories = {}
    for feat in cat_features:
        cats = df[feat].unique().tolist()
        cat_categories[feat] = cats

    # Convert to numeric
    for feat in cat_features:
        df[feat] = df[feat].astype("category").cat.codes

    # Targets: delta_next_hour_pct, tte_hours, energy_saved_by_alert
    targets = ["delta_next_hour_pct", "tte_5pct_hours", "energy_saved_by_alert"]

    # Create sequences
    sequences = []
    labels = []
    for i in range(len(df) - seq_len):
        seq = []
        for t in range(seq_len):
            row = df.iloc[i + t]
            vec = [row[f] for f in num_features + cat_features]
            seq.append(vec)
        label = [df.iloc[i + seq_len][t] for t in targets]
        sequences.append(seq)
        labels.append(label)

    # Save preprocessing params
    os.makedirs("runs/gru", exist_ok=True)
    with open("runs/gru/preproc_export.json", "w") as f:
        json.dump({
            "num_features": num_features,
            "num_mins": [scalers[f]["min"] for f in num_features],
            "num_scales": [scalers[f]["scale"] for f in num_features],
            "cat_features": cat_features,
            "cat_categories": [cat_categories[f] for f in cat_features]
        }, f)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Training loop
def train_model(data_csv, seq_len=6, epochs=5, batch_size=32, hidden_dim=64):
    X, y = preprocess_data(data_csv, seq_len)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[-1]
    output_dim = y.shape[-1]
    model = BatteryPredictor(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model, "runs/gru/latest.pt")
    print("Model saved to runs/gru/latest.pt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV data path")
    ap.add_argument("--model", default="gru")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    train_model(args.data, epochs=args.epochs)
