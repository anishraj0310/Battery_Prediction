"""
Export PyTorch model to ONNX.
Detects model type from checkpoint and exports accordingly.
"""

import torch
import torch.nn as nn
import json
import numpy as np
import argparse
import os

# Model definition (must match train.py)
class BatteryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

def export_onnx(ckpt_path, out_path, seq_len=6):
    # Load model
    model = torch.load(ckpt_path, weights_only=False)
    model.eval()

    # Load preprocessing to get input dim
    preproc_path = os.path.join(os.path.dirname(ckpt_path), "preproc_export.json")
    with open(preproc_path, "r") as f:
        preproc = json.load(f)

    # Input dimension: number of numeric features + number of categorical features (each encoded as single int)
    input_dim = len(preproc["num_features"]) + len(preproc["cat_features"]) if "num_features" in preproc and "cat_features" in preproc else 11

    # Create dummy input: [batch=1, seq_len, input_dim]
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=11,  # Compatible with onnxruntime-mobile
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model exported to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="PyTorch checkpoint path")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--seq_len", type=int, default=6)
    args = ap.parse_args()

    export_onnx(args.ckpt, args.out, args.seq_len)
