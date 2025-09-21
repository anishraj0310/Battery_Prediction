"""
Quantize an ONNX model for smaller size and faster CPU inference.

Supports:
- dynamic (weight-only INT8)  -> widely compatible, no calibration needed
- qlinear (static, Q/DQ ops) -> INT8 activations + weights using simple calibration from CSV

Examples:
  # Dynamic (recommended first pass)
  python src/quantize_onnx.py --model runs/gru/model.onnx --out runs/gru/model.int8.onnx --mode dynamic

  # Static QLinear using calibration from your CSV (matching training columns)
  python src/quantize_onnx.py --model runs/gru/model.onnx --out runs/gru/model.qlinear.onnx \
      --mode qlinear --preproc runs/gru/preproc_export.json --calib_csv data/synthetic_sessions.csv --num-samples 2000 --seq-len 6
"""

import argparse, os, json, csv, random
import numpy as np

from onnxruntime.quantization import (
    quantize_dynamic, quantize_static, CalibrationDataReader,
    QuantType, CalibrationMethod
)

class _SeqDataReader(CalibrationDataReader):
    """
    Minimal calibration reader for our seq model with a single input named 'input' of shape [B,T,x_dim].
    Generates batches from preprocessed CSV rows using the same preprocessing json.
    """
    def __init__(self, csv_path: str, preproc_json: str, batch_size: int = 16, seq_len: int = 6, num_samples: int = 1024):
        with open(preproc_json, "r") as f:
            self.P = json.load(f)
        self.seq_len = seq_len
        self.bs = batch_size
        # load a subset for calibration
        rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for i,row in enumerate(reader):
                rows.append(row)
                if len(rows) >= num_samples: break
        random.shuffle(rows)
        self.samples = [self._vectorize(r) for r in rows]
        self._i = 0

    def _vectorize(self, row: dict):
        # numeric
        nums = []
        for k,mean,scale in zip(self.P["num_features"], self.P["num_means"], self.P["num_scales"]):
            v = float(row[k])
            z = (v - mean) / (scale if scale != 0 else 1.0)
            nums.append(z)
        # categorical (exact match on strings/ints as CSV provides)
        cats = []
        for feat, cats_list in zip(self.P["cat_features"], self.P["cat_categories"]):
            v = row[feat]
            try:
                # convert to int if the category list is ints and CSV field looks like int
                if all(isinstance(c, (int, np.integer)) for c in cats_list):
                    v = int(v)
            except: pass
            for c in cats_list:
                cats.append(1.0 if v == c else 0.0)
        x = np.array(nums + cats, dtype=np.float32)[None, self.seq_len, :]  # [1,xdim] placeholder
        x = np.repeat(x, self.seq_len, axis=1).reshape(1, self.seq_len, -1) # [1,T,xdim]
        return x

    def get_next(self):
        if self._i >= len(self.samples):
            return None
        # build batch
        xs = []
        for _ in range(self.bs):
            if self._i >= len(self.samples): break
            xs.append(self.samples[self._i]); self._i += 1
        batch = np.concatenate(xs, axis=0) if len(xs)>1 else xs[0]
        return {"input": batch}

def quant_dynamic(in_path: str, out_path: str):
    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    print(f"[OK] Dynamic quantized model -> {out_path}")

def quant_qlinear(in_path: str, out_path: str, preproc_json: str, calib_csv: str, num_samples: int, seq_len: int):
    reader = _SeqDataReader(calib_csv, preproc_json, batch_size=16, seq_len=seq_len, num_samples=num_samples)
    quantize_static(
        model_input=in_path,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format="QOperator",  # QLinear ops
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Entropy,
    )
    print(f"[OK] QLinear (static) quantized model -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fp32 ONNX")
    ap.add_argument("--out", required=True, help="Path to output INT8 ONNX")
    ap.add_argument("--mode", choices=["dynamic","qlinear"], default="dynamic")
    ap.add_argument("--preproc", help="preproc_export.json (required for qlinear)")
    ap.add_argument("--calib_csv", help="CSV for calibration (required for qlinear)")
    ap.add_argument("--num-samples", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=6)
    args = ap.parse_args()

    if args.mode == "dynamic":
        quant_dynamic(args.model, args.out)
    else:
        if not (args.preproc and args.calib_csv):
            raise SystemExit("For --mode qlinear you must pass --preproc and --calib_csv")
        quant_qlinear(args.model, args.out, args.preproc, args.calib_csv, args.num_samples, args.seq_len)

if __name__ == "__main__":
    main()
