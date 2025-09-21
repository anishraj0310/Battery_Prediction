

### Quantization (dynamic + QLinear)

Install:
```bash
pip install onnxruntime onnxruntime-tools
```

**Dynamic (weight-only INT8)**
```bash
python src/quantize_onnx.py --model runs/gru/model.onnx --out runs/gru/model.int8.onnx --mode dynamic
```

**Static QLinear (INT8 activations + weights)**
```bash
python src/quantize_onnx.py --model runs/gru/model.onnx --out runs/gru/model.qlinear.onnx       --mode qlinear --preproc runs/gru/preproc_export.json --calib_csv data/synthetic_sessions.csv --num-samples 2000 --seq-len 6
```


### LSTM variant export

Train an LSTM checkpoint and export the same way:
```bash
python src/train.py --data data/synthetic_sessions.csv --model lstm --epochs 5
python src/export_onnx.py --ckpt runs/gru/latest.pt --out runs/gru/model_lstm.onnx --seq_len 6
```
(The exporter detects model type from the checkpoint and exports accordingly.)


### Android ORT quickstart

See `android/README_ANDROID.md` and drop `BatteryOrtClient.kt` into your app. Place `model.onnx` and `preproc_export.json` under `app/src/main/assets/`.

