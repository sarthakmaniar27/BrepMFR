# BrepMFR Lite ONNX PyG Demo

This package runs the included 3-class Thread + Text ONNX model on an existing
**lite** PyTorch-Geometric (`.pt`) graph:

| Class ID | Label |
| --- | --- |
| 0 | Stock |
| 1 | Thread |
| 2 | Text |

The model predicts one label for every B-rep face.

## Package contents

- `brepmfr_lite.onnx` — exported ONNX model.
- `run_onnx_pyg_inference.py` — command-line inference runner.
- `label_map.json` — class ID to label mapping.
- `model_config.json` — exact model input contract.
- `requirements.txt` — Python packages needed for the demo.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For NVIDIA GPU inference, replace `onnxruntime` with the compatible
`onnxruntime-gpu` distribution, then confirm the runner prints
`CUDAExecutionProvider`.

## Run one graph

```powershell
python run_onnx_pyg_inference.py `
  --input "C:\graphs\part.pt" `
  --output-dir "C:\onnx_results"
```

## Run a folder

```powershell
python run_onnx_pyg_inference.py `
  --input "C:\graphs" `
  --output-dir "C:\onnx_results" `
  --max-files 10
```

Use `--recursive` to scan subdirectories. Omit `--max-files` to process every
graph in the directory.

## Output

The runner writes:

- `<graph>_predictions.csv`: one row per face with class ID, label, confidence,
  and all class probabilities.
- `onnx_inference_summary.csv`: per-graph status and prediction-file location.

## Input requirements

The input must be a PyTorch-Geometric graph created by the Thread + Text
pipeline with `--inference_profile lite`. The runner rejects graphs containing
A1/A2/A3 tensors and skips zero-face graphs, because the included model was
trained and exported for the lite profile only.

This package does not directly read STEP or macro JSON files. Convert those to
lite `.pt` graphs with the matching preprocessing pipeline before inference.
