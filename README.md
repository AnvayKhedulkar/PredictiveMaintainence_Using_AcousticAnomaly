# Audio Anomaly Detection with MIMII / MIMII DG

This project is a VS Code-ready end-to-end scaffold for industrial audio anomaly detection using the MIMII and MIMII DG dataset families.

## Project goal
Train anomaly detection models on mostly normal machine sounds and detect abnormal sounds from raw audio recordings.

## Included
- Data download helper commands and setup notes
- Metadata builder
- Audio preprocessing and segmentation
- Feature extraction (MFCC + spectral + log-Mel)
- Isolation Forest baseline
- Autoencoder baseline
- Evaluation pipeline
- Inference script
- VS Code setup steps

## Recommended workflow
1. Create the Python environment.
2. Install dependencies.
3. Download MIMII or MIMII DG manually into `data/raw/`.
4. Build metadata.
5. Extract features.
6. Train baseline models.
7. Evaluate.
8. Run inference on a file/folder.

## Dataset notes
MIMII is freely available on Zenodo and contains sounds from valves, pumps, fans, and slide rails.[web:79][web:83]
MIMII DG is a domain generalization benchmark for anomalous sound detection on industrial machine audio.[web:81][web:99]

Because these datasets are large, this project does **not** bundle them. You should download them separately and place the extracted folders under `data/raw/`.

## Suggested raw data layout

### Option A: MIMII
```text
data/raw/
  mimii/
    fan/
    pump/
    slider/
    valve/
```

### Option B: MIMII DG
```text
data/raw/
  mimii_dg/
    fan/
    gearbox/
    bearing/
    slider/
    valve/
```

## Quick start

### 1) Create environment

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Build metadata
```bash
python src/build_metadata.py --data-root data/raw/mimii --dataset-family mimii --output data/processed/metadata.csv
```

For MIMII DG:
```bash
python src/build_metadata.py --data-root data/raw/mimii_dg --dataset-family mimii_dg --output data/processed/metadata.csv
```

### 3) Extract features
```bash
python src/extract_features.py --metadata data/processed/metadata.csv --output-dir data/features --segment-seconds 5 --hop-seconds 2.5
```

### 4) Train Isolation Forest baseline
```bash
python src/train_isolation_forest.py --features data/features/tabular_features.csv --model-out models/isolation_forest.joblib
```

### 5) Train Autoencoder baseline
```bash
python src/train_autoencoder.py --features data/features/tabular_features.csv --model-out models/autoencoder.pt
```

### 6) Evaluate
```bash
python src/evaluate.py --features data/features/tabular_features.csv --if-model models/isolation_forest.joblib --ae-model models/autoencoder.pt --output-dir outputs
```

### 7) Inference
```bash
python src/infer.py --input path/to/audio.wav --if-model models/isolation_forest.joblib --ae-model models/autoencoder.pt --thresholds outputs/thresholds.json
```

## VS Code setup
1. Open the project folder in VS Code.
2. Install the Python extension.
3. Press `Ctrl+Shift+P` → `Python: Select Interpreter` → choose `.venv`.
4. Open the integrated terminal.
5. Run the commands from the Quick start section.
6. Use the Run panel or terminal to execute scripts.

## What you need before starting
- Python 3.10+ installed.
- Enough disk space for MIMII/MIMII DG; MIMII archives can be several GB each on Zenodo.[web:79]
- You must download the dataset yourself because redistribution is not included in this project package.

## Notes on methodology
- DCASE Task 2 and MIMII baseline work use autoencoder-style anomaly scoring based on reconstruction error.[web:76][web:86]
- MIMII consists of machine categories including valves, pumps, fans, and slide rails.[web:79][web:83]
- MIMII DG is intended to benchmark anomaly detection under domain shifts.[web:81][web:99]

