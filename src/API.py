from __future__ import annotations

from pathlib import Path
import json
import tempfile

import joblib
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse

from src.infer import build_feature_df
from src.train_autoencoder import AE


# Base directories: src is the parent folder of this file
SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = SRC_DIR / "models"
OUTPUTS_DIR = SRC_DIR / "outputs"

app = FastAPI(title="Pump Audio Anomaly API")

# Load models and thresholds once at startup
IF_BUNDLE = joblib.load(MODELS_DIR / "isolation_forest.joblib")
AE_CKPT = torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu")
SCALER_BUNDLE = joblib.load(MODELS_DIR / "autoencoder.scaler.joblib")
THRESHOLDS = json.load(open(OUTPUTS_DIR / "thresholds.json", "r", encoding="utf-8"))

AE_MODEL = AE(AE_CKPT["input_dim"])
AE_MODEL.load_state_dict(AE_CKPT["state_dict"])
AE_MODEL.eval()


# ─────────────────────────────────────────────
# Home page: file upload form
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Pump Anomaly Detector</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .card {
                background: #1e293b;
                border-radius: 16px;
                padding: 40px;
                width: 100%;
                max-width: 480px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            }
            h1 {
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 8px;
                color: #f1f5f9;
            }
            p.subtitle {
                font-size: 0.9rem;
                color: #94a3b8;
                margin-bottom: 28px;
            }
            .upload-area {
                border: 2px dashed #334155;
                border-radius: 12px;
                padding: 32px 20px;
                text-align: center;
                cursor: pointer;
                transition: border-color 0.2s;
                margin-bottom: 20px;
            }
            .upload-area:hover { border-color: #6366f1; }
            .upload-area input[type="file"] {
                display: none;
            }
            .upload-area label {
                cursor: pointer;
                display: block;
            }
            .upload-icon { font-size: 2.5rem; margin-bottom: 10px; }
            .upload-text { font-size: 0.95rem; color: #94a3b8; }
            .file-name {
                font-size: 0.85rem;
                color: #6366f1;
                margin-top: 8px;
                min-height: 20px;
            }
            button[type="submit"] {
                width: 100%;
                padding: 14px;
                background: #6366f1;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            button[type="submit"]:hover { background: #4f46e5; }
            button[type="submit"]:disabled { background: #334155; cursor: not-allowed; }

            /* Result card */
            #result { margin-top: 28px; display: none; }
            .result-label {
                font-size: 2rem;
                font-weight: 800;
                text-align: center;
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 16px;
            }
            .label-normal { background: #064e3b; color: #6ee7b7; }
            .label-anomaly { background: #7f1d1d; color: #fca5a5; }
            .scores {
                background: #0f172a;
                border-radius: 10px;
                padding: 16px;
                font-size: 0.85rem;
                color: #94a3b8;
            }
            .scores table { width: 100%; border-collapse: collapse; }
            .scores td { padding: 6px 4px; }
            .scores td:last-child { text-align: right; color: #e2e8f0; }
            .loading {
                text-align: center;
                color: #6366f1;
                font-size: 0.9rem;
                margin-top: 20px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Pump Anomaly Detector</h1>
            <p class="subtitle">Upload a .wav recording of your pump to check if it sounds normal or anomalous.</p>

            <form id="uploadForm">
                <div class="upload-area">
                    <label for="fileInput">
                        <div class="upload-icon">🎙️</div>
                        <div class="upload-text">Tap to select a .wav file</div>
                        <div class="file-name" id="fileName">No file selected</div>
                    </label>
                    <input type="file" id="fileInput" name="file" accept=".wav,audio/*" required />
                </div>

                <button type="submit" id="submitBtn">Analyze Recording</button>
            </form>

            <div class="loading" id="loading">⏳ Analyzing... this may take a moment</div>

            <div id="result">
                <div class="result-label" id="resultLabel"></div>
                <div class="scores">
                    <table>
                        <tr>
                            <td>Isolation Forest Score</td>
                            <td id="ifScore"></td>
                        </tr>
                        <tr>
                            <td>IF Threshold</td>
                            <td id="ifThreshold"></td>
                        </tr>
                        <tr>
                            <td>IF Decision</td>
                            <td id="ifDecision"></td>
                        </tr>
                        <tr><td colspan="2" style="padding:6px 0;"></td></tr>
                        <tr>
                            <td>Autoencoder Score</td>
                            <td id="aeScore"></td>
                        </tr>
                        <tr>
                            <td>AE Threshold</td>
                            <td id="aeThreshold"></td>
                        </tr>
                        <tr>
                            <td>AE Decision</td>
                            <td id="aeDecision"></td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            fileInput.addEventListener('change', () => {
                fileName.textContent = fileInput.files[0]
                    ? fileInput.files[0].name
                    : 'No file selected';
            });

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const file = fileInput.files[0];
                if (!file) return;

                submitBtn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    const label = data.predicted_label;
                    const labelEl = document.getElementById('resultLabel');
                    labelEl.textContent = label.toUpperCase();
                    labelEl.className = 'result-label ' +
                        (label === 'normal' ? 'label-normal' : 'label-anomaly');

                    document.getElementById('ifScore').textContent =
                        data.isolation_forest_mean_score.toFixed(4);
                    document.getElementById('ifThreshold').textContent =
                        data.isolation_forest_threshold.toFixed(4);
                    document.getElementById('ifDecision').textContent =
                        data.if_model_decision;

                    document.getElementById('aeScore').textContent =
                        data.autoencoder_mean_score.toFixed(4);
                    document.getElementById('aeThreshold').textContent =
                        data.autoencoder_threshold.toFixed(4);
                    document.getElementById('aeDecision').textContent =
                        data.ae_model_decision;

                    result.style.display = 'block';
                } catch (err) {
                    alert('Error: ' + err.message);
                } finally {
                    submitBtn.disabled = false;
                    loading.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────
# Predict endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    df = build_feature_df(tmp_path)

    if_scores = -IF_BUNDLE["pipeline"].score_samples(df[IF_BUNDLE["feature_columns"]])
    if_mean = float(np.mean(if_scores))

    X_ae = SCALER_BUNDLE["scaler"].transform(
        df[SCALER_BUNDLE["feature_columns"]].values
    )
    xb = torch.tensor(X_ae, dtype=torch.float32)
    with torch.no_grad():
        recon = AE_MODEL(xb).numpy()
        ae_scores = np.mean((recon - X_ae) ** 2, axis=1)
    ae_mean = float(np.mean(ae_scores))

    if_flag = bool(if_mean >= THRESHOLDS["isolation_forest"])
    ae_flag = bool(ae_mean >= THRESHOLDS["autoencoder"])

    if_result = "anomaly" if if_flag else "normal"
    ae_result = "anomaly" if ae_flag else "normal"

    if_diff = abs(if_mean - THRESHOLDS["isolation_forest"])
    ae_diff = abs(ae_mean - THRESHOLDS["autoencoder"])

    label = if_result if if_diff >= ae_diff else ae_result

    return JSONResponse(
        {
            "filename": file.filename,
            "predicted_label": label,
            "isolation_forest_mean_score": if_mean,
            "isolation_forest_threshold": THRESHOLDS["isolation_forest"],
            "autoencoder_mean_score": ae_mean,
            "autoencoder_threshold": THRESHOLDS["autoencoder"],
            "if_model_decision": if_result,
            "ae_model_decision": ae_result,
        }
    )


# from __future__ import annotations

# from pathlib import Path
# import json
# import tempfile

# import joblib
# import numpy as np
# import torch
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse

# from src.infer import build_feature_df
# from src.train_autoencoder import AE


# # Base directories: src is the parent folder of this file
# SRC_DIR = Path(__file__).resolve().parent
# MODELS_DIR = SRC_DIR / "models"
# OUTPUTS_DIR = SRC_DIR / "outputs"

# app = FastAPI(title="Pump Audio Anomaly API")

# # Load models and thresholds once at startup
# IF_BUNDLE = joblib.load(MODELS_DIR / "isolation_forest.joblib")
# AE_CKPT = torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu")
# SCALER_BUNDLE = joblib.load(MODELS_DIR / "autoencoder.scaler.joblib")
# THRESHOLDS = json.load(open(OUTPUTS_DIR / "thresholds.json", "r", encoding="utf-8"))

# AE_MODEL = AE(AE_CKPT["input_dim"])
# AE_MODEL.load_state_dict(AE_CKPT["state_dict"])
# AE_MODEL.eval()


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     suffix = Path(file.filename).suffix
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = Path(tmp.name)

#     df = build_feature_df(tmp_path)

#     if_scores = -IF_BUNDLE["pipeline"].score_samples(df[IF_BUNDLE["feature_columns"]])
#     if_mean = float(np.mean(if_scores))

#     X_ae = SCALER_BUNDLE["scaler"].transform(
#         df[SCALER_BUNDLE["feature_columns"]].values
#     )
#     xb = torch.tensor(X_ae, dtype=torch.float32)
#     with torch.no_grad():
#         recon = AE_MODEL(xb).numpy()
#         ae_scores = np.mean((recon - X_ae) ** 2, axis=1)
#     ae_mean = float(np.mean(ae_scores))

#     if_flag = bool(if_mean >= THRESHOLDS["isolation_forest"])
#     ae_flag = bool(ae_mean >= THRESHOLDS["autoencoder"])

#     if_result = "anomaly" if if_flag else "normal"
#     ae_result = "anomaly" if ae_flag else "normal"

#     if_diff = abs(if_mean - THRESHOLDS["isolation_forest"])
#     ae_diff = abs(ae_mean - THRESHOLDS["autoencoder"])

#     label = if_result if if_diff >= ae_diff else ae_result

#     return JSONResponse(
#         {
#             "filename": file.filename,
#             "predicted_label": label,
#             "isolation_forest_mean_score": if_mean,
#             "isolation_forest_threshold": THRESHOLDS["isolation_forest"],
#             "autoencoder_mean_score": ae_mean,
#             "autoencoder_threshold": THRESHOLDS["autoencoder"],
#             "if_model_decision": if_result,
#             "ae_model_decision": ae_result,
#         }
#     )