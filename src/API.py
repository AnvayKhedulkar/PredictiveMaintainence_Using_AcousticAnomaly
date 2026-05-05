from __future__ import annotations
from pathlib import Path
import tempfile
import json

import numpy as np
import joblib
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.infer import build_feature_df
from src.train_autoencoder import AE

app = FastAPI(title="Pump Audio Anomaly API")

# Load models and thresholds once at startup
IF_BUNDLE = joblib.load("models/isolation_forest.joblib")
AE_CKPT = torch.load("models/autoencoder.pt", map_location="cpu")
SCALER_BUNDLE = joblib.load("models/autoencoder.scaler.joblib")
THRESHOLDS = json.load(open("outputs/thresholds.json", "r", encoding="utf-8"))

AE_MODEL = AE(AE_CKPT["input_dim"])
AE_MODEL.load_state_dict(AE_CKPT["state_dict"])
AE_MODEL.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to a temp path
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    # Build features exactly like infer.py
    df = build_feature_df(tmp_path)

    # IF scores
    if_scores = -IF_BUNDLE["pipeline"].score_samples(df[IF_BUNDLE["feature_columns"]])
    if_mean = float(np.mean(if_scores))

    # AE scores
    X_ae = SCALER_BUNDLE["scaler"].transform(
        df[SCALER_BUNDLE["feature_columns"]].values
    )
    xb = torch.tensor(X_ae, dtype=torch.float32)
    with torch.no_grad():
        recon = AE_MODEL(xb).numpy()
        ae_scores = np.mean((recon - X_ae) ** 2, axis=1)
    ae_mean = float(np.mean(ae_scores))

    # flags and final label (use your latest logic or the simple OR)
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