from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.infer import build_feature_df
from src.train_autoencoder import AE


def compute_scores_for_file(
    file_path: str,
    if_bundle,
    ae_ckpt,
    scaler_bundle,
) -> tuple[float, float]:
    """
    Compute mean IF and AE scores for one file, using exactly the same
    pipeline as infer.py.
    """
    df = build_feature_df(Path(file_path))

    # Isolation Forest scores (same as infer.py)
    if_scores = -if_bundle["pipeline"].score_samples(df[if_bundle["feature_columns"]])
    if_mean = float(np.mean(if_scores))

    # Autoencoder scores (same as infer.py)
    X_ae = scaler_bundle["scaler"].transform(
        df[scaler_bundle["feature_columns"]].values
    )
    model = AE(ae_ckpt["input_dim"])
    model.load_state_dict(ae_ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_ae, dtype=torch.float32)
        recon = model(xb).numpy()
        ae_scores = np.mean((recon - X_ae) ** 2, axis=1)
    ae_mean = float(np.mean(ae_scores))

    return if_mean, ae_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--if-model", required=True)
    parser.add_argument("--ae-model", required=True)
    parser.add_argument("--ae-scaler", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("DEBUG: loading metadata:", args.metadata)
    meta = pd.read_csv(args.metadata)
    meta = meta[meta["split"].isin(["train", "test"])]

    print("DEBUG: loading models")
    if_bundle = joblib.load(args.if_model)
    ae_ckpt = torch.load(args.ae_model, map_location="cpu")
    scaler_bundle = joblib.load(args.ae_scaler)

    print("DEBUG: loading thresholds")
    thresholds = json.load(open(args.thresholds, "r", encoding="utf-8"))
    if_th = thresholds["isolation_forest"]
    ae_th = thresholds["autoencoder"]

    rows = []
    for _, row in meta.iterrows():
        file_path = row["file_path"]
        condition = row["condition"]
        machine_id = row["machine_id"]
        machine_type = row["machine_type"]

        print(f"DEBUG: processing {file_path}")
        if_mean, ae_mean = compute_scores_for_file(
            file_path, if_bundle, ae_ckpt, scaler_bundle
        )

        if_flag = if_mean >= if_th
        ae_flag = ae_mean >= ae_th
        combined_label = "anomaly" if (if_flag or ae_flag) else "normal"

        rows.append(
            {
                "file_path": file_path,
                "machine_type": machine_type,
                "machine_id": machine_id,
                "condition": condition,
                "if_score": if_mean,
                "ae_score": ae_mean,
                "if_is_anomaly": if_flag,
                "ae_is_anomaly": ae_flag,
                "combined_label": combined_label,
            }
        )

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print("Saved scores to", args.output)


if __name__ == "__main__":
    main()