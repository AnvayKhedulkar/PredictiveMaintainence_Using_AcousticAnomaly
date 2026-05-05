from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.train_autoencoder import AE

META_COLS = {"file_path","segment_index","dataset_family","machine_type","machine_id","condition","domain","split"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--if-model", required=True)
    parser.add_argument("--ae-model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    df = df[df["split"] == "test"].reset_index(drop=True)
    y = (df["condition"] == "anomaly").astype(int).values

    if_bundle = joblib.load(args.if_model)
    X_if = df[if_bundle["feature_columns"]]
    if_scores = -if_bundle["pipeline"].score_samples(X_if)

    ae_ckpt = torch.load(args.ae_model, map_location="cpu")
    scaler_bundle = joblib.load(str(Path(args.ae_model).with_suffix('.scaler.joblib')))
    X_ae = scaler_bundle["scaler"].transform(df[scaler_bundle["feature_columns"]].values)
    model = AE(ae_ckpt["input_dim"])
    model.load_state_dict(ae_ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_ae, dtype=torch.float32)
        recon = model(xb).numpy()
        ae_scores = np.mean((recon - X_ae) ** 2, axis=1)

    results = []
    thresholds = {}
    for name, scores in [("isolation_forest", if_scores), ("autoencoder", ae_scores)]:
        roc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else np.nan
        pr = average_precision_score(y, scores) if len(np.unique(y)) > 1 else np.nan
        thr = float(np.quantile(scores[y == 0], 0.95)) if np.any(y == 0) else float(np.quantile(scores, 0.95))
        pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        cm = confusion_matrix(y, pred)
        results.append({"model": name, "roc_auc": roc, "pr_auc": pr, "precision": p, "recall": r, "f1": f1, "threshold": thr})
        thresholds[name] = thr

        plt.figure(figsize=(6,4))
        sns.histplot(scores[y == 0], color='steelblue', label='normal', stat='density', bins=40, alpha=0.6)
        if np.any(y == 1):
            sns.histplot(scores[y == 1], color='tomato', label='anomaly', stat='density', bins=40, alpha=0.6)
        plt.axvline(thr, color='black', linestyle='--', label='threshold')
        plt.title(f'{name} score distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'{name}_score_distribution.png', dpi=160)
        plt.close()

        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} confusion matrix')
        plt.tight_layout()
        plt.savefig(out_dir / f'{name}_confusion_matrix.png', dpi=160)
        plt.close()

    pd.DataFrame(results).to_csv(out_dir / 'metrics.csv', index=False)
    with open(out_dir / 'thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2)
    print(f"saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
