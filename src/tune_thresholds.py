from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def best_threshold(scores, y_true, model_name: str):
    # scores: higher = more anomalous
    # use quantiles as candidate thresholds
    qs = np.linspace(0.1, 0.9, 81)
    thresholds = np.quantile(scores, qs)

    best = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best.update(
                {
                    "threshold": float(th),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )

    print(f"{model_name} best threshold search result:", best)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default="outputs/inference_scores.csv")
    parser.add_argument("--thresholds", type=str, default="outputs/thresholds.json")
    args = parser.parse_args()

    df = pd.read_csv(args.scores)

    # true labels: 1 = anomaly, 0 = normal
    y_true = (df["condition"] == "anomaly").astype(int).values

    if_scores = df["if_score"].values
    ae_scores = df["ae_score"].values

    print("IF ROC-AUC:", roc_auc_score(y_true, if_scores))
    print("AE ROC-AUC:", roc_auc_score(y_true, ae_scores))

    if_best = best_threshold(if_scores, y_true, "Isolation Forest")
    ae_best = best_threshold(ae_scores, y_true, "Autoencoder")

    # load current thresholds and update
    with open(args.thresholds, "r", encoding="utf-8") as f:
        th = json.load(f)

    th["isolation_forest"] = if_best["threshold"]
    th["autoencoder"] = ae_best["threshold"]

    with open(args.thresholds, "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2)

    print("Updated thresholds.json with tuned values")


if __name__ == "__main__":
    main()