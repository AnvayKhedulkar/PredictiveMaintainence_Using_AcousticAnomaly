from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = {"file_path","segment_index","dataset_family","machine_type","machine_id","condition","domain","split"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.features)

    # use only normal train segments for IF
    train_df = df[(df["condition"] == "normal") & (df["split"] == "train")].copy()
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X = train_df[feature_cols].values

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=args.n_estimators,
                contamination=args.contamination,
                random_state=42,
            ))
        ]
    )

    pipe.fit(X)

    bundle = {
        "pipeline": pipe,
        "feature_columns": feature_cols,
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print(f"saved IF model to {args.model_out}")


if __name__ == "__main__":
    main()