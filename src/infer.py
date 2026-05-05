from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.common import load_audio, segment_audio, extract_tabular_features
from src.train_autoencoder import AE


from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import json

# imports for features/AE remain as above

def build_feature_df(input_path: Path, sr: int = 16000, segment_seconds: float = 5.0, hop_seconds: float = 2.5):
    print("DEBUG: build_feature_df called with", input_path)
    y, sr = load_audio(input_path, sr=sr)
    print("DEBUG: loaded audio, len(y) =", len(y), "sr =", sr)
    segments = segment_audio(y, sr, segment_seconds, hop_seconds)
    print("DEBUG: num segments =", len(segments))
    rows = []
    for i, seg in enumerate(segments):
        print("DEBUG: extracting features for segment", i)
        feats = extract_tabular_features(seg, sr)
        print("DEBUG: segment", i, "num_features =", len(feats))
        feats['file_path'] = str(input_path)
        feats['segment_index'] = i
        rows.append(feats)
    df = pd.DataFrame(rows)
    print("DEBUG: feature df created, shape =", df.shape)
    return df


def main():
    print("DEBUG: inside main()")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--if-model', required=True)
    parser.add_argument('--ae-model', required=True)
    parser.add_argument('--thresholds', required=True)
    args = parser.parse_args()
    print("DEBUG: parsed args:", args)

    df = build_feature_df(Path(args.input))
    print("DEBUG: built feature df, shape =", df.shape)

    import joblib
    if_bundle = joblib.load(args.if_model)
    print("DEBUG: loaded IF model")

    if_scores = -if_bundle['pipeline'].score_samples(df[if_bundle['feature_columns']])
    print("DEBUG: computed IF scores, len =", len(if_scores))
    # df = build_feature_df(Path(args.input))
    # if_bundle = joblib.load(args.if_model)
    # if_scores = -if_bundle['pipeline'].score_samples(df[if_bundle['feature_columns']])

    import torch
    ae_ckpt = torch.load(args.ae_model, map_location='cpu')
    print("DEBUG: loaded AE checkpoint")

    scaler_path = str(Path(args.ae_model).with_suffix('.scaler.joblib'))
    print("DEBUG: AE scaler path =", scaler_path)
    scaler_bundle = joblib.load(scaler_path)
    print("DEBUG: loaded AE scaler")

    import numpy as np
    X_ae = scaler_bundle['scaler'].transform(df[scaler_bundle['feature_columns']].values)
    print("DEBUG: transformed features for AE, shape =", X_ae.shape)

    
    model = AE(ae_ckpt['input_dim'])
    model.load_state_dict(ae_ckpt['state_dict'])
    model.eval()
    print("DEBUG: AE model ready")

    with torch.no_grad():
        xb = torch.tensor(X_ae, dtype=torch.float32)
        recon = model(xb).numpy()
        ae_scores = np.mean((recon - X_ae) ** 2, axis=1)
    print("DEBUG: computed AE scores, len =", len(ae_scores))

    import json
    thresholds = json.load(open(args.thresholds, 'r', encoding='utf-8'))
    print("DEBUG: thresholds loaded:", thresholds)

    if_flag = bool(np.mean(if_scores) >= thresholds['isolation_forest'])
    ae_flag = bool(np.mean(ae_scores) >= thresholds['autoencoder'])

    # keep the original dict if you like
    print({
        'isolation_forest_mean_score': float(np.mean(if_scores)),
        'isolation_forest_is_anomaly': if_flag,
        'autoencoder_mean_score': float(np.mean(ae_scores)),
        'autoencoder_is_anomaly': ae_flag
    })
    if if_flag == False:
        if_result = "normal"
    else:
        if_result = "anomaly"
    if ae_flag == False:
        ae_result = "normal"
    else:
        ae_result = "anomaly"

    # NEW: simple label output
    # label = "anomaly" if (if_flag or ae_flag) else "normal"
    print(f"IF_Distance: {abs(float(np.mean(if_scores)) - thresholds['isolation_forest'])}")
    print(f"AE_Distance: {abs(float(np.mean(ae_scores)) - thresholds['autoencoder'])}")
    label = if_result if ((abs(float(np.mean(if_scores)) - thresholds['isolation_forest'])) >= (abs(float(np.mean(ae_scores)) - thresholds['autoencoder']))) else ae_result
    print(f"File: {args.input}")
    print(f"Predicted label: {label}")
    print(f"IF score: {np.mean(if_scores):.4f}, threshold: {thresholds['isolation_forest']:.4f}")
    print(f"AE score: {np.mean(ae_scores):.4f}, threshold: {thresholds['autoencoder']:.4f}")
    


if __name__ == '__main__':
    print("DEBUG: starting infer main()")
    main()
