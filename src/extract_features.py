from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.common import load_audio, segment_audio, extract_tabular_features, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=2.5)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    out_dir = ensure_dir(args.output_dir)
    rows = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        path = row["file_path"]
        try:
            y, sr = load_audio(path, sr=args.sr)
            segments = segment_audio(y, sr, args.segment_seconds, args.hop_seconds)
            for idx, seg in enumerate(segments):
                feats = extract_tabular_features(seg, sr)
                feats.update({
                    "file_path": path,
                    "segment_index": idx,
                    "dataset_family": row.get("dataset_family", "unknown"),
                    "machine_type": row.get("machine_type", "unknown"),
                    "machine_id": row.get("machine_id", "unknown"),
                    "condition": row.get("condition", "unknown"),
                    "domain": row.get("domain", "na"),
                    "split": row.get("split", "train"),
                })
                rows.append(feats)
        except Exception as e:
            print(f"ERROR: failed on file {path}: {repr(e)}")
            raise

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(out_dir / "tabular_features.csv", index=False)
    print(f"saved features: {out_dir / 'tabular_features.csv'} rows={len(feat_df)}")


if __name__ == "__main__":
    main()
