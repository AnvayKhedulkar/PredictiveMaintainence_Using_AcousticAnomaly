from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import soundfile as sf


def infer_condition(parts):
    s = " ".join(parts).lower()
    if "abnormal" in s or "anomaly" in s:
        return "anomaly"
    if "normal" in s:
        return "normal"
    return "unknown"


def infer_domain(parts):
    s = " ".join(parts).lower()
    for token in ["source", "target", "domain_a", "domain_b"]:
        if token in s:
            return token
    return "unknown"


def build_metadata(data_root: Path, dataset_family: str) -> pd.DataFrame:
    rows = []
    wavs = list(data_root.rglob("*.wav"))
    for wav in wavs:
        rel = wav.relative_to(data_root)
        parts = rel.parts
        machine_type = parts[0] if len(parts) > 0 else "unknown"
        machine_id = next((p for p in parts if p.lower().startswith("id_")), "unknown")
        condition = infer_condition(parts)
        domain = infer_domain(parts) if dataset_family.lower() == "mimii_dg" else "na"
        info = sf.info(str(wav))
        rows.append({
            "file_path": str(wav),
            "dataset_family": dataset_family,
            "machine_type": machine_type,
            "machine_id": machine_id,
            "condition": condition,
            "domain": domain,
            "duration_sec": info.duration,
            "sample_rate": info.samplerate,
            "split": "train" if condition == "normal" else "test"
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["machine_type", "machine_id", "file_path"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-family", required=True, choices=["mimii", "mimii_dg"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_metadata(Path(args.data_root), args.dataset_family)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"saved metadata: {args.output} rows={len(df)}")


if __name__ == "__main__":
    main()
