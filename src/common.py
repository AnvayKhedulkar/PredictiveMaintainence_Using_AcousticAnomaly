from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import librosa
import soundfile as sf

NUMERIC_FEATURE_COLUMNS = []


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_audio(path: str | Path, sr: int = 16000, mono: bool = True):
    y, orig_sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return y.astype(np.float32), sr


def segment_audio(y: np.ndarray, sr: int, segment_seconds: float, hop_seconds: float) -> List[np.ndarray]:
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(y) < seg_len:
        pad = seg_len - len(y)
        y = np.pad(y, (0, pad))
    segments = []
    for start in range(0, max(1, len(y) - seg_len + 1), hop_len):
        seg = y[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    if not segments:
        segments = [y[:seg_len] if len(y) >= seg_len else np.pad(y, (0, seg_len - len(y)))]
    return segments


def extract_tabular_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel + 1e-9)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)

        for i in range(mfcc.shape[0]):
            feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

        for i in range(chroma.shape[0]):
            feats[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))

        feats["log_mel_mean"] = float(np.mean(log_mel))
        feats["log_mel_std"] = float(np.std(log_mel))
        feats["spec_centroid_mean"] = float(np.mean(spec_cent))
        feats["spec_centroid_std"] = float(np.std(spec_cent))
        feats["spec_bw_mean"] = float(np.mean(spec_bw))
        feats["spec_bw_std"] = float(np.std(spec_bw))
        feats["rolloff_mean"] = float(np.mean(spec_roll))
        feats["rolloff_std"] = float(np.std(spec_roll))
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"] = float(np.std(zcr))
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))
        feats["flatness_mean"] = float(np.mean(flatness))
        feats["flatness_std"] = float(np.std(flatness))

        # IMPORTANT: no tempo here
        # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # feats["tempo"] = float(tempo)

    except Exception as e:
        print("DEBUG: extract_tabular_features error:", repr(e))
        raise

    return feats


def save_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
