# Project plan

## Scope
- Baseline anomaly detection for industrial machine audio.
- Start with MIMII.
- Extend to MIMII DG after the first baseline works.

## Models
- Isolation Forest on aggregated tabular acoustic features.
- Feedforward autoencoder on scaled tabular acoustic features.

## Feature families
- MFCC
- Chroma
- Log-Mel statistics
- Spectral centroid/bandwidth/rolloff
- RMS, flatness, ZCR, tempo

## Outputs
- metrics.csv
- thresholds.json
- score distribution plots
- confusion matrix plots
