from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

META_COLS = {"file_path","segment_index","dataset_family","machine_type","machine_id","condition","domain","split"}

class AE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    train_df = df[df["condition"] == "normal"].copy()
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X = train_df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = AE(Xs.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for (xb,) in dl:
            pred = model(xb)
            loss = loss_fn(pred, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch={epoch+1} loss={total/len(ds):.6f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": Xs.shape[1]}, args.model_out)
    joblib.dump({"scaler": scaler, "feature_columns": feature_cols}, str(Path(args.model_out).with_suffix('.scaler.joblib')))
    print(f"saved model: {args.model_out}")


if __name__ == "__main__":
    main()
