"""
data_prep.py

Converts a raw timestamped sensor-reading CSV into windowed sequences
suitable for training RNN/GRU/LSTM forecasters (and flattened versions for
the Linear Regression baseline), predicting the value 15 minutes ahead.

Usage:
    python src/data_prep.py --data data/sensor_readings.csv --window 12
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_windows(series, window, horizon=1):
    """Return (X, y) where X[i] is a window-length sequence and y[i] is the
    value horizon steps after the end of that window."""
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i: i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Prepare windowed time-series data.")
    parser.add_argument("--data", required=True, help="CSV with a timestamp column and sensor readings.")
    parser.add_argument("--target-col", default="temperature")
    parser.add_argument("--window", type=int, default=12, help="Number of past readings used per sample.")
    parser.add_argument("--horizon", type=int, default=1, help="Steps ahead to predict (1 step = 15 min).")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    series = df[args.target_col].values.astype("float32")

    split_idx = int(len(series) * 0.8)
    mean, std = series[:split_idx].mean(), series[:split_idx].std()
    series_norm = (series - mean) / std

    X, y = build_windows(series_norm, args.window, args.horizon)
    split_idx_windows = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx_windows], X[split_idx_windows:]
    y_train, y_test = y[:split_idx_windows], y[split_idx_windows:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "windowed_data.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        mean=mean, std=std,
    )

    print("Prepared", len(X_train), "training and", len(X_test), "test windows.")
    print("Saved to", out_dir, "/windowed_data.npz")


if __name__ == "__main__":
    main()
