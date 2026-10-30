"""Reliability curve and proper scoring rules for the fitted model."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    preds = pd.read_parquet(DATA_DIR / "xg_predictions.parquet")

    # True outcomes come from the features pipeline (single source of truth)
    df = pd.read_parquet(DATA_DIR / "shots.parquet")
    y_true = build_features(df)["goal"].to_numpy()
    p = preds["xg_ours"].to_numpy()

    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, p)
    ll = log_loss(y_true, p)

    print(f"Brier score: {brier:.4f}")
    print(f"Log loss:    {ll:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label="our model")
    ax.set_xlabel("Mean predicted xG (bin)")
    ax.set_ylabel("Observed goal rate (bin)")
    ax.set_title("Reliability curve — 2018 WC shots")
    ax.legend()
    fig.savefig(DATA_DIR / "reliability_curve.png", dpi=150)
    print("Saved data/reliability_curve.png")


if __name__ == "__main__":
    main()