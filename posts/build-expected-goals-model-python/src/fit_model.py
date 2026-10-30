"""Fit a logistic regression xG model with statsmodels."""
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "shots.parquet")
    features = build_features(df)

    X = features.drop(columns="goal")
    y = features["goal"]

    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=0)
    print(model.summary())

    preds = model.predict(X_const)
    out = df[["match_id", "minute", "shot.statsbomb_xg"]].copy()
    out["xg_ours"] = preds.to_numpy()
    out.to_parquet(DATA_DIR / "xg_predictions.parquet", index=False)
    print(f"\nSaved predictions. Mean xG: {preds.mean():.4f}, "
          f"goal rate: {y.mean():.4f}")


if __name__ == "__main__":
    main()