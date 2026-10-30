"""Offline unit tests for feature engineering."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import build_features, shot_angle


def test_angle_bounds() -> None:
    for loc in ([50, 34], [104, 34], [90, 5], [105, 34]):
        a = shot_angle(np.array(loc, dtype=float))
        assert 0.0 <= a <= np.pi


def test_angle_pen_spot_wider_than_byline() -> None:
    pen_spot = shot_angle(np.array([94.0, 34.0]))
    byline = shot_angle(np.array([94.0, 60.0]))
    assert pen_spot > byline


def test_goal_centre_angle_is_pi() -> None:
    assert shot_angle(np.array([105.0, 34.0])) == pytest.approx(np.pi)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 2],
            "minute": [10, 20],
            "shot.statsbomb_xg": [0.05, 0.4],
            "location": [[94.0, 34.0], [105.0, 34.0]],
            "shot.body_part.name": ["Head", "Right Foot"],
            "shot.type.name": ["Open Play", "Free Kick"],
            "shot.outcome.name": ["Off Target", "Goal"],
        }
    )


def test_feature_frame_shape_and_no_nans(tiny_df: pd.DataFrame) -> None:
    feats = build_features(tiny_df)
    assert len(feats) == len(tiny_df)
    assert not feats.isna().any().any()


def test_distance_non_negative_and_labels_correct(tiny_df: pd.DataFrame) -> None:
    feats = build_features(tiny_df)
    assert (feats["distance"] >= 0).all()
    assert feats["goal"].tolist() == [0, 1]
    assert feats["is_header"].tolist() == [1, 0]
    assert feats["is_set_piece"].tolist() == [0, 1]