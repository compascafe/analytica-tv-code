"""Engineer features for the xG model."""
import numpy as np
import pandas as pd

GOAL_CENTRE = np.array([105.0, 34.0])
GOAL_HALF_WIDTH = 7.32 / 2


def shot_angle(loc: np.ndarray) -> float:
    """Angle subtended by the goal mouth from the shot location, in [0, pi]."""
    left = np.array([105.0, 34.0 - GOAL_HALF_WIDTH])
    right = np.array([105.0, 34.0 + GOAL_HALF_WIDTH])
    v1, v2 = left - loc, right - loc
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    loc = np.vstack(df["location"].to_numpy())
    dist = np.linalg.norm(loc - GOAL_CENTRE, axis=1)
    angle = np.array([shot_angle(row) for row in loc])

    out = pd.DataFrame(
        {
            "distance": dist,
            "angle": angle,
            "is_header": (df["shot.body_part.name"] == "Head").astype(int),
            "is_set_piece": (
                df["shot.type.name"].isin(["Free Kick", "Corner"])
            ).astype(int),
            "goal": (df["shot.outcome.name"] == "Goal").astype(int),
        }
    )
    return out.reset_index(drop=True)