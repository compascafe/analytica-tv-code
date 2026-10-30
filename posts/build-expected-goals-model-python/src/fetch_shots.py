"""Download 2018 World Cup shots from StatsBomb open data."""
from pathlib import Path

import pandas as pd
from statsbombpy import sb

DATA_DIR = Path("data")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    competitions = sb.competitions()
    wc = competitions[
        (competitions["competition_name"] == "World Cup")
        & (competitions["season_name"] == "2018")
    ]
    if wc.empty:
        raise RuntimeError("2018 World Cup not found in competition list")

    match_ids = sb.matches(
        competition_id=int(wc.iloc[0]["competition_id"]),
        season_id=int(wc.iloc[0]["season_id"]),
    )["match_id"].tolist()
    print(f"Found {len(match_ids)} matches")

    frames = []
    for mid in match_ids:
        events = sb.events(match_id=mid)
        shots = events[events["type.name"] == "Shot"].copy()
        keep = [
            "match_id",
            "minute",
            "shot.statsbomb_xg",  # reference values for comparison only
            "location",
            "shot.body_part.name",
            "shot.type.name",
            "shot.outcome.name",
        ]
        frames.append(shots[keep])

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["location"])
    df.to_parquet(DATA_DIR / "shots.parquet", index=False)
    print(f"Saved {len(df)} shots to data/shots.parquet")


if __name__ == "__main__":
    main()