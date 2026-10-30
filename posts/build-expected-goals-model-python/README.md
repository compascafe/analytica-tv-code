# Build an expected goals model in Python with logistic regression

A reproducible walkthrough: pull StatsBomb open-data shots, engineer features, fit a logistic regression xG model in Python, and check calibration with reliability curves and Brier score.

Article: https://analytica.tv/en/blog/build-expected-goals-model-python/

## Files

- `requirements.txt` (text)
- `src/fetch_shots.py` (python)
- `src/features.py` (python)
- `src/fit_model.py` (python)
- `src/calibration.py` (python)
- `tests/test_features.py` (python)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Published as open code for the AnalyticaTv article. Adapt it to your own data.
