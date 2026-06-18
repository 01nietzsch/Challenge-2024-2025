from __future__ import annotations

import random

import joblib
import pandas as pd

from .config import CANDIDATES_CSV, FEATURE_COLUMNS, MODELS_DIR, SUMMARY_DIR


def _generate_compositions(n_samples: int = 1000) -> pd.DataFrame:
    bounds = {
        "fe": (50, 95),
        "c": (0, 2),
        "mn": (0, 10),
        "ni": (0, 5),
        "co": (0, 2),
        "cr": (0, 15),
        "mo": (0, 5),
        "v": (0, 2),
        "n": (0, 0.5),
        "nb": (0, 2),
        "w": (0, 5),
        "al": (0, 2),
        "ti": (0, 2),
        "si": (0, 5),
    }
    rows = []
    for _ in range(n_samples):
        row = {element: random.uniform(*bounds[element]) for element in bounds}
        total = sum(row.values())
        rows.append({k: (v / total) * 100 for k, v in row.items()})
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


def run_optimization(n_samples: int = 1000) -> pd.DataFrame:
    tensile_model = joblib.load(MODELS_DIR / "tensile_strength_regressor.pkl")
    yield_model = joblib.load(MODELS_DIR / "yield_strength_regressor.pkl")
    elongation_model = joblib.load(MODELS_DIR / "elongation_classifier.pkl")

    candidates = _generate_compositions(n_samples=n_samples)
    candidates["pred_tensile_strength"] = tensile_model.predict(candidates[FEATURE_COLUMNS])
    candidates["pred_yield_strength"] = yield_model.predict(candidates[FEATURE_COLUMNS])
    candidates["pred_strength_rating"] = elongation_model.predict(candidates[FEATURE_COLUMNS])

    filtered = candidates[
        (candidates["pred_tensile_strength"] >= 2000)
        & (candidates["pred_yield_strength"] >= 1500)
        & (candidates["pred_strength_rating"].isin(["Medium", "Strong"]))
        & (candidates["ni"] <= 2)
        & (candidates["co"] <= 2)
    ].sort_values(["ni", "co"])

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    filtered.head(25).to_csv(CANDIDATES_CSV, index=False)
    return filtered
