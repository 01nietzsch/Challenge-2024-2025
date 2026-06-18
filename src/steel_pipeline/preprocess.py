from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    FINAL_DATASET,
    IMPUTED_DATASET,
    INTERIM_NORMALIZED,
    INTERIM_DATA_DIR,
    MECHANICAL_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_DATASET,
)


@dataclass
class PreprocessArtifacts:
    normalized_path: str
    imputed_path: str
    final_path: str


def _classify_strength(elongation: float) -> str:
    if pd.isna(elongation):
        return "Unknown"
    if elongation < 5:
        return "Fragile"
    if elongation <= 10:
        return "Medium"
    return "Strong"


def _load_raw_dataset() -> pd.DataFrame:
    if not RAW_DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {RAW_DATASET}")
    data = pd.read_csv(RAW_DATASET)
    data.columns = [col.strip().lower() for col in data.columns]
    if "formula" not in data.columns:
        data = pd.read_csv(RAW_DATASET, skiprows=1)
        data.columns = [col.strip().lower() for col in data.columns]
    if "formula" not in data.columns:
        raise ValueError("Raw dataset must contain a 'formula' column")
    return data


def _prepare_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data["fe"].isna().any():
        non_fe = [c for c in FEATURE_COLUMNS if c != "fe"]
        data.loc[data["fe"].isna(), "fe"] = (100 - data.loc[data["fe"].isna(), non_fe].fillna(0).sum(axis=1)).clip(lower=0)

    feature_only = data[FEATURE_COLUMNS].copy()
    feature_only = feature_only.fillna(0)
    scaled = StandardScaler().fit_transform(feature_only)
    normalized = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=data.index)
    return normalized


def run_preprocessing() -> PreprocessArtifacts:
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = _load_raw_dataset()
    normalized_features = _prepare_feature_matrix(raw)

    normalized_full = pd.concat([raw[["formula"]], normalized_features, raw[[c for c in MECHANICAL_COLUMNS if c in raw.columns]]], axis=1)
    normalized_full.to_excel(INTERIM_NORMALIZED, index=False)

    imputed = normalized_full.copy()
    for col in MECHANICAL_COLUMNS:
        if col not in imputed.columns:
            imputed[col] = pd.NA

    imputer = KNNImputer(n_neighbors=5)
    imputed[MECHANICAL_COLUMNS] = imputer.fit_transform(imputed[MECHANICAL_COLUMNS])
    imputed.to_excel(IMPUTED_DATASET, index=False)

    final_df = imputed.copy()
    final_df["strength_rating"] = final_df["elongation"].apply(_classify_strength)
    final_df.to_excel(FINAL_DATASET, index=False)

    return PreprocessArtifacts(
        normalized_path=str(INTERIM_NORMALIZED),
        imputed_path=str(IMPUTED_DATASET),
        final_path=str(FINAL_DATASET),
    )
