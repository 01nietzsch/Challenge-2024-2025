from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

from .config import FEATURE_COLUMNS, FIGURES_DIR, FINAL_DATASET, METRICS_JSON, MODELS_DIR, SUMMARY_DIR


def _save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> str:
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    output = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return str(output)


def run_training() -> dict:
    if not FINAL_DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {FINAL_DATASET}. Run preprocessing first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(FINAL_DATASET)
    data.columns = [col.strip().lower() for col in data.columns]
    data = data[data["strength_rating"].ne("Unknown")]

    X = data[FEATURE_COLUMNS]

    reg_models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(),
        "lasso_regression": Lasso(),
        "random_forest_regression": RandomForestRegressor(n_estimators=100, random_state=42),
        "decision_tree_regression": DecisionTreeRegressor(random_state=42),
    }

    regression_targets = {
        "tensile_strength": data["tensile strength"],
        "yield_strength": data["yield strength"],
        "elongation": data["elongation"],
    }

    metrics: dict[str, dict] = {"regression": {}, "classification": {}}

    for target_name, y in regression_targets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_name, best_model, best_r2 = None, None, float("-inf")
        metrics["regression"][target_name] = {}

        for model_name, model in reg_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            metrics["regression"][target_name][model_name] = {"r2": r2, "mse": mse}
            if r2 > best_r2:
                best_r2 = r2
                best_name, best_model = model_name, model

        joblib.dump(best_model, MODELS_DIR / f"{target_name}_regressor.pkl")
        metrics["regression"][target_name]["best_model"] = best_name

    y = data["strength_rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cls_models = {
        "k_nearest_neighbors": KNeighborsClassifier(),
        "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }

    best_name, best_model, best_acc = None, None, float("-inf")
    for model_name, model in cls_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        figure_path = _save_confusion_matrix(y_test, pred, model_name)
        metrics["classification"][model_name] = {"accuracy": acc, "confusion_matrix": figure_path}
        if acc > best_acc:
            best_acc = acc
            best_name, best_model = model_name, model

    joblib.dump(best_model, MODELS_DIR / "elongation_classifier.pkl")
    metrics["classification"]["best_model"] = best_name

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
