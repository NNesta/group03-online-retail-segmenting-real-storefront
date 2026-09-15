"""
Churn prediction: train/test split, model training (Logistic Regression,
Random Forest, XGBoost), evaluation, and comparison.

Extracted, unmodified in logic, from `notebooks/supervised_learning.ipynb`,
"Supervised Learning" section, steps 4-13 (cells 59-68).
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import CHURN_FEATURE_COLS, RANDOM_STATE


def make_train_test_split(
    features: pd.DataFrame,
    feature_cols=CHURN_FEATURE_COLS,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split on the churn feature table.

    Mirrors:
        X = features[feature_cols]
        y = features['Churned']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
    """
    X = features[feature_cols]
    y = features["Churned"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on X_train, transform both train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Instantiate the three candidate models with the notebook's hyperparameters.

    Mirrors the `models = {...}` dict in the notebook exactly: Logistic
    Regression, Random Forest, XGBoost.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,       # L2 regularization
            gamma=0.1,            # min loss reduction to split
            min_child_weight=3,   # min sum of instance weight in a child
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


def train_and_evaluate(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
):
    """Fit each model, predict, and compute metrics + train/test AUC gap.

    Logistic Regression uses the scaled features (distance/gradient-based);
    the tree models use raw features. Mirrors the notebook's training loop
    and `train_test_gap` overfitting check exactly.

    Returns
    -------
    (results, train_test_gap)
        results        -- dict[model_name] -> {model, y_pred, y_proba, accuracy,
                           precision, recall, f1, roc_auc}
        train_test_gap -- dict[model_name] -> {train_auc, test_auc, gap}
    """
    results = {}
    train_test_gap = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_proba_train = model.predict_proba(X_train_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_proba_train = model.predict_proba(X_train)[:, 1]

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        train_test_gap[name] = {
            "train_auc": roc_auc_score(y_train, y_proba_train),
            "test_auc": roc_auc_score(y_test, y_proba),
        }
        train_test_gap[name]["gap"] = (
            train_test_gap[name]["train_auc"] - train_test_gap[name]["test_auc"]
        )

    return results, train_test_gap


def build_comparison_table(results: Dict[str, dict]) -> pd.DataFrame:
    """Metric comparison table across models (accuracy/precision/recall/f1/roc_auc)."""
    comparison = pd.DataFrame({
        name: {k: v for k, v in res.items() if k not in ("model", "y_pred", "y_proba")}
        for name, res in results.items()
    }).T
    return comparison


def pick_best_model(results: Dict[str, dict], comparison: pd.DataFrame) -> Tuple[str, dict]:
    """Pick the model with the highest ROC-AUC on the test set."""
    best_name = comparison["roc_auc"].idxmax()
    return best_name, results[best_name]


def feature_importance(best_name: str, best_result: dict, feature_cols=CHURN_FEATURE_COLS) -> pd.Series:
    """Feature importance/coefficients for the best model, ranked by magnitude.

    Mirrors the notebook's final "Feature importance (for the Model
    Interpretation step)" cell.
    """
    if best_name == "Random Forest" or best_name == "XGBoost":
        importance = pd.Series(
            best_result["model"].feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
    else:
        importance = pd.Series(
            best_result["model"].coef_[0], index=feature_cols
        ).sort_values(key=abs, ascending=False)
    return importance
