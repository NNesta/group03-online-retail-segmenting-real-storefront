"""
Forward-looking churn label + feature engineering.

Extracted, unmodified in logic, from `notebooks/supervised_learning.ipynb`,
"Supervised Learning" section, steps 1-3:

    Target: will a customer who was active BEFORE the cutoff date make at
    least one more purchase in the 6 months AFTER it? If not, they're
    labeled "churned". Features are built only from pre-cutoff data, so
    the model never sees the future it's trying to predict (no data
    leakage).
"""
from typing import Tuple

import pandas as pd

from src.config import HOLDOUT_DAYS


def time_based_split(df_clean: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split cleaned sales into a pre-cutoff feature window and a post-cutoff outcome window.

    Mirrors:
        cutoff = df_clean['InvoiceDate'].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        train_window = df_clean[df_clean['InvoiceDate'] < cutoff]
        outcome_window = df_clean[df_clean['InvoiceDate'] >= cutoff]
    """
    cutoff = df_clean["InvoiceDate"].max() - pd.Timedelta(days=holdout_days)
    train_window = df_clean[df_clean["InvoiceDate"] < cutoff]
    outcome_window = df_clean[df_clean["InvoiceDate"] >= cutoff]
    return train_window, outcome_window, cutoff


def build_churn_features(train_window: pd.DataFrame, outcome_window: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Build pre-cutoff RFM-style features and the forward-looking Churned label.

    Mirrors:
        snapshot_date = cutoff
        features = train_window.groupby('Customer ID').agg(
            Recency=..., Frequency=..., Monetary=...,
            FirstPurchase=..., UniqueProducts=..., TotalItems=...,
        ).reset_index()
        features['Tenure'] = (snapshot_date - features['FirstPurchase']).dt.days
        features['AvgOrderValue'] = features['Monetary'] / features['Frequency']
        features['AvgItemsPerOrder'] = features['TotalItems'] / features['Frequency']
        features['PurchaseRate'] = features['Frequency'] / features['Tenure'].replace(0, 1)
        features['Churned'] = (~features['Customer ID'].isin(active_after_cutoff)).astype(int)
    """
    snapshot_date = cutoff

    features = train_window.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        UniqueProducts=("StockCode", "nunique"),
        TotalItems=("Quantity", "sum"),
    ).reset_index()

    features["Tenure"] = (snapshot_date - features["FirstPurchase"]).dt.days
    features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]
    features["AvgItemsPerOrder"] = features["TotalItems"] / features["Frequency"]
    features["PurchaseRate"] = features["Frequency"] / features["Tenure"].replace(0, 1)
    features = features.drop(columns=["FirstPurchase"])

    active_after_cutoff = set(outcome_window["Customer ID"].unique())
    features["Churned"] = (~features["Customer ID"].isin(active_after_cutoff)).astype(int)

    return features
