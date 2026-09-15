"""
RFM (Recency, Frequency, Monetary) feature engineering.

Extracted, unmodified in logic, from
`notebooks/unsupervised_learning.ipynb`, section
"1. RFM feature engineering":

    snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
"""
import pandas as pd


def compute_rfm(df_clean: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """Compute per-customer Recency, Frequency, Monetary features + quintile scores.

    Parameters
    ----------
    df_clean : cleaned SALES transactions (output of `src.data.clean.clean_transactions`).
    snapshot_date : the "as of" date recency is measured against. Defaults
        to one day after the last transaction in df_clean, exactly as in
        the notebook.

    Returns
    -------
    DataFrame, one row per Customer ID, with columns:
        Recency, Frequency, Monetary, R_Score, F_Score, M_Score
    """
    if snapshot_date is None:
        snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()

    # Quintile scores (1-5), used later to translate clusters into business labels.
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    return rfm
