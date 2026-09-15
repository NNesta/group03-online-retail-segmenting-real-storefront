"""
Data cleaning pipeline.

Extracted, unmodified in logic, from the "## Data Cleaning" section that is
identical in both `notebooks/unsupervised_learning.ipynb` and
`notebooks/supervised_learning.ipynb` (cells 13-32):

    1. Fix data types
    2. Remove exact duplicate rows
    3. Drop rows with missing Customer ID
    4. Drop rows with missing Description
    5. Remove non-product / administrative stock codes
    6. Separate cancellations (invoices starting with 'C') into df_returns
    7. Remove non-positive quantities and prices
    8. (optional sanity cap on extreme prices - left as a no-op, as in the
       notebooks, where it's commented out)
    9. Add the Revenue column

Each step mirrors the notebook cell-for-cell so the two are easy to
cross-check; the only change is wrapping it as a function that returns
(df_clean, df_returns) instead of printing to a notebook cell.
"""
from typing import Tuple

import pandas as pd

from src.config import ADMIN_STOCK_CODES


def clean_transactions(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline on raw transaction rows.

    Parameters
    ----------
    df : raw concatenated transactions, as returned by
        `src.data.load.load_raw_transactions`.
    verbose : print the same step-by-step summary the notebooks print.

    Returns
    -------
    (df_clean, df_returns)
        df_clean   -- cleaned SALES rows only (cancellations removed),
                      with a `Revenue` column added.
        df_returns -- cancelled-order rows (Invoice starting with 'C'),
                      kept separately in case they're needed later.
    """
    df = df.copy()

    # ---------------------------------------------------------------
    # 1. Fix data types
    # ---------------------------------------------------------------
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    # ---------------------------------------------------------------
    # 2. Remove exact duplicate rows
    # ---------------------------------------------------------------
    before = len(df)
    df_clean = df.drop_duplicates(list(df.columns))
    if verbose:
        print(f"Dropped {before - len(df_clean)} exact duplicate rows -> {len(df_clean)} left")

    # ---------------------------------------------------------------
    # 3. Drop rows with missing Customer ID
    # ---------------------------------------------------------------
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["Customer ID"])
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    if verbose:
        print(f"Dropped {before - len(df_clean)} rows with missing Customer ID -> {len(df_clean)} left")

    # ---------------------------------------------------------------
    # 4. Drop rows with missing Description
    # ---------------------------------------------------------------
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["Description"])
    if verbose:
        print(f"Dropped {before - len(df_clean)} rows with missing Description -> {len(df_clean)} left")

    # ---------------------------------------------------------------
    # 5. Remove non-product / administrative stock codes
    # ---------------------------------------------------------------
    # These codes are postage, fees, manual adjustments, bank charges,
    # samples, discounts and test entries -- not real products, so they'd
    # distort RFM, segmentation and product-level analysis.
    before = len(df_clean)
    df_clean = df_clean[~df_clean["StockCode"].isin(ADMIN_STOCK_CODES)]
    if verbose:
        print(f"Dropped {before - len(df_clean)} rows with admin/non-product stock codes -> {len(df_clean)} left")

    # ---------------------------------------------------------------
    # 6. Separate cancellations (returns) from the sales dataset
    # ---------------------------------------------------------------
    # Invoices starting with 'C' are cancellations. Keep them in a separate
    # dataframe in case you want to analyze returns later, but exclude them
    # from the "sales" dataset used for RFM/segmentation/prediction.
    is_cancelled = df_clean["Invoice"].str.startswith("C")
    df_returns = df_clean[is_cancelled].copy()
    df_clean = df_clean[~is_cancelled].copy()
    if verbose:
        print(f"Separated {len(df_returns)} cancelled-order rows into df_returns -> {len(df_clean)} sales rows left")

    # ---------------------------------------------------------------
    # 7. Remove non-positive quantities and prices
    # ---------------------------------------------------------------
    # A handful of negative-quantity rows aren't flagged as cancellations,
    # and zero/negative prices are data errors or free/adjustment items
    # with no real revenue signal.
    before = len(df_clean)
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
    if verbose:
        print(f"Dropped {before - len(df_clean)} rows with Quantity <= 0 or Price <= 0 -> {len(df_clean)} left")

    # ---------------------------------------------------------------
    # 8. Remove extreme price outliers (optional sanity cap)
    # ---------------------------------------------------------------
    # A few Price values are in the thousands and clearly not per-unit
    # retail prices. The notebooks leave this as a flag-only comment
    # rather than dropping rows, so we do the same here (no-op).
    # price_cap = df_clean['Price'].quantile(0.999)
    # n_outliers = (df_clean['Price'] > price_cap).sum()

    # ---------------------------------------------------------------
    # 9. Add the Revenue column (needed for EDA/RFM downstream)
    # ---------------------------------------------------------------
    df_clean = df_clean.drop(columns=["SourceSheet"])
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["Price"]

    df_clean = df_clean.reset_index(drop=True)
    df_returns = df_returns.reset_index(drop=True)

    if verbose:
        print("\n==== CLEANING SUMMARY ====")
        print(f"Raw rows:        {len(df):,}")
        print(f"Clean sales rows:{len(df_clean):,}")
        print(f"Returns rows:    {len(df_returns):,}")

    return df_clean, df_returns
