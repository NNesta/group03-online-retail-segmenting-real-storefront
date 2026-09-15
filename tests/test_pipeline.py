"""
Unit tests for the refactored pipeline modules.

These run on a small synthetic transaction table, so they don't need the
45 MB Excel workbook and finish in seconds. They check the behaviours the
notebooks rely on: cancellations are split off, bad rows are dropped, RFM
is computed correctly, and the churn label is built without leakage.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.clean import clean_transactions  # noqa: E402
from src.features.churn import build_churn_features, time_based_split  # noqa: E402
from src.features.rfm import compute_rfm  # noqa: E402
from src.models.clustering import business_segment  # noqa: E402


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny raw table exercising every cleaning rule."""
    rows = [
        # normal sales for customer 1
        ("500001", "85123A", "WHITE HANGING HEART", 6, "2010-01-04 10:00", 2.55, 1.0, "United Kingdom"),
        ("500002", "85123A", "WHITE HANGING HEART", 3, "2010-06-04 10:00", 2.55, 1.0, "United Kingdom"),
        # exact duplicate of the row above -> should be dropped
        ("500002", "85123A", "WHITE HANGING HEART", 3, "2010-06-04 10:00", 2.55, 1.0, "United Kingdom"),
        # customer 2, single order
        ("500003", "71053", "WHITE METAL LANTERN", 2, "2010-02-10 11:00", 3.39, 2.0, "France"),
        # missing Customer ID -> dropped
        ("500004", "71053", "WHITE METAL LANTERN", 2, "2010-03-10 11:00", 3.39, np.nan, "United Kingdom"),
        # admin stock code -> dropped
        ("500005", "POST", "POSTAGE", 1, "2010-03-11 11:00", 18.0, 1.0, "France"),
        # cancellation -> moved to returns
        ("C500006", "85123A", "WHITE HANGING HEART", -6, "2010-03-12 11:00", 2.55, 1.0, "United Kingdom"),
        # non-positive quantity, not a cancellation -> dropped
        ("500007", "22423", "REGENCY CAKESTAND", -1, "2010-03-13 11:00", 12.75, 2.0, "France"),
        # zero price -> dropped
        ("500008", "22423", "REGENCY CAKESTAND", 4, "2010-03-14 11:00", 0.0, 2.0, "France"),
        # customer 3, recent repeat buyer
        ("500009", "22423", "REGENCY CAKESTAND", 4, "2010-09-01 09:00", 12.75, 3.0, "United Kingdom"),
        ("500010", "22423", "REGENCY CAKESTAND", 4, "2010-12-01 09:00", 12.75, 3.0, "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=[
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ])
    df["SourceSheet"] = "2009-2010"
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def test_cleaning_drops_and_splits_correctly(raw_df):
    df_clean, df_returns = clean_transactions(raw_df, verbose=False)

    # cancellations are separated, not deleted
    assert len(df_returns) == 1
    assert df_returns["Invoice"].str.startswith("C").all()
    assert not df_clean["Invoice"].str.startswith("C").any()

    # no missing customer IDs, no admin codes, no non-positive qty/price
    assert df_clean["Customer ID"].notna().all()
    assert "POST" not in set(df_clean["StockCode"])
    assert (df_clean["Quantity"] > 0).all()
    assert (df_clean["Price"] > 0).all()

    # 5 valid sales rows remain (two for customer 1 after the duplicate is
    # dropped, one for customer 2, two for customer 3)
    assert len(df_clean) == 5

    # Revenue is added and correct
    assert "Revenue" in df_clean.columns
    np.testing.assert_allclose(
        df_clean["Revenue"], df_clean["Quantity"] * df_clean["Price"]
    )

    # helper column dropped, Customer ID is an int
    assert "SourceSheet" not in df_clean.columns
    assert pd.api.types.is_integer_dtype(df_clean["Customer ID"])


def test_cleaning_is_idempotent_on_clean_input(raw_df):
    df_clean, _ = clean_transactions(raw_df, verbose=False)
    df_clean2 = df_clean.copy()
    df_clean2["SourceSheet"] = "x"  # the function expects this column
    again, returns_again = clean_transactions(df_clean2, verbose=False)
    assert len(again) == len(df_clean)
    assert returns_again.empty


# --------------------------------------------------------------------------
# RFM
# --------------------------------------------------------------------------
def test_compute_rfm_values(raw_df):
    df_clean, _ = clean_transactions(raw_df, verbose=False)
    snapshot = pd.Timestamp("2011-01-01")
    rfm = compute_rfm(df_clean, snapshot_date=snapshot)

    assert set(rfm.columns) >= {"Customer ID", "Recency", "Frequency", "Monetary"}
    assert len(rfm) == df_clean["Customer ID"].nunique()

    # customer 3 ordered twice, most recently 2010-12-01
    c3 = rfm[rfm["Customer ID"] == 3].iloc[0]
    assert c3["Frequency"] == 2
    assert c3["Recency"] == (snapshot - pd.Timestamp("2010-12-01 09:00")).days
    assert c3["Monetary"] == pytest.approx(2 * 4 * 12.75)

    # Monetary must equal the customer's total cleaned revenue
    totals = df_clean.groupby("Customer ID")["Revenue"].sum()
    merged = rfm.set_index("Customer ID")["Monetary"]
    pd.testing.assert_series_equal(
        merged.sort_index(), totals.sort_index(), check_names=False
    )


def test_compute_rfm_defaults_snapshot_to_day_after_last_sale(raw_df):
    df_clean, _ = clean_transactions(raw_df, verbose=False)
    rfm = compute_rfm(df_clean)
    # the most recent buyer should have Recency of 1 day
    assert rfm["Recency"].min() == 1


# --------------------------------------------------------------------------
# Business segment rules
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scores,expected", [
    ({"R_Score": 5, "F_Score": 5, "M_Score": 5, "Frequency": 12}, "VIP"),
    ({"R_Score": 3, "F_Score": 3, "M_Score": 3, "Frequency": 4}, "Loyal Customer"),
    ({"R_Score": 5, "F_Score": 1, "M_Score": 1, "Frequency": 2}, "New Customer"),
    ({"R_Score": 1, "F_Score": 5, "M_Score": 5, "Frequency": 9}, "Win-Back Priority"),
    ({"R_Score": 1, "F_Score": 3, "M_Score": 2, "Frequency": 5}, "At Risk"),
    ({"R_Score": 1, "F_Score": 1, "M_Score": 1, "Frequency": 2}, "Lost / Let Go"),
    # Frequency == 1 short-circuits everything else
    ({"R_Score": 5, "F_Score": 5, "M_Score": 5, "Frequency": 1}, "One-Time Buyer"),
])
def test_business_segment_rules(scores, expected):
    assert business_segment(pd.Series(scores)) == expected


# --------------------------------------------------------------------------
# Churn features / no leakage
# --------------------------------------------------------------------------
def test_time_based_split_and_churn_label(raw_df):
    df_clean, _ = clean_transactions(raw_df, verbose=False)
    train_window, outcome_window, cutoff = time_based_split(df_clean, holdout_days=180)

    # windows partition the data at the cutoff
    assert (train_window["InvoiceDate"] < cutoff).all()
    assert (outcome_window["InvoiceDate"] >= cutoff).all()
    assert len(train_window) + len(outcome_window) == len(df_clean)

    features = build_churn_features(train_window, outcome_window, cutoff)

    # one row per customer active BEFORE the cutoff only
    assert set(features["Customer ID"]) == set(train_window["Customer ID"])

    # label matches "no purchase in the outcome window"
    active_after = set(outcome_window["Customer ID"])
    for _, row in features.iterrows():
        expected = 0 if row["Customer ID"] in active_after else 1
        assert row["Churned"] == expected

    # engineered features are sane
    assert (features["Frequency"] > 0).all()
    assert (features["AvgOrderValue"] > 0).all()
    assert features["Churned"].isin([0, 1]).all()
    # no future information leaked into the feature window
    assert features["Recency"].min() >= 0


# --------------------------------------------------------------------------
# Cluster -> segment naming
# --------------------------------------------------------------------------
def _rfm_stub(rule_segments, clusters):
    """Build a minimal RFM frame whose business_segment() output is controlled.

    Frequency == 1 forces 'One-Time Buyer'; the R/F/M score combinations below
    force each of the other labels we need for the collision test.
    """
    presets = {
        "VIP": (5, 5, 5, 10),
        "Loyal Customer": (3, 3, 3, 4),
        "At Risk": (1, 3, 2, 5),
        "One-Time Buyer": (5, 5, 5, 1),
    }
    rows = []
    for i, (seg, cluster) in enumerate(zip(rule_segments, clusters)):
        r, f, m, freq = presets[seg]
        rows.append({
            "Customer ID": i,
            "Recency": 10, "Frequency": freq, "Monetary": 100.0,
            "R_Score": r, "F_Score": f, "M_Score": m,
        })
    return pd.DataFrame(rows), np.array(clusters)


def test_assign_segments_labels_every_customer_on_name_collision():
    """Two clusters sharing a modal rule segment must both still get a name.

    This is the bug inherited from the notebook: its dedup logic dropped the
    colliding cluster from the mapping, leaving those customers with NaN.
    """
    from src.models.clustering import assign_segments

    # clusters 0 and 1 are both modally 'VIP'; cluster 1 also contains
    # 'Loyal Customer', so it should fall back to that rather than go unnamed.
    rule_segments = ["VIP"] * 5 + ["VIP"] * 3 + ["Loyal Customer"] * 2
    clusters = [0] * 5 + [1] * 5
    rfm, labels = _rfm_stub(rule_segments, clusters)

    out, mapping = assign_segments(rfm, labels)

    assert out["Segment"].notna().all()
    assert len(set(mapping.values())) == len(mapping)  # names are unique
    assert mapping[0] == "VIP"                         # bigger/first cluster keeps it
    assert mapping[1] == "Loyal Customer"              # collision falls back


def test_assign_segments_simple_case_matches_modal_rule_segment():
    from src.models.clustering import assign_segments

    rule_segments = ["VIP"] * 4 + ["One-Time Buyer"] * 4
    clusters = [0] * 4 + [1] * 4
    rfm, labels = _rfm_stub(rule_segments, clusters)

    out, mapping = assign_segments(rfm, labels)
    assert mapping == {0: "VIP", 1: "One-Time Buyer"}
    assert out.loc[out["Cluster"] == 0, "Segment"].eq("VIP").all()


def test_action_for_handles_suffixed_names():
    from src.models.clustering import action_for
    from src.config import ACTION_MAP

    assert action_for("VIP") == ACTION_MAP["VIP"]
    assert action_for("VIP (2)") == ACTION_MAP["VIP"]
    assert action_for("Something Unmapped")  # never returns None/empty
