"""
Interactive CRM Customer Explorer (Streamlit).

Consumes only the artifacts written by `python scripts/run_pipeline.py`
(data/processed/*.csv and models/*.joblib) -- it does no training of its
own, so the app starts in seconds and always reflects the last pipeline
run.

Run with:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

st.set_page_config(page_title="Retail CRM Customer Explorer", page_icon="🛍️", layout="wide")


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_explorer() -> pd.DataFrame:
    return pd.read_csv(config.CUSTOMER_EXPLORER_PATH)


@st.cache_data(show_spinner=False)
def load_profile() -> pd.DataFrame:
    return pd.read_csv(config.SEGMENT_PROFILE_PATH, index_col=0)


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(config.MODEL_COMPARISON_PATH, index_col=0)


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    with open(config.CHURN_MODEL_META_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_rules() -> pd.DataFrame:
    if config.ASSOCIATION_RULES_PATH.exists():
        return pd.read_csv(config.ASSOCIATION_RULES_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_clean_sales() -> pd.DataFrame:
    """Cleaned line items -- used for the per-customer purchase history tab."""
    df = pd.read_csv(config.CLEAN_SALES_PATH, parse_dates=["InvoiceDate"])
    return df


def artifacts_missing() -> list:
    required = [
        config.CUSTOMER_EXPLORER_PATH,
        config.SEGMENT_PROFILE_PATH,
        config.MODEL_COMPARISON_PATH,
        config.CHURN_MODEL_META_PATH,
    ]
    return [p for p in required if not p.exists()]


missing = artifacts_missing()
if missing:
    st.title("🛍️ Retail CRM Customer Explorer")
    st.error("Pipeline artifacts are missing. Run the pipeline first:")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.write("Missing files:")
    for p in missing:
        st.write(f"- `{p.relative_to(config.ROOT_DIR)}`")
    st.stop()


explorer = load_explorer()
profile = load_profile()
comparison = load_comparison()
meta = load_meta()
rules = load_rules()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🛍️ Retail CRM")
page = st.sidebar.radio(
    "View",
    ["Customer Explorer", "Segment Overview", "Churn Model", "Market Basket", "About"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"{len(explorer):,} customers · {len(profile)} segments\n\n"
    f"Best churn model: **{meta['best_model']}** "
    f"(ROC-AUC {meta['metrics']['roc_auc']:.3f})"
)


SEGMENT_COLORS = px.colors.qualitative.Set2


# ==========================================================================
# 1. CUSTOMER EXPLORER
# ==========================================================================
if page == "Customer Explorer":
    st.title("Customer Explorer")
    st.caption("Pick a customer to see their segment, RFM profile and predicted next action.")

    col_a, col_b = st.columns([2, 3])
    with col_a:
        customer_ids = sorted(explorer["Customer ID"].unique())
        customer_id = st.selectbox("Customer ID", customer_ids, index=0)
    with col_b:
        seg_filter = st.multiselect(
            "Filter the list by segment (optional)",
            sorted(explorer["Segment"].dropna().unique()),
        )
        if seg_filter:
            subset = sorted(explorer[explorer["Segment"].isin(seg_filter)]["Customer ID"].unique())
            st.caption(f"{len(subset):,} customers match — pick one above or below.")
            if subset:
                customer_id = st.selectbox("Filtered Customer ID", subset, key="filtered")

    row = explorer[explorer["Customer ID"] == customer_id].iloc[0]

    st.markdown("---")

    # --- Headline segment + action ---
    segment = row["Segment"] if pd.notna(row["Segment"]) else "Unsegmented"
    action = row.get("RecommendedAction")
    st.subheader(f"Customer {int(customer_id)} — {segment}")
    if pd.notna(action):
        st.info(f"**Recommended CRM action:** {action}")

    # --- RFM metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recency (days)", f"{int(row['Recency'])}", help="Days since last purchase")
    m2.metric("Frequency (orders)", f"{int(row['Frequency'])}")
    m3.metric("Monetary (£)", f"{row['Monetary']:,.0f}")
    m4.metric("RFM score", f"{int(row['R_Score'])}{int(row['F_Score'])}{int(row['M_Score'])}")

    # --- Churn prediction ---
    st.markdown("### Predicted next action")
    if pd.notna(row.get("ChurnProbability")):
        prob = float(row["ChurnProbability"])
        c1, c2 = st.columns([1, 2])
        with c1:
            verdict = "Likely to churn" if prob >= 0.5 else "Likely to return"
            st.metric("Churn probability", f"{prob:.0%}", delta=verdict, delta_color="off")
        with c2:
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(
                f"Probability this customer makes **no** purchase in the next "
                f"{meta['holdout_days']} days, per the {meta['best_model']} model."
            )
    else:
        st.warning(
            "No churn prediction for this customer — they had no purchases before the "
            f"model cutoff ({meta['cutoff_date']}), so there were no pre-cutoff features to score."
        )

    # --- Where they sit vs everyone else ---
    st.markdown("### Position in the customer base")
    tab1, tab2 = st.tabs(["PCA position", "RFM percentiles"])

    with tab1:
        fig = px.scatter(
            explorer, x="PCA1", y="PCA2", color="Segment",
            opacity=0.35, color_discrete_sequence=SEGMENT_COLORS,
            title="Customer segments (PCA projection of RFM features)",
        )
        fig.add_scatter(
            x=[row["PCA1"]], y=[row["PCA2"]], mode="markers",
            marker=dict(size=18, color="red", symbol="x", line=dict(width=2)),
            name=f"Customer {int(customer_id)}",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, width='stretch')

    with tab2:
        pct = {
            "Recency": (explorer["Recency"] <= row["Recency"]).mean() * 100,
            "Frequency": (explorer["Frequency"] <= row["Frequency"]).mean() * 100,
            "Monetary": (explorer["Monetary"] <= row["Monetary"]).mean() * 100,
        }
        pct_df = pd.DataFrame({"Metric": list(pct), "Percentile": list(pct.values())})
        fig = px.bar(
            pct_df, x="Percentile", y="Metric", orientation="h", range_x=[0, 100],
            text=pct_df["Percentile"].map(lambda v: f"{v:.0f}th"),
            title="Where this customer ranks against all customers",
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Note: for Recency, a *high* percentile means a long gap since the last "
            "purchase — i.e. worse, not better."
        )

    # --- Purchase history (loaded lazily: the clean sales file is large) ---
    with st.expander("Purchase history (loads the full transaction file)"):
        sales = load_clean_sales()
        hist = sales[sales["Customer ID"] == customer_id].sort_values("InvoiceDate", ascending=False)
        st.write(f"**{hist['Invoice'].nunique():,} orders · {len(hist):,} line items**")
        monthly = (
            hist.set_index("InvoiceDate")["Revenue"]
            .resample("ME").sum().reset_index()
        )
        if not monthly.empty:
            st.plotly_chart(
                px.bar(monthly, x="InvoiceDate", y="Revenue", title="Monthly spend (£)"),
                width='stretch',
            )
        st.dataframe(
            hist[["InvoiceDate", "Invoice", "StockCode", "Description", "Quantity", "Price", "Revenue"]].head(200),
            width='stretch', hide_index=True,
        )

    # --- Download the full action list ---
    st.markdown("---")
    st.download_button(
        "⬇️ Download full customer action list (CSV)",
        explorer.to_csv(index=False).encode("utf-8"),
        file_name="customer_action_list.csv",
        mime="text/csv",
    )


# ==========================================================================
# 2. SEGMENT OVERVIEW
# ==========================================================================
elif page == "Segment Overview":
    st.title("Segment Overview")
    st.caption("Who to court, who to win back, who to let go.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(explorer):,}")
    k2.metric("Total revenue", f"£{explorer['Monetary'].sum():,.0f}")
    k3.metric("Segments", f"{len(profile)}")

    st.markdown("### Segment profiles")
    display = profile.copy()
    display["AvgRecency"] = display["AvgRecency"].round(0)
    display["AvgFrequency"] = display["AvgFrequency"].round(1)
    display["AvgMonetary"] = display["AvgMonetary"].round(0)
    display["TotalRevenue"] = display["TotalRevenue"].round(0)
    display["PctCustomers"] = display["PctCustomers"].round(1)
    display["PctRevenue"] = display["PctRevenue"].round(1)
    st.dataframe(display, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            profile.reset_index().sort_values("Customers"),
            x="Customers", y="Segment", orientation="h",
            color="Segment", color_discrete_sequence=SEGMENT_COLORS,
            title="Customers per segment",
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.bar(
            profile.reset_index().sort_values("TotalRevenue"),
            x="TotalRevenue", y="Segment", orientation="h",
            color="Segment", color_discrete_sequence=SEGMENT_COLORS,
            title="Revenue per segment (£)",
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, width='stretch')

    st.markdown("### PCA projection of all customers")
    fig = px.scatter(
        explorer, x="PCA1", y="PCA2", color="Segment",
        opacity=0.5, color_discrete_sequence=SEGMENT_COLORS,
        hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Recommended CRM action per segment")
    for seg, r in profile.iterrows():
        st.markdown(
            f"**{seg}** — {int(r['Customers']):,} customers "
            f"({r['PctCustomers']:.1f}% of base, {r['PctRevenue']:.1f}% of revenue)  \n"
            f"{r['RecommendedAction']}"
        )


# ==========================================================================
# 3. CHURN MODEL
# ==========================================================================
elif page == "Churn Model":
    st.title("Churn Model")
    st.caption(
        f"Target: no repeat purchase in the {meta['holdout_days']} days after "
        f"{meta['cutoff_date']}. Features come only from pre-cutoff data, so the "
        "model never sees the future it's predicting."
    )

    st.markdown("### Model comparison")
    st.dataframe(comparison.round(3), width='stretch')

    fig = px.bar(
        comparison[["accuracy", "precision", "recall", "f1", "roc_auc"]].reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score"
        ),
        x="Metric", y="Score", color="index", barmode="group", range_y=[0, 1],
        labels={"index": "Model"}, title="Metric comparison across models",
    )
    st.plotly_chart(fig, width='stretch')

    m = meta["metrics"]
    st.markdown(f"### Best model: {meta['best_model']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("Accuracy", f"{m['accuracy']:.3f}")
    c3.metric("Precision", f"{m['precision']:.3f}")
    c4.metric("Recall", f"{m['recall']:.3f}")
    c5.metric("F1", f"{m['f1']:.3f}")

    gap = meta["train_test_auc_gap"]
    st.caption(
        f"Overfitting check — train AUC {gap['train_auc']:.3f} vs test AUC "
        f"{gap['test_auc']:.3f} (gap {gap['gap']:.3f}). A large gap suggests "
        "overfitting; a small gap with low scores overall suggests underfitting."
    )

    st.markdown("### Feature importance")
    imp = pd.Series(meta["feature_importance"]).sort_values()
    fig = px.bar(
        x=imp.values, y=imp.index, orientation="h",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Churn risk across the customer base")
    scored = explorer.dropna(subset=["ChurnProbability"])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.histogram(scored, x="ChurnProbability", nbins=40, title="Predicted churn probability"),
            width='stretch',
        )
    with c2:
        by_seg = scored.groupby("Segment")["ChurnProbability"].mean().sort_values()
        st.plotly_chart(
            px.bar(x=by_seg.values, y=by_seg.index, orientation="h",
                   labels={"x": "Mean churn probability", "y": "Segment"},
                   title="Average churn risk by segment"),
            width='stretch',
        )

    st.markdown("### Highest-value customers at risk")
    threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.5, 0.05)
    at_risk = (
        scored[scored["ChurnProbability"] >= threshold]
        .sort_values("Monetary", ascending=False)
        .head(50)
    )
    st.dataframe(
        at_risk[["Customer ID", "Segment", "Recency", "Frequency", "Monetary",
                 "ChurnProbability", "RecommendedAction"]].round(3),
        width='stretch', hide_index=True,
    )
    st.download_button(
        "⬇️ Download at-risk list (CSV)",
        at_risk.to_csv(index=False).encode("utf-8"),
        file_name=f"at_risk_customers_p{threshold:.2f}.csv",
        mime="text/csv",
    )


# ==========================================================================
# 4. MARKET BASKET
# ==========================================================================
elif page == "Market Basket":
    st.title("Market Basket Analysis")
    st.caption("Customers who bought X also bought Y — association rules mined with mlxtend.")

    if rules.empty:
        st.warning(
            "No association rules found. Re-run the pipeline with the market-basket "
            "step enabled, or lower `min_support` / `min_confidence` in "
            "`src/pipeline.py`."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            min_lift = st.slider(
                "Minimum lift", 1.0, float(max(2.0, rules["lift"].max())), 1.0, 0.1
            )
        with c2:
            min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05)

        filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)]
        st.write(f"**{len(filtered)} rules** match.")
        st.dataframe(filtered.round(3), width='stretch', hide_index=True)

        if not filtered.empty:
            top = filtered.head(20).copy()
            top["Rule"] = top["antecedents"].str.slice(0, 40) + " → " + top["consequents"].str.slice(0, 40)
            fig = px.bar(
                top.sort_values("lift"), x="lift", y="Rule", orientation="h",
                title="Top rules by lift",
            )
            fig.update_layout(height=620)
            st.plotly_chart(fig, width='stretch')

            st.download_button(
                "⬇️ Download rules (CSV)",
                filtered.to_csv(index=False).encode("utf-8"),
                file_name="association_rules.csv",
                mime="text/csv",
            )


# ==========================================================================
# 5. ABOUT
# ==========================================================================
else:
    st.title("About this project")
    st.markdown(
        f"""
**Scenario.** Consulting for an online retailer's new CRM team, who want to know
which customers to court, which to win back, and which to let go before the next
marketing budget is set.

**Data.** Online Retail II (UCI) — over 1,000,000 real transactions from a UK-based
online retailer, Dec 2009–Dec 2011.

**Pipeline.**

1. **Clean** — drop duplicates and rows with no Customer ID, strip out
   administrative stock codes (postage, fees, adjustments), split cancelled
   invoices (`C…`) into a separate returns table, and drop non-positive
   quantities and prices.
2. **Segment (unsupervised)** — engineer Recency / Frequency / Monetary per
   customer, log-transform the skewed F and M, standardize, then K-Means.
   Clusters are named in business terms (VIP, At Risk, One-Time Buyer, …) and
   projected to 2D with PCA.
3. **Predict (supervised)** — a forward-looking churn label: will a customer
   active before **{meta['cutoff_date']}** buy again in the following
   {meta['holdout_days']} days? Features come only from pre-cutoff data.
   Logistic Regression, Random Forest and XGBoost are compared; the best by
   ROC-AUC ({meta['best_model']}) is saved and used here.
4. **Bonus** — market-basket association rules over the best-selling products.

**Reproduce it.**
"""
    )
    st.code("python scripts/run_pipeline.py\nstreamlit run app/streamlit_app.py", language="bash")
    st.markdown(
        "All modelling code lives in `src/` (refactored out of the original "
        "notebooks in `notebooks/`); this app only reads the artifacts the "
        "pipeline writes to `data/processed/` and `models/`."
    )
