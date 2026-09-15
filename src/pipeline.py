"""
End-to-end pipeline orchestration.

Wires together the extracted notebook logic (src/data, src/features,
src/models) into one reproducible run:

    raw Excel
      -> clean_transactions()                    [src/data/clean.py]
      -> compute_rfm() + K-Means segmentation     [src/features/rfm.py, src/models/clustering.py]
      -> time_based_split() + churn features      [src/features/churn.py]
      -> train/compare churn models               [src/models/churn.py]
      -> (bonus) market-basket association rules  [src/models/market_basket.py]
      -> persist processed data + models + a
         merged "customer explorer" table for the
         Streamlit app                            [src/config.py paths]

Run via `python scripts/run_pipeline.py` (thin CLI wrapper around
`run_full_pipeline` below), not by importing this module inside a
notebook -- it's meant to be the single source of truth for regenerating
every artifact under data/processed/ and models/.
"""
import json
import time

import joblib
import pandas as pd

from src import config
from src.data.clean import clean_transactions
from src.data.load import load_raw_transactions
from src.features.churn import build_churn_features, time_based_split
from src.features.rfm import compute_rfm
from src.models.churn import (
    build_comparison_table,
    build_models,
    feature_importance,
    make_train_test_split,
    pick_best_model,
    scale_features,
    train_and_evaluate,
)
from src.models.clustering import (
    action_for,
    add_pca_projection,
    assign_segments,
    fit_best_kmeans,
    scale_rfm_features,
    segment_profile,
)

try:
    from src.models.market_basket import build_basket_matrix, mine_association_rules
    _MLXTEND_AVAILABLE = True
except ImportError:
    _MLXTEND_AVAILABLE = False


def _step(label: str):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")


def run_full_pipeline(
    excel_path=config.RAW_EXCEL_PATH,
    run_market_basket: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the full pipeline and persist every artifact the Streamlit app needs.

    Returns a dict of summary stats, useful for logging / tests.
    """
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary = {}

    # ------------------------------------------------------------------
    # 1. Load + clean
    # ------------------------------------------------------------------
    _step("1/6  Loading raw transactions")
    df_raw = load_raw_transactions(excel_path)
    print(f"Raw rows: {len(df_raw):,}")

    _step("2/6  Cleaning transactions")
    df_clean, df_returns = clean_transactions(df_raw, verbose=verbose)
    df_clean.to_csv(config.CLEAN_SALES_PATH, index=False)
    df_returns.to_csv(config.RETURNS_PATH, index=False)
    summary["clean_rows"] = len(df_clean)
    summary["returns_rows"] = len(df_returns)

    # ------------------------------------------------------------------
    # 2. RFM + segmentation (unsupervised)
    # ------------------------------------------------------------------
    _step("3/6  RFM feature engineering + K-Means segmentation")
    rfm = compute_rfm(df_clean)
    X, rfm_scaler, X_scaled = scale_rfm_features(rfm)
    kmeans, best_k, labels = fit_best_kmeans(X_scaled)
    rfm, cluster_map = assign_segments(rfm, labels)
    rfm, pca = add_pca_projection(rfm, X_scaled)
    profile = segment_profile(rfm)

    rfm.to_csv(config.RFM_SEGMENTS_PATH, index=False)
    profile.to_csv(config.SEGMENT_PROFILE_PATH)
    joblib.dump(rfm_scaler, config.RFM_SCALER_PATH)
    joblib.dump(kmeans, config.KMEANS_MODEL_PATH)
    joblib.dump(pca, config.PCA_MODEL_PATH)

    print(f"Best k = {best_k} -> segments: {sorted(set(cluster_map.values()))}")
    summary["n_customers"] = len(rfm)
    summary["best_k"] = best_k
    summary["segments"] = sorted(set(cluster_map.values()))

    # ------------------------------------------------------------------
    # 3. Churn features + labels (supervised)
    # ------------------------------------------------------------------
    _step("4/6  Building forward-looking churn features + labels")
    train_window, outcome_window, cutoff = time_based_split(df_clean)
    churn_features = build_churn_features(train_window, outcome_window, cutoff)
    churn_features.to_csv(config.CHURN_FEATURES_PATH, index=False)
    print(f"Cutoff: {cutoff.date()} | churn rate: {churn_features['Churned'].mean():.1%}")
    summary["cutoff_date"] = str(cutoff.date())
    summary["churn_rate"] = float(churn_features["Churned"].mean())

    # ------------------------------------------------------------------
    # 4. Train + compare churn models
    # ------------------------------------------------------------------
    _step("5/6  Training + comparing churn models")
    X_train, X_test, y_train, y_test = make_train_test_split(churn_features)
    X_train_scaled, X_test_scaled, churn_scaler = scale_features(X_train, X_test)

    models = build_models()
    results, train_test_gap = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )
    comparison = build_comparison_table(results)
    comparison.to_csv(config.MODEL_COMPARISON_PATH)
    print(comparison[["accuracy", "precision", "recall", "f1", "roc_auc"]])

    best_name, best_result = pick_best_model(results, comparison)
    importance = feature_importance(best_name, best_result)
    print(f"\nBest model: {best_name} (ROC-AUC={best_result['roc_auc']:.3f})")

    joblib.dump(best_result["model"], config.CHURN_MODEL_PATH)
    joblib.dump(churn_scaler, config.CHURN_SCALER_PATH)
    with open(config.CHURN_MODEL_META_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "feature_cols": config.CHURN_FEATURE_COLS,
            "needs_scaling": best_name == "Logistic Regression",
            "metrics": {k: float(v) for k, v in results[best_name].items()
                        if k not in ("model", "y_pred", "y_proba")},
            "train_test_auc_gap": {k: float(v) for k, v in train_test_gap[best_name].items()},
            "feature_importance": importance.to_dict(),
            "cutoff_date": str(cutoff.date()),
            "holdout_days": config.HOLDOUT_DAYS,
        }, f, indent=2)

    summary["best_model"] = best_name
    summary["best_model_roc_auc"] = float(best_result["roc_auc"])

    # ------------------------------------------------------------------
    # 5. (Bonus) market-basket association rules
    # ------------------------------------------------------------------
    if run_market_basket and _MLXTEND_AVAILABLE:
        _step("6/6  (Bonus) market-basket association rules")
        try:
            basket = build_basket_matrix(df_clean, top_n_products=200)
            rules = mine_association_rules(basket, min_support=0.02, min_confidence=0.3)
            rules.to_csv(config.ASSOCIATION_RULES_PATH, index=False)
            print(f"Found {len(rules)} association rules")
            summary["association_rules"] = len(rules)
        except Exception as exc:  # pragma: no cover - best-effort bonus step
            print(f"Market-basket step skipped ({exc})")
            summary["association_rules"] = 0
    else:
        _step("6/6  (Bonus) market-basket association rules -- skipped")
        summary["association_rules"] = 0

    # ------------------------------------------------------------------
    # 6. Build the merged "customer explorer" table for Streamlit
    # ------------------------------------------------------------------
    _step("Building customer explorer table")
    all_features = churn_features.copy()
    proba_lookup = None
    if best_name == "Logistic Regression":
        X_all_scaled = churn_scaler.transform(all_features[config.CHURN_FEATURE_COLS])
        proba_lookup = best_result["model"].predict_proba(X_all_scaled)[:, 1]
    else:
        proba_lookup = best_result["model"].predict_proba(all_features[config.CHURN_FEATURE_COLS])[:, 1]
    all_features["ChurnProbability"] = proba_lookup
    all_features["ChurnPrediction"] = (all_features["ChurnProbability"] >= 0.5).astype(int)

    explorer = rfm.merge(
        all_features[["Customer ID", "ChurnProbability", "ChurnPrediction"]],
        on="Customer ID", how="left",
    )
    explorer["RecommendedAction"] = explorer["Segment"].map(action_for)
    explorer.to_csv(config.CUSTOMER_EXPLORER_PATH, index=False)
    summary["explorer_rows"] = len(explorer)

    elapsed = time.time() - t0
    _step(f"Pipeline complete in {elapsed:.1f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    run_full_pipeline()
