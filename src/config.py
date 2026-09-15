"""
Central configuration: filesystem paths and constants shared across the
pipeline (data loading, feature engineering, clustering, churn modelling).

Keeping these in one place means every module (and the Streamlit app)
points at the same files, so `python scripts/run_pipeline.py` and
`streamlit run app/streamlit_app.py` never get out of sync.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"

RAW_EXCEL_PATH = RAW_DATA_DIR / "online_retail_II.xlsx"

# Cleaned line-item transactions (post data-cleaning, pre-feature-engineering)
CLEAN_SALES_PATH = PROCESSED_DATA_DIR / "clean_sales.csv.gz"
RETURNS_PATH = PROCESSED_DATA_DIR / "returns.csv.gz"

# Customer-level RFM + segmentation output
RFM_SEGMENTS_PATH = PROCESSED_DATA_DIR / "rfm_segments.csv"
SEGMENT_PROFILE_PATH = PROCESSED_DATA_DIR / "segment_profile.csv"

# Churn modelling features / predictions
CHURN_FEATURES_PATH = PROCESSED_DATA_DIR / "churn_features.csv"
MODEL_COMPARISON_PATH = PROCESSED_DATA_DIR / "model_comparison.csv"

# Final table used by the Streamlit "Customer Explorer"
CUSTOMER_EXPLORER_PATH = PROCESSED_DATA_DIR / "customer_explorer.csv"

# Market-basket association rules (bonus)
ASSOCIATION_RULES_PATH = PROCESSED_DATA_DIR / "association_rules.csv"

# Saved model artifacts
RFM_SCALER_PATH = MODELS_DIR / "rfm_scaler.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_model.joblib"
PCA_MODEL_PATH = MODELS_DIR / "pca_model.joblib"

CHURN_SCALER_PATH = MODELS_DIR / "churn_scaler.joblib"
CHURN_MODEL_PATH = MODELS_DIR / "churn_best_model.joblib"
CHURN_MODEL_META_PATH = MODELS_DIR / "churn_model_meta.json"

# --------------------------------------------------------------------------
# Cleaning constants (mirrors the notebooks exactly)
# --------------------------------------------------------------------------
ADMIN_STOCK_CODES = [
    "POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES",
    "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "TEST001", "TEST002", "B",
]

# --------------------------------------------------------------------------
# Modelling constants (mirrors the notebooks exactly)
# --------------------------------------------------------------------------
MIN_K_FOR_BUSINESS_USE = 4
K_RANGE = range(3, 9)
RANDOM_STATE = 42

HOLDOUT_DAYS = 180  # "will they buy again in the next 6 months?"

CHURN_FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "Tenure",
    "UniqueProducts", "TotalItems", "AvgOrderValue",
    "AvgItemsPerOrder", "PurchaseRate",
]

ACTION_MAP = {
    "VIP": "Court: white-glove service, early access, loyalty rewards",
    "Loyal Customer": "Court: upsell/cross-sell, referral incentives",
    "New Customer": "Court: onboarding series, second-purchase discount",
    "Win-Back Priority": "Win back: personal outreach, high-value reactivation offer",
    "At Risk": "Win back: targeted discount, re-engagement email flow",
    "Needs Attention": "Monitor: moderate-touch nurture campaign",
    "One-Time Buyer": "Convert: post-purchase follow-up, incentivize 2nd order",
    "Lost / Let Go": "Let go: exclude from paid marketing spend, low-cost win-back only",
}
