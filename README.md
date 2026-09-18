# Retail CRM Analytics — Online Retail II

Customer segmentation and churn prediction for a UK-based online retailer, developed for a CRM team that needs to understand customer value, identify customers at risk of churning, and prioritize retention activities.

## Project Overview

This Group 03 project uses the **Online Retail II** dataset to:

- Clean and prepare more than one million transaction records.
- Explore sales, customer, product, and country-level patterns.
- Engineer customer-level **Recency, Frequency, and Monetary (RFM)** features.
- Segment customers using **K-Means clustering**.
- Predict customer churn using supervised machine learning.
- Discover product associations through market-basket analysis.
- Provide an interactive **Streamlit Customer Explorer** application.

### Project Pipeline

```text
Raw Excel data
      │
      ▼
Data cleaning
      │
      ▼
RFM feature engineering
      │
      ▼
K-Means customer segmentation ───► PCA visualization
      │
      ▼
Churn feature engineering
      │
      ▼
Logistic Regression / Random Forest / XGBoost
      │
      ▼
Processed outputs and trained models
      │
      ▼
Streamlit Customer Explorer
```

## Contributors

| Contributor | RegN0 | Focus Area | Key Contributions |
|---|---|---|---|
| **Nestor Ngabonziza** | **20251MBI022** | Project integration | K-Means clustering fix, Streamlit app setup, development container setup, pipeline integration, and README maintenance |
| **Justine Mudahogora** | **20251MBI016** | Streamlit and documentation | Initial README, Streamlit dashboard structure, compiled model artifacts, and repository cleanup |
| **Arthur Butera** | **20251MBI054** |Supervised learning and environment | Churn prediction models, project environment, requirements, and pipeline scripts |
| **NIYOMUFASHA Emmerance** | **20251MBI019** | Streamlit support | Helper functions used by the Customer Explorer application |
| **Nsabimana Jean Paul** | **20251MBI008** | Model Cross checking | Model test pipeline ,Unsupervised model , and final project report |

Contributions included peer review, collaboration, and pull-request integration across the team.

## Project Contents

| Component | Description |
|---|---|
| Data cleaning | Handles duplicates, missing Customer IDs, administrative stock codes, cancellations, invalid quantities, and invalid prices |
| Exploratory Data Analysis | Examines revenue trends, top products, country distribution, basket sizes, cancellations, and customer concentration |
| Customer segmentation | Uses RFM features, transformations, K-Means clustering, business segment labels, and PCA |
| Churn prediction | Compares Logistic Regression, Random Forest, and XGBoost using a time-based churn definition |
| Market-basket analysis | Mines product association rules using `mlxtend` |
| Streamlit application | Allows users to explore customer profiles, segments, churn probabilities, and recommended actions |

## Project Structure

```text
online-retail-crm/
├── README.md
├── requirements.txt
├── environment.yml
├── Makefile
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   └── processed/
│       ├── clean_sales.csv.gz
│       ├── returns.csv.gz
│       ├── rfm_segments.csv
│       ├── segment_profile.csv
│       ├── churn_features.csv
│       ├── model_comparison.csv
│       ├── association_rules.csv
│       └── customer_explorer.csv
│
├── models/
│   ├── rfm_scaler.joblib
│   ├── kmeans_model.joblib
│   ├── pca_model.joblib
│   ├── churn_scaler.joblib
│   ├── churn_best_model.joblib
│   └── churn_model_meta.json
│
├── notebooks/
│   ├── unsupervised_learning.ipynb
│   └── supervised_learning.ipynb
│
│
├── Report/
│   └── Final_Project_Report.pdf
│
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── data/
│   │   ├── load.py
│   │   └── clean.py
│   ├── features/
│   │   ├── rfm.py
│   │   └── churn.py
│   └── models/
│       ├── clustering.py
│       ├── churn.py
│       └── market_basket.py
│
├── scripts/
│   └── run_pipeline.py
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    └── test_pipeline.py
```

## Requirements

- Python **3.9 or higher**
- Python 3.11 is recommended
- Approximately 2 GB of free memory for the complete pipeline
- The Online Retail II Excel workbook

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/NNesta/group03-online-retail-segmenting-real-storefront
cd group03-online-retail-segmenting-real-storefront
```

### 2. Create and activate an environment

Using `venv`:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, using Conda:

```bash
conda env create -f environment.yml
conda activate retail-crm
```

### 3. Add the dataset

Download the **Online Retail II** dataset from the UCI Machine Learning Repository and place it at:

```text
data/raw/online_retail_II.xlsx
```

The workbook must contain the following sheets:

```text
Year 2009-2010
Year 2010-2011
```

### 4. Run the pipeline

```bash
python scripts/run_pipeline.py
```

### 5. Launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

The application can also be accessed through the deployed Streamlit page:

<https://group03-online-retail.streamlit.app/>

## Using Make

If `make` is available, the following commands can be used:

```bash
make install
make pipeline
make app
```

## Data Source

The project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

- **Dataset:** Online Retail II
- **Source:** UCI Machine Learning Repository
- **Period:** December 2009 to December 2011
- **Domain:** UK-based online retailer selling gifts and homeware
- **Raw records:** 1,067,371 transaction lines

Dataset page:

<https://archive.ics.uci.edu/dataset/502/online+retail+ii>

## Running the Pipeline

The main pipeline can be executed with:

```bash
python scripts/run_pipeline.py
```

### Available options

```bash
python scripts/run_pipeline.py --no-market-basket
python scripts/run_pipeline.py --excel path/to/online_retail_II.xlsx
python scripts/run_pipeline.py --quiet
```

| Option | Description |
|---|---|
| `--no-market-basket` | Skips the optional market-basket analysis |
| `--excel` | Specifies a custom path to the Excel workbook |
| `--quiet` | Suppresses cleaning-step logs |

The pipeline is deterministic, using `random_state=42`, and is designed to be idempotent. Re-running it regenerates the processed outputs and model artifacts.

### Pipeline Steps

1. Load and combine both Excel sheets.
2. Clean the transaction data.
3. Save cleaned sales and returns data.
4. Calculate customer-level RFM features.
5. Fit K-Means clustering and assign business segment names.
6. Generate PCA coordinates for visualization.
7. Build time-based churn features and labels.
8. Train and evaluate the three churn models.
9. Save the best-performing model and metadata.
10. Mine market-basket association rules.
11. Merge customer information into the file used by the Streamlit app.

## Data Cleaning

The cleaning process is implemented in `src/data/clean.py`.

| Step | Action | Purpose |
|---|---|---|
| 1 | Fix data types and standardize string columns | Ensures consistent processing |
| 2 | Remove exact duplicate rows | Prevents duplicate transactions from distorting results |
| 3 | Remove rows without a Customer ID | Ensures customer-level analysis is possible |
| 4 | Remove rows without a product description | Removes incomplete product records |
| 5 | Remove administrative stock codes | Excludes postage, fees, adjustments, and test entries |
| 6 | Separate cancelled invoices beginning with `C` | Keeps returns available for separate analysis |
| 7 | Remove non-positive quantities and prices | Excludes invalid sales records |
| 8 | Flag extreme price values | Preserves potentially meaningful outliers |
| 9 | Add `Revenue = Quantity × Price` | Creates the main revenue measure |

The complete run produced approximately:

- **1,067,371** raw transaction rows
- **790,721** clean sales rows
- **17,879** returns rows
- **5,852** customers with a Customer ID

## Customer Segmentation

### RFM Features

For each customer, the following features are calculated:

- **Recency:** Number of days since the customer’s most recent order.
- **Frequency:** Number of distinct invoices placed.
- **Monetary:** Total revenue generated by the customer.

Frequency and Monetary are log-transformed before standardization because both variables are strongly right-skewed.

### Clustering Method

K-Means clustering is evaluated for values of `k` from 3 to 8. The final solution uses:

```text
k = 5
```

The choice balances clustering quality with the CRM requirement for sufficiently detailed and actionable customer groups.

PCA is used to project the standardized RFM space into two dimensions for visualization.

### Customer Segments

| Segment | Customers | % of Base | Avg. Recency (Days) | Avg. Orders | Avg. Spend (£) | % of Revenue |
|---|---:|---:|---:|---:|---:|---:|
| VIP | 900 | 15.4% | 42 | 22.8 | 13,537 | 70.1% |
| Loyal Customer | 1,692 | 28.9% | 64 | 5.7 | 1,941 | 18.9% |
| At Risk | 840 | 14.4% | 394 | 3.4 | 1,291 | 6.2% |
| One-Time Buyer | 1,340 | 22.9% | 99 | 1.7 | 412 | 3.2% |
| Lost / Let Go | 1,080 | 18.5% | 521 | 1.2 | 254 | 1.6% |

### Suggested CRM Actions

| Segment | Suggested Action |
|---|---|
| VIP | White-glove service, early access, and loyalty rewards |
| Loyal Customer | Upselling, cross-selling, and referral incentives |
| At Risk | Targeted discounts and re-engagement campaigns |
| One-Time Buyer | Post-purchase follow-up and second-order incentives |
| Lost / Let Go | Low-cost automated win-back communication and limited paid retention spend |

## Churn Prediction

### Prediction Definition

A customer is labelled as **churned** if they make no purchase during the 180 days following the selected cutoff date.

The model uses only information available before the cutoff date to avoid data leakage.

### Features

The churn models use nine customer-level features:

```text
Recency
Frequency
Monetary
Tenure
UniqueProducts
TotalItems
AvgOrderValue
AvgItemsPerOrder
PurchaseRate
```

### Models Compared

| Model | Description |
|---|---|
| Logistic Regression | Interpretable baseline trained on standardized features |
| Random Forest | Ensemble of decision trees that captures non-linear relationships |
| XGBoost | Gradient-boosted tree model with regularization |

The models are evaluated using a stratified 25% test split and the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.726 | 0.722 | 0.699 | 0.711 | 0.797 |
| **Random Forest** | **0.725** | 0.706 | 0.734 | 0.720 | **0.807** |
| XGBoost | 0.725 | 0.702 | 0.741 | 0.721 | 0.804 |

Random Forest achieved the highest ROC-AUC in the reported evaluation and was saved as the best model. XGBoost achieved the highest recall, while Logistic Regression remained the most interpretable model.

### Most Important Churn Features

The most influential features in the Random Forest model were:

| Feature | Importance |
|---|---:|
| Recency | 0.23 |
| PurchaseRate | 0.18 |
| Monetary | 0.14 |
| TotalItems | 0.13 |

Recency was the strongest predictor, indicating that the time since a customer’s last purchase is particularly important for identifying potential churn.

## Market-Basket Analysis

The bonus market-basket analysis uses `mlxtend`'s Apriori algorithm on a one-hot invoice-by-product matrix.

To keep the analysis tractable, it is restricted to the top 200 best-selling products in the UK market.

The resulting association rules include:

- Antecedents
- Consequents
- Support
- Confidence
- Lift

The Streamlit application allows users to filter the rules by lift and confidence.

## Streamlit Customer Explorer

The Streamlit application reads the outputs generated by the pipeline and does not retrain the models.

It contains five main views:

| View | Description |
|---|---|
| **Customer Explorer** | Search for a customer and view segment, RFM metrics, churn probability, recommended action, PCA position, percentile ranks, and purchase history |
| **Segment Overview** | Explore segment sizes, revenue contribution, PCA visualization, and recommended actions |
| **Churn Model** | Review model performance, feature importance, overfitting checks, and high-value customers at risk |
| **Market Basket** | Explore product association rules using lift and confidence filters |
| **About** | View a summary of the methodology |

If the required pipeline artifacts are missing, the application displays setup instructions.

## Outputs Reference

| File | Level | Main Contents |
|---|---|---|
| `data/processed/clean_sales.csv.gz` | Transaction line | Clean sales transactions and revenue |
| `data/processed/returns.csv.gz` | Transaction line | Cancelled invoices |
| `data/processed/rfm_segments.csv` | Customer | RFM values, scores, cluster, segment, and PCA coordinates |
| `data/processed/segment_profile.csv` | Segment | Segment statistics and recommended actions |
| `data/processed/churn_features.csv` | Customer | Churn features and churn label |
| `data/processed/model_comparison.csv` | Model | Accuracy, precision, recall, F1-score, and ROC-AUC |
| `data/processed/association_rules.csv` | Product rule | Support, confidence, and lift |
| `data/processed/customer_explorer.csv` | Customer | Segment, churn probability, prediction, and recommended action |
| `models/churn_model_meta.json` | Model metadata | Best model, metrics, AUC gap, and feature importance |

## Notebooks and Source Code

The notebooks contain the original exploratory analysis and visualizations:

```text
notebooks/unsupervised_learning.ipynb
notebooks/supervised_learning.ipynb
```

Reusable logic was extracted into the `src/` package:

| Functionality | Module |
|---|---|
| Data loading | `src/data/load.py` |
| Data cleaning | `src/data/clean.py` |
| RFM engineering | `src/features/rfm.py` |
| Churn feature engineering | `src/features/churn.py` |
| Clustering and PCA | `src/models/clustering.py` |
| Churn model training and evaluation | `src/models/churn.py` |
| Market-basket analysis | `src/models/market_basket.py` |
| Pipeline orchestration | `src/pipeline.py` |

The refactored clustering implementation also ensures that every customer receives a valid segment label, including cases where multiple clusters initially receive the same business label.

## Testing

Run the test suite with:

```bash
pytest -q
```

The tests use a small synthetic transaction dataset and cover:

- Data cleaning rules
- Duplicate removal
- Missing Customer ID handling
- Administrative stock-code removal
- Cancellation separation
- Invalid quantity and price handling
- Revenue calculation
- Cleaning idempotency
- RFM calculations
- Business segment rules
- Time-based churn splitting
- Churn label correctness
- Leakage checks
- Segment-label assignment

## Troubleshooting

### Raw data not found

Ensure the workbook is located at:

```text
data/raw/online_retail_II.xlsx
```

Alternatively, provide a custom path:

```bash
python scripts/run_pipeline.py --excel path/to/online_retail_II.xlsx
```

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root:

```bash
python scripts/run_pipeline.py
streamlit run app/streamlit_app.py
```

### Streamlit reports missing artifacts

Run the pipeline first:

```bash
python scripts/run_pipeline.py
```

### Pipeline is slow or runs out of memory

The Excel import is the main bottleneck and may require approximately 2 GB of free memory.

You can skip the optional market-basket step:

```bash
python scripts/run_pipeline.py --no-market-basket
```

### No association rules are found

Lower the `min_support` or `min_confidence` settings in the market-basket configuration.

## Limitations and Future Work

The project has several limitations:

1. The UK accounts for most of the dataset’s revenue, so the findings mainly represent UK customer behavior.
2. The features are primarily transaction-based and do not include marketing interactions, customer service contacts, demographics, or product preferences.
3. Tree-based models show a train-test performance gap, indicating some overfitting.
4. The 180-day churn window is a business-defined threshold and may affect the results.
5. Segment and churn labels should be used for marketing prioritization, not to reduce the quality of customer service.

Potential future improvements include:

- Integrating marketing campaign and customer service data.
- Testing alternative churn windows, such as 90 and 365 days.
- Improving tree-model regularization.
- Evaluating the impact of retention campaigns.
- Developing an internal dashboard for direct customer lookup and monitoring.

## References

1. Chen, D., & UCI Machine Learning Repository. **Online Retail II Data Set**.  
   <https://archive.ics.uci.edu/dataset/502/online+retail+ii>

2. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python**. *Journal of Machine Learning Research, 12*, 2825–2830.

3. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

---

**Group 03 — Retail CRM Analytics**
