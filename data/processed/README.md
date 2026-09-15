# Processed data

**Everything in this folder is generated.** Don't edit these files by hand —
regenerate them with:

```bash
python scripts/run_pipeline.py
```

| File | Grain | What it is |
|---|---|---|
| `clean_sales.csv.gz` | line item | Cleaned sales transactions (cancellations and junk rows removed, `Revenue` added) |
| `returns.csv.gz` | line item | Cancelled invoices (`C…`), set aside during cleaning |
| `rfm_segments.csv` | customer | RFM values, quintile scores, K-Means cluster, business segment, PCA coordinates |
| `segment_profile.csv` | segment | Size, average RFM, revenue share and recommended CRM action per segment |
| `churn_features.csv` | customer | Pre-cutoff features plus the forward-looking `Churned` label |
| `model_comparison.csv` | model | Accuracy / precision / recall / F1 / ROC-AUC for all three classifiers |
| `association_rules.csv` | rule | Market-basket rules with support, confidence and lift |
| `customer_explorer.csv` | customer | Segments joined to churn probabilities — the table the Streamlit app reads |

The two largest files are gzipped; `pandas.read_csv` handles `.gz`
transparently, so no extra step is needed to read them.
