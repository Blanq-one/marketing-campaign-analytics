# Marketing Campaign Analytics

Predict which customers will respond to a marketing campaign using machine learning, explain the key drivers with SHAP, and export results to Power BI / Tableau dashboards.

Built with **Python**, **scikit-learn**, **LightGBM**, **XGBoost**, and **SMOTE** for class-imbalance handling.

## Dataset

The pipeline uses a bank-marketing-style campaign dataset (~11k rows) with features like:

| Feature | Description |
|---------|-------------|
| `age`, `job`, `marital`, `education` | Customer demographics |
| `balance`, `income` | Financial indicators |
| `contact`, `month`, `campaign` | Campaign interaction details |
| `pdays`, `previous`, `poutcome` | Previous campaign history |
| `response` | Target variable (1 = responded, 0 = did not) |

## Notebooks

The analysis is broken into four sequential notebooks:

| Notebook | Description |
|----------|-------------|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Explore distributions, correlations, response rates by segment, and campaign trends |
| [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | Build engineered features — tenure, aggregates, ratios, interaction terms, customer segments |
| [`03_model_training.ipynb`](notebooks/03_model_training.ipynb) | Train logistic regression baseline and XGBoost with SMOTE oversampling |
| [`04_model_evaluation.ipynb`](notebooks/04_model_evaluation.ipynb) | Evaluate holdout performance — ROC-AUC, precision-recall, confusion matrix |

## Results

Evaluated on the campaign dataset using LightGBM:

| Metric    | Holdout | 5-Fold CV (mean +/- std) |
|-----------|---------|--------------------------|
| ROC-AUC   | 0.93    | 0.925 +/- 0.002          |
| Precision | 0.83    | 0.83 +/- 0.004           |
| Recall    | 0.88    | 0.88 +/- 0.006           |
| F1        | 0.86    | 0.85 +/- 0.005           |

LightGBM improves ROC-AUC by ~2% over the logistic regression baseline while keeping a strong precision-recall balance, enabling marketing teams to target high-value customers and cut wasted outreach.

## Getting Started

```bash
# clone the repo
git clone https://github.com/<your-username>/marketing-campaign-analytics.git
cd marketing-campaign-analytics

# set up environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# install dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python -m src.pipeline --csv data/raw/campaign_data.csv --target response
```

Metrics and a confusion matrix print to the terminal. Artifacts (model, predictions, plots, SHAP) are saved to `data/processed/run_YYYYMMDD_HHMMSS/`.

### CLI flags

| Flag | Description |
|------|-------------|
| `--model logistic\|xgboost\|lightgbm` | Model selection (default: `lightgbm`) |
| `--cv` | Run 5-fold cross-validation with baseline comparison |
| `--no-shap` | Skip SHAP explanations (faster) |
| `--no-smote` | Disable SMOTE oversampling |
| `--db-url` / `--table` | Load data from PostgreSQL instead of CSV |

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation & validation
│   ├── 03_model_training.ipynb       # Model training & comparison
│   └── 04_model_evaluation.ipynb     # Evaluation & visualization
├── src/
│   ├── pipeline.py                   # End-to-end CLI pipeline
│   ├── config/settings.py            # Environment-based configuration
│   ├── models/                       # Logistic, XGBoost, LightGBM wrappers
│   ├── services/                     # Data loading, preprocessing, training, evaluation
│   └── views/                        # Plotting utilities & SHAP explainer
├── dashboards/                       # BI export scripts (Power BI / Tableau)
├── data/
│   ├── raw/campaign_data.csv         # Input dataset
│   └── processed/                    # Pipeline run outputs
├── sql/schema.sql                    # Example database schema
├── docs/API.md                       # Module & function reference
└── requirements.txt
```

## Dashboard Exports

Generate Power BI / Tableau-ready CSVs from any pipeline run:

```bash
python -m dashboards.export_dashboard_data
python -m dashboards.segment_dashboard
python -m dashboards.feature_importance_dashboard
```

Outputs land in `dashboards/exports/`. See the [Power BI template guide](dashboards/POWER_BI_TEMPLATE.md) for step-by-step dashboard setup.

## Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Data:** pandas, NumPy, SQLAlchemy
- **Visualization:** matplotlib, seaborn
- **Database:** PostgreSQL (optional)
