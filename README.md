# UIDAI Intelligence Studio

This project turns the full `datahack.ipynb` analysis into a reusable UIDAI analytics pipeline and a professional Streamlit dashboard. The goal is not to discard notebook work, but to preserve every meaningful analytical block and expose it as a production-style exploration surface.

## What the project now includes

- Data cleaning for enrolment, demographic update, and biometric update files
- State and UT standardization, including merged UT normalization
- Analysis-ready datasets with time attributes and total activity fields
- State-month analytical features:
  - update-to-enrolment ratios
  - demographic and biometric update intensity
  - lifecycle shares for child enrolment and adult updates
  - total activity and stress proxies
- State-level analytical outputs:
  - Aadhaar Maturity Index (AMI)
  - Update Pressure Index (UPI)
  - Volatility Stress Index (VSI)
  - Temporal Predictability Score (TPS)
  - policy categories
  - governance status and recommended policy actions
  - anomaly flags and priority scoring
- Additional notebook-derived outputs:
  - activity correlation matrix
  - indicator correlation matrix
  - rank persistence tables
  - lag features
  - Pareto concentration summary
  - national monthly summary

## Project structure

- [datahack.ipynb](/C:/Users/nagah/Projects/UIDAI-data/datahack.ipynb)
  Source notebook containing the original analysis flow.
- [src/data_cleaning.py](/C:/Users/nagah/Projects/UIDAI-data/src/data_cleaning.py)
  Cleans raw UIDAI files and standardizes state names.
- [src/analytics.py](/C:/Users/nagah/Projects/UIDAI-data/src/analytics.py)
  Shared analytics engine that reproduces the notebook logic.
- [src/feature_engineering.py](/C:/Users/nagah/Projects/UIDAI-data/src/feature_engineering.py)
  Builds the full analysis output bundle for the dashboard.
- [dashboard/app.py](/C:/Users/nagah/Projects/UIDAI-data/dashboard/app.py)
  Dynamic Streamlit app with filter-driven recomputation.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place raw UIDAI files in [data/raw](/C:/Users/nagah/Projects/UIDAI-data/data/raw):

- `api_data_aadhar_enrolment.zip`
- `api_data_aadhar_demographic.zip`
- `api_data_aadhar_biometric.zip`

3. Run the pipeline:

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/visualization.py
```

4. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Dashboard sections

- Executive Overview
- Governance Diagnostics
- Lifecycle and Operations
- Anomalies and Risk
- State Drilldown
- Data Export

Each view is designed to use the notebook’s analytical outputs directly, and the dashboard recomputes metrics when the user changes date ranges or state scope.

## Core analytical interpretation

- `AMI`
  Higher means more mature Aadhaar activity footprint.
- `UPI`
  Higher means stronger transition from enrolment to maintenance/update burden.
- `VSI`
  Higher means more unstable update pressure over time.
- `TPS`
  Higher means more predictable temporal relationship between enrolment and later update behavior.

## Important note

The dashboard is production-ready from a code perspective, but it still needs the real UIDAI datasets in `data/raw/` to generate analysis outputs. The repository does not currently include those raw files.
