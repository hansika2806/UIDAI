<<<<<<< HEAD
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
=======
# 🇮🇳 UIDAI Data Hackathon 2026  
## Aadhaar System Maturity, Stress & Governance Intelligence Framework

> Transforming Aadhaar Enrollment & Update Data into Policy-Grade Governance Intelligence

---

## 📌 Overview

India’s Aadhaar ecosystem serves over **1.3 billion residents** and has transitioned from an expansion-driven system to a **mature, maintenance-intensive digital infrastructure**.

This project builds a **Governance Intelligence Framework** that transforms raw Aadhaar enrolment and update datasets into:

- System maturity diagnostics  
- Operational stress indicators  
- Maintenance burden metrics  
- Lifecycle-aware forecasting signals  
- Policy-ready state classifications  

The focus shifts from:

> “How many enrolled?”  
to  
> “How efficiently is the system functioning?”

---

## 🎯 Problem Statement

Traditional Aadhaar analytics emphasize enrollment volumes. However:

- Demographic updates exceed enrollments by **5–10x** in several states.
- Update demand shows significantly higher volatility than enrollment.
- Large states experience disproportionate maintenance stress.
- Demographic lifecycle shifts predict future biometric update surges.

This project bridges the gap between raw transaction data and governance intelligence.

---

## 📊 Datasets Used

| Dataset | Description | Role in Analysis |
|----------|------------|------------------|
| Aadhaar Enrolment Dataset | New Aadhaar registrations | Expansion baseline |
| Demographic Update Dataset | Address, name, DOB updates | Maintenance burden |
| Biometric Update Dataset | Fingerprint, iris, face updates | Authentication upkeep |

All datasets were cleaned, standardized, and aggregated at **state–month granularity**.

---

## ⚙️ Data Engineering Pipeline

### Data Cleaning
- Date standardization  
- State & UT normalization  
- Administrative reconciliation  
- Duplicate removal  
- Numeric validation  
- Categorical treatment of pincodes  

### Feature Engineering
- Child vs Adult enrolment shares  
- Total Aadhaar activity  
- Update-to-enrolment ratios  
- Volatility measures  
- Lifecycle demographic indicators  

---

## 🧠 Analytical Framework

### 🔹 Tier 1 — Univariate Analysis
Identifies update dominance and system maturity transition.

### 🔹 Tier 2 — Bivariate Analysis

**Correlation Results**

| Activity Pair | Pearson r |
|---------------|-----------|
| Enrolment ↔ Demographic Updates | 0.61 |
| Demographic Updates ↔ Biometric Updates | 0.57 |
| Enrolment ↔ Biometric Updates | 0.53 |

Spearman rank stability ≈ **0.95–0.97**

**Volatility Diagnostics**

- Enrollment CV ≈ 30–40%  
- Update Pressure CV ≈ 80–120%  

---

### 🔹 Tier 3 — Trivariate Governance Mapping

Three key dimensions:

- **Scale** → Total Aadhaar Activity  
- **Pressure** → Update Pressure Index (UPI)  
- **Stress** → Volatility Stress Index (VSI)  

---

## 📈 Policy-Grade Indicators

### 1️⃣ Aadhaar Maturity Index (AMI)
Composite normalized maturity score.

### 2️⃣ Update Pressure Index (UPI)

```python
UPI = log(Updates_total + 1) - log(Enrolments_total + 1)
```

Measures maintenance intensity.

### 3️⃣ Volatility Stress Index (VSI)
Coefficient of variation of update pressure over time.

### 4️⃣ Temporal Predictability Score (TPS)
Lag-based stability metric between enrolment and updates.

All indicators normalized using Min–Max scaling.

---

## 🚨 Governance Classification

States categorized into:

- Stable Mature Systems  
- High Maintenance Stress  
- Expansion Phase States  
- Unpredictable Systems  
- Balanced / Transitional  

Each category maps directly to targeted policy actions.

---

## 📊 Quantitative Highlights

| Metric | Value |
|--------|-------|
| Pearson (Enrolment–Demographic) | 0.61 |
| Spearman Rank Stability | 0.97 |
| Enrollment Volume CV | ~30–40% |
| Update Pressure CV | ~80–120% |

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Statistical diagnostics (Correlation, CV)  
- Log-based normalization  
- Rank-based composite scoring  
- Euclidean anomaly detection  

---

## 📂 Project Structure

```
├── data/
│   ├── enrolment_dataset.csv
│   ├── demographic_updates.csv
│   └── biometric_updates.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── analysis_framework.ipynb
│   └── indicator_engineering.ipynb
│
├── outputs/
│   ├── state_indicators.csv
│   └── governance_classification.csv
│
└── README.md
```

---

## 🚀 Key Contributions

- Shifted Aadhaar analytics from descriptive reporting to governance intelligence  
- Engineered interpretable policy-grade composite indicators  
- Integrated volatility-aware capacity diagnostics  
- Developed anomaly detection for governance irregularities  
- Enabled lifecycle-based infrastructure forecasting  

---

## 🏁 Conclusion

This project reframes Aadhaar as a mature, lifecycle-driven digital infrastructure requiring intelligent maintenance governance rather than enrollment-only expansion policy.

It delivers a scalable, interpretable, and policy-aligned analytical framework for proactive decision-making.
>>>>>>> 819f66ac64300732888a50e32c6d037cde6a10b5
