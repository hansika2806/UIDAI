# UIDAI Governance Intelligence Framework
### Aadhaar System Maturity, Stress & Policy Intelligence — UIDAI Data Hackathon 2026

> Transforming raw Aadhaar enrolment and update data into policy-grade governance intelligence at state–month granularity.

---

## Overview

India's Aadhaar ecosystem serves over **1.3 billion residents** and has long since transitioned from an expansion-driven system into a **mature, maintenance-intensive digital infrastructure**. Traditional analytics ask *"how many enrolled?"* — this project asks a harder question:

> **How efficiently is the system functioning, and where is it under stress?**

This framework engineers four composite governance indicators, classifies every Indian state into policy-action categories, and surfaces everything through a production-ready Streamlit dashboard with filter-driven recomputation.

---

## Analytical Framework

The analysis proceeds in three tiers.

**Tier 1 — Univariate Analysis**  
Establishes update dominance and system maturity transition signals. In several states, demographic updates exceed enrolments by 5–10×.

**Tier 2 — Bivariate Analysis**  
Quantifies inter-activity relationships and volatility asymmetry.

| Activity Pair | Pearson r |
|---|---|
| Enrolment ↔ Demographic Updates | 0.61 |
| Demographic Updates ↔ Biometric Updates | 0.57 |
| Enrolment ↔ Biometric Updates | 0.53 |

Spearman rank stability across states: **ρ ≈ 0.95–0.97**  
Enrolment CV: ~30–40% · Update Pressure CV: ~80–120%

**Tier 3 — Trivariate Governance Mapping**  
Maps each state on three dimensions simultaneously: Scale (Total Aadhaar Activity), Pressure (UPI), and Stress (VSI) — enabling clustered policy recommendations.

---

## Policy-Grade Composite Indicators

Four indicators are derived for each state and normalized via Min–Max scaling.

**Aadhaar Maturity Index (AMI)**  
Composite normalized score reflecting the overall maturity of a state's Aadhaar activity footprint. Higher = more mature.

**Update Pressure Index (UPI)**  
```
UPI = log(Updates_total + 1) − log(Enrolments_total + 1)
```
Measures the degree to which a state has shifted from enrolment expansion to maintenance burden. Higher = stronger pressure.

**Volatility Stress Index (VSI)**  
Coefficient of variation of update pressure over time. Higher = more unstable, harder to plan infrastructure for.

**Temporal Predictability Score (TPS)**  
Lag-based stability metric between enrolment and subsequent update behavior. Higher = more predictable future demand signal.

---

## Governance Classification

States are classified into five policy categories with targeted action recommendations:

| Category | Signal |
|---|---|
| Stable Mature Systems | High AMI, moderate UPI, low VSI |
| High Maintenance Stress | High UPI + high VSI |
| Expansion Phase States | Low AMI, low UPI |
| Unpredictable Systems | High VSI, low TPS |
| Balanced / Transitional | Mixed signals across all four indicators |

Anomaly flags and priority scores are computed using Euclidean anomaly detection against the national indicator distribution.

---

## Data Engineering Pipeline

**Datasets**

| Dataset | Role |
|---|---|
| Aadhaar Enrolment | Expansion baseline |
| Demographic Update (name, address, DOB) | Maintenance burden proxy |
| Biometric Update (fingerprint, iris, face) | Authentication upkeep |

All datasets cleaned, standardized, and aggregated at **state–month granularity**.

**Cleaning steps:** date standardization, state/UT normalization, UT merger reconciliation, duplicate removal, numeric validation, pincode treatment.

**Engineered features:** child vs adult enrolment shares, total Aadhaar activity, update-to-enrolment ratios, volatility measures, lifecycle demographic indicators, Pareto concentration summaries, lag features, rank persistence tables, national monthly summary.

---

## Project Structure

```
UIDAI-data/
├── data/
│   └── raw/                          # Raw UIDAI zip files (not committed)
│       ├── api_data_aadhar_enrolment.zip
│       ├── api_data_aadhar_demographic.zip
│       └── api_data_aadhar_biometric.zip
├── src/
│   ├── data_cleaning.py              # Cleans raw files, standardizes state names
│   ├── analytics.py                  # Core analytics engine (reproduces notebook logic)
│   └── feature_engineering.py        # Builds full analysis output bundle
├── dashboard/
│   └── app.py                        # Streamlit app with filter-driven recomputation
├── datahack.ipynb                    # Original analysis notebook
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw UIDAI zip files in data/raw/

# 3. Run the pipeline
python src/data_cleaning.py
python src/feature_engineering.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

> **Note:** The repository does not include raw UIDAI datasets. Request access through the official UIDAI data portal.

---

## Dashboard

Six views, each recomputing from filtered data when the user changes date range or state scope:

- **Executive Overview** — national-level maturity and stress summary
- **Governance Diagnostics** — indicator heatmaps and state rankings
- **Lifecycle & Operations** — child/adult enrolment splits, update intensity
- **Anomalies & Risk** — Euclidean anomaly flags, priority scores
- **State Drilldown** — full indicator profile for a selected state
- **Data Export** — filtered outputs as CSV

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Streamlit · Log-normalization · Rank-based composite scoring · Spearman rank analysis · Euclidean anomaly detection · Pareto concentration diagnostics

---

## Key Contributions

- Shifted Aadhaar analytics from descriptive volume reporting to interpretable governance intelligence
- Engineered four policy-grade composite indicators (AMI, UPI, VSI, TPS) grounded in log-normalization and rank-based scoring
- Built volatility-aware capacity diagnostics that surface states with high infrastructure planning risk
- Developed anomaly detection for governance irregularities at state level
- Delivered a production-ready dashboard with filter-driven metric recomputation across three datasets
