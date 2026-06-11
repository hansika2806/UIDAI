# 🇮🇳 UIDAI Data Hackathon: Aadhaar Governance Intelligence Framework

This project turns the raw transaction records of the Aadhaar ecosystem into a reusable analytics pipeline and a professional Streamlit dashboard, the **UIDAI Intelligence Studio**.

As India's Aadhaar ecosystem grows to serve over **1.3 billion residents**, the system's operational demands are shifting from expansion-oriented registrations to mature maintenance-oriented updates. This framework shifts focus from simply measuring enrollment volume to analyzing system maturity, operational pressure, and capacity stress at a **state-month granularity**.

---

## 📌 Overview & Problem Statement

Traditional Aadhaar dashboards focus heavily on enrollment metrics. However:
- Demographic updates (addresses, names, DOB) now exceed enrollments by **5–10x** in mature states.
- Update demand exhibits significantly higher volatility than enrollment, creating operational bottlenecks.
- Larger states face distinct maintenance pressure patterns, where child enrollments strongly correlate with adult update requirements at predictable lags.

This framework bridges the gap by computing **Policy-Grade Composite Indicators** and applying **Staff-Level ML Engineering** to categorise and prioritize states for targeted governance actions.

---

## ⚙️ Enterprise ML Architecture

The pipeline is built for production readiness, incorporating rigorous data validation, model registries, and drift monitoring.

1. **Data Ingestion & Contracts (`src/data_cleaning.py`, `src/schemas.py`)**:
   - Memory-efficient chunked loading of raw zip files.
   - Enforces statistical data contracts using **Pandera** to validate schemas, ensure non-negative constraints, and catch structural faults before analysis begins.
   
2. **Feature Engineering & Core Analytics (`src/feature_engineering.py`, `src/analytics.py`)**:
   - Aggregates metrics at a **state-month** level.
   - Computes rank persistence, temporal predictability (lags), and composite indices.

3. **Advanced ML Modules**:
   - **Clustering (`src/clustering.py`)**: Gaussian Mixture Models (GMM) with dynamic AICc component selection for state profiling.
   - **Causal Inference (`src/causal.py`)**: Granger causality with Augmented Dickey-Fuller (ADF) stationarity checks and CUSUM structural break detection.
   - **Demand Forecasting (`src/forecasting.py`)**: Walk-forward probabilistic SARIMA forecasting.
   - **Anomaly Detection (`src/anomaly.py`)**: Ensemble anomaly detection combining Isolation Forests and ECOD to flag severe operational spikes.
   - **Interpretability (`src/interpretability.py`)**: SHAP value calculation using LightGBM to explain what drives the Aadhaar priority indices.

4. **Model Ops & Reliability (`src/registry.py`, `src/drift.py`)**:
   - **Model Registry**: Fitted estimators (GMMs, Scalers, IsolationForests) are serialized to `models/v1/` via joblib for decoupled inference.
   - **Drift Monitoring**: Continuous two-sample Kolmogorov-Smirnov (KS) tests for covariate shift and Population Stability Index (PSI) for concept drift.

---

## 🧠 Policy-Grade Composite Indicators

The framework relies on four primary composite indices (min-max scaled between `0` and `1`):

- **Aadhaar Maturity Index (AMI)**: Measures the maturity of the state's Aadhaar footprint based on rank stability.
- **Update Pressure Index (UPI)**: Represents maintenance intensity vs new signups.
- **Volatility Stress Index (VSI)**: Highlights unpredictable surges in operational demand.
- **Temporal Predictability Score (TPS)**: Measures the correlation between child enrollment cohorts and subsequent adult biometric update waves.

---

## 📂 Project Structure

```
├── data/
│   ├── raw/                       # Put zip files here: api_data_aadhar_*.zip
│   ├── cleaned/                   # Generated cleaned CSV outputs
│   └── analysis/                  # Analytical CSVs, drift reports, models
│
├── models/
│   └── v1/                        # Joblib serialized ML estimators (Registry)
│
├── src/
│   ├── config.py                  # Configurations and static variables
│   ├── schemas.py                 # Pandera data contracts
│   ├── data_cleaning.py           # Raw data ingestion and cleaning
│   ├── feature_engineering.py     # Main ML pipeline orchestrator
│   ├── analytics.py               # Core indicator computations
│   ├── robust_indicators.py       # MAD and Entropy calculations
│   ├── clustering.py              # GMM state profiling
│   ├── causal.py                  # Granger + CUSUM analysis
│   ├── forecasting.py             # SARIMA models
│   ├── anomaly.py                 # Isolation Forest + ECOD ensemble
│   ├── interpretability.py        # LightGBM + SHAP explanations
│   ├── registry.py                # Model serialization management
│   ├── drift.py                   # KS-test and PSI drift detection
│   └── visualization.py           # Static Plotly charts
│
├── dashboard/
│   ├── app.py                     # Streamlit application entrypoint
│   └── components/                # Modular UI components (metrics, charts, panels)
│
├── tests/
│   └── test_staff_enhancements.py # Pytest suite for ML engineering features
│
└── requirements.txt               # Dependencies
```

---

## 🚀 Setup & Launch Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets
Place the raw zip datasets in the `data/raw/` directory:
- `api_data_aadhar_enrolment.zip`
- `api_data_aadhar_demographic.zip`
- `api_data_aadhar_biometric.zip`

### 3. Run the ML Pipeline
Execute the end-to-end pipeline. This will clean the data, validate schemas, fit and serialize models, compute SHAP/anomalies, and generate drift reports.
```bash
python src/data_cleaning.py
python src/feature_engineering.py
```

### 4. Run Tests (Optional but Recommended)
Validate the structural integrity of the project:
```bash
python -m pytest tests/test_staff_enhancements.py -v
```

### 5. Launch the Intelligence Studio
Explore the interactive visual dashboard:
```bash
streamlit run dashboard/app.py
```
