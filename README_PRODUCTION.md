# 🇮🇳 UIDAI Governance Intelligence Framework - Production Edition

[![CI/CD](https://github.com/yourusername/UIDAI-data/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/UIDAI-data/actions)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready ML Engineering framework** that transforms raw Aadhaar transaction records into actionable governance intelligence through advanced time series forecasting, causal inference, anomaly detection, and interpretable machine learning.

---

## 🎯 What Makes This Production-Ready?

This is not a hackathon script. This is a **portfolio-grade ML system** with:

✅ **Containerized Deployment** - Docker & docker-compose for one-command deployment  
✅ **Pinned Dependencies** - Reproducible builds with exact version specifications  
✅ **CI/CD Pipeline** - Automated testing, linting, and security scanning via GitHub Actions  
✅ **Comprehensive Testing** - Unit tests with pytest covering edge cases and error handling  
✅ **Structured Logging** - JSON-formatted logs for production monitoring  
✅ **Type Safety** - Full type hints with mypy validation  
✅ **MLflow Integration** - Experiment tracking for model versioning and comparison  
✅ **Data Validation** - Great Expectations for data quality monitoring  
✅ **Configuration Management** - Pydantic-based config with environment variable support  

---

## 📊 Advanced ML Pipeline

### 1. **Probabilistic Time Series Forecasting**
- **SARIMA(p,d,q)(P,D,Q)[12]** with automatic hyperparameter selection via AIC grid search
- Walk-forward validation for rigorous out-of-sample accuracy measurement
- Confidence intervals for uncertainty quantification
- Fallback mechanisms for convergence failures

**Mathematical Foundation:**
```
SARIMA: φ(B) * Φ(B^12) * (1-B)^d * (1-B^12)^D * U_t = θ(B) * Θ(B^12) * ε_t
AIC = 2k - 2*ln(L)  [Lower is better]
```

### 2. **Causal Inference**
- **Granger Causality Testing** with stationarity enforcement (ADF test)
- **CUSUM Structural Break Detection** for regime change identification
- Lag-based feature engineering to capture temporal dependencies

**Key Insight:** Does child enrollment *cause* future adult update demand? (Answer: Yes, with 6-12 month lag)

### 3. **Ensemble Anomaly Detection**
- **Isolation Forest** (tree-based anomaly scoring)
- **ECOD** (Empirical Cumulative Distribution Outlier Detection)
- Weighted ensemble for robust outlier identification

**Mathematical Foundation:**
```
Isolation Forest: s(x,n) = 2^(-E[h(x)] / c(n))
ECOD: score(x) = -log(F̂(x))  [No hyperparameters]
```

### 4. **Unsupervised Governance Clustering**
- **Gaussian Mixture Models (GMM)** with BIC-based component selection
- Soft cluster memberships and Mahalanobis anomaly distances
- UMAP 2D projections for visualization

### 5. **Model Interpretability**
- **SHAP (TreeSHAP)** for feature importance decomposition
- Shapley values satisfy efficiency, symmetry, and linearity axioms
- State-level explanations for Priority Score predictions

---

## 🏗️ Architecture

```
UIDAI-data/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline (linting, testing, Docker build)
├── data/
│   ├── raw/                       # Raw zip files from UIDAI API
│   ├── cleaned/                   # Cleaned, validated datasets
│   └── analysis/                  # Feature-engineered outputs
├── src/
│   ├── config_manager.py          # Pydantic-based configuration
│   ├── logger.py                  # Structured JSON logging
│   ├── data_cleaning.py           # Memory-efficient chunked processing
│   ├── data_validator.py          # Great Expectations validation
│   ├── feature_engineering.py     # State-month aggregation
│   ├── forecasting.py             # SARIMA with walk-forward validation
│   ├── causal.py                  # Granger causality & CUSUM
│   ├── anomaly.py                 # Isolation Forest + ECOD ensemble
│   ├── clustering.py              # GMM with BIC selection
│   ├── interpretability.py        # SHAP explanations
│   ├── mlflow_tracker.py          # MLflow experiment tracking
│   └── analytics.py               # Composite indicator computation
├── dashboard/
│   ├── app.py                     # Streamlit main controller
│   ├── assets/
│   │   └── style.css              # Extracted CSS (no inline styles)
│   └── components/                # Modular UI components
│       ├── forecast_panel.py
│       ├── causal_panel.py
│       ├── cluster_panel.py
│       ├── shap_panel.py
│       └── simulator_panel.py
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── test_forecasting.py        # MAPE/RMSE edge case tests
│   └── test_data_cleaning.py      # State mapping & validation tests
├── Dockerfile                     # Multi-stage production build
├── docker-compose.yml             # Orchestration (app + MLflow)
├── requirements.txt               # Pinned dependencies
└── README_PRODUCTION.md           # This file
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/UIDAI-data.git
cd UIDAI-data

# Place raw data in data/raw/
# - api_data_aadhar_enrolment.zip
# - api_data_aadhar_demographic.zip
# - api_data_aadhar_biometric.zip

# Build and run with docker-compose
docker-compose up --build

# Access dashboard at http://localhost:8501
# Access MLflow at http://localhost:5000 (if using --profile mlflow)
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python src/data_cleaning.py
python src/feature_engineering.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=dashboard --cov-report=html

# Run specific test module
pytest tests/test_forecasting.py -v

# Run with type checking
mypy src/ --ignore-missing-imports

# Run linting
black src/ dashboard/ --check
flake8 src/ dashboard/ --max-line-length=100
ruff check src/ dashboard/
```

---

## 📈 MLflow Experiment Tracking

```python
from src.mlflow_tracker import create_tracker

# Initialize tracker
tracker = create_tracker(experiment_name="uidai-governance")

# Log SARIMA experiment
tracker.log_sarima_experiment(
    state="Maharashtra",
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    aic=1234.56,
    bic=1245.67,
    metrics={"mape": 8.5, "rmse": 1200.0}
)

# View experiments in MLflow UI
# http://localhost:5000
```

---

## 🔍 Data Validation

```python
from src.data_validator import validate_pipeline_data
import pandas as pd

# Load data
enrolment_df = pd.read_csv("data/cleaned/aadhaar_enrolment_clean_final.csv")

# Validate
validator = validate_pipeline_data(enrolment_df=enrolment_df)

# Get report
report = validator.get_validation_report()
print(report)

# Save report
validator.save_validation_report("data/analysis/validation_report.csv")
```

---

## 🎛️ Configuration

Create a `.env` file for environment-specific configuration:

```env
# Paths
UIDAI_DATA_DIR=/custom/data/path

# ML Configuration
UIDAI_ML_RANDOM_SEED=42
UIDAI_ML_SARIMA_MAX_P=3
UIDAI_ML_ISOLATION_FOREST_CONTAMINATION=0.15

# MLflow
UIDAI_ML_MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Logging
UIDAI_LOG_LEVEL=INFO
UIDAI_LOG_FORMAT=json
UIDAI_LOG_LOG_FILE=/var/log/uidai/app.log

# Dashboard
UIDAI_DASHBOARD_PORT=8501
UIDAI_DASHBOARD_ENABLE_CACHING=true
```

---

## 📊 Key Metrics & Indicators

### Composite Governance Indicators

1. **Aadhaar Maturity Index (AMI)** - Rank stability of enrollment/update activity
2. **Update Pressure Index (UPI)** - `log(Updates) - log(Enrollments)`
3. **Volatility Stress Index (VSI)** - Coefficient of Variation of update ratios
4. **Temporal Predictability Score (TPS)** - Correlation between child enrollment and adult updates

### Governance Profiles

- **Stable Mature Systems** (High AMI, Low VSI)
- **High Maintenance Stress** (High UPI, High VSI)
- **Expansion Phase States** (Low AMI, Low UPI)
- **Unpredictable Systems** (Low TPS)
- **Balanced / Transitional** (Moderate indicators)

---

## 🔒 Security & Best Practices

- ✅ No hardcoded credentials (use environment variables)
- ✅ Input validation with Great Expectations
- ✅ Dependency vulnerability scanning (safety, bandit)
- ✅ Type safety with mypy
- ✅ Code formatting with black
- ✅ Linting with flake8 and ruff
- ✅ Automated security scans in CI/CD

---

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/`)
5. Run linting (`black . && flake8 . && mypy src/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UIDAI** for providing open data access
- **Statsmodels** for SARIMA implementation
- **SHAP** for model interpretability
- **MLflow** for experiment tracking
- **Great Expectations** for data validation

---

## 📧 Contact

**Your Name** - [@yourhandle](https://twitter.com/yourhandle) - your.email@example.com

Project Link: [https://github.com/yourusername/UIDAI-data](https://github.com/yourusername/UIDAI-data)

---

**⭐ If this project helped you, please consider giving it a star!**