import glob
import os
import sys
from typing import Dict, List
import pandas as pd
import streamlit as st

# Ensure src and dashboard directories are in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
for directory in [SRC_DIR, DASHBOARD_DIR]:
    if directory not in sys.path:
        sys.path.append(directory)

from analytics import build_analysis_outputs
from data_cleaning import main as run_data_cleaning
from feature_engineering import main as run_feature_pipeline

from components.metrics import metric_card, render_kpis
from components.layout import (
    render_executive_overview,
    render_governance_diagnostics,
    render_lifecycle_operations,
    render_anomalies_risk,
    render_state_drilldown,
    render_data_export,
)
from components.forecast_panel import render_forecast_panel
from components.cluster_panel import render_cluster_panel
from components.causal_panel import render_causal_panel
from components.shap_panel import render_shap_panel
from components.simulator_panel import render_simulator_panel


RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ANALYSIS_DATA_DIR = os.path.join(BASE_DIR, "data", "analysis")
CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")

SOURCE_FILES = {
    "enrolment": ["enrolment_analysis_ready.csv", "aadhaar_enrolment_clean_final.csv"],
    "demographic": [
        "demographic_analysis_ready.csv",
        "aadhaar_demographic_update_clean_final.csv",
    ],
    "biometric": ["biometric_analysis_ready.csv", "aadhaar_biometric_update_clean_final.csv"],
}

EXPECTED_RAW_FILES = [
    "api_data_aadhar_enrolment.zip or matching enrolment CSV files",
    "api_data_aadhar_demographic.zip or matching demographic CSV files",
    "api_data_aadhar_biometric.zip or matching biometric CSV files",
]

# Set Streamlit configurations
st.set_page_config(
    page_title="UIDAI Intelligence Studio",
    page_icon="ID",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI design
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(14,116,144,0.12), transparent 28%),
            radial-gradient(circle at top left, rgba(245,158,11,0.10), transparent 24%),
            linear-gradient(180deg, #f8fafc 0%, #eef4f7 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: "Aptos Display", "Trebuchet MS", sans-serif;
        letter-spacing: -0.02em;
        color: #102a43;
    }
    p, div, span, label {
        font-family: "Aptos", "Segoe UI", sans-serif;
    }
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(16,42,67,0.96), rgba(14,116,144,0.90));
        color: white;
        box-shadow: 0 18px 45px rgba(16,42,67,0.18);
        margin-bottom: 1rem;
    }
    .hero p {
        color: rgba(255,255,255,0.82);
        margin-bottom: 0;
    }
    .metric-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #102a43;
        margin: 0.1rem 0;
    }
    .metric-note {
        font-size: 0.88rem;
        color: #475569;
    }
    .panel-note {
        background: rgba(255,255,255,0.8);
        border-left: 4px solid #0e7490;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_source_frames() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for key, candidates in SOURCE_FILES.items():
        frame = None
        for filename in candidates:
            base_dir = ANALYSIS_DATA_DIR if "analysis_ready" in filename else CLEANED_DATA_DIR
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                frame = pd.read_csv(path, parse_dates=["date"])
                break
        if frame is None:
            raise FileNotFoundError(f"Missing source dataset for {key}.")
        frames[key] = frame
    return frames


def analysis_assets_ready() -> bool:
    required = [
        os.path.join(ANALYSIS_DATA_DIR, "state_policy_indicators_full.csv"),
        os.path.join(ANALYSIS_DATA_DIR, "state_month_master.csv"),
    ]
    return all(os.path.exists(path) for path in required)


def cleaned_assets_ready() -> bool:
    required = [
        os.path.join(CLEANED_DATA_DIR, "aadhaar_enrolment_clean_final.csv"),
        os.path.join(CLEANED_DATA_DIR, "aadhaar_demographic_update_clean_final.csv"),
        os.path.join(CLEANED_DATA_DIR, "aadhaar_biometric_update_clean_final.csv"),
    ]
    return all(os.path.exists(path) for path in required)


def raw_assets_ready() -> bool:
    patterns = [
        "api_data_aadhar_enrolment*.zip",
        "api_data_aadhar_enrolment*.csv",
        "api_data_aadhar_demographic*.zip",
        "api_data_aadhar_demographic*.csv",
        "api_data_aadhar_biometric*.zip",
        "api_data_aadhar_biometric*.csv",
    ]
    return any(glob.glob(os.path.join(RAW_DATA_DIR, pattern)) for pattern in patterns)


def render_setup_status() -> None:
    raw_ok = raw_assets_ready()
    cleaned_ok = cleaned_assets_ready()
    analysis_ok = analysis_assets_ready()

    st.subheader("Project Status")
    status_cols = st.columns(3)
    with status_cols[0]:
        metric_card("Raw UIDAI Files", "Ready" if raw_ok else "Missing", "Zip or CSV drops in data/raw")
    with status_cols[1]:
        metric_card("Cleaned Datasets", "Ready" if cleaned_ok else "Missing", "Outputs from data_cleaning.py")
    with status_cols[2]:
        metric_card("Analysis Bundle", "Ready" if analysis_ok else "Missing", "Outputs from feature_engineering.py")

    if not raw_ok:
        st.error("Raw UIDAI files are not present yet. The app cannot build cleaned or analysis datasets until those files are added.")
        st.markdown("Expected files in `data/raw/`:")
        for item in EXPECTED_RAW_FILES:
            st.markdown(f"- `{item}`")
        return

    if raw_ok and not cleaned_ok:
        st.warning("Raw files are present, but cleaned datasets are missing.")
        if st.button("Run Data Cleaning", type="primary"):
            with st.spinner("Running raw data cleaning pipeline..."):
                try:
                    run_data_cleaning()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                else:
                    st.cache_data.clear()
                    if cleaned_assets_ready():
                        st.success("Data cleaning finished. You can now generate the analysis bundle.")
                        st.rerun()
                    else:
                        st.error("Data cleaning completed without producing all cleaned datasets.")
        return

    if cleaned_ok and not analysis_ok:
        st.warning("Cleaned datasets are ready. Generate the analysis bundle to unlock the dashboard.")
        if st.button("Generate Analysis Outputs", type="primary"):
            with st.spinner("Generating analytical features..."):
                try:
                    run_feature_pipeline()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                else:
                    st.cache_data.clear()
                    if analysis_assets_ready():
                        st.success("Analysis pipeline finished. Loading dashboard...")
                        st.rerun()
                    else:
                        st.error("The pipeline ran but did not create the required analysis files.")
        return


def filter_frames(
    frames: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    selected_states: List[str],
) -> Dict[str, pd.DataFrame]:
    filtered: Dict[str, pd.DataFrame] = {}
    for key, frame in frames.items():
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]
        if selected_states:
            data = data[data["state"].isin(selected_states)]
        filtered[key] = data
    return filtered


def build_outputs_for_filters(
    frames: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    selected_states: List[str],
) -> Dict[str, pd.DataFrame]:
    filtered = filter_frames(frames, start_date, end_date, selected_states)
    for key, frame in filtered.items():
        if frame.empty:
            raise ValueError(f"No rows remain in {key} for the selected filters.")
    return build_analysis_outputs(
        enrolment=filtered["enrolment"],
        demographic=filtered["demographic"],
        biometric=filtered["biometric"],
    )


# Main layout structure
st.markdown(
    """
    <div class="hero">
        <h1>UIDAI Intelligence Studio</h1>
        <p>Aadhaar Maturity, Stress & Governance composite indicators and policy mapping workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not analysis_assets_ready():
    render_setup_status()
    st.stop()

try:
    source_frames = load_source_frames()
except FileNotFoundError:
    st.error(
        "Source datasets are missing. Run the raw data cleaning pipeline first."
    )
    st.stop()

# Date range selection
all_states = sorted(source_frames["enrolment"]["state"].dropna().unique().tolist())
all_dates = pd.concat([frame["date"] for frame in source_frames.values()])
default_start = all_dates.min().to_pydatetime()
default_end = all_dates.max().to_pydatetime()

st.sidebar.header("Analysis Controls")
selected_dates = st.sidebar.date_input(
    "Date Window",
    value=(default_start, default_end),
    min_value=default_start,
    max_value=default_end,
)
if len(selected_dates) != 2:
    st.sidebar.error("Select both start and end dates.")
    st.stop()

selected_states = st.sidebar.multiselect(
    "State Scope",
    options=all_states,
    default=all_states,
    help="All downstream indicators are recomputed over this exact state selection.",
)

start_date = pd.Timestamp(selected_dates[0])
end_date = pd.Timestamp(selected_dates[1])

# Recompute index features based on filters
with st.spinner("Recomputing metrics for the active filters..."):
    try:
        outputs = build_outputs_for_filters(source_frames, start_date, end_date, selected_states)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

state_master = outputs["state_master_full"].copy()
state_month = outputs["state_month_master"].copy()
anomalies = outputs["state_focus_summary"].copy()
national = outputs["national_monthly_summary"].copy()
pareto = outputs["pareto_activity"].copy()

policy_categories = sorted(state_master["Policy_Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Policy Categories",
    options=policy_categories,
    default=policy_categories,
)
if selected_categories:
    state_master = state_master[state_master["Policy_Category"].isin(selected_categories)]
    anomalies = anomalies[anomalies["Policy_Category"].isin(selected_categories)]
    state_month = state_month[state_month["state"].isin(state_master["state"])]
    pareto = pareto[pareto["state"].isin(state_master["state"])]

if state_master.empty:
    st.error("No states remain after category filtering.")
    st.stop()

selected_state = st.sidebar.selectbox("State Drilldown", options=sorted(state_master["state"].tolist()))

# Render top-level metric blocks
render_kpis(state_master, state_month, anomalies)

# Create layout tabs
tabs = st.tabs(
    [
        "Executive Overview",
        "Governance Diagnostics",
        "Lifecycle & Operations",
        "Anomalies & Risk",
        "State Drilldown",
        "Advanced ML Insights",
        "Policy Simulator",
        "Data Export",
    ]
)

with tabs[0]:
    render_executive_overview(national, state_master, pareto, anomalies)

with tabs[1]:
    render_governance_diagnostics(state_master, selected_state, outputs)

with tabs[2]:
    render_lifecycle_operations(national, state_master, outputs)

with tabs[3]:
    render_anomalies_risk(anomalies)

with tabs[4]:
    render_state_drilldown(state_master, state_month, selected_state)

with tabs[5]:
    st.markdown("## 🧠 Advanced ML Governance Insights")
    ml_tabs = st.tabs(
        [
            "Probabilistic Forecasting",
            "Governance Clustering",
            "Causal Inference",
            "Priority Attribution (SHAP)",
        ]
    )
    with ml_tabs[0]:
        render_forecast_panel(state_month)
    with ml_tabs[1]:
        render_cluster_panel()
    with ml_tabs[2]:
        render_causal_panel()
    with ml_tabs[3]:
        render_shap_panel()

with tabs[6]:
    render_simulator_panel(state_master, state_month)

with tabs[7]:
    render_data_export(state_master, state_month, anomalies, national, pareto, outputs)
