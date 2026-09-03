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
    "biometric": [
        "biometric_analysis_ready.csv",
        "aadhaar_biometric_update_clean_final.csv",
    ],
}

PRECOMPUTED_OUTPUT_FILES = {
    "state_month_master": "state_month_master.csv",
    "state_stability": "state_stability.csv",
    "lag_features": "lag_features.csv",
    "rank_persistence": "rank_persistence.csv",
    "rank_correlations": "rank_correlations.csv",
    "state_master_full": "state_master_full.csv",
    "state_anomalies": "state_anomalies.csv",
    "state_focus_summary": "state_focus_summary.csv",
    "national_monthly_summary": "national_monthly_summary.csv",
    "pareto_activity": "pareto_activity.csv",
    "activity_correlation_matrix": "activity_correlation_matrix.csv",
    "indicator_correlation_matrix": "indicator_correlation_matrix.csv",
}

STATE_SCOPED_OUTPUTS = {
    "state_month_master",
    "state_stability",
    "lag_features",
    "rank_persistence",
    "state_master_full",
    "state_anomalies",
    "state_focus_summary",
    "pareto_activity",
}

EXPECTED_RAW_FILES = [
    "api_data_aadhar_enrolment.zip or matching enrolment CSV files",
    "api_data_aadhar_demographic.zip or matching demographic CSV files",
    "api_data_aadhar_biometric.zip or matching biometric CSV files",
]

# ── Streamlit page configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="UIDAI Intelligence Studio",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load external stylesheet ──────────────────────────────────────────────────
_CSS_PATH = os.path.join(DASHBOARD_DIR, "assets", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Inline Google Font import (Streamlit strips @import from CSS files) ──────
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── Risk classification helpers ───────────────────────────────────────────────
def _risk_level(state: str, state_master: pd.DataFrame) -> tuple[str, str, str]:
    """Return (level_label, css_class, dot_color) for the state's risk."""
    row = state_master[state_master["state"] == state]
    if row.empty:
        return "Unknown", "risk-stable", "#94a3b8"
    cat = str(row["Policy_Category"].iloc[0])
    if "Maintenance Stress" in cat or "Overloaded" in cat:
        return "Critical", "risk-critical", "#dc2626"
    if "Unpredictable" in cat:
        return "Warning", "risk-warning", "#d97706"
    return "Stable", "risk-stable", "#16a34a"


# ── Alert banner ──────────────────────────────────────────────────────────────
def render_alert_banner(
    state_master: pd.DataFrame,
    anomalies: pd.DataFrame,
    analysis_dir: str,
) -> None:
    """Render the top-level operational threat summary banner."""
    alerts = []

    # Capacity breach alert from forecast model summary
    model_summary_path = os.path.join(analysis_dir, "forecast_model_summary.csv")
    if os.path.exists(model_summary_path):
        ms = pd.read_csv(model_summary_path)
        if "months_to_breach" in ms.columns:
            breached = ms[ms["months_to_breach"].notna() & (ms["months_to_breach"] > -1)]
            if not breached.empty:
                earliest = int(breached["months_to_breach"].min())
                alerts.append(
                    f"⚡ <strong>{len(breached)} state(s)</strong> are projected to breach infrastructure "
                    f"capacity within 12 months — earliest in <strong>{earliest} month(s)</strong>."
                )

    # High-anomaly states
    if not anomalies.empty and "Anomaly_Flag_Count" in anomalies.columns:
        high_risk_count = int((anomalies["Anomaly_Flag_Count"] >= 2).sum())
        if high_risk_count > 0:
            alerts.append(
                f"🚨 <strong>{high_risk_count} state(s)</strong> have triggered 2 or more simultaneous anomaly "
                f"rules — high-priority governance intervention needed."
            )

    # Stress category alert
    if not state_master.empty and "Policy_Category" in state_master.columns:
        stress_count = int(state_master["Policy_Category"].str.contains("Maintenance Stress", na=False).sum())
        if stress_count > 0:
            alerts.append(
                f"📊 <strong>{stress_count} state(s)</strong> are classified under High Maintenance Stress — "
                f"update demand is outpacing enrollment-era infrastructure."
            )

    if not alerts:
        return

    items_html = "".join(f'<div class="alert-banner-item">• {a}</div>' for a in alerts)
    st.markdown(
        f"""
        <div class="alert-banner">
            <div class="alert-banner-title">🔴 Operational Intelligence Alert</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Data pipeline helpers ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_source_frames() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for key, candidates in SOURCE_FILES.items():
        frame = None
        for filename in candidates:
            base_dir = (
                ANALYSIS_DATA_DIR if "analysis_ready" in filename else CLEANED_DATA_DIR
            )
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                frame = pd.read_csv(path, parse_dates=["date"])
                break
        if frame is None:
            raise FileNotFoundError(f"Missing source dataset for {key}.")
        frames[key] = frame
    return frames


@st.cache_data(show_spinner=False)
def load_precomputed_outputs() -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    for key, filename in PRECOMPUTED_OUTPUT_FILES.items():
        path = os.path.join(ANALYSIS_DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing analysis output: {filename}.")
        outputs[key] = pd.read_csv(path)
    return outputs


def filter_precomputed_outputs(
    outputs: Dict[str, pd.DataFrame],
    selected_states: List[str],
) -> Dict[str, pd.DataFrame]:
    filtered = {key: frame.copy() for key, frame in outputs.items()}
    if not selected_states:
        return filtered

    for key in STATE_SCOPED_OUTPUTS:
        frame = filtered.get(key)
        if frame is not None and "state" in frame.columns:
            filtered[key] = frame[frame["state"].isin(selected_states)].copy()

    state_master = filtered["state_master_full"].copy()
    if "Total_Activity" in state_master.columns:
        pareto = state_master.sort_values("Total_Activity", ascending=False).reset_index(drop=True)
        pareto["activity_rank"] = range(1, len(pareto) + 1)
        total_activity = pareto["Total_Activity"].sum()
        pareto["cum_share"] = (
            pareto["Total_Activity"].cumsum() / total_activity
            if total_activity
            else 0.0
        )
        filtered["pareto_activity"] = pareto

    return filtered


def analysis_assets_ready() -> bool:
    required = [
        os.path.join(ANALYSIS_DATA_DIR, filename)
        for filename in PRECOMPUTED_OUTPUT_FILES.values()
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
        metric_card(
            "Raw UIDAI Files",
            "Ready" if raw_ok else "Missing",
            "Zip or CSV drops in data/raw",
        )
    with status_cols[1]:
        metric_card(
            "Cleaned Datasets",
            "Ready" if cleaned_ok else "Missing",
            "Outputs from data_cleaning.py",
        )
    with status_cols[2]:
        metric_card(
            "Analysis Bundle",
            "Ready" if analysis_ok else "Missing",
            "Outputs from feature_engineering.py",
        )

    if not raw_ok:
        st.error(
            "Raw UIDAI files are not present yet. The app cannot build cleaned or analysis datasets until those files are added."
        )
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
                        st.success(
                            "Data cleaning finished. You can now generate the analysis bundle."
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Data cleaning completed without producing all cleaned datasets."
                        )
        return

    if cleaned_ok and not analysis_ok:
        st.warning(
            "Cleaned datasets are ready. Generate the analysis bundle to unlock the dashboard."
        )
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
                        st.error(
                            "The pipeline ran but did not create the required analysis files."
                        )
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


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1 style="color:white;font-size:1.7rem;margin-bottom:0.3rem;">🆔 UIDAI Intelligence Studio</h1>
        <p>Aadhaar Maturity, Stress &amp; Governance composite indicators and policy mapping workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not analysis_assets_ready():
    render_setup_status()
    st.stop()

using_precomputed_outputs = False
try:
    source_frames = load_source_frames()
except FileNotFoundError:
    source_frames = {}
    using_precomputed_outputs = True
    precomputed_outputs = load_precomputed_outputs()
else:
    precomputed_outputs = {}

# ── Sidebar controls ──────────────────────────────────────────────────────────
if using_precomputed_outputs:
    control_state_month = precomputed_outputs["state_month_master"].copy()
    all_states = sorted(control_state_month["state"].dropna().unique().tolist())
    all_dates = pd.to_datetime(control_state_month["year_month"].astype(str) + "-01")
else:
    all_states = sorted(source_frames["enrolment"]["state"].dropna().unique().tolist())
    all_dates = pd.concat([frame["date"] for frame in source_frames.values()])

default_start = all_dates.min().to_pydatetime()
default_end = all_dates.max().to_pydatetime()

st.sidebar.markdown(
    '<div style="font-size:1rem;font-weight:500;color:#102a43;margin-bottom:0.5rem;">⚙️ Analysis Controls</div>',
    unsafe_allow_html=True,
)
selected_dates = st.sidebar.date_input(
    "Date Window",
    value=(default_start, default_end),
    min_value=default_start,
    max_value=default_end,
    disabled=using_precomputed_outputs,
)
if len(selected_dates) != 2:
    st.sidebar.error("Select both start and end dates.")
    st.stop()

selected_states = st.sidebar.multiselect(
    "State Scope",
    options=all_states,
    default=all_states,
    help=(
        "Uses the bundled precomputed analysis outputs in deployment mode."
        if using_precomputed_outputs
        else "All downstream indicators are recomputed over this exact state selection."
    ),
)

start_date = pd.Timestamp(selected_dates[0])
end_date = pd.Timestamp(selected_dates[1])

# ── Recompute metrics ─────────────────────────────────────────────────────────
if using_precomputed_outputs:
    outputs = filter_precomputed_outputs(precomputed_outputs, selected_states)
else:
    with st.spinner("Recomputing metrics for the active filters..."):
        try:
            outputs = build_outputs_for_filters(
                source_frames, start_date, end_date, selected_states
            )
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
    state_master = state_master[
        state_master["Policy_Category"].isin(selected_categories)
    ]
    anomalies = anomalies[anomalies["Policy_Category"].isin(selected_categories)]
    state_month = state_month[state_month["state"].isin(state_master["state"])]
    pareto = pareto[pareto["state"].isin(state_master["state"])]

if state_master.empty:
    st.error("No states remain after category filtering.")
    st.stop()

# ── State selector + sidebar risk card ───────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.85rem;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">State Drilldown</div>',
    unsafe_allow_html=True,
)
selected_state = st.sidebar.selectbox(
    "Select State",
    options=sorted(state_master["state"].tolist()),
    label_visibility="collapsed",
)

# Sidebar risk card
risk_label, risk_css, risk_dot_color = _risk_level(selected_state, state_master)
_state_row = state_master[state_master["state"] == selected_state]
_policy_cat = str(_state_row["Policy_Category"].iloc[0]) if not _state_row.empty else "—"
_policy_action = str(_state_row["Policy_Action"].iloc[0]) if not _state_row.empty else "—"
st.sidebar.markdown(
    f"""
    <div class="sidebar-risk-card {risk_css}">
        <div class="sidebar-risk-title">Selected State Risk</div>
        <div class="sidebar-risk-value">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                         background:{risk_dot_color};flex-shrink:0;"></span>
            <span style="color:{risk_dot_color};">{risk_label}</span>
            &nbsp;·&nbsp;
            <span style="color:#475569;font-size:0.9rem;">{selected_state}</span>
        </div>
        <div style="font-size:0.78rem;color:#64748b;margin-top:0.4rem;">{_policy_cat}</div>
        <div style="font-size:0.82rem;color:#334155;margin-top:0.3rem;font-style:italic;">"{_policy_action}"</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Alert banner ──────────────────────────────────────────────────────────────
render_alert_banner(state_master, anomalies, ANALYSIS_DATA_DIR)

# ── Top KPI strip ─────────────────────────────────────────────────────────────
render_kpis(state_master, state_month, anomalies)

# ── Question-driven navigation (6 tabs) ──────────────────────────────────────
tabs = st.tabs(
    [
        "🔍 What needs attention now?",
        "🧠 Why is this state flagged?",
        "📉 What will happen if we do nothing?",
        "🎛️ How can we intervene?",
        "📊 How do states compare?",
        "📦 Export data",
    ]
)

# Tab 1 — Attention
with tabs[0]:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_executive_overview(national, state_master, pareto, anomalies)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_anomalies_risk(anomalies)

# Tab 2 — Diagnosis
with tabs[1]:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_state_drilldown(state_master, state_month, selected_state)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_shap_panel()

# Tab 3 — Forecasting
with tabs[2]:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_forecast_panel(state_month)

# Tab 4 — Simulator
with tabs[3]:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_simulator_panel(state_master, state_month)

# Tab 5 — Comparison
with tabs[4]:
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_governance_diagnostics(state_master, selected_state, outputs)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_lifecycle_operations(national, state_master, outputs)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_cluster_panel()

# Tab 6 — Export
with tabs[5]:
    render_data_export(state_master, state_month, anomalies, national, pareto, outputs)
