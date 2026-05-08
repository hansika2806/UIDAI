import glob
import os
import sys
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from analytics import build_analysis_outputs  # noqa: E402
from data_cleaning import main as run_data_cleaning  # noqa: E402
from feature_engineering import main as run_feature_pipeline  # noqa: E402
from notebook_mode import load_notebook_artifacts, notebook_exists  # noqa: E402

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

PALETTE = {
    "navy": "#102a43",
    "teal": "#0e7490",
    "gold": "#f59e0b",
    "coral": "#ea580c",
    "green": "#15803d",
    "slate": "#475569",
    "mist": "#f8fafc",
}

EXPECTED_RAW_FILES = [
    "api_data_aadhar_enrolment.zip or matching enrolment CSV files",
    "api_data_aadhar_demographic.zip or matching demographic CSV files",
    "api_data_aadhar_biometric.zip or matching biometric CSV files",
]


st.set_page_config(
    page_title="UIDAI Intelligence Studio",
    page_icon="ID",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def format_indian_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"{value / 1e5:.2f} L"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f} K"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
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
            try:
                run_data_cleaning()
            except FileNotFoundError as exc:
                st.error(str(exc))
            else:
                st.cache_data.clear()
                if cleaned_assets_ready():
                    st.success("Data cleaning finished. You can now generate the analysis bundle.")
                else:
                    st.error("Data cleaning completed without producing all cleaned datasets.")
        return

    if cleaned_ok and not analysis_ok:
        st.warning("Cleaned datasets are ready. Generate the analysis bundle to unlock the dashboard.")
        if st.button("Generate Analysis Outputs", type="primary"):
            try:
                run_feature_pipeline()
            except FileNotFoundError as exc:
                st.error(str(exc))
            else:
                st.cache_data.clear()
                if analysis_assets_ready():
                    st.success("Analysis pipeline finished. Reload the app to pick up the new outputs.")
                else:
                    st.error("The pipeline ran but did not create the required analysis files.")
        return


@st.cache_data(show_spinner=False)
def load_notebook_mode_artifacts():
    return load_notebook_artifacts()


def render_notebook_mode() -> None:
    artifacts = load_notebook_mode_artifacts()
    summary = artifacts["summary"]
    tables = artifacts["tables"]
    images = artifacts["images"]

    st.info(
        "Notebook Mode is active. This version uses executed notebook artifacts directly, so it stays faithful to your analysis while remaining lighter than Full Data Mode."
    )

    note_cols = st.columns(5)
    with note_cols[0]:
        metric_card("Notebook States", str(summary.get("states_aggregated") or "-"), "Recovered from executed notebook")
    with note_cols[1]:
        metric_card("State Rows", str(summary.get("state_master_rows") or "-"), "Recovered indicator rows")
    with note_cols[2]:
        metric_card("State-Month Rows", str(summary.get("state_month_rows") or "-"), "Recovered monthly observations")
    with note_cols[3]:
        metric_card("Anomalous States", str(summary.get("anomaly_state_count") or "-"), "Recovered anomaly count")
    with note_cols[4]:
        metric_card(
            "Lifecycle Corr",
            f"{summary['lifecycle_correlation']:.3f}" if summary.get("lifecycle_correlation") is not None else "-",
            "Child enrollment share vs adult updates",
        )

    tabs = st.tabs(
        [
            "Notebook Overview",
            "Recovered Tables",
            "Recovered Charts",
            "Recovered Risks",
            "Upgrade Path",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown(
                '<div class="panel-note">This mode is driven from the executed `.ipynb` itself. It preserves your notebook outputs, charts, category logic, and anomaly findings even though the underlying CSV files are not available in the repo.</div>',
                unsafe_allow_html=True,
            )
            if summary.get("policy_category_counts"):
                cat_df = pd.DataFrame(
                    [
                        {"Policy Category": key, "Count": value}
                        for key, value in summary["policy_category_counts"].items()
                    ]
                ).sort_values("Count", ascending=True)
                fig = px.bar(
                    cat_df,
                    x="Count",
                    y="Policy Category",
                    orientation="h",
                    template="plotly_white",
                    title="Recovered Policy Category Counts",
                    color="Count",
                    color_continuous_scale="Tealgrn",
                )
                fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Recovered Highlights")
            highlights = [
                f"Enrollment vs demographic rank persistence: {summary['rank_rho_enrolment_demographic']:.3f}" if summary.get("rank_rho_enrolment_demographic") is not None else None,
                f"Enrollment vs biometric rank persistence: {summary['rank_rho_enrolment_biometric']:.3f}" if summary.get("rank_rho_enrolment_biometric") is not None else None,
                f"Pressure-maturity mismatch state: {summary['pressure_mismatch_state']}" if summary.get("pressure_mismatch_state") else None,
                "High-stress IQR anomaly screen returned no states.",
            ]
            for item in highlights:
                if item:
                    st.markdown(f"- {item}")

            if summary.get("pressure_mismatch_state"):
                mismatch_df = pd.DataFrame(
                    [
                        {
                            "state": summary["pressure_mismatch_state"],
                            "UPI": summary["pressure_mismatch_upi"],
                            "AMI": summary["pressure_mismatch_ami"],
                        }
                    ]
                )
                st.dataframe(mismatch_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        table_options = {
            "State-Month Ratio Sample": "state_month_ratios_sample",
            "Activity Correlation Matrix": "activity_correlation_matrix",
            "State Stability Sample": "state_stability_sample",
            "Lag Feature Sample": "lag_features_sample",
            "Lifecycle State Sample": "state_lifecycle_sample",
            "Lifecycle Summary Stats": "lifecycle_summary_stats",
            "Indicator Correlation Matrix": "indicator_correlation_matrix",
            "Policy Table Sample": "policy_table_sample",
        }
        selected = st.selectbox("Recovered notebook table", list(table_options.keys()))
        table_key = table_options[selected]
        if table_key in tables:
            st.dataframe(tables[table_key], use_container_width=True, hide_index=True)
        else:
            st.warning("That table was not recoverable from notebook outputs.")

    with tabs[2]:
        st.markdown(
            '<div class="panel-note">These are embedded notebook chart outputs. They are not regenerated from data files here; they are recovered from the executed notebook exactly as captured.</div>',
            unsafe_allow_html=True,
        )
        if not images:
            st.warning("No embedded notebook charts were found.")
        else:
            chart_names = [f"{item['title']} (Cell {item['cell_index']})" for item in images]
            selected_chart_name = st.selectbox("Recovered chart", chart_names)
            selected_chart = images[chart_names.index(selected_chart_name)]
            st.subheader(selected_chart["title"])
            st.caption(selected_chart["caption"])
            st.image(selected_chart["bytes"], use_container_width=True)

            preview_cols = st.columns(3)
            for idx, image in enumerate(images[:9]):
                with preview_cols[idx % 3]:
                    st.image(image["bytes"], caption=image["title"], use_container_width=True)

    with tabs[3]:
        left, right = st.columns([1.15, 1])
        with left:
            anomaly_names = artifacts["anomaly_names"][:9]
            if anomaly_names:
                anomaly_df = pd.DataFrame({"Recovered anomalous states": anomaly_names})
                st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
            else:
                st.info("Anomaly names were only partially recoverable from notebook text output.")
        with right:
            policy_table = tables.get("policy_table_sample")
            if policy_table is not None:
                st.dataframe(policy_table, use_container_width=True, hide_index=True)
            else:
                st.info("Policy table sample was not recoverable.")

        chart_lookup = {item["title"]: item for item in images}
        for preferred in ["Pressure-Maturity Anomaly", "Operational Risk Map", "Indicator Heatmap"]:
            if preferred in chart_lookup:
                st.subheader(preferred)
                st.image(chart_lookup[preferred]["bytes"], use_container_width=True)

    with tabs[4]:
        st.markdown(
            """
            To unlock Full Data Mode later, add one of the following:

            - raw UIDAI zip/csv files into `data/raw/`
            - cleaned datasets into `data/cleaned/`
            - analysis outputs into `data/analysis/`

            Notebook Mode keeps the project presentable and notebook-faithful now, while leaving the door open for full recomputation later.
            """
        )


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


def build_pareto_chart(pareto_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=pareto_df["state"],
            y=pareto_df["Total_Activity"],
            name="Total Activity",
            marker_color=PALETTE["navy"],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=pareto_df["state"],
            y=pareto_df["cum_share"],
            mode="lines+markers",
            name="Cumulative Share",
            line=dict(color=PALETTE["coral"], width=3),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Activity Concentration by State",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Total Activity", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Share", tickformat=".0%", secondary_y=True)
    return fig


def build_indicator_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    matrix = corr_df.set_index("metric")
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_quadrant_chart(state_master: pd.DataFrame, highlight_state: str = "") -> go.Figure:
    fig = px.scatter(
        state_master,
        x="AMI",
        y="UPI",
        color="Policy_Category",
        size="VSI",
        size_max=44,
        hover_name="state",
        hover_data={
            "TPS": ":.3f",
            "Total_Activity": ":,.0f",
            "Governance_Status": True,
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        template="plotly_white",
        title="Governance Quadrant: Maturity vs Maintenance Pressure",
    )
    fig.add_vline(x=state_master["AMI"].median(), line_dash="dash", line_color="#64748b")
    fig.add_hline(y=state_master["UPI"].median(), line_dash="dash", line_color="#64748b")
    if highlight_state and highlight_state in set(state_master["state"]):
        chosen = state_master[state_master["state"] == highlight_state]
        fig.add_trace(
            go.Scatter(
                x=chosen["AMI"],
                y=chosen["UPI"],
                mode="markers+text",
                text=chosen["state"],
                textposition="top center",
                marker=dict(size=18, color=PALETTE["coral"], line=dict(width=2, color="white")),
                name="Selected State",
            )
        )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=70, b=20))
    return fig


def build_state_month_chart(state_series: pd.DataFrame, state_name: str) -> go.Figure:
    chart_df = state_series.sort_values("year_month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["year_month"], y=chart_df["E_total"], mode="lines", name="Enrollments"))
    fig.add_trace(go.Scatter(x=chart_df["year_month"], y=chart_df["D_total"], mode="lines", name="Demographic Updates"))
    fig.add_trace(go.Scatter(x=chart_df["year_month"], y=chart_df["B_total"], mode="lines", name="Biometric Updates"))
    fig.update_layout(
        title=f"{state_name}: Monthly Activity Mix",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_ratio_trend_chart(national: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=national["year_month"],
            y=national["demo_update_ratio"],
            mode="lines",
            name="Demographic / Enrollment",
            line=dict(color=PALETTE["teal"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=national["year_month"],
            y=national["biometric_update_ratio"],
            mode="lines",
            name="Biometric / Enrollment",
            line=dict(color=PALETTE["gold"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=national["year_month"],
            y=national["update_to_enrol_ratio"],
            mode="lines",
            name="Combined Update / Enrollment",
            line=dict(color=PALETTE["coral"], width=4),
        )
    )
    fig.update_layout(
        title="Maintenance Intensity Through Time",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_stress_scale_chart(state_master: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        state_master,
        x="Total_Activity",
        y="VSI",
        color="AMI",
        size="UPI",
        size_max=40,
        hover_name="state",
        hover_data=["TPS", "Governance_Status"],
        color_continuous_scale="Viridis",
        log_x=True,
        template="plotly_white",
        title="Stress vs Scale Diagnostic",
    )
    fig.add_hline(y=state_master["VSI"].median(), line_dash="dash", line_color="#94a3b8")
    fig.add_vline(x=max(state_master["Total_Activity"].median(), 1), line_dash="dash", line_color="#94a3b8")
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_lifecycle_chart(state_master: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        state_master,
        x="E_child_share",
        y="D_adult_share",
        size="Total_Activity",
        color="TPS",
        hover_name="state",
        size_max=50,
        color_continuous_scale="Tealgrn",
        template="plotly_white",
        title="Lifecycle Transition Pattern",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(title="Child Enrollment Share (0-5)")
    fig.update_yaxes(title="Adult Demographic Update Share (17+)")
    return fig


def build_anomaly_chart(anomalies: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        anomalies,
        x="AMI",
        y="UPI",
        size="Anomaly_Flag_Count",
        color="Anomaly_Flag_Count",
        hover_name="state",
        hover_data=["VSI", "TPS", "Governance_Status"],
        size_max=48,
        color_continuous_scale="OrRd",
        template="plotly_white",
        title="Anomaly Pressure Map",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_peer_comparison(state_master: pd.DataFrame, selected_state: str) -> go.Figure:
    row = state_master[state_master["state"] == selected_state].iloc[0]
    metrics = ["AMI", "UPI", "VSI", "TPS"]
    peer = state_master[metrics].median()
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=[row[m] for m in metrics], theta=metrics, fill="toself", name=selected_state)
    )
    fig.add_trace(
        go.Scatterpolar(r=[peer[m] for m in metrics], theta=metrics, fill="toself", name="Median Peer")
    )
    fig.update_layout(
        title=f"{selected_state}: Indicator Profile vs Median",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


st.markdown(
    """
    <div class="hero">
        <h1>UIDAI Intelligence Studio</h1>
        <p>Notebook-derived governance analytics, anomaly diagnostics, lifecycle interpretation, and policy action mapping in one dynamic workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not analysis_assets_ready():
    render_setup_status()
    if notebook_exists():
        render_notebook_mode()
    st.stop()

try:
    source_frames = load_source_frames()
except FileNotFoundError:
    st.error(
        "Source datasets are missing. Run `python src/data_cleaning.py` and `python src/feature_engineering.py` after placing the UIDAI files in `data/raw/`."
    )
    st.stop()

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

with st.spinner("Recomputing notebook metrics for the active filter window..."):
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

kpi_cols = st.columns(5)
with kpi_cols[0]:
    metric_card("States in Scope", str(state_master["state"].nunique()), "Filtered analytical universe")
with kpi_cols[1]:
    metric_card("Months Covered", str(state_month["year_month"].nunique()), "Active state-month observations")
with kpi_cols[2]:
    metric_card(
        "Total Activity",
        format_indian_number(state_master["Total_Activity"].sum()),
        "Enrollments plus demographic and biometric updates",
    )
with kpi_cols[3]:
    metric_card(
        "Flagged States",
        str((anomalies["Anomaly_Flag_Count"] > 0).sum()),
        "At least one anomaly rule triggered",
    )
with kpi_cols[4]:
    top_category = state_master["Policy_Category"].mode().iloc[0]
    metric_card("Dominant Category", top_category, "Most common governance profile")

tabs = st.tabs(
    [
        "Executive Overview",
        "Governance Diagnostics",
        "Lifecycle & Operations",
        "Anomalies & Risk",
        "State Drilldown",
        "Data Export",
    ]
)

with tabs[0]:
    st.markdown(
        '<div class="panel-note">This view keeps the big picture intact: national movement, state concentration, category spread, and the policy queue that deserves attention first.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.6, 1])
    with left:
        trend_df = national.melt(
            id_vars="year_month",
            value_vars=["E_total", "D_total", "B_total"],
            var_name="Series",
            value_name="Value",
        )
        fig = px.line(
            trend_df,
            x="year_month",
            y="Value",
            color="Series",
            template="plotly_white",
            title="National UIDAI Activity Trend",
            color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"], PALETTE["gold"]],
        )
        fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        category_counts = (
            state_master["Policy_Category"].value_counts().rename_axis("Policy_Category").reset_index(name="Count")
        )
        fig = px.bar(
            category_counts,
            x="Count",
            y="Policy_Category",
            orientation="h",
            color="Policy_Category",
            template="plotly_white",
            title="Policy Category Mix",
        )
        fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.4, 1.1])
    with left:
        st.plotly_chart(build_pareto_chart(pareto), use_container_width=True)
    with right:
        st.subheader("Priority States")
        preview_cols = [
            "state",
            "Priority_Score",
            "Governance_Status",
            "Policy_Action",
            "Anomaly_Flag_Count",
        ]
        st.dataframe(
            anomalies[preview_cols].head(12),
            use_container_width=True,
            hide_index=True,
        )

with tabs[1]:
    st.plotly_chart(build_quadrant_chart(state_master, selected_state), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_stress_scale_chart(state_master), use_container_width=True)
    with right:
        indicator_corr = outputs["indicator_correlation_matrix"]
        st.plotly_chart(
            build_indicator_heatmap(indicator_corr, "Indicator Correlation Heatmap"),
            use_container_width=True,
        )

    compare_mode = st.radio(
        "Rank Persistence View",
        options=["Demographic Rank", "Biometric Rank"],
        horizontal=True,
    )
    rank_df = outputs["rank_persistence"].copy()
    y_metric = "D_rank" if compare_mode == "Demographic Rank" else "B_rank"
    fig = px.scatter(
        rank_df,
        x="E_rank",
        y=y_metric,
        hover_name="state",
        template="plotly_white",
        title=f"Enrollment Rank vs {compare_mode}",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(build_ratio_trend_chart(national), use_container_width=True)
    with right:
        top_vol = state_master.nlargest(12, "VSI")[["state", "VSI", "UPI", "TPS"]]
        fig = px.bar(
            top_vol.sort_values("VSI"),
            x="VSI",
            y="state",
            orientation="h",
            color="UPI",
            template="plotly_white",
            title="Highest Volatility States",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.3, 1.05])
    with left:
        st.plotly_chart(build_lifecycle_chart(state_master), use_container_width=True)
    with right:
        activity_corr = outputs["activity_correlation_matrix"]
        st.plotly_chart(
            build_indicator_heatmap(activity_corr, "Activity and Lifecycle Correlations"),
            use_container_width=True,
        )

with tabs[3]:
    st.plotly_chart(build_anomaly_chart(anomalies), use_container_width=True)
    left, right = st.columns([1.2, 1])
    with left:
        flag_cols = [
            "state",
            "Anomaly_Flag_Count",
            "Anomaly_High_Stress",
            "Anomaly_Pressure_Mismatch",
            "Anomaly_Scale_Volatility",
            "Anomaly_Governance_Outlier",
            "Volatility_Excess",
            "Governance_Distance",
        ]
        st.dataframe(
            anomalies[flag_cols].sort_values("Anomaly_Flag_Count", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        pressure_mismatch = anomalies[anomalies["Anomaly_Pressure_Mismatch"]]
        if pressure_mismatch.empty:
            st.info("No pressure-maturity mismatch states under the active filter window.")
        else:
            fig = px.bar(
                pressure_mismatch.sort_values("Priority_Score"),
                x="Priority_Score",
                y="state",
                color="UPI",
                orientation="h",
                template="plotly_white",
                title="Pressure-Maturity Mismatch Queue",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    state_row = state_master[state_master["state"] == selected_state].iloc[0]
    state_series = state_month[state_month["state"] == selected_state].copy()

    st.subheader(selected_state)
    st.caption(
        f"{state_row['Governance_Status']} | {state_row['Policy_Category']} | Priority score {state_row['Priority_Score']:.2f}"
    )
    drill_cols = st.columns(4)
    with drill_cols[0]:
        metric_card("AMI", f"{state_row['AMI']:.2f}", "Maturity index")
    with drill_cols[1]:
        metric_card("UPI", f"{state_row['UPI']:.2f}", "Update pressure index")
    with drill_cols[2]:
        metric_card("VSI", f"{state_row['VSI']:.2f}", "Volatility stress index")
    with drill_cols[3]:
        metric_card("TPS", f"{state_row['TPS']:.2f}", "Temporal predictability")

    st.markdown(
        f'<div class="panel-note"><strong>Recommended action:</strong> {state_row["Policy_Action"]}</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(build_state_month_chart(state_series, selected_state), use_container_width=True)
    with right:
        st.plotly_chart(build_peer_comparison(state_master, selected_state), use_container_width=True)

    ratio_series = state_series[["year_month", "demo_update_ratio", "biometric_update_ratio", "update_to_enrol_ratio"]]
    fig = px.line(
        ratio_series.melt(id_vars="year_month", var_name="Metric", value_name="Value"),
        x="year_month",
        y="Value",
        color="Metric",
        template="plotly_white",
        title=f"{selected_state}: Update Intensity Through Time",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Download Analytical Outputs")
    export_map = {
        "State Master": state_master,
        "State Month Master": state_month,
        "State Focus Summary": anomalies,
        "National Monthly Summary": national,
        "Pareto Activity": pareto,
        "Rank Persistence": outputs["rank_persistence"],
        "Lag Features": outputs["lag_features"],
    }
    export_choice = st.selectbox("Choose dataset", list(export_map.keys()))
    export_df = export_map[export_choice]
    st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        label=f"Download {export_choice} CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{export_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
