"""Cluster panel: GMM cluster visualization with UMAP 2D projection and dynamic interpreters."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "analysis",
)

# Human-readable cluster profile descriptions (anchored by AMI centroid)
CLUSTER_PROFILES = {
    0: {
        "name": "Expansion Phase",
        "color": "#0e7490",
        "interpreter": (
            "This state's governance profile places it in the <strong>Expansion Phase</strong> group — "
            "low Aadhaar maturity with growing update volumes. "
            "Investment priority is <strong>enrollment infrastructure and outreach</strong>. "
            "Focus on geographic coverage gaps and underserved demographic segments."
        ),
    },
    1: {
        "name": "Unpredictable System",
        "color": "#dc2626",
        "interpreter": (
            "This state behaves unlike most others in its group. Its high Mahalanobis distance "
            "means it is an <strong>outlier even within the unpredictable cluster</strong> — "
            "its governance profile does not fit any standard template. "
            "A <strong>custom diagnostic approach</strong> is recommended before applying any standard intervention."
        ),
    },
}


def _load_clustered_data():
    path = os.path.join(ANALYSIS_DIR, "clustered_states.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_bic_data():
    path = os.path.join(ANALYSIS_DIR, "bic_selection.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_centroids():
    path = os.path.join(ANALYSIS_DIR, "cluster_centroids.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def build_umap_chart(clustered: pd.DataFrame) -> go.Figure:
    """UMAP 2D projection coloured by GMM cluster label."""
    if "UMAP_1" not in clustered.columns:
        return go.Figure().update_layout(title="UMAP projection not available")

    plot_df = clustered.copy()
    plot_df["Mahalanobis_Distance"] = (
        plot_df["Mahalanobis_Distance"].astype(float).tolist()
    )

    # Color map aligned to cluster profiles
    color_map = {}
    for cluster_id, profile in CLUSTER_PROFILES.items():
        color_map[profile["name"]] = profile["color"]

    fig = px.scatter(
        plot_df,
        x="UMAP_1",
        y="UMAP_2",
        color="GMM_Governance_Label",
        size="Mahalanobis_Distance",
        size_max=40,
        hover_name="state",
        hover_data={
            "AMI": ":.3f",
            "UPI": ":.3f",
            "VSI": ":.3f",
            "TPS": ":.3f",
            "Mahalanobis_Distance": ":.2f",
        },
        color_discrete_map=color_map,
        template="plotly_white",
        title="UMAP Projection of Governance Indicator Space",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.2)", title="UMAP Dimension 1"
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.2)", title="UMAP Dimension 2"
        ),
        paper_bgcolor="rgba(255,255,255,0.92)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    )
    return fig


def build_bic_chart(bic_df: pd.DataFrame) -> go.Figure:
    """BIC/AIC/AICc model selection chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bic_df["K"],
            y=bic_df["BIC"],
            mode="lines+markers",
            name="BIC",
            line=dict(color="#102a43", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bic_df["K"],
            y=bic_df["AIC"],
            mode="lines+markers",
            name="AIC",
            line=dict(color="#0e7490", width=2, dash="dash"),
        )
    )

    if "AICc" in bic_df.columns:
        fig.add_trace(
            go.Scatter(
                x=bic_df["K"],
                y=bic_df["AICc"],
                mode="lines+markers",
                name="AICc (Selected)",
                line=dict(color="#d97706", width=2.5, dash="dashdot"),
            )
        )
        optimal_k = int(bic_df.loc[bic_df["AICc"].idxmin(), "K"])
    else:
        optimal_k = int(bic_df.loc[bic_df["BIC"].idxmin(), "K"])

    fig.add_vline(
        x=optimal_k,
        line_dash="dot",
        line_color="#ea580c",
        annotation_text=f"Optimal K={optimal_k}",
        annotation_font_color="#ea580c",
    )

    fig.update_layout(
        title="GMM Model Selection: BIC, AIC & AICc vs K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Information Criterion",
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    )
    return fig


def build_mahalanobis_chart(clustered: pd.DataFrame) -> go.Figure:
    """Bar chart of Mahalanobis distances sorted by magnitude."""
    top = clustered.nlargest(15, "Mahalanobis_Distance")

    color_map = {}
    for _, profile in CLUSTER_PROFILES.items():
        color_map[profile["name"]] = profile["color"]

    fig = px.bar(
        top.sort_values("Mahalanobis_Distance"),
        x="Mahalanobis_Distance",
        y="state",
        orientation="h",
        color="GMM_Governance_Label",
        template="plotly_white",
        title="Top 15 States by Mahalanobis Distance",
        color_discrete_map=color_map,
        labels={"Mahalanobis_Distance": "Distance from Cluster Centroid", "state": ""},
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    )
    return fig


def render_cluster_panel() -> None:
    """Render the full clustering panel with dynamic interpreters."""
    st.markdown(
        '<div class="headline">🔬 Unsupervised Governance Clustering (GMM)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="interpret-box" style="margin-top:0;">'
        "<strong>Gaussian Mixture Model</strong> with BIC-selected K classifies each state into "
        "governance profiles. Each state receives a <em>soft</em> cluster membership probability "
        "and a <strong>Mahalanobis distance</strong> anomaly score — states with high distances "
        "are governance outliers even within their own cluster."
        "</div>",
        unsafe_allow_html=True,
    )

    clustered = _load_clustered_data()
    bic_df = _load_bic_data()
    centroids = _load_centroids()

    if clustered is None:
        st.warning(
            "Clustering data not available. Run the advanced analytics pipeline first."
        )
        return

    # ── UMAP projection ───────────────────────────────────────────────────────
    st.plotly_chart(build_umap_chart(clustered), width="stretch")

    # ── Cluster profile interpreter cards ─────────────────────────────────────
    st.markdown("### Governance Cluster Profiles")
    if "GMM_Cluster" in clustered.columns:
        unique_clusters = sorted(clustered["GMM_Cluster"].dropna().unique())
        cluster_cols = st.columns(len(unique_clusters))
        for i, cluster_id in enumerate(unique_clusters):
            cluster_states = clustered[clustered["GMM_Cluster"] == cluster_id]
            profile = CLUSTER_PROFILES.get(
                int(cluster_id),
                {
                    "name": f"Cluster {int(cluster_id)}",
                    "color": "#475569",
                    "interpreter": "No description available.",
                },
            )
            med_mahal = (
                cluster_states["Mahalanobis_Distance"].median()
                if "Mahalanobis_Distance" in cluster_states
                else 0
            )
            with cluster_cols[i]:
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.9);border-radius:16px;padding:1.25rem;
                                border-left:4px solid {profile['color']};
                                box-shadow:0 8px 24px rgba(15,23,42,0.06);margin-bottom:1rem;">
                        <div style="font-size:0.78rem;color:#64748b;text-transform:uppercase;
                                    letter-spacing:0.05em;">Cluster {int(cluster_id)}</div>
                        <div style="font-size:1.2rem;font-weight:500;color:{profile['color']};
                                    margin:0.3rem 0;">{profile['name']}</div>
                        <div style="font-size:0.85rem;color:#334155;">
                            <strong>{len(cluster_states)}</strong> states &nbsp;·&nbsp;
                            Median Mahalanobis: <strong>{med_mahal:.2f}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── State selector for individual profile interpretation ──────────────────
    st.markdown("### State-Level Cluster Interpretation")
    all_states = sorted(clustered["state"].unique())
    sel_state = st.selectbox(
        "Select state for cluster interpretation:",
        all_states,
        key="cluster_state_select",
    )
    state_cluster_row = clustered[clustered["state"] == sel_state]

    if not state_cluster_row.empty:
        cluster_id = int(state_cluster_row["GMM_Cluster"].iloc[0])
        mahal = (
            float(state_cluster_row["Mahalanobis_Distance"].iloc[0])
            if "Mahalanobis_Distance" in state_cluster_row.columns
            else 0.0
        )
        profile = CLUSTER_PROFILES.get(
            cluster_id,
            {
                "name": f"Cluster {cluster_id}",
                "color": "#475569",
                "interpreter": "No description available.",
            },
        )

        # Substitute mahalanobis distance into Cluster 1 interpreter dynamically
        interp_text = profile["interpreter"]
        if "Mahalanobis distance" in interp_text and cluster_id == 1:
            interp_text = interp_text.replace(
                "Its high Mahalanobis distance",
                f"Its Mahalanobis distance of <strong>{mahal:.2f}</strong>",
            )

        st.markdown(
            f'<div class="interpret-box" style="border-left-color:{profile["color"]}">'
            f'<strong>{sel_state}</strong> belongs to <strong>{profile["name"]}</strong> '
            f"(Cluster {cluster_id}) with a Mahalanobis distance of {mahal:.2f}.<br><br>"
            f"{interp_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Charts side by side ───────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        if bic_df is not None:
            st.plotly_chart(build_bic_chart(bic_df), width="stretch")
    with right:
        st.plotly_chart(build_mahalanobis_chart(clustered), width="stretch")

    # ── Technical details behind expander ─────────────────────────────────────
    with st.expander("Show mathematical validation details"):
        st.info(
            "🔒 **Cluster labels are deterministically anchored by AMI centroid (descending).** "
            "Cluster 0 always represents the highest-maturity governance profile regardless of EM initialisation."
        )
        if centroids is not None and not centroids.empty:
            st.markdown("**Cluster Centroids (Original Indicator Scale)**")
            st.dataframe(centroids, width="stretch", hide_index=True)

        prob_cols = [c for c in clustered.columns if c.startswith("GMM_Prob_")]
        if prob_cols:
            st.markdown("**Soft Cluster Membership Probabilities**")
            display_cols = [
                "state",
                "GMM_Governance_Label",
                "Mahalanobis_Distance",
            ] + prob_cols
            st.dataframe(
                clustered[display_cols].sort_values(
                    "Mahalanobis_Distance", ascending=False
                ),
                width="stretch",
                hide_index=True,
            )
