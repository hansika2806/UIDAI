"""Cluster panel: GMM cluster visualization with UMAP 2D projection."""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "analysis")


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

    # Coerce size column to plain list to prevent narwhals.Series crash in plotly
    plot_df = clustered.copy()
    plot_df["Mahalanobis_Distance"] = plot_df["Mahalanobis_Distance"].astype(float).tolist()

    fig = px.scatter(
        plot_df,
        x="UMAP_1",
        y="UMAP_2",
        color="GMM_Governance_Label",
        size="Mahalanobis_Distance",
        size_max=40,
        hover_name="state",
        hover_data=["AMI", "UPI", "VSI", "TPS", "Mahalanobis_Distance"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white",
        title="UMAP Projection of Governance Indicator Space (Clusters anchored by AMI)",
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(title="UMAP Dimension 1")
    fig.update_yaxes(title="UMAP Dimension 2")
    return fig


def build_bic_chart(bic_df: pd.DataFrame) -> go.Figure:
    """BIC/AIC/AICc model selection chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bic_df["K"], y=bic_df["BIC"],
        mode="lines+markers", name="BIC",
        line=dict(color="#102a43", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=bic_df["K"], y=bic_df["AIC"],
        mode="lines+markers", name="AIC",
        line=dict(color="#0e7490", width=2, dash="dash"),
    ))

    if "AICc" in bic_df.columns:
        fig.add_trace(go.Scatter(
            x=bic_df["K"], y=bic_df["AICc"],
            mode="lines+markers", name="AICc (Selected)",
            line=dict(color="#d97706", width=2.5, dash="dashdot"),
        ))
        optimal_k = int(bic_df.loc[bic_df["AICc"].idxmin(), "K"])
    else:
        optimal_k = int(bic_df.loc[bic_df["BIC"].idxmin(), "K"])

    fig.add_vline(x=optimal_k, line_dash="dot", line_color="#ea580c",
                  annotation_text=f"Optimal K={optimal_k}")

    fig.update_layout(
        title="GMM Model Selection: BIC, AIC, and AICc vs K",
        xaxis_title="Number of Components (K)",
        yaxis_title="Information Criterion",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_mahalanobis_chart(clustered: pd.DataFrame) -> go.Figure:
    """Bar chart of Mahalanobis distances sorted by magnitude."""
    top = clustered.nlargest(15, "Mahalanobis_Distance")
    fig = px.bar(
        top.sort_values("Mahalanobis_Distance"),
        x="Mahalanobis_Distance",
        y="state",
        orientation="h",
        color="GMM_Governance_Label",
        template="plotly_white",
        title="Top 15 States by Mahalanobis Distance (Cluster Outliers)",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def render_cluster_panel() -> None:
    """Render the full clustering panel."""
    st.subheader("🔬 Unsupervised Governance Clustering (GMM)")
    st.markdown(
        '<div class="panel-note"><strong>Gaussian Mixture Model</strong> with BIC-selected K replaces '
        'hard-threshold classification. Each state receives a <em>soft</em> cluster membership probability '
        'and a <strong>Mahalanobis distance</strong> anomaly score from its cluster centroid.</div>',
        unsafe_allow_html=True,
    )

    clustered = _load_clustered_data()
    bic_df = _load_bic_data()
    centroids = _load_centroids()

    if clustered is None:
        st.warning("Clustering data not available. Run the advanced analytics pipeline first.")
        return

    # Label anchoring info (Fix 2)
    st.info(
        "🔒 **Cluster labels are deterministically anchored by AMI centroid (descending).** "
        "Cluster 0 always represents the highest-maturity governance profile regardless of EM initialisation. "
        "This prevents label-switching across pipeline runs."
    )

    # UMAP projection
    st.plotly_chart(build_umap_chart(clustered), use_container_width=True)

    left, right = st.columns(2)

    with left:
        # BIC selection
        if bic_df is not None:
            st.plotly_chart(build_bic_chart(bic_df), use_container_width=True)

    with right:
        # Mahalanobis distance ranking
        st.plotly_chart(build_mahalanobis_chart(clustered), use_container_width=True)

    # Cluster centroids
    if centroids is not None and not centroids.empty:
        st.markdown("**Cluster Centroids (Original Indicator Scale)**")
        st.dataframe(centroids, use_container_width=True, hide_index=True)

    # Soft membership probabilities
    prob_cols = [c for c in clustered.columns if c.startswith("GMM_Prob_")]
    if prob_cols:
        st.markdown("**Soft Cluster Membership Probabilities**")
        display_cols = ["state", "GMM_Governance_Label", "Mahalanobis_Distance"] + prob_cols
        st.dataframe(
            clustered[display_cols].sort_values("Mahalanobis_Distance", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
