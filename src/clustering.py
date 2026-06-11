"""
Unsupervised Governance Clustering for UIDAI Framework.

Replaces hard-threshold classification with data-driven Gaussian Mixture
Models (GMM), selects K via BIC, computes soft cluster memberships,
Mahalanobis anomaly distances, and UMAP 2D projections.

Mathematical foundations:
    GMM likelihood:     p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
    EM maximises:       L(theta) = sum_i log sum_k pi_k * N(x_i | mu_k, Sigma_k)
    BIC:                BIC = -2 * L + p * log(n)
    Mahalanobis:        D_M(x, mu_k) = sqrt((x - mu_k)^T Sigma_k^{-1} (x - mu_k))
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Governance cluster label map (assigned post-hoc to GMM clusters by centroid properties)
GOVERNANCE_LABELS = [
    "Stable Mature System",
    "High Maintenance Stress",
    "Expansion Phase",
    "Unpredictable System",
    "Balanced / Transitional",
]

INDICATOR_COLS = ["AMI", "UPI", "VSI", "TPS"]


def select_k_via_bic(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 8),
    n_init: int = 10,
) -> Tuple[int, pd.DataFrame]:
    """Select optimal number of GMM components using AICc.

    AICc = AIC + (2 * P^2 + 2 * P) / (n_samples - P - 1)
    where P is the number of parameters.

    Lower AICc is better. Returns the optimal K and a dataframe of
    all K values with their BIC, AIC, and AICc scores.
    """
    n_samples = X.shape[0]
    D = X.shape[1]

    # Dynamic K cap: if N < 50, limit K range to max 4
    if n_samples < 50:
        k_max = min(4, k_range[1])
        k_min = min(k_range[0], k_max)
        k_range = (k_min, k_max)
        logger.info(
            f"Small dataset detected (N={n_samples} < 50). Capping GMM K range to {k_range}."
        )

    results = []
    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            random_state=42,
        )
        gmm.fit(X)

        # Calculate number of GMM parameters for full covariance:
        # P = K * (1 + D + D*(D+1)/2) - 1
        n_params = k * (1 + D + D * (D + 1) // 2) - 1
        aic = gmm.aic(X)
        bic = gmm.bic(X)

        if n_samples - n_params - 1 > 0:
            aicc = aic + (2.0 * n_params**2 + 2.0 * n_params) / (
                n_samples - n_params - 1
            )
        else:
            aicc = np.inf

        results.append(
            {
                "K": k,
                "BIC": bic,
                "AIC": aic,
                "AICc": aicc,
                "log_likelihood": gmm.score(X) * n_samples,
            }
        )

    bic_df = pd.DataFrame(results)
    optimal_k = int(bic_df.loc[bic_df["AICc"].idxmin(), "K"])
    logger.info(f"AICc-optimal K = {optimal_k}")
    return optimal_k, bic_df


def fit_gmm(
    X: np.ndarray,
    n_components: int,
    n_init: int = 10,
) -> GaussianMixture:
    """Fit a Gaussian Mixture Model with full covariance."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        random_state=42,
    )
    gmm.fit(X)
    return gmm


def anchor_cluster_order(
    gmm: GaussianMixture,
    indicator_cols: list,
    anchor_col: str = "AMI",
) -> np.ndarray:
    """Return a permutation index that sorts GMM clusters by a chosen centroid dimension.

    LABEL ANCHORING RATIONALE:
        GMM uses random initialisation. Even with a fixed random_state, running
        on different data subsets can produce the same clusters but in a different
        order (the classic 'label switching' problem). This makes cluster 0 this
        month mean something different from cluster 0 last month.

        Solution: sort cluster indices by the value of `anchor_col` in the
        cluster centroid (descending). Cluster 0 is always the highest-AMI
        (most mature) cluster; cluster 1 is the next-highest; and so on.
        This ordering is data-driven and deterministic regardless of EM init.

    Returns:
        new_order : np.ndarray of shape (n_components,)
            Permutation such that gmm.means_[new_order] is sorted by anchor_col.
    """
    try:
        anchor_idx = indicator_cols.index(anchor_col)
    except ValueError:
        anchor_idx = 0  # fallback: sort by first column

    # Sort by anchor column descending (highest AMI first = most mature cluster)
    new_order = np.argsort(gmm.means_[:, anchor_idx])[::-1]
    return new_order


def remap_gmm_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    new_order: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remap labels and probabilities to the anchored cluster ordering."""
    # Build reverse mapping: old cluster k -> new cluster rank
    old_to_new = np.empty_like(new_order)
    for new_rank, old_k in enumerate(new_order):
        old_to_new[old_k] = new_rank

    anchored_labels = old_to_new[labels]
    anchored_probs = probabilities[:, new_order]  # reorder probability columns
    return anchored_labels, anchored_probs


def compute_mahalanobis_distances(
    X: np.ndarray,
    gmm: GaussianMixture,
) -> np.ndarray:
    """Compute Mahalanobis distance from each point to its assigned cluster centroid.

    D_M(x, mu_k) = sqrt((x - mu_k)^T Sigma_k^{-1} (x - mu_k))

    This is a principled anomaly metric: high D_M means the state is
    far from the typical profile of its own governance cluster.
    """
    labels = gmm.predict(X)
    distances = np.zeros(len(X))

    for k in range(gmm.n_components):
        mask = labels == k
        if not np.any(mask):
            continue
        mu_k = gmm.means_[k]
        sigma_k = gmm.covariances_[k]

        # Regularize covariance to prevent singular matrix
        sigma_k_reg = sigma_k + np.eye(sigma_k.shape[0]) * 1e-6
        sigma_k_inv = np.linalg.inv(sigma_k_reg)

        diff = X[mask] - mu_k
        # Vectorized Mahalanobis: sqrt(diag(diff @ Sigma_inv @ diff.T))
        distances[mask] = np.sqrt(np.sum(diff @ sigma_k_inv * diff, axis=1))

    return distances


def assign_governance_labels(
    gmm: GaussianMixture,
    indicator_cols: list,
) -> Dict[int, str]:
    """Assign interpretable governance labels to GMM clusters based on centroid properties.

    Heuristic: rank clusters by their centroid characteristics:
        - High AMI + Low VSI => Stable Mature
        - High UPI + High VSI => High Maintenance Stress
        - Low AMI + Low UPI => Expansion Phase
        - Low TPS => Unpredictable
        - Otherwise => Balanced / Transitional
    """
    n_components = gmm.n_components
    centroids = pd.DataFrame(gmm.means_, columns=indicator_cols)
    label_map = {}
    used_labels = set()

    # Score each cluster for each governance profile
    for k in range(n_components):
        c = centroids.iloc[k]
        scores = {}

        # Stable Mature: high AMI, low VSI
        scores["Stable Mature System"] = c["AMI"] - c["VSI"]
        # High Maintenance: high UPI, high VSI
        scores["High Maintenance Stress"] = c["UPI"] + c["VSI"]
        # Expansion Phase: low AMI, low UPI
        scores["Expansion Phase"] = (1 - c["AMI"]) + (1 - c["UPI"])
        # Unpredictable: low TPS
        scores["Unpredictable System"] = 1 - c["TPS"]
        # Balanced: moderate everything
        scores["Balanced / Transitional"] = (
            1 - abs(c["AMI"] - 0.5) - abs(c["UPI"] - 0.5)
        )

        # Sort by score descending, pick first unused
        for label in sorted(scores, key=scores.get, reverse=True):
            if label not in used_labels:
                label_map[k] = label
                used_labels.add(label)
                break

    # Fill any remaining clusters with generic labels
    for k in range(n_components):
        if k not in label_map:
            label_map[k] = f"Cluster {k}"

    return label_map


def compute_umap_projection(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project indicator space to 2D using UMAP.

    UMAP preserves both local and global structure better than t-SNE,
    making it ideal for visualizing governance clusters.
    """
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, X.shape[0] - 1),
            min_dist=min_dist,
            random_state=42,
            metric="euclidean",
        )
        return reducer.fit_transform(X)
    except ImportError:
        logger.warning("umap-learn not installed; returning PCA fallback.")
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(X)
    except Exception as e:
        logger.warning(f"UMAP failed ({e}); returning PCA fallback.")
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(X)


def run_governance_clustering(
    state_master: pd.DataFrame,
    indicator_cols: Optional[list] = None,
    k_range: Tuple[int, int] = (2, 8),
) -> Dict[str, object]:
    """Full clustering pipeline.

    1. Standardize indicators
    2. Select K via BIC
    3. Fit GMM
    4. Compute soft memberships + Mahalanobis distances
    5. Assign governance labels
    6. Compute UMAP projection

    Returns dict with all outputs.
    """
    if indicator_cols is None:
        indicator_cols = INDICATOR_COLS

    logger.info("Running governance clustering pipeline...")

    # 1. Standardize
    X_raw = state_master[indicator_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 2. Select K
    optimal_k, bic_df = select_k_via_bic(X, k_range=k_range)

    # 3. Fit GMM
    gmm = fit_gmm(X, n_components=optimal_k)

    # 4. Soft memberships and distances
    probabilities = gmm.predict_proba(X)
    hard_labels = gmm.predict(X)
    mahalanobis_dist = compute_mahalanobis_distances(X, gmm)

    # ── Anchor cluster ordering by AMI centroid (Fix: label switching) ──
    new_order = anchor_cluster_order(gmm, indicator_cols, anchor_col="AMI")
    hard_labels, probabilities = remap_gmm_predictions(
        hard_labels, probabilities, new_order
    )
    # Reorder GMM means/covariances to match the anchored order
    anchored_means = gmm.means_[new_order]
    # Build anchored label map (anchored cluster 0 = highest AMI = most mature)
    anchored_centroids_df = pd.DataFrame(
        scaler.inverse_transform(anchored_means),
        columns=indicator_cols,
    )
    anchored_centroids_df["Cluster"] = range(optimal_k)

    # 5. Governance labels (on anchored centroids)
    label_map = assign_governance_labels(gmm, indicator_cols)
    # Remap label_map keys to anchored ordering
    anchored_label_map = {
        new_rank: label_map.get(int(old_k), f"Cluster {new_rank}")
        for new_rank, old_k in enumerate(new_order)
    }

    # 6. UMAP
    umap_coords = compute_umap_projection(X)

    # Build result DataFrame
    result = state_master.copy()
    result["GMM_Cluster"] = hard_labels
    result["GMM_Governance_Label"] = result["GMM_Cluster"].map(anchored_label_map)
    result["Mahalanobis_Distance"] = mahalanobis_dist

    for k in range(optimal_k):
        result[f"GMM_Prob_Cluster_{k}"] = probabilities[:, k]

    result["UMAP_1"] = umap_coords[:, 0]
    result["UMAP_2"] = umap_coords[:, 1]

    # Cluster centroid summary (anchored)
    anchored_centroids_df["Governance_Label"] = anchored_centroids_df["Cluster"].map(
        anchored_label_map
    )
    anchored_centroids_df["Cluster_Size"] = [
        int((hard_labels == k).sum()) for k in range(optimal_k)
    ]

    logger.info(
        f"Clustering complete: {optimal_k} clusters (anchored by AMI centroid), "
        f"labels: {anchored_label_map}"
    )

    return {
        "state_master_clustered": result,
        "bic_selection": bic_df,
        "centroids": anchored_centroids_df,
        "gmm_model": gmm,
        "scaler": scaler,
        "label_map": anchored_label_map,
    }
