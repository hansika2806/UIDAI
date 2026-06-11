"""
MLflow Experiment Tracking for UIDAI Governance Framework.

Tracks SARIMA model experiments, hyperparameters, and performance metrics.
"""

from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for UIDAI forecasting models.

    Tracks:
    - SARIMA hyperparameters (p, d, q, P, D, Q, s)
    - Model selection metrics (AIC, BIC)
    - Forecast accuracy (MAPE, RMSE, MAE)
    - Walk-forward validation results
    """

    def __init__(
        self,
        experiment_name: str = "uidai-governance",
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (None for local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracker initialized for experiment: {experiment_name}")

    def log_sarima_experiment(
        self,
        state: str,
        order: tuple,
        seasonal_order: tuple,
        aic: float,
        bic: float,
        metrics: Optional[Dict[str, float]] = None,
        model: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Log a SARIMA model experiment.

        Args:
            state: State name
            order: SARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            aic: Akaike Information Criterion
            bic: Bayesian Information Criterion
            metrics: Optional validation metrics (MAPE, RMSE, MAE)
            model: Optional fitted model object
            tags: Optional tags for the run

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name=f"SARIMA_{state}") as run:
            # Log parameters
            p, d, q = order
            P, D, Q, s = seasonal_order

            mlflow.log_param("state", state)
            mlflow.log_param("p", p)
            mlflow.log_param("d", d)
            mlflow.log_param("q", q)
            mlflow.log_param("P", P)
            mlflow.log_param("D", D)
            mlflow.log_param("Q", Q)
            mlflow.log_param("seasonal_period", s)
            mlflow.log_param("order", str(order))
            mlflow.log_param("seasonal_order", str(seasonal_order))

            # Log model selection metrics
            mlflow.log_metric("aic", aic)
            mlflow.log_metric("bic", bic)

            # Log validation metrics if provided
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if not np.isnan(metric_value):
                        mlflow.log_metric(metric_name, metric_value)

            # Log tags
            default_tags = {
                "model_type": "SARIMA",
                "framework": "statsmodels",
                "state": state,
            }
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            # Log model if provided
            if model is not None:
                try:
                    mlflow.statsmodels.log_model(model, "model")
                except Exception as e:
                    logger.warning(f"Could not log model: {e}")

            logger.info(
                f"Logged SARIMA experiment for {state}",
                extra={
                    "run_id": run.info.run_id,
                    "state": state,
                    "aic": aic,
                    "bic": bic,
                },
            )

            return run.info.run_id

    def log_grid_search_results(
        self, state: str, results_df: pd.DataFrame, best_params: Dict[str, Any]
    ) -> None:
        """
        Log SARIMA grid search results.

        Args:
            state: State name
            results_df: DataFrame with columns [order, seasonal_order, aic, bic]
            best_params: Best parameters selected
        """
        with mlflow.start_run(run_name=f"GridSearch_{state}"):
            # Log best parameters
            mlflow.log_params(best_params)

            # Log grid search summary
            mlflow.log_metric("n_models_tested", len(results_df))
            mlflow.log_metric("best_aic", results_df["aic"].min())
            mlflow.log_metric("best_bic", results_df["bic"].min())

            # Save grid search results as artifact
            results_path = f"grid_search_{state}.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)

            mlflow.set_tag("search_type", "grid_search")
            mlflow.set_tag("state", state)

            logger.info(f"Logged grid search results for {state}")

    def log_anomaly_detection(
        self,
        method: str,
        n_anomalies: int,
        contamination: float,
        parameters: Dict[str, Any],
        anomaly_scores: Optional[np.ndarray] = None,
    ) -> str:
        """
        Log anomaly detection experiment.

        Args:
            method: Detection method (e.g., 'IsolationForest', 'ECOD')
            n_anomalies: Number of anomalies detected
            contamination: Contamination parameter
            parameters: Model parameters
            anomaly_scores: Optional anomaly scores array

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name=f"Anomaly_{method}") as run:
            mlflow.log_param("method", method)
            mlflow.log_param("contamination", contamination)
            mlflow.log_params(parameters)

            mlflow.log_metric("n_anomalies", n_anomalies)

            if anomaly_scores is not None:
                mlflow.log_metric("mean_anomaly_score", float(np.mean(anomaly_scores)))
                mlflow.log_metric("max_anomaly_score", float(np.max(anomaly_scores)))

            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("method", method)

            logger.info(f"Logged anomaly detection: {method}")

            return run.info.run_id

    def log_clustering_results(
        self,
        n_clusters: int,
        bic: float,
        aic: float,
        silhouette_score: Optional[float] = None,
        cluster_sizes: Optional[Dict[int, int]] = None,
    ) -> str:
        """
        Log clustering experiment results.

        Args:
            n_clusters: Number of clusters
            bic: Bayesian Information Criterion
            aic: Akaike Information Criterion
            silhouette_score: Optional silhouette score
            cluster_sizes: Optional cluster size distribution

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name=f"Clustering_K{n_clusters}") as run:
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("method", "GMM")

            mlflow.log_metric("bic", bic)
            mlflow.log_metric("aic", aic)

            if silhouette_score is not None:
                mlflow.log_metric("silhouette_score", silhouette_score)

            if cluster_sizes:
                for cluster_id, size in cluster_sizes.items():
                    mlflow.log_metric(f"cluster_{cluster_id}_size", size)

            mlflow.set_tag("task", "clustering")
            mlflow.set_tag("algorithm", "GMM")

            logger.info(f"Logged clustering results: K={n_clusters}")

            return run.info.run_id

    @staticmethod
    def get_best_run(
        experiment_name: str, metric: str = "aic"
    ) -> Optional[mlflow.entities.Run]:
        """
        Get the best run from an experiment based on a metric.

        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize (lower is better)

        Returns:
            Best MLflow run or None
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1,
        )

        if len(runs) == 0:
            return None

        return runs.iloc[0]


# Convenience function
def create_tracker(
    experiment_name: str = "uidai-governance", tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """Create and return an MLflow tracker instance."""
    return MLflowTracker(experiment_name, tracking_uri)


# Made with Bob
