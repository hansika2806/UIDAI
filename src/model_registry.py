"""
Model Persistence & Registry for UIDAI Governance Framework.

Implements model serialization, versioning, and lifecycle management using
MLflow Model Registry. Decouples training from inference to prevent:
- Coordinate sliding in clustering (GMM centroids shift on refit)
- Label switching in classification
- Inconsistent predictions across runs

Architecture:
- Train once → Serialize → Register → Deploy
- Inference mode loads pre-trained models
- A/B testing and rollback support
- Model lineage tracking
"""

import joblib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    MLflow-based model registry for UIDAI governance models.
    
    Manages:
    - SARIMA forecasting models (per state)
    - GMM clustering models (governance profiles)
    - Gradient Boosting models (priority scoring)
    - Anomaly detection models (Isolation Forest, ECOD)
    """
    
    def __init__(
        self,
        registry_uri: Optional[str] = None,
        experiment_name: str = "uidai-governance"
    ):
        """
        Initialize model registry.
        
        Args:
            registry_uri: MLflow tracking URI (None for local)
            experiment_name: MLflow experiment name
        """
        if registry_uri:
            mlflow.set_tracking_uri(registry_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"Model registry initialized: {experiment_name}")
    
    def register_sarima_model(
        self,
        model: Any,
        state: str,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        metrics: Dict[str, float],
        stage: str = "Staging"
    ) -> str:
        """
        Register a SARIMA forecasting model.
        
        Args:
            model: Fitted SARIMA model
            state: State name
            order: SARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            metrics: Validation metrics (MAPE, RMSE, etc.)
            stage: Model stage ('Staging', 'Production', 'Archived')
        
        Returns:
            Model version string
        """
        model_name = f"sarima_{state.lower().replace(' ', '_')}"
        
        with mlflow.start_run(run_name=f"SARIMA_{state}") as run:
            # Log parameters
            p, d, q = order
            P, D, Q, s = seasonal_order
            
            mlflow.log_params({
                "state": state,
                "p": p, "d": d, "q": q,
                "P": P, "D": D, "Q": Q,
                "seasonal_period": s,
                "model_type": "SARIMA"
            })
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if not np.isnan(metric_value):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.statsmodels.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=[])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage
            )
            
            logger.info(
                f"Registered SARIMA model for {state}",
                extra={
                    "model_name": model_name,
                    "version": model_version,
                    "stage": stage,
                    "aic": metrics.get("aic"),
                    "mape": metrics.get("mape")
                }
            )
            
            return f"{model_name}/v{model_version}"
    
    def register_clustering_model(
        self,
        model: BaseEstimator,
        n_clusters: int,
        metrics: Dict[str, float],
        feature_names: List[str],
        stage: str = "Staging"
    ) -> str:
        """
        Register a GMM clustering model.
        
        CRITICAL: Clustering models must be serialized to prevent coordinate
        sliding. Re-fitting GMM on new data can cause cluster labels to switch,
        breaking downstream analysis.
        
        Args:
            model: Fitted GMM model
            n_clusters: Number of clusters
            metrics: Model metrics (BIC, AIC, silhouette)
            feature_names: List of feature names used
            stage: Model stage
        
        Returns:
            Model version string
        """
        model_name = "governance_clustering"
        
        with mlflow.start_run(run_name=f"GMM_K{n_clusters}") as run:
            # Log parameters
            mlflow.log_params({
                "n_clusters": n_clusters,
                "model_type": "GMM",
                "features": ",".join(feature_names),
                "covariance_type": getattr(model, "covariance_type", "full")
            })
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Save cluster centroids as artifact
            if hasattr(model, "means_"):
                centroids_df = pd.DataFrame(
                    model.means_,
                    columns=feature_names
                )
                centroids_path = "cluster_centroids.csv"
                centroids_df.to_csv(centroids_path, index=False)
                mlflow.log_artifact(centroids_path)
            
            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=[])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage
            )
            
            logger.info(
                f"Registered clustering model",
                extra={
                    "model_name": model_name,
                    "version": model_version,
                    "n_clusters": n_clusters,
                    "bic": metrics.get("bic")
                }
            )
            
            return f"{model_name}/v{model_version}"
    
    def register_priority_model(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        metrics: Dict[str, float],
        stage: str = "Staging"
    ) -> str:
        """
        Register a priority scoring model (Gradient Boosting).
        
        Args:
            model: Fitted GBM model
            feature_names: List of feature names
            metrics: Model metrics (R², RMSE, etc.)
            stage: Model stage
        
        Returns:
            Model version string
        """
        model_name = "priority_scoring"
        
        with mlflow.start_run(run_name="PriorityScoring") as run:
            # Log parameters
            mlflow.log_params({
                "model_type": "GradientBoosting",
                "features": ",".join(feature_names),
                "n_estimators": getattr(model, "n_estimators", None),
                "max_depth": getattr(model, "max_depth", None)
            })
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=[])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage
            )
            
            logger.info(
                f"Registered priority scoring model",
                extra={
                    "model_name": model_name,
                    "version": model_version,
                    "r2": metrics.get("r2")
                }
            )
            
            return f"{model_name}/v{model_version}"
    
    def load_sarima_model(self, state: str, stage: str = "Production") -> Any:
        """
        Load a SARIMA model from registry.
        
        Args:
            state: State name
            stage: Model stage to load
        
        Returns:
            Loaded SARIMA model
        """
        model_name = f"sarima_{state.lower().replace(' ', '_')}"
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.statsmodels.load_model(model_uri)
            logger.info(f"Loaded SARIMA model for {state} from {stage}")
            return model
        except Exception as e:
            logger.error(f"Failed to load SARIMA model for {state}: {e}")
            return None
    
    def load_clustering_model(self, stage: str = "Production") -> Optional[BaseEstimator]:
        """
        Load clustering model from registry.
        
        Args:
            stage: Model stage to load
        
        Returns:
            Loaded GMM model
        """
        model_name = "governance_clustering"
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded clustering model from {stage}")
            return model
        except Exception as e:
            logger.error(f"Failed to load clustering model: {e}")
            return None
    
    def load_priority_model(self, stage: str = "Production") -> Optional[BaseEstimator]:
        """
        Load priority scoring model from registry.
        
        Args:
            stage: Model stage to load
        
        Returns:
            Loaded GBM model
        """
        model_name = "priority_scoring"
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded priority scoring model from {stage}")
            return model
        except Exception as e:
            logger.error(f"Failed to load priority scoring model: {e}")
            return None
    
    def promote_to_production(self, model_name: str, version: str) -> None:
        """
        Promote a model version to production.
        
        Args:
            model_name: Name of the model
            version: Version to promote
        """
        client = mlflow.tracking.MlflowClient()
        
        # Archive current production model
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for prod_version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived"
                )
        except Exception:
            pass  # No production model exists yet
        
        # Promote new version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(
            f"Promoted model to production",
            extra={"model_name": model_name, "version": version}
        )
    
    def rollback_model(self, model_name: str, target_version: str) -> None:
        """
        Rollback to a previous model version.
        
        Args:
            model_name: Name of the model
            target_version: Version to rollback to
        """
        self.promote_to_production(model_name, target_version)
        logger.warning(
            f"Rolled back model",
            extra={"model_name": model_name, "version": target_version}
        )
    
    def list_models(self, stage: Optional[str] = None) -> List[Dict]:
        """
        List all registered models.
        
        Args:
            stage: Filter by stage (None for all)
        
        Returns:
            List of model metadata dictionaries
        """
        client = mlflow.tracking.MlflowClient()
        models = []
        
        for rm in client.search_registered_models():
            for mv in rm.latest_versions:
                if stage is None or mv.current_stage == stage:
                    models.append({
                        "name": rm.name,
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "run_id": mv.run_id,
                        "creation_time": mv.creation_timestamp
                    })
        
        return models


# Convenience functions
def save_model_local(model: Any, path: Path, metadata: Optional[Dict] = None) -> None:
    """
    Save model locally using joblib (fallback if MLflow unavailable).
    
    Args:
        model: Model to save
        path: Save path
        metadata: Optional metadata dictionary
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {"model": model}
    if metadata:
        save_dict["metadata"] = metadata
    
    joblib.dump(save_dict, path)
    logger.info(f"Saved model locally to {path}")


def load_model_local(path: Path) -> Tuple[Any, Optional[Dict]]:
    """
    Load model from local file.
    
    Args:
        path: Model file path
    
    Returns:
        Tuple of (model, metadata)
    """
    save_dict = joblib.load(path)
    model = save_dict.get("model")
    metadata = save_dict.get("metadata")
    
    logger.info(f"Loaded model from {path}")
    return model, metadata

# Made with Bob
