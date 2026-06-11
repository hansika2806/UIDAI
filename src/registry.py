"""
Model Registry: Save and Load fitted ML estimators using joblib.

Provides a simple key-value artifact store backed by the local filesystem
(models/v1/). Entries are versioned by name; overwriting is explicit.

Usage:
    from registry import save_model_artifact, load_model_artifact

    # During training:
    save_model_artifact(fitted_scaler, "clustering_scaler")
    save_model_artifact(fitted_gmm, "gmm_model")

    # During inference:
    scaler = load_model_artifact("clustering_scaler")
    gmm    = load_model_artifact("gmm_model")
"""

import logging
import os
from typing import Any, List

import joblib

logger = logging.getLogger(__name__)

# Default model store directory (relative to project root)
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "v1",
)


def get_model_dir() -> str:
    """Return the active model store directory (creates it if missing)."""
    os.makedirs(_DEFAULT_MODEL_DIR, exist_ok=True)
    return _DEFAULT_MODEL_DIR


def save_model_artifact(model: Any, name: str) -> str:
    """Serialize a fitted estimator to disk using joblib.

    Parameters
    ----------
    model : Any
        The fitted sklearn/scipy/custom estimator to persist.
    name : str
        Logical name for the artifact (e.g. "gmm_model", "anomaly_scaler").

    Returns
    -------
    str
        Absolute path of the saved artifact file.
    """
    model_dir = get_model_dir()
    path = os.path.join(model_dir, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info(f"[Registry] Saved model artifact: {name} → {path}")
    return path


def load_model_artifact(name: str) -> Any:
    """Load a previously serialized estimator from disk.

    Parameters
    ----------
    name : str
        Logical name used when saving (without the .joblib extension).

    Returns
    -------
    Any
        The deserialized estimator object.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist in the model store.
    """
    model_dir = get_model_dir()
    path = os.path.join(model_dir, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Registry] Model artifact '{name}' not found at {path}. "
            "Run the full training pipeline first."
        )
    model = joblib.load(path)
    logger.info(f"[Registry] Loaded model artifact: {name} ← {path}")
    return model


def list_artifacts() -> List[str]:
    """Return the names of all persisted artifacts in the model store."""
    model_dir = get_model_dir()
    return [
        f.replace(".joblib", "")
        for f in os.listdir(model_dir)
        if f.endswith(".joblib")
    ]
