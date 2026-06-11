"""
Configuration Management for UIDAI Governance Intelligence Framework.

Uses Pydantic for type-safe configuration with validation.
Supports environment variables and .env files.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathConfig(BaseSettings):
    """File path configuration."""

    model_config = SettingsConfigDict(
        env_prefix="UIDAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Base directory (project root)
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.absolute()
    )

    # Data directories
    data_dir: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    cleaned_data_dir: Optional[Path] = None
    analysis_data_dir: Optional[Path] = None

    @field_validator("data_dir", mode="before")
    @classmethod
    def set_data_dir(cls, v, info):
        if v is None:
            return info.data["base_dir"] / "data"
        return Path(v)

    @field_validator("raw_data_dir", mode="before")
    @classmethod
    def set_raw_data_dir(cls, v, info):
        if v is None:
            return info.data["data_dir"] / "raw"
        return Path(v)

    @field_validator("cleaned_data_dir", mode="before")
    @classmethod
    def set_cleaned_data_dir(cls, v, info):
        if v is None:
            return info.data["data_dir"] / "cleaned"
        return Path(v)

    @field_validator("analysis_data_dir", mode="before")
    @classmethod
    def set_analysis_data_dir(cls, v, info):
        if v is None:
            return info.data["data_dir"] / "analysis"
        return Path(v)

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.raw_data_dir,
            self.cleaned_data_dir,
            self.analysis_data_dir,
        ]:
            if dir_path:
                dir_path.mkdir(parents=True, exist_ok=True)


class MLConfig(BaseSettings):
    """Machine Learning configuration."""

    model_config = SettingsConfigDict(
        env_prefix="UIDAI_ML_", env_file=".env", env_file_encoding="utf-8"
    )

    # Random seed for reproducibility
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # SARIMA forecasting
    sarima_max_p: int = Field(default=2, ge=0, le=5)
    sarima_max_d: int = Field(default=1, ge=0, le=2)
    sarima_max_q: int = Field(default=2, ge=0, le=5)
    sarima_max_P: int = Field(default=1, ge=0, le=2)
    sarima_max_D: int = Field(default=1, ge=0, le=2)
    sarima_max_Q: int = Field(default=1, ge=0, le=2)
    sarima_seasonal_period: int = Field(default=12, ge=1)

    # Anomaly detection
    isolation_forest_contamination: float = Field(default=0.1, ge=0.0, le=0.5)
    isolation_forest_n_estimators: int = Field(default=200, ge=50, le=500)

    # Clustering
    gmm_min_components: int = Field(default=2, ge=2, le=10)
    gmm_max_components: int = Field(default=8, ge=2, le=10)
    gmm_n_init: int = Field(default=10, ge=1, le=50)

    # MLflow tracking
    mlflow_tracking_uri: Optional[str] = Field(default=None)
    mlflow_experiment_name: str = Field(default="uidai-governance")


class DashboardConfig(BaseSettings):
    """Streamlit dashboard configuration."""

    model_config = SettingsConfigDict(
        env_prefix="UIDAI_DASHBOARD_", env_file=".env", env_file_encoding="utf-8"
    )

    # Server settings
    port: int = Field(default=8501, ge=1024, le=65535)
    host: str = Field(default="0.0.0.0")

    # Cache settings
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    # UI settings
    page_title: str = Field(default="UIDAI Intelligence Studio")
    page_icon: str = Field(default="🇮🇳")
    layout: str = Field(default="wide")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="UIDAI_LOG_", env_file=".env", env_file_encoding="utf-8"
    )

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(default="json", pattern="^(json|text)$")
    log_file: Optional[Path] = None

    @field_validator("log_file", mode="before")
    @classmethod
    def set_log_file(cls, v):
        if v is None:
            return None
        return Path(v)


class Config:
    """Main configuration container."""

    def __init__(self):
        self.paths = PathConfig()
        self.ml = MLConfig()
        self.dashboard = DashboardConfig()
        self.logging = LoggingConfig()

        # Ensure directories exist
        self.paths.ensure_directories()

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  paths={self.paths}\n"
            f"  ml={self.ml}\n"
            f"  dashboard={self.dashboard}\n"
            f"  logging={self.logging}\n"
            f")"
        )


# Global configuration instance
config = Config()


# Backward compatibility with old config.py
BASE_DIR = str(config.paths.base_dir)
DATA_DIR = str(config.paths.data_dir)
RAW_DATA_DIR = str(config.paths.raw_data_dir)
CLEANED_DATA_DIR = str(config.paths.cleaned_data_dir)
ANALYSIS_DATA_DIR = str(config.paths.analysis_data_dir)

# Made with Bob
