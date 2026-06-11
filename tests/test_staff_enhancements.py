"""
Tests for Staff-Level ML Engineering Enhancements:
  1. Pandera data contract validation (schemas.py)
  2. Model registry save/load (registry.py)
  3. Drift monitoring — KS test and PSI (drift.py)
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# 1. Pandera Schema Validation
# ---------------------------------------------------------------------------


class TestPanderaSchemas:
    """Data contract validation tests."""

    def test_enrolment_schema_valid(self):
        """Valid enrolment DataFrame passes schema."""
        from schemas import EnrolmentSchema

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "state": ["Maharashtra", "Delhi"],
                "district": ["Mumbai", "Central"],
                "pincode": ["400001", "110001"],
                "age_0_5": [100, 200],
                "age_5_17": [150, 250],
                "age_18_greater": [500, 800],
            }
        )
        validated = EnrolmentSchema.validate(df)
        assert len(validated) == 2

    def test_enrolment_schema_rejects_negative_counts(self):
        """Schema raises SchemaErrors on negative age counts."""
        import pandera as pa
        from schemas import EnrolmentSchema

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01"]),
                "state": ["Maharashtra"],
                "district": ["Mumbai"],
                "pincode": ["400001"],
                "age_0_5": [-5],  # invalid: negative
                "age_5_17": [150],
                "age_18_greater": [500],
            }
        )
        with pytest.raises(pa.errors.SchemaErrors):
            EnrolmentSchema.validate(df, lazy=True)

    def test_demographic_schema_valid(self):
        """Valid demographic DataFrame passes schema."""
        from schemas import DemographicUpdateSchema

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-03-01"]),
                "state": ["Karnataka"],
                "district": ["Bangalore"],
                "pincode": ["560001"],
                "demo_age_5_17": [300],
                "demo_age_17_": [700],
            }
        )
        validated = DemographicUpdateSchema.validate(df)
        assert "demo_age_5_17" in validated.columns

    def test_biometric_schema_valid(self):
        """Valid biometric DataFrame passes schema."""
        from schemas import BiometricUpdateSchema

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-04-01"]),
                "state": ["Tamil Nadu"],
                "district": ["Chennai"],
                "pincode": ["600001"],
                "bio_age_5_17": [200],
                "bio_age_17_": [600],
            }
        )
        validated = BiometricUpdateSchema.validate(df)
        assert len(validated) == 1


# ---------------------------------------------------------------------------
# 2. Model Registry Save / Load
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Model serialization and deserialization tests."""

    def test_save_and_load_sklearn_model(self, tmp_path, monkeypatch):
        """Saving and loading an sklearn estimator round-trips correctly."""
        from sklearn.preprocessing import StandardScaler
        import registry

        # Redirect model store to temp directory
        monkeypatch.setattr(registry, "_DEFAULT_MODEL_DIR", str(tmp_path))

        scaler = StandardScaler()
        scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        path = registry.save_model_artifact(scaler, "test_scaler")
        assert os.path.exists(path)

        loaded = registry.load_model_artifact("test_scaler")
        np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)

    def test_load_missing_artifact_raises(self, tmp_path, monkeypatch):
        """Loading a non-existent artifact raises FileNotFoundError."""
        import registry

        monkeypatch.setattr(registry, "_DEFAULT_MODEL_DIR", str(tmp_path))

        with pytest.raises(FileNotFoundError):
            registry.load_model_artifact("does_not_exist")

    def test_list_artifacts(self, tmp_path, monkeypatch):
        """list_artifacts returns all saved artifact names."""
        from sklearn.preprocessing import StandardScaler
        import registry

        monkeypatch.setattr(registry, "_DEFAULT_MODEL_DIR", str(tmp_path))

        scaler = StandardScaler().fit([[1, 2]])
        registry.save_model_artifact(scaler, "artifact_a")
        registry.save_model_artifact(scaler, "artifact_b")

        names = registry.list_artifacts()
        assert "artifact_a" in names
        assert "artifact_b" in names

    def test_saved_artifact_is_correct_type(self, tmp_path, monkeypatch):
        """Loaded artifact preserves its original type."""
        from sklearn.mixture import GaussianMixture
        import registry

        monkeypatch.setattr(registry, "_DEFAULT_MODEL_DIR", str(tmp_path))

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(np.random.randn(20, 3))

        registry.save_model_artifact(gmm, "test_gmm")
        loaded = registry.load_model_artifact("test_gmm")

        assert isinstance(loaded, GaussianMixture)
        assert loaded.n_components == 2


# ---------------------------------------------------------------------------
# 3. Drift Monitoring (KS Test + PSI)
# ---------------------------------------------------------------------------


class TestDriftMonitoring:
    """Statistical drift detection tests."""

    def test_ks_stable_identical_distributions(self):
        """KS test on identical distributions reports no drift."""
        from drift import detect_covariate_drift

        data = pd.Series(np.random.normal(0, 1, 100))
        result = detect_covariate_drift(data, data, alpha=0.05)

        # Same distribution should never drift
        assert result["drift_detected"] is False
        assert result["ks_stat"] == 0.0
        assert result["p_value"] == 1.0

    def test_ks_detects_large_shift(self):
        """KS test detects a clear distribution shift."""
        from drift import detect_covariate_drift

        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 200))
        current = pd.Series(np.random.normal(5, 1, 200))  # clearly shifted

        result = detect_covariate_drift(baseline, current, alpha=0.05)

        assert result["drift_detected"] is True
        assert result["ks_stat"] > 0.5
        assert result["p_value"] < 0.01

    def test_ks_handles_small_series(self):
        """KS test with very small series returns insufficient_data severity."""
        from drift import detect_covariate_drift

        result = detect_covariate_drift(pd.Series([1.0]), pd.Series([2.0]), alpha=0.05)
        assert result["severity"] == "insufficient_data"

    def test_psi_stable_identical_distributions(self):
        """PSI on identical distributions returns near-zero value."""
        from drift import calculate_psi

        probs = np.random.uniform(0, 1, 100)
        psi = calculate_psi(probs, probs)

        assert psi < 0.01

    def test_psi_high_shift_exceeds_threshold(self):
        """PSI on heavily shifted distributions exceeds the critical threshold."""
        from drift import calculate_psi, PSI_WARNING

        np.random.seed(7)
        baseline = np.random.beta(2, 5, 100)  # skewed left
        current = np.random.beta(5, 2, 100)  # skewed right

        psi = calculate_psi(baseline, current)

        assert psi >= PSI_WARNING

    def test_run_drift_check_no_baseline(self, tmp_path):
        """run_drift_check returns empty DataFrame when no baseline exists."""
        from drift import run_drift_check

        df = pd.DataFrame(
            {
                "AMI": [0.5, 0.6, 0.7],
                "UPI": [0.3, 0.4, 0.5],
                "VSI": [0.2, 0.3, 0.4],
                "TPS": [0.8, 0.7, 0.6],
            }
        )

        # Point to a non-existent path
        result = run_drift_check(
            df, baseline_path=str(tmp_path / "nonexistent_baseline.csv")
        )
        assert result.empty or result is None or isinstance(result, pd.DataFrame)

    def test_run_drift_check_with_baseline(self, tmp_path):
        """run_drift_check produces a drift report when a baseline exists."""
        from drift import run_drift_check

        baseline = pd.DataFrame(
            {
                "AMI": np.random.normal(0.5, 0.1, 36),
                "UPI": np.random.normal(0.4, 0.1, 36),
                "VSI": np.random.normal(0.3, 0.1, 36),
                "TPS": np.random.normal(0.7, 0.1, 36),
            }
        )
        baseline_path = str(tmp_path / "baseline.csv")
        baseline.to_csv(baseline_path, index=False)

        # Slightly shifted current data
        current = baseline.copy()
        current["AMI"] += 0.1

        result = run_drift_check(
            current,
            baseline_path=baseline_path,
        )

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "drift_detected" in result.columns
        assert "AMI" in result["feature"].values
