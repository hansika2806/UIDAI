"""
Pytest configuration and shared fixtures for UIDAI test suite.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_time_series():
    """Generate a sample time series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=36, freq="M")
    values = 1000 + np.cumsum(np.random.randn(36) * 50)
    return pd.DataFrame({"date": dates, "value": values})


@pytest.fixture
def sample_state_data():
    """Generate sample state-level data."""
    states = ["State A", "State B", "State C"]
    dates = pd.date_range(start="2020-01-01", periods=12, freq="M")

    data = []
    for state in states:
        for date in dates:
            data.append(
                {
                    "state": state,
                    "date": date,
                    "enrolment": np.random.randint(1000, 5000),
                    "demographic_update": np.random.randint(500, 3000),
                    "biometric_update": np.random.randint(300, 2000),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_indicators():
    """Generate sample governance indicators."""
    return pd.DataFrame(
        {
            "state": ["State A", "State B", "State C"],
            "AMI": [0.8, 0.6, 0.4],
            "UPI": [0.7, 0.5, 0.3],
            "VSI": [0.2, 0.4, 0.6],
            "TPS": [0.9, 0.7, 0.5],
        }
    )


@pytest.fixture
def mock_cleaned_data(tmp_path):
    """Create mock cleaned data files in a temporary directory."""
    data_dir = tmp_path / "data" / "cleaned"
    data_dir.mkdir(parents=True)

    # Create mock enrolment data
    enrolment_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "registrar": ["State A"] * 100,
            "aadhaar_generated": np.random.randint(100, 1000, 100),
        }
    )
    enrolment_df.to_csv(data_dir / "aadhaar_enrolment_clean_final.csv", index=False)

    return data_dir


# Made with Bob
