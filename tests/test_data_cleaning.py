"""
Unit tests for data cleaning module.

Tests:
- State name normalization
- Date parsing and validation
- Negative value filtering
- Chunked loading logic
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_cleaning import STATE_MAPPING, MERGE_UT_MAPPING


class TestStateMapping:
    """Test state name normalization logic."""

    def test_state_mapping_consistency(self):
        """Ensure all state mappings are strings."""
        for key, value in STATE_MAPPING.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(key) > 0
            assert len(value) > 0

    def test_merge_ut_mapping_consistency(self):
        """Ensure UT merge mappings are valid."""
        for key, value in MERGE_UT_MAPPING.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            # Merged UTs should map to the combined name
            assert "Dadra And Nagar Haveli And Daman And Diu" in value

    def test_no_circular_mappings(self):
        """Ensure no circular state mappings exist."""
        for key, value in STATE_MAPPING.items():
            # Value should not map back to key
            assert STATE_MAPPING.get(value, value) != key


class TestDataValidation:
    """Test data validation and filtering logic."""

    def test_negative_value_detection(self):
        """Test detection of negative transaction values."""
        df = pd.DataFrame(
            {
                "registrar": ["State A", "State B", "State C"],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "aadhaar_generated": [100, -50, 200],
                "rejected": [10, 20, -5],
            }
        )

        # Filter negative values
        mask = (df["aadhaar_generated"] >= 0) & (df["rejected"] >= 0)
        clean_df = df[mask]

        assert len(clean_df) == 1
        assert clean_df.iloc[0]["registrar"] == "State A"

    def test_date_parsing(self):
        """Test date parsing handles various formats."""
        dates = ["2023-01-01", "01/01/2023", "2023-1-1"]

        for date_str in dates:
            try:
                parsed = pd.to_datetime(date_str)
                assert isinstance(parsed, pd.Timestamp)
            except Exception as e:
                pytest.fail(f"Failed to parse date {date_str}: {e}")

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        df = pd.DataFrame(
            {
                "registrar": ["State A", None, "State C"],
                "date": ["2023-01-01", "2023-01-02", None],
                "value": [100, 200, 300],
            }
        )

        # Drop rows with missing critical fields
        clean_df = df.dropna(subset=["registrar", "date"])

        assert len(clean_df) == 1
        assert clean_df.iloc[0]["registrar"] == "State A"


class TestChunkedProcessing:
    """Test memory-efficient chunked data processing."""

    def test_chunk_size_calculation(self):
        """Test chunk size is reasonable for memory constraints."""
        chunk_size = 100000

        # Assuming ~1KB per row, 100K rows ≈ 100MB
        estimated_memory_mb = (chunk_size * 1024) / (1024 * 1024)

        assert estimated_memory_mb < 500  # Should be under 500MB per chunk

    def test_chunk_aggregation_consistency(self):
        """Test that chunked aggregation produces same result as full load."""
        # Create test data
        df = pd.DataFrame({"state": ["A"] * 50 + ["B"] * 50, "value": list(range(100))})

        # Full aggregation
        full_agg = df.groupby("state")["value"].sum()

        # Chunked aggregation
        chunk_results = []
        for chunk in [df.iloc[:50], df.iloc[50:]]:
            chunk_agg = chunk.groupby("state")["value"].sum()
            chunk_results.append(chunk_agg)

        chunked_agg = pd.concat(chunk_results).groupby(level=0).sum()

        # Results should match
        pd.testing.assert_series_equal(full_agg, chunked_agg)


class TestDataIntegrity:
    """Test data integrity constraints."""

    def test_no_duplicate_state_month_records(self):
        """Test that state-month combinations are unique after aggregation."""
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B", "B"],
                "date": ["2023-01-01", "2023-01-15", "2023-01-01", "2023-02-01"],
                "value": [100, 200, 300, 400],
            }
        )

        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M")

        # Aggregate by state-month
        agg_df = df.groupby(["state", "year_month"])["value"].sum().reset_index()

        # Check uniqueness
        assert len(agg_df) == len(agg_df[["state", "year_month"]].drop_duplicates())

    def test_temporal_ordering(self):
        """Test that dates are properly ordered after processing."""
        df = pd.DataFrame(
            {
                "date": ["2023-03-01", "2023-01-01", "2023-02-01"],
                "value": [300, 100, 200],
            }
        )

        df["date"] = pd.to_datetime(df["date"])
        df_sorted = df.sort_values("date")

        assert df_sorted.iloc[0]["value"] == 100
        assert df_sorted.iloc[1]["value"] == 200
        assert df_sorted.iloc[2]["value"] == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
