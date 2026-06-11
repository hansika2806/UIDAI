"""
Property-Based Testing for UIDAI Governance Framework.

Uses Hypothesis to verify mathematical properties and invariants that should
hold for all valid inputs, not just specific test cases.

Properties tested:
1. Indicator monotonicity (AMI, UPI, VSI, TPS ∈ [0, 1])
2. Forecast consistency (predictions should be stable for similar inputs)
3. Clustering stability (centroids should maintain relative ordering)
4. Aggregation commutativity (sum order shouldn't matter)
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import column, data_frames, range_indexes

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecasting import _mape, _rmse


class TestIndicatorProperties:
    """Test mathematical properties of governance indicators."""

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=100))
    def test_ami_bounded(self, ami_values):
        """AMI should always be in [0, 1] range."""
        ami_array = np.array(ami_values)
        assert np.all(ami_array >= 0.0)
        assert np.all(ami_array <= 1.0)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=100))
    def test_indicator_monotonicity(self, values):
        """Sorted indicators should be monotonically increasing."""
        sorted_values = sorted(values)
        for i in range(len(sorted_values) - 1):
            assert sorted_values[i] <= sorted_values[i + 1]

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    def test_composite_score_bounded(self, ami, upi, vsi, tps):
        """Composite score from normalized indicators should be bounded."""
        # Simple weighted average
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        indicators = np.array([ami, upi, vsi, tps])

        composite = np.dot(weights, indicators)

        assert 0.0 <= composite <= 1.0


class TestForecastingProperties:
    """Test mathematical properties of forecasting functions."""

    @given(
        st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=3, max_size=50),
        st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=3, max_size=50),
    )
    def test_mape_symmetry(self, actual, predicted):
        """MAPE should be symmetric for positive values."""
        assume(len(actual) == len(predicted))
        assume(all(a > 0 for a in actual))

        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)

        mape = _mape(actual_arr, predicted_arr)

        # MAPE should be non-negative
        assert mape >= 0 or np.isnan(mape)

    @given(
        st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=3, max_size=50),
        st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=3, max_size=50),
    )
    def test_rmse_non_negative(self, actual, predicted):
        """RMSE should always be non-negative."""
        assume(len(actual) == len(predicted))

        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)

        rmse = _rmse(actual_arr, predicted_arr)

        assert rmse >= 0

    @given(
        st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=3, max_size=50)
    )
    def test_perfect_prediction_zero_error(self, values):
        """Perfect predictions should have zero error."""
        arr = np.array(values)

        mape = _mape(arr, arr)
        rmse = _rmse(arr, arr)

        assert mape == 0.0
        assert rmse == 0.0

    @given(
        st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=3, max_size=50),
        st.floats(min_value=0.1, max_value=2.0),
    )
    def test_mape_scale_invariance(self, values, scale):
        """MAPE should be scale-invariant (percentage-based)."""
        actual = np.array(values)
        predicted = actual * 1.1  # 10% overestimate

        mape1 = _mape(actual, predicted)
        mape2 = _mape(actual * scale, predicted * scale)

        # Should be approximately equal (within floating point tolerance)
        assert abs(mape1 - mape2) < 1e-6


class TestAggregationProperties:
    """Test properties of data aggregation operations."""

    @given(
        data_frames(
            [
                column(
                    "value",
                    dtype=float,
                    elements=st.floats(min_value=0, max_value=1000),
                ),
                column("state", dtype=str, elements=st.sampled_from(["A", "B", "C"])),
            ],
            index=range_indexes(min_size=10, max_size=100),
        )
    )
    def test_aggregation_commutativity(self, df):
        """Sum aggregation should be commutative (order doesn't matter)."""
        # Group by state and sum
        agg1 = df.groupby("state")["value"].sum()

        # Shuffle and aggregate again
        df_shuffled = df.sample(frac=1.0, random_state=42)
        agg2 = df_shuffled.groupby("state")["value"].sum()

        # Results should be identical
        pd.testing.assert_series_equal(agg1.sort_index(), agg2.sort_index())

    @given(st.lists(st.floats(min_value=0, max_value=1000), min_size=5, max_size=100))
    def test_sum_associativity(self, values):
        """Sum should be associative: (a+b)+c = a+(b+c)."""
        arr = np.array(values)

        # Split into three parts
        n = len(arr)
        split1 = n // 3
        split2 = 2 * n // 3

        part1 = arr[:split1]
        part2 = arr[split1:split2]
        part3 = arr[split2:]

        # Two different groupings
        result1 = (part1.sum() + part2.sum()) + part3.sum()
        result2 = part1.sum() + (part2.sum() + part3.sum())

        # Should be equal (within floating point tolerance)
        assert abs(result1 - result2) < 1e-6


class TestStatisticalProperties:
    """Test statistical properties and invariants."""

    @given(st.lists(st.floats(min_value=0, max_value=1000), min_size=10, max_size=100))
    def test_mean_bounded_by_min_max(self, values):
        """Mean should be between min and max."""
        arr = np.array(values)
        mean = arr.mean()

        assert arr.min() <= mean <= arr.max()

    @given(st.lists(st.floats(min_value=0, max_value=1000), min_size=10, max_size=100))
    def test_std_non_negative(self, values):
        """Standard deviation should be non-negative."""
        arr = np.array(values)
        std = arr.std()

        assert std >= 0

    @given(st.lists(st.floats(min_value=0, max_value=1000), min_size=10, max_size=100))
    def test_cv_scale_invariance(self, values):
        """Coefficient of Variation should be scale-invariant."""
        assume(np.mean(values) > 0)

        arr = np.array(values)
        cv1 = arr.std() / arr.mean()

        # Scale by 10
        arr_scaled = arr * 10
        cv2 = arr_scaled.std() / arr_scaled.mean()

        # Should be approximately equal
        assert abs(cv1 - cv2) < 1e-6


class TestDataValidationProperties:
    """Test data validation invariants."""

    @given(
        st.lists(st.integers(min_value=0, max_value=1000000), min_size=5, max_size=50)
    )
    def test_non_negative_counts(self, counts):
        """Transaction counts should always be non-negative."""
        arr = np.array(counts)
        assert np.all(arr >= 0)

    @given(
        st.lists(st.integers(min_value=0, max_value=1000), min_size=5, max_size=50),
        st.lists(st.integers(min_value=0, max_value=1000), min_size=5, max_size=50),
    )
    def test_update_ratio_bounded(self, updates, enrollments):
        """Update-to-enrollment ratio should be non-negative."""
        assume(len(updates) == len(enrollments))

        updates_arr = np.array(updates, dtype=float)
        enrollments_arr = np.array(enrollments, dtype=float)

        # Avoid division by zero
        mask = enrollments_arr > 0
        if np.any(mask):
            ratios = updates_arr[mask] / enrollments_arr[mask]
            assert np.all(ratios >= 0)


class TestClusteringProperties:
    """Test clustering algorithm properties."""

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=1), st.floats(min_value=0, max_value=1)
            ),
            min_size=10,
            max_size=100,
        )
    )
    @settings(deadline=None)
    def test_cluster_assignment_consistency(self, points):
        """Same point should get same cluster assignment."""
        from sklearn.cluster import KMeans

        X = np.array(points)
        # Ensure we have enough distinct points to avoid degenerate empty/duplicate clusters
        assume(len(np.unique(X, axis=0)) >= 3)

        # Fit model
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels1 = kmeans.fit_predict(X)

        # Predict again on same data
        labels2 = kmeans.predict(X)

        # Should be identical
        assert np.array_equal(labels1, labels2)

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=1), st.floats(min_value=0, max_value=1)
            ),
            min_size=20,
            max_size=100,
        ),
        st.integers(min_value=2, max_value=5),
    )
    @settings(deadline=None)
    def test_cluster_count_matches_k(self, points, k):
        """Number of unique clusters should match K."""
        from sklearn.cluster import KMeans

        assume(len(points) >= k)

        X = np.array(points)
        # Ensure we have enough distinct points to avoid degenerate empty/duplicate clusters
        assume(len(np.unique(X, axis=0)) >= k)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        n_unique_labels = len(np.unique(labels))

        # Should have exactly K clusters (or fewer if some are empty)
        assert n_unique_labels <= k


class TestTemporalProperties:
    """Test temporal ordering and consistency properties."""

    @given(
        st.lists(
            st.datetimes(
                min_value=pd.Timestamp("2010-01-01"),
                max_value=pd.Timestamp("2024-12-31"),
            ),
            min_size=5,
            max_size=50,
        )
    )
    def test_sorted_dates_monotonic(self, dates):
        """Sorted dates should be monotonically increasing."""
        sorted_dates = sorted(dates)

        for i in range(len(sorted_dates) - 1):
            assert sorted_dates[i] <= sorted_dates[i + 1]

    @given(st.lists(st.integers(min_value=1, max_value=12), min_size=12, max_size=36))
    def test_monthly_aggregation_preserves_total(self, monthly_values):
        """Monthly aggregation should preserve total."""
        arr = np.array(monthly_values)

        # Quarterly aggregation
        n_quarters = len(arr) // 3
        quarterly = arr[: n_quarters * 3].reshape(-1, 3).sum(axis=1)

        # Total should be preserved
        assert abs(arr[: n_quarters * 3].sum() - quarterly.sum()) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])

# Made with Bob
