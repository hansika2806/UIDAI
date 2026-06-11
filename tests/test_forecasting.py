"""
Unit tests for forecasting module.

Tests critical edge cases:
- Division by zero in MAPE calculation
- Empty/insufficient data handling
- SARIMA fallback behavior
- Walk-forward validation logic
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from forecasting import _mape, _rmse, walk_forward_backtest


class TestMAPE:
    """Test Mean Absolute Percentage Error calculation."""
    
    def test_mape_normal_case(self):
        """Test MAPE with normal positive values."""
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        
        mape = _mape(actual, predicted)
        
        # Expected: mean of [10%, 5%, 3.33%] ≈ 6.11%
        assert 5.0 < mape < 7.0
    
    def test_mape_with_zeros(self):
        """Test MAPE when actual contains zeros (should return nan)."""
        actual = np.array([0, 100, 200])
        predicted = np.array([10, 110, 190])
        
        mape = _mape(actual, predicted)
        
        # Should skip zeros and compute on non-zero values only
        assert not np.isnan(mape)
    
    def test_mape_all_zeros(self):
        """Test MAPE when all actuals are zero (should return nan)."""
        actual = np.array([0, 0, 0])
        predicted = np.array([10, 20, 30])
        
        mape = _mape(actual, predicted)
        
        assert np.isnan(mape)
    
    def test_mape_perfect_prediction(self):
        """Test MAPE with perfect predictions."""
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        
        mape = _mape(actual, predicted)
        
        assert mape == 0.0


class TestRMSE:
    """Test Root Mean Squared Error calculation."""
    
    def test_rmse_normal_case(self):
        """Test RMSE with normal values."""
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        
        rmse = _rmse(actual, predicted)
        
        # Expected: sqrt(mean([100, 100, 100])) = 10
        assert 9.0 < rmse < 11.0
    
    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        
        rmse = _rmse(actual, predicted)
        
        assert rmse == 0.0
    
    def test_rmse_with_negatives(self):
        """Test RMSE handles negative values correctly."""
        actual = np.array([-100, -200, -300])
        predicted = np.array([-110, -190, -310])
        
        rmse = _rmse(actual, predicted)
        
        assert 9.0 < rmse < 11.0


class TestWalkForwardBacktest:
    """Test walk-forward validation logic."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data for train/test split."""
        series = np.array([1, 2, 3])  # Only 3 points (below the absolute minimum of 5 for scaling)
        
        result = walk_forward_backtest(series, test_size=3, min_train=8)
        
        # Should return nan metrics when data is insufficient
        assert np.isnan(result['mape'])
        assert np.isnan(result['rmse'])
        assert result['n_train'] == 0
        assert result['n_test'] == 0
    
    @patch('forecasting.forecast_state')
    def test_forecast_failure_handling(self, mock_forecast):
        """Test handling when forecast_state returns None."""
        mock_forecast.return_value = None
        
        series = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        
        result = walk_forward_backtest(series, test_size=3, min_train=8)
        
        # Should return nan metrics but valid train/test counts
        assert np.isnan(result['mape'])
        assert np.isnan(result['rmse'])
        assert result['n_train'] == 9
        assert result['n_test'] == 3
    
    @patch('forecasting.forecast_state')
    def test_successful_backtest(self, mock_forecast):
        """Test successful walk-forward validation."""
        # Mock successful forecast
        mock_forecast.return_value = {
            'forecast_mean': np.array([105, 110, 115])
        }
        
        series = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        
        result = walk_forward_backtest(series, test_size=3, min_train=8)
        
        # Should return valid metrics
        assert not np.isnan(result['mape'])
        assert not np.isnan(result['rmse'])
        assert result['n_train'] == 9
        assert result['n_test'] == 3
        assert len(result['actual']) == 3
        assert len(result['predicted']) == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_value_series(self):
        """Test handling of single-value time series."""
        series = np.array([100])
        
        result = walk_forward_backtest(series, test_size=1, min_train=1)
        
        # Should handle gracefully
        assert isinstance(result, dict)
    
    def test_constant_series(self):
        """Test handling of constant time series."""
        actual = np.array([100, 100, 100])
        predicted = np.array([100, 100, 100])
        
        mape = _mape(actual, predicted)
        rmse = _rmse(actual, predicted)
        
        assert mape == 0.0
        assert rmse == 0.0
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        actual = np.array([1e10, 1e11, 1e12])
        predicted = np.array([1.1e10, 0.9e11, 1.05e12])
        
        mape = _mape(actual, predicted)
        rmse = _rmse(actual, predicted)
        
        # Should not overflow or return inf
        assert not np.isinf(mape)
        assert not np.isinf(rmse)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Made with Bob
