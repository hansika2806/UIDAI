"""
Data Validation for UIDAI Governance Framework.

Uses Great Expectations for data quality validation and monitoring.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

from logger import get_logger

logger = get_logger(__name__)


class UidaiDataValidator:
    """
    Data validator for UIDAI datasets using Great Expectations.
    
    Validates:
    - Schema compliance (column names, types)
    - Data quality (nulls, ranges, distributions)
    - Business rules (state names, date ranges)
    - Referential integrity
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_results: Dict[str, Dict] = {}
    
    def validate_enrolment_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate enrolment dataset.
        
        Expected schema:
        - registrar: string (state name)
        - date: datetime
        - aadhaar_generated: int >= 0
        - rejected: int >= 0
        - resident_type: string (optional)
        """
        logger.info("Validating enrolment data")
        
        ge_df = PandasDataset(df)
        results = {}
        
        # Schema validation
        results['has_registrar'] = ge_df.expect_column_to_exist('registrar')
        results['has_date'] = ge_df.expect_column_to_exist('date')
        results['has_aadhaar_generated'] = ge_df.expect_column_to_exist('aadhaar_generated')
        
        # Data type validation
        if 'date' in df.columns:
            results['date_is_datetime'] = ge_df.expect_column_values_to_be_of_type(
                'date', 'datetime64'
            )
        
        # Value range validation
        if 'aadhaar_generated' in df.columns:
            results['aadhaar_generated_non_negative'] = ge_df.expect_column_values_to_be_between(
                'aadhaar_generated', min_value=0, max_value=None
            )
        
        if 'rejected' in df.columns:
            results['rejected_non_negative'] = ge_df.expect_column_values_to_be_between(
                'rejected', min_value=0, max_value=None
            )
        
        # Null validation
        results['registrar_not_null'] = ge_df.expect_column_values_to_not_be_null('registrar')
        results['date_not_null'] = ge_df.expect_column_values_to_not_be_null('date')
        
        # Business rule: Date should be within reasonable range
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            min_date = pd.Timestamp('2010-01-01')
            max_date = pd.Timestamp.now()
            results['date_in_range'] = ge_df.expect_column_values_to_be_between(
                'date', min_value=min_date, max_value=max_date
            )
        
        self.validation_results['enrolment'] = results
        self._log_validation_summary('enrolment', results)
        
        return results
    
    def validate_update_data(self, df: pd.DataFrame, update_type: str) -> Dict:
        """
        Validate demographic or biometric update dataset.
        
        Args:
            df: DataFrame to validate
            update_type: 'demographic' or 'biometric'
        
        Expected schema:
        - registrar: string (state name)
        - date: datetime
        - total_updates: int >= 0
        - age_group: string (optional)
        """
        logger.info(f"Validating {update_type} update data")
        
        ge_df = PandasDataset(df)
        results = {}
        
        # Schema validation
        results['has_registrar'] = ge_df.expect_column_to_exist('registrar')
        results['has_date'] = ge_df.expect_column_to_exist('date')
        
        # Check for update count columns
        update_columns = [col for col in df.columns if 'update' in col.lower()]
        results['has_update_columns'] = len(update_columns) > 0
        
        # Value range validation for numeric columns
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if col != 'date':
                results[f'{col}_non_negative'] = ge_df.expect_column_values_to_be_between(
                    col, min_value=0, max_value=None
                )
        
        # Null validation
        results['registrar_not_null'] = ge_df.expect_column_values_to_not_be_null('registrar')
        results['date_not_null'] = ge_df.expect_column_values_to_not_be_null('date')
        
        # Uniqueness check for state-date combinations
        if 'registrar' in df.columns and 'date' in df.columns:
            duplicate_count = df.duplicated(subset=['registrar', 'date']).sum()
            results['no_duplicate_state_dates'] = duplicate_count == 0
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate state-date combinations")
        
        self.validation_results[update_type] = results
        self._log_validation_summary(update_type, results)
        
        return results
    
    def validate_state_master(self, df: pd.DataFrame) -> Dict:
        """
        Validate state master dataset (aggregated indicators).
        
        Expected schema:
        - state: string
        - AMI, UPI, VSI, TPS: float [0, 1]
        - Priority_Score: float
        - Governance_Profile: string
        """
        logger.info("Validating state master data")
        
        ge_df = PandasDataset(df)
        results = {}
        
        # Schema validation
        results['has_state'] = ge_df.expect_column_to_exist('state')
        
        # Indicator validation (should be between 0 and 1)
        for indicator in ['AMI', 'UPI', 'VSI', 'TPS']:
            if indicator in df.columns:
                results[f'{indicator}_in_range'] = ge_df.expect_column_values_to_be_between(
                    indicator, min_value=0.0, max_value=1.0
                )
        
        # State uniqueness
        results['states_unique'] = ge_df.expect_column_values_to_be_unique('state')
        
        # No nulls in critical columns
        critical_cols = ['state', 'AMI', 'UPI', 'VSI', 'TPS']
        for col in critical_cols:
            if col in df.columns:
                results[f'{col}_not_null'] = ge_df.expect_column_values_to_not_be_null(col)
        
        self.validation_results['state_master'] = results
        self._log_validation_summary('state_master', results)
        
        return results
    
    def validate_time_series(self, df: pd.DataFrame, date_col: str = 'date') -> Dict:
        """
        Validate time series properties.
        
        Args:
            df: DataFrame with time series data
            date_col: Name of the date column
        """
        logger.info("Validating time series properties")
        
        results = {}
        
        if date_col not in df.columns:
            results['error'] = f"Date column '{date_col}' not found"
            return results
        
        # Check if dates are sorted
        is_sorted = df[date_col].is_monotonic_increasing
        results['dates_sorted'] = is_sorted
        
        # Check for gaps in time series
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            date_diffs = df[date_col].diff().dropna()
            if len(date_diffs) > 0:
                median_diff = date_diffs.median()
                max_diff = date_diffs.max()
                results['median_gap_days'] = median_diff.days
                results['max_gap_days'] = max_diff.days
                results['has_large_gaps'] = max_diff > median_diff * 3
        
        # Check for duplicate dates
        duplicate_dates = df[date_col].duplicated().sum()
        results['duplicate_dates'] = duplicate_dates
        results['no_duplicate_dates'] = duplicate_dates == 0
        
        self.validation_results['time_series'] = results
        self._log_validation_summary('time_series', results)
        
        return results
    
    def _log_validation_summary(self, dataset_name: str, results: Dict) -> None:
        """Log validation summary."""
        passed = sum(1 for v in results.values() if isinstance(v, dict) and v.get('success', False))
        total = len([v for v in results.values() if isinstance(v, dict)])
        
        logger.info(
            f"Validation summary for {dataset_name}",
            extra={
                "dataset": dataset_name,
                "passed": passed,
                "total": total,
                "success_rate": passed / total if total > 0 else 0
            }
        )
    
    def get_validation_report(self) -> pd.DataFrame:
        """
        Generate a validation report DataFrame.
        
        Returns:
            DataFrame with validation results
        """
        report_data = []
        
        for dataset, results in self.validation_results.items():
            for check, result in results.items():
                if isinstance(result, dict):
                    report_data.append({
                        'dataset': dataset,
                        'check': check,
                        'success': result.get('success', False),
                        'details': str(result)
                    })
                else:
                    report_data.append({
                        'dataset': dataset,
                        'check': check,
                        'success': result,
                        'details': ''
                    })
        
        return pd.DataFrame(report_data)
    
    def save_validation_report(self, output_path: Path) -> None:
        """Save validation report to CSV."""
        report = self.get_validation_report()
        report.to_csv(output_path, index=False)
        logger.info(f"Validation report saved to {output_path}")


def validate_pipeline_data(
    enrolment_df: Optional[pd.DataFrame] = None,
    demographic_df: Optional[pd.DataFrame] = None,
    biometric_df: Optional[pd.DataFrame] = None,
    state_master_df: Optional[pd.DataFrame] = None
) -> UidaiDataValidator:
    """
    Convenience function to validate all pipeline data.
    
    Args:
        enrolment_df: Enrolment dataset
        demographic_df: Demographic update dataset
        biometric_df: Biometric update dataset
        state_master_df: State master dataset
    
    Returns:
        Validator instance with results
    """
    validator = UidaiDataValidator()
    
    if enrolment_df is not None:
        validator.validate_enrolment_data(enrolment_df)
    
    if demographic_df is not None:
        validator.validate_update_data(demographic_df, 'demographic')
    
    if biometric_df is not None:
        validator.validate_update_data(biometric_df, 'biometric')
    
    if state_master_df is not None:
        validator.validate_state_master(state_master_df)
    
    return validator

# Made with Bob
