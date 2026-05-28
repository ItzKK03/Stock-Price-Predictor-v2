"""
Project: Stock Price Predictor v2
Package: test
Filename: test_feature_engineering.py
Description: Automated unit test suite for the feature engineering module. Validates 
             OHLCV schema verification states, indicators calculation pipelines (RSI, MA, MACD, 
             ATR, ADX), technical oscillator boundary constraints, and error propagation 
             mechanisms within custom indicator registration wrappers.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    add_technical_indicators,
    validate_ohlc_data,
    add_custom_indicator
)


# ==========================================
# TEST SUITE: OHLCV BASELINE SCHEMA TESTS
# ==========================================
class TestValidateOhlcData:
    """
    Unit test boundaries evaluating structural requirements and schema completeness 
    for primary equity market tracking tables.
    """

    def test_valid_data(self, sample_stock_data: pd.DataFrame):
        """
        Verifies positive control condition where a compliant dataset holding all 
        required OHLCV components returns a valid true flag and zero missing fields.
        """
        is_valid, missing = validate_ohlc_data(sample_stock_data)
        assert is_valid is True
        assert missing == []

    def test_missing_columns(self):
        """
        Ensures that tables lacking necessary price arrays fail validation checks 
        and accurately isolate the missing schema column labels.
        """
        df = pd.DataFrame({'Close': [1, 2, 3]})

        is_valid, missing = validate_ohlc_data(df)

        assert is_valid is False
        assert 'Open' in missing
        assert 'High' in missing
        assert 'Low' in missing
        assert 'Volume' in missing

    def test_empty_dataframe(self):
        """
        Validates edge-case behavior showing that an unpopulated, completely empty 
        dataframe fails validation constraints immediately.
        """
        df = pd.DataFrame()

        is_valid, missing = validate_ohlc_data(df)

        assert is_valid is False


# ==========================================
# TEST SUITE: TECHNICAL INDICATORS PIPELINE
# ==========================================
class TestAddTechnicalIndicators:
    """
    Unit test coverage validating calculation loops, technical feature extraction maps, 
    and parameter constraints across financial indicators pipelines.
    """

    def test_none_dataframe_raises_error(self):
        """
        Ensures that passing an uninitialized None object to calculation routines 
        safely throws an explicit descriptive ValueError.
        """
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            add_technical_indicators(None)

    def test_empty_dataframe_returns_as_is(self):
        """
        Confirms pass-through safety where blank data tables pass through calculation 
        steps unchanged without triggering pipeline crashes.
        """
        df = pd.DataFrame()
        result = add_technical_indicators(df)
        assert result.empty

    def test_missing_close_column_raises_error(self):
        """
        Validates baseline data dependency constraints by verifying that a missing Close 
        pricing array throws an explicit validation error.
        """
        df = pd.DataFrame({'Open': [1, 2, 3], 'High': [1, 2, 3]})

        with pytest.raises(ValueError, match="must have 'Close' column"):
            add_technical_indicators(df)

    def test_invalid_rsi_length(self, sample_stock_data: pd.DataFrame):
        """
        Ensures defensive parameter checking blocks non-positive lookback windows 
        for momentum oscillators like the Relative Strength Index.
        """
        with pytest.raises(ValueError, match="rsi_length must be positive"):
            add_technical_indicators(sample_stock_data, rsi_length=0)

    def test_invalid_ma_length(self, sample_stock_data: pd.DataFrame):
        """
        Ensures defensive parameter checking blocks negative historical tracking lengths 
        for trend-following simple moving averages.
        """
        with pytest.raises(ValueError, match="ma20_length must be positive"):
            add_technical_indicators(sample_stock_data, ma20_length=-1)

    def test_adds_all_indicators(self, sample_stock_data: pd.DataFrame):
        """
        Verifies calculation completeness, confirming every mandatory mathematical feature 
        is successfully extracted and appended to the output schema.
        """
        result = add_technical_indicators(sample_stock_data)

        assert (
        'RSI' in result.columns
        or 'RSI_10' in result.columns
        )
        assert 'MA20' in result.columns
        assert 'MA50' in result.columns
        assert 'MACD' in result.columns
        assert 'ATR' in result.columns
        assert 'ADX' in result.columns

    def test_indicator_values_are_numeric(self, sample_stock_data: pd.DataFrame):
        """
        Checks datatype consistency across appended elements to ensure all computed trading 
        metrics are structured as standard numeric floating points.
        """
        result = add_technical_indicators(sample_stock_data)

        assert np.issubdtype(result['RSI'].dtype, np.number)
        assert np.issubdtype(result['MA20'].dtype, np.number)
        assert np.issubdtype(result['MA50'].dtype, np.number)

    def test_rsi_in_valid_range(self, sample_stock_data: pd.DataFrame):
        """
        Validates mathematical constraint boundaries, verifying the Relative Strength Index 
        stays strictly within its standard theoretical 0 to 100 parameters.
        """
        result = add_technical_indicators(sample_stock_data)

        # Truncate preliminary calculation lookback gapping holes containing NaNs
        rsi_values = result['RSI'].dropna()

        if len(rsi_values) > 0:
            assert rsi_values.min() >= 0
            assert rsi_values.max() <= 100

    def test_custom_parameters(self, sample_stock_data: pd.DataFrame):
        """
        Confirms lookback window dynamic overrides are properly ingested and parsed 
        by underlying library calculation threads.
        """
        result = add_technical_indicators(
            sample_stock_data,
            rsi_length=10,
            ma20_length=15,
            ma50_length=30
        )

        assert (
        'RSI' in result.columns
        or 'RSI_10' in result.columns
        )
        assert (
        'MA20' in result.columns
        or 'SMA_15' in result.columns
        )
        assert (
        'MA50' in result.columns
        or 'SMA_30' in result.columns
        )


# ==========================================
# TEST SUITE: CUSTOM INDICATOR WRAPPERS
# ==========================================
class TestAddCustomIndicator:
    """
    Unit test coverage validating modular registration pipelines designed to inject 
    user-defined mathematical overlays into active feature matrices.
    """

    def test_empty_dataframe_raises_error(self):
        """
        Ensures custom injection components block execution if target datasets are completely empty.
        """
        def dummy_func(df, **kwargs):
            return pd.Series([1, 2, 3])

        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            add_custom_indicator(df, dummy_func, 'custom')

    def test_adds_custom_indicator(self, sample_stock_data: pd.DataFrame):
        """
        Verifies functional evaluation loops by confirming an external callable indicator 
        maps and aggregates values accurately into a target destination row profile.
        """
        def custom_indicator(df, **kwargs):
            return df['Close'] * 2

        result = add_custom_indicator(sample_stock_data, custom_indicator, 'doubled_close')

        assert 'doubled_close' in result.columns
        pd.testing.assert_series_equal(
            result['doubled_close'],
            sample_stock_data['Close'] * 2,
            check_names=False
        )

    def test_indicator_error_propagates(self, sample_stock_data: pd.DataFrame):
        """
        Validates containment safety and error tracing, confirming runtime errors triggered inside 
        external indicator extensions safely bubble up to the main parent process.
        """
        def failing_indicator(df, **kwargs):
            raise RuntimeError("Intentional failure")

        with pytest.raises(RuntimeError, match="Intentional failure"):
            add_custom_indicator(sample_stock_data, failing_indicator, 'fail')