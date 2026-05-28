"""
Project: Stock Price Predictor v2
Package: test
Filename: test_data_collection.py
Description: Automated unit test suite for the data collection and network resilience module. 
             Validates fault-tolerant exponential backoff retry routines, robust yfinance data ingestion, 
             MultiIndex schema flattening rules, and defensive parameter checking.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_collection import get_stock_data, get_stock_news, _retry_with_backoff
from tests.conftest import sample_stock_data


# ==========================================
# TEST SUITE: BACKOFF RETRY LOGIC
# ==========================================
class TestRetryWithBackoff:
    """
    Unit test coverage for the network layer's exponential backoff and fault-tolerance architecture.
    """

    def test_successful_first_attempt(self):
        """
        Verifies that the retry routine executes successfully and immediately returns 
        the outcome when no operational exceptions or transient drops are encountered.
        """
        result = _retry_with_backoff(lambda: 42, max_retries=3)
        assert result == 42

    def test_successful_after_retry(self):
        """
        Validates the state tracking wrapper by simulating a transient endpoint dropout 
        and verifying that successful recovery occurs on subsequent iteration frames.
        """
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = _retry_with_backoff(flaky_func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count[0] == 2

    def test_all_retries_fail(self):
        """
        Ensures that if connection exceptions persist continuously across all assigned allocation 
        cycles, the system bubbles up the final tracked runtime exception to the parent thread.
        """
        def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            _retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    def test_jitter_applied(self):
        """
        Verifies that the stochastic random jitter parameter is successfully ingested 
        and handled without causing structural calculation exceptions.
        """
        result = _retry_with_backoff(lambda: 1, jitter=True)
        assert result == 1


# ==========================================
# TEST SUITE: MARKET OHLCV DATA INGESTION
# ==========================================
class TestGetStockData:
    """
    Unit test coverage checking historical time-series extraction pipelines and data alignment.
    """

    def test_invalid_ticker_empty(self):
        """
        Ensures defensive boundary checking triggers a ValueError when an empty ticker string is supplied.
        """
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            get_stock_data("", "2024-01-01", "2024-01-31")

    def test_invalid_ticker_none(self):
        """
        Ensures defensive boundary checking triggers a ValueError when a None parameter is passed as a ticker symbol.
        """
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            get_stock_data(None, "2024-01-01", "2024-01-31")  # type: ignore

    def test_invalid_start_date(self):
        """
        Ensures defensive boundary checking triggers a ValueError when the timeline initiation constraint is empty.
        """
        with pytest.raises(ValueError, match="Start date must be a non-empty string"):
            get_stock_data("AAPL", "", "2024-01-31")

    def test_invalid_end_date(self):
        """
        Ensures defensive boundary checking triggers a ValueError when the timeline termination constraint is empty.
        """
        with pytest.raises(ValueError, match="End date must be a non-empty string"):
            get_stock_data("AAPL", "2024-01-01", "")

    @patch('src.data_collection.yf.download')
    def test_successful_fetch(self, mock_download: MagicMock, sample_stock_data: pd.DataFrame):
        """
        Validates clean execution threads under nominal operational circumstances, confirming 
        the returned DataFrame aligns with architectural expectation indices.
        """
        mock_download.return_value = sample_stock_data

        result = get_stock_data("AAPL", "2024-01-01", "2024-01-31")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert 'Close' in result.columns
        mock_download.assert_called_once()

    @patch('src.data_collection.yf.download')
    def test_empty_data_raises_error(self, mock_download: MagicMock):
        """
        Validates exception safety handling when API calls return structural blanks, 
        ensuring an empty layout properly throws a clear descriptive ValueError.
        """
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data returned"):
            get_stock_data("INVALID", "2024-01-01", "2024-01-31")

    @patch('src.data_collection.yf.download')
    def test_multiindex_columns_flattened(self, mock_download: MagicMock, sample_stock_data: pd.DataFrame):
        """
        Validates the critical flattening conversion routines designed to remove multi-index hierarchies, 
        safeguarding the feature engineering layers against axis dimension conflicts.
        """
        # Form an explicit multi-index structure simulating complex yfinance output tables
        multi_columns = pd.MultiIndex.from_tuples([
            ('Close', 'AAPL'), ('Open', 'AAPL'), ('High', 'AAPL')
        ])
        sample_stock_data = sample_stock_data.iloc[:, :3]
        sample_stock_data.columns = multi_columns
        mock_download.return_value = sample_stock_data

        result = get_stock_data("AAPL", "2024-01-01", "2024-01-31")

        assert not isinstance(result.columns, pd.MultiIndex)
        assert 'Close' in result.columns


# ==========================================
# TEST SUITE: MARKET TEXT DATA COLLECTION
# ==========================================
class TestGetStockNews:
    """
    Unit test coverage validating unstructured natural language textual stream downloads and mock payloads.
    """

    def test_invalid_ticker(self):
        """
        Ensures malformed or blank entry arguments return gracefully as an empty array 
        instead of terminating execution pipelines.
        """
        result = get_stock_news("")
        assert result == []

        result = get_stock_news(None)  # type: ignore
        assert result == []

    @patch('src.data_collection.yf.Ticker')
    def test_successful_fetch(self, mock_ticker: MagicMock, mock_news_data: list):
        """
        Validates parsing stability when processing structured json mock structures returned by financial services.
        """
        mock_stock = MagicMock()
        mock_stock.news = mock_news_data
        mock_ticker.return_value = mock_stock

        result = get_stock_news("AAPL")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]['title'] == 'Apple Reports Record Q4 Earnings'

    @patch('src.data_collection.yf.Ticker')
    def test_no_news(self, mock_ticker: MagicMock):
        """
        Ensures structural robustness during periods of market inactivity when media endpoints 
        return empty arrays, verifying that clean fallback lists are provided.
        """
        mock_stock = MagicMock()
        mock_stock.news = []
        mock_ticker.return_value = mock_stock

        result = get_stock_news("AAPL")

        assert result == []

    @patch('src.data_collection.yf.Ticker')
    def test_api_error_returns_empty_list(self, mock_ticker: MagicMock):
        """
        Verifies fault containment loops inside the collection thread, checking that unexpected internal 
        failures within external library structures degrade gracefully into neutral empty records.
        """
        mock_stock = MagicMock()
        mock_stock.news = None
        mock_ticker.return_value = mock_stock

        result = get_stock_news("AAPL")

        assert result == []