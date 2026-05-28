"""
Project: Stock Price Predictor v2
Package: test
Filename: conftest.py
Description: Centralized Pytest configuration and shared fixtures pipeline. Generates 
             deterministic synthetic data arrays, mock multi-source API payloads, 
             3D tensor sequence spaces, and ephemeral filesystem paths to validate 
             the integrity of model pipelines, indicators, and evaluation metrics.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Any


# ==========================================
# TIME-SERIES OHLCV SIMULATION FIXTURES
# ==========================================
@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """
    Generates a deterministic 100-day historical stock price panel matrix with standard 
    OHLCV parameters to simulate live market data ingestion.

    Returns:
        pd.DataFrame: Tabular data frame containing indexed random-walk asset trajectories.
    """
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Enforce deterministic tracking behaviors by hardcoding pseudo-random number generation seeds
    np.random.seed(42)

    # Initialize baseline financial conditions and apply a 2% daily variance scaling parameter
    base_price = 150.0
    returns = np.random.randn(100) * 0.02  
    prices = base_price * np.cumprod(1 + returns)

    # Compile tracking frames by mapping stochastic spreads around computed close prices
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(100) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    df.index.name = 'Date'
    return df


@pytest.fixture
def sample_data_with_indicators(sample_stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms baseline OHLCV price tables by appending synthetic technical metrics, 
    simulating feature extraction pipelines.

    Args:
        sample_stock_data (pd.DataFrame): Core actual market tracking matrix fixture.

    Returns:
        pd.DataFrame: Augmented matrix holding technical momentum and trend indicators.
    """
    df = sample_stock_data.copy()

    # Generate a bounded momentum signal (0-100) to replicate standard RSI profiles
    df['RSI'] = 50 + np.random.randn(100) * 10
    df['RSI'] = df['RSI'].clip(0, 100)

    # Process rolling baseline constraints across historical price windows
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    return df


# ==========================================
# TEXT SENTIMENT & API MOCK FIXTURES
# ==========================================
@pytest.fixture
def sample_sentiment_data() -> pd.DataFrame:
    """
    Assembles a isolated daily timeline tracking text evaluation confidence indices.

    Returns:
        pd.DataFrame: Chronological panel table containing continuous numerical sentiment signals.
    """
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    sentiment = np.random.randn(50)

    df = pd.DataFrame({'Sentiment': sentiment}, index=dates)
    df.index.name = 'Date'
    return df


@pytest.fixture
def mock_news_data() -> list:
    """
    Compiles structured JSON-like mock dictionaries mimicking raw unstructured response payloads 
    returned by remote yfinance server endpoints.

    Returns:
        list: Collection of nested dictionary records containing headline strings and publication metadata.
    """
    return [
        {
            'title': 'Apple Reports Record Q4 Earnings',
            'publisher': 'Financial Times',
            'link': 'https://example.com/news1',
            'providerPublishTime': 1704067200,
            'type': 'STORY',
            'thumbnail': {'resolutions': []},
            'relatedTickers': ['AAPL']
        },
        {
            'title': 'Tech Stocks Rally on Positive Outlook',
            'publisher': 'Reuters',
            'link': 'https://example.com/news2',
            'providerPublishTime': 1704153600,
            'type': 'STORY',
            'thumbnail': {'resolutions': []},
            'relatedTickers': ['AAPL', 'GOOGL']
        },
        {
            'title': 'Market Analysis: AAPL Price Target Raised',
            'publisher': 'Bloomberg',
            'link': 'https://example.com/news3',
            'providerPublishTime': 1704240000,
            'type': 'STORY',
            'thumbnail': {'resolutions': []},
            'relatedTickers': ['AAPL']
        }
    ]


# ==========================================
# TENSOR VECTOR SHAPING & KPI FIXTURES
# ==========================================
@pytest.fixture
def sample_scaled_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes normalized 2D numerical matrix pairs to check pipeline scale limits.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Isolated input feature arrays and target label metrics.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, 1)

    return X, y


@pytest.fixture
def sample_time_series_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs 3D array blocks configured to replicate sequential LSTM tensor format maps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Input tensor shaped as [Samples, Sequence Time Steps, Features].
    """
    np.random.seed(42)
    samples = 50
    time_steps = 10
    features = 5

    X = np.random.rand(samples, time_steps, features)
    y = np.random.rand(samples, 1)

    return X, y


@pytest.fixture
def sample_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates co-dependent pricing random walks representing actual actual values and simulated 
    model outputs to validate regression scores.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Aligned vectors containing ground truths and predictions.
    """
    np.random.seed(42)
    n = 100

    y_true = 100 + np.cumsum(np.random.randn(n))
    y_pred = y_true + np.random.randn(n) * 5

    return y_true, y_pred


# ==========================================
# INDUSTRIAL FILESYSTEM WORKSPACE FIXTURES
# ==========================================
@pytest.fixture
def temp_model_path(tmp_path: Any) -> str:
    """
    Allocates an isolated, sandboxed file tracking reference location path to check deep learning 
    serialization and persistence behaviors.

    Args:
        tmp_path (Any): Native platform-agnostic Pytest directory environment context provider.

    Returns:
        str: Absolute system filepath string targeting a mock .keras file artifact.
    """
    return str(tmp_path / "test_model.keras")


@pytest.fixture
def temp_scaler_path(tmp_path: Any) -> str:
    """
    Allocates an isolated, sandboxed file tracking reference location path to validate normalization 
    scaler configuration saves.

    Args:
        tmp_path (Any): Native platform-agnostic Pytest directory environment context provider.

    Returns:
        str: Absolute system filepath string targeting a mock .joblib file artifact.
    """
    return str(tmp_path / "test_scaler.joblib")