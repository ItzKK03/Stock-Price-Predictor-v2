"""
Project: Stock Price Predictor v2
Package: src
Filename: feature_engineering.py
Description: Advanced quantitative feature engineering module. Calculates mathematical 
             technical indicators, momentum oscillators, volatility bands, and volume overlays 
             using the pandas-ta architecture to expand the input dimension space for sequential models.
Author: Kartik Kant (AI/ML Engineer)
"""

import pandas as pd
import pandas_ta as ta
from typing import Optional
from src.logger import get_logger

# Initialize centralized logging context for feature calculation telemetry
logger = get_logger(__name__)


# ==========================================
# DEFULATORY OHLCV SCHEMA VALIDATION
# ==========================================
def validate_ohlc_data(df: pd.DataFrame) -> tuple:
    """
    Validates that the source DataFrame contains the standard price parameters required 
    for calculating comprehensive quantitative trading indicators.

    Args:
        df (pd.DataFrame): Tabular financial asset market dataset.

    Returns:
        tuple: (bool indicating structural validity, list detailing missing column names)
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


# ==========================================
# QUANTITATIVE INDICATOR PIPELINE
# ==========================================
def add_technical_indicators(
    df: pd.DataFrame,
    rsi_length: int = 14,
    ma20_length: int = 20,
    ma50_length: int = 50,
    skip_validation: bool = False
) -> pd.DataFrame:
    """
    Ingests raw price matrices and vectors, then computes and appends momentum, trend, 
    volatility, and volume-based indicators into a single unified mathematical matrix.

    Args:
        df (pd.DataFrame): Input market data containing standard OHLCV price paths.
        rsi_length (int): Smoothing window period for the Relative Strength Index oscillator.
        ma20_length (int): Lookback period window for the short-term simple moving average.
        ma50_length (int): Lookback period window for the long-term simple moving average.
        skip_validation (bool): Bypasses rigorous schema checks if set to True.

    Returns:
        pd.DataFrame: Augmented DataFrame enriched with multi-dimensional technical features.

    Raises:
        ValueError: If input matrices are structurally empty or missing critical columns.
    """
    if df is None:
        logger.error("DataFrame is None")
        raise ValueError("DataFrame cannot be None")

    if df.empty:
        logger.warning("Empty DataFrame provided, returning as-is")
        return df

    if rsi_length <= 0:
        logger.error(f"Invalid rsi_length: {rsi_length}")
        raise ValueError("rsi_length must be positive")

    if ma20_length <= 0:
        logger.error(f"Invalid ma20_length: {ma20_length}")
        raise ValueError("ma20_length must be positive")

    if ma50_length <= 0:
        logger.error(f"Invalid ma50_length: {ma50_length}")
        raise ValueError("ma50_length must be positive")

    if not skip_validation:
        if 'Close' not in df.columns:
            logger.error("Missing required 'Close' column")
            raise ValueError("DataFrame must have 'Close' column")

    logger.debug(
        f"Adding technical indicators: RSI({rsi_length}), "
        f"SMA({ma20_length}), SMA({ma50_length})"
    )

    df = df.copy()

    try:
        # Compute Relative Strength Index (RSI) to capture momentum overbought/oversold boundaries
        df.ta.rsi(close=df['Close'], length=rsi_length, append=True)

        # Compute Short-Term Simple Moving Average (SMA) to capture macro price momentum velocity
        df.ta.sma(close=df['Close'], length=ma20_length, append=True)

        # Compute Long-Term Simple Moving Average (SMA) to map foundational institutional support lines
        df.ta.sma(close=df['Close'], length=ma50_length, append=True)

        # Compute Moving Average Convergence Divergence (MACD) trend-following momentum signals
        df.ta.macd(
        close=df['Close'],
        append=True
        )

        # Compute Bollinger Bands to dynamically map localized volatility and price envelope expansions
        df.ta.bbands(
            close=df['Close'],
            append=True
        )

        # Compute Average True Range (ATR) to quantify market structural volatility and noise scales
        df.ta.atr(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            append=True
        )

        # Compute Average Directional Index (ADX) to determine structural trend strength regardless of direction
        df.ta.adx(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            append=True
        )

        # Compute On-Balance Volume (OBV) to quantify directional momentum using institutional capital flow accumulation
        df.ta.obv(
            close=df['Close'],
            volume=df['Volume'],
            append=True
        )

        # Map auto-generated mathematical library column labels into clean, standardized database feature properties
        df.rename(columns={
        'RSI_14':'RSI',
        'SMA_20':'MA20',
        'SMA_50':'MA50',

        'MACD_12_26_9':'MACD',
        'MACDs_12_26_9':'MACD_SIGNAL',
        'MACDh_12_26_9':'MACD_HIST',

        'BBL_5_2.0_2.0':'BB_LOWER',
        'BBM_5_2.0_2.0':'BB_MIDDLE',
        'BBU_5_2.0_2.0':'BB_UPPER',
        'BBB_5_2.0_2.0':'BB_WIDTH',
        'BBP_5_2.0_2.0':'BB_PERCENT',

        'ATRr_14':'ATR',
        'ADX_14':'ADX'
        }, inplace=True)

        # Define and verify the final feature vector space schema configuration
        indicators_added = [
        'RSI',
        'MA20',
        'MA50',
        'MACD',
        'MACD_SIGNAL',
        'MACD_HIST',
        'BB_UPPER',
        'BB_LOWER',
        'BB_WIDTH',
        'ATR',
        'ADX'
        ]
        missing_indicators = [col for col in indicators_added if col not in df.columns]

        if missing_indicators:
            logger.warning(f"Some indicators could not be calculated: {missing_indicators}")

        logger.debug("Technical indicators added successfully")
        return df

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise


# ==========================================
# CUSTOM PLUG-IN EXTENSION INTERFACE
# ==========================================
def add_custom_indicator(
    df: pd.DataFrame,
    indicator_func: callable,
    column_name: str,
    **kwargs
) -> pd.DataFrame:
    """
    Provides a modular abstraction layer to dynamically register and inject custom user-defined 
    mathematical overlays into the feature pipeline.

    Args:
        df (pd.DataFrame): Target price data matrix.
        indicator_func (callable): Custom algorithm calculation routine to execute.
        column_name (str): Associated string matrix identity key map for output columns.
        **kwargs: Arbitrary keyword variables passed straight into the indicator execution thread.

    Returns:
        pd.DataFrame: Data layout extended with the custom indicator configurations.

    Raises:
        ValueError: If baseline operational structures are uninitialized or blank.
    """
    if df is None or df.empty:
        logger.error("Cannot add indicator to empty DataFrame")
        raise ValueError("DataFrame cannot be empty")

    logger.debug(f"Adding custom indicator: {column_name}")

    df = df.copy()

    try:
        result = indicator_func(df, **kwargs)
        df[column_name] = result
        logger.debug(f"Custom indicator '{column_name}' added successfully")
        return df
    except Exception as e:
        logger.error(f"Error adding custom indicator '{column_name}': {e}")
        raise