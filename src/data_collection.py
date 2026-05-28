"""
Project: Stock Price Predictor v2
Package: src
Filename: data_collection.py
Description: Data ingestion and network resilience module. Handles asynchronous 
             fetching of historical stock arrays and market news feeds via the 
             Yahoo Finance API, incorporating exponential backoff retry algorithms 
             with random jitter to manage connection throttling and transient dropouts.
Author: Kartik Kant (AI/ML Engineer)
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from src.logger import get_logger
import time
import random

# Initialize centralized logging context for telemetry tracking
logger = get_logger(__name__)


# ==========================================
# NETWORK RESILIENCE CORE ARCHITECTURE
# ==========================================
def _retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True
) -> Any:
    """
    Executes an arbitrary ingestion callable inside an exponential backoff wrapper 
    to provide fault tolerance against API rate limits and network degradation.

    Args:
        func (callable): Targeted operations routine to run.
        max_retries (int): Total distribution scale allocation for execution re-entries.
        base_delay (float): Fundamental time coefficient determining baseline backoff steps.
        max_delay (float): Upper bounding threshold capping structural latency delays.
        jitter (bool): Toggles inclusion of stochastic multipliers to distribute network concurrency.

    Returns:
        Any: Evaluated outcome of the wrapped processing function execution.

    Raises:
        Exception: Propagates structural runtime exceptions if all allocated retries are exhausted.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Calculate logarithmic scaling intervals based on current attempt counter
                delay = min(base_delay * (2 ** attempt), max_delay)

                # Stochastic adjustment calculation to disperse simultaneous client re-entries
                if jitter:
                    delay = delay * (0.5 + random.random())

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


# ==========================================
# TIME-SERIES OHLCV DATA COLLECTION PIPELINE
# ==========================================
def get_stock_data(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Extracts structured historical equity records matching specified data bounds.

    Args:
        ticker (str): Target stock label asset representation string.
        start (str): Ingestion point initial date boundary ('YYYY-MM-DD').
        end (str): Ingestion point termination date boundary ('YYYY-MM-DD').
        max_retries (int): Fault-tolerance constraint managing network retry parameters.

    Returns:
        pd.DataFrame: Structured tabular panel matrix holding datetime-indexed market frames.

    Raises:
        ValueError: Triggers validation faults if parameter attributes are malformed or blank.
    """
    # Defensive parameter checks to validate operational variable schemas
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        raise ValueError("Ticker must be a non-empty string")

    if not start or not isinstance(start, str):
        logger.error(f"Invalid start date: {start}")
        raise ValueError("Start date must be a non-empty string")

    if not end or not isinstance(end, str):
        logger.error(f"Invalid end date: {end}")
        raise ValueError("End date must be a non-empty string")

    logger.info(f"Fetching data for {ticker} from {start} to {end}...")

    def _fetch() -> pd.DataFrame:
        """Isolated operational closure mapping API collection calls."""
        data = yf.download(ticker, start=start, end=end, group_by='column', progress=False)

        # Normalization layer to strip out hierarchically stacked multi-index schemas.
        # Resolves coordinate mapping problems in deep learning pipeline feature merges.
        if isinstance(data.columns, pd.MultiIndex):
            logger.debug("Flattening MultiIndex columns...")
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        return data

    try:
        # Route execution threads into the fault-tolerance retry layer
        data = _retry_with_backoff(_fetch, max_retries=max_retries)
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker} after retries: {e}")
        raise

    # Standardize temporal index properties to unified timestamp representations
    data.index = pd.to_datetime(data.index)

    logger.info(f"Fetched {len(data)} rows of data for {ticker}")
    return data


# ==========================================
# MEDIA STREAM EXTRACTION PIPELINE
# ==========================================
def get_stock_news(
    ticker: str,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Collects raw textual commentary data payloads relating to the specified token parameter.

    Args:
        ticker (str): Equity symbol identification parameter.
        max_retries (int): Baseline threshold managing connection retry limitations.

    Returns:
        List[Dict[str, Any]]: Array of dictionary structures containing unstructured text attributes.
    """
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        return []

    logger.debug(f"Fetching news for {ticker}...")

    def _fetch() -> List[Dict[str, Any]]:
        """Isolated structural wrapper mapping ticker property reads."""
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []

    try:
        news = _retry_with_backoff(_fetch, max_retries=max_retries)
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []

    if not news:
        logger.warning(f"No news found for {ticker}")
        return []

    logger.debug(f"Fetched {len(news)} news articles for {ticker}")
    return news