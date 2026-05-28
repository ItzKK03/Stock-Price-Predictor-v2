"""
Project: Stock Price Predictor v2
Package: src
Filename: sentiment_analysis.py
Description: Natural Language Processing (NLP) sentiment analytics engine. Initializes 
             and caches the FinBERT transformer model pipeline to evaluate financial news headlines 
             in memory-safe batches, generating rolling, confidence-weighted daily sentiment indices.
Author: Kartik Kant (AI/ML Engineer)
"""

from functools import lru_cache
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Any, Optional
from src.logger import get_logger

from .data_collection import get_stock_news

# Initialize centralized logging context for transformer execution telemetry
logger = get_logger(__name__)

# Core Hugging Face weights path targeting the domain-optimized FinBERT model
MODEL_NAME: str = "ProsusAI/finbert"


# ==========================================
# MODEL INITIALIZATION & CACHING LAYER
# ==========================================
@lru_cache(maxsize=1)
def initialize_sentiment_model(model_name: str = MODEL_NAME) -> Any:
    """
    Loads and compiles the transformer pipeline into memory, applying an LRU cache 
    mechanism to protect against redundant weight reloads across application states.

    Args:
        model_name (str): Hugging Face model repository identifier token.

    Returns:
        Any: Initialized tokenization and sequence classification pipeline instance.

    Raises:
        Exception: Cascades initialization errors if the binary weights fail to pull down or compile.
    """
    logger.info(f"Initializing sentiment model: {model_name}...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        logger.info("Sentiment model initialized successfully.")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize sentiment model: {e}")
        raise


# ==========================================
# BATCH-PROCESSED SENTIMENT INFERENCE ENGINE
# ==========================================
def get_sentiment_scores(
    headlines: List[str],
    sentiment_pipeline: Any,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Executes batch text classification inference across an array of headlines 
    to manage memory footprints and accelerate hardware tensor processing.

    Args:
        headlines (List[str]): Array containing raw text headline parameters.
        sentiment_pipeline (Any): Initialized transformers sequence classifier pipeline.
        batch_size (int): Segment step distribution limit for processing text elements.

    Returns:
        List[Dict[str, Any]]: Collection of dictionaries tracking predicted labels and softmax weights.

    Raises:
        ValueError: Triggers evaluation blocks if parameters are structurally malformed.
    """
    if not headlines:
        logger.warning("Empty headlines list provided")
        return []

    if not isinstance(headlines, list):
        logger.error(f"headlines must be a list, got {type(headlines)}")
        raise ValueError("headlines must be a list")

    # Clean the collection by stripping out non-string parameters or empty spaces
    valid_headlines = [h for h in headlines if h and isinstance(h, str)]
    if not valid_headlines:
        logger.warning("No valid headlines after filtering")
        return []

    logger.debug(f"Analyzing sentiment for {len(valid_headlines)} headlines in batches of {batch_size}")

    all_sentiments: List[Dict[str, Any]] = []

    try:
        # Step through the text array boundaries slicing inputs into processing segments
        for i in range(0, len(valid_headlines), batch_size):
            batch = valid_headlines[i:i + batch_size]
            batch_sentiments = sentiment_pipeline(batch)
            all_sentiments.extend(batch_sentiments)

        logger.debug(f"Sentiment analysis complete for {len(all_sentiments)} headlines")
        return all_sentiments

    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        raise


# ==========================================
# DAILY DATA AGGREGATION & POOLING LAYER
# ==========================================
def get_daily_sentiment(
    ticker: str,
    sentiment_pipeline: Any
) -> pd.DataFrame:
    """
    Ingests unstructured news metadata streams, processes natural language inferences, 
    and applies confidence weights to aggregate text analytics into a structured daily time-series.

    Args:
        ticker (str): Target asset identification tag representation string.
        sentiment_pipeline (Any): Pre-loaded transformers model pipeline context.

    Returns:
        pd.DataFrame: Chronological DatetimeIndex data table holding the combined metric 'Sentiment'.
    """
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    logger.debug(f"Fetching daily sentiment for {ticker}")

    news = get_stock_news(ticker)

    if not news:
        logger.warning("No news returned, returning empty sentiment DataFrame")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    # Structural mapping closure converting fluctuating schema payloads to reliable keys
    normalized_news = []
    for item in news:
        if not isinstance(item, dict):
            continue
            
        content = item.get('content', item) if isinstance(item.get('content'), dict) else {}
        
        title = item.get('title') or content.get('title') or item.get('headline')
        pub_time = item.get('publish_time') or item.get('providerPublishTime') or content.get('pubDate') or item.get('publishedAt')
        
        if title and pub_time:
            normalized_news.append({'title': title, 'publish_time': pub_time})

    df = pd.DataFrame(normalized_news)

    if 'title' not in df.columns:
        logger.warning("News found, but could not extract 'title' column from the API response structure.")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    df = df.dropna(subset=['title'])
    df['title'] = df['title'].astype(str)

    if df.empty:
        logger.warning("News titles were empty after cleaning")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    if 'publish_time' not in df.columns:
        logger.warning("News found, but could not extract 'publish_time' column")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    # Safely align variable temporal objects (Unix fields vs ISO formats) down to unified timestamps
    try:
        if pd.api.types.is_numeric_dtype(df['publish_time']):
            df['publish_time'] = pd.to_datetime(df['publish_time'], unit='s')
        else:
            df['publish_time'] = pd.to_datetime(df['publish_time'])
            
        if df['publish_time'].dt.tz is not None:
            df['publish_time'] = df['publish_time'].dt.tz_localize(None)
    except Exception as e:
        logger.error(f"Error parsing dates: {e}")
        return pd.DataFrame(columns=['Sentiment']).rename_axis('Date')

    df = df.set_index('publish_time')

    # Convert table rows to string sequences to prepare for model execution
    headlines = df['title'].tolist()
    logger.debug(f"Analyzing {len(headlines)} news headlines")
    sentiments = get_sentiment_scores(headlines, sentiment_pipeline)

    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    # Map discrete textual outcomes into standardized computational sign factors
    label_to_score: Dict[str, int] = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['numeric_sentiment'] = df['sentiment_label'].map(label_to_score)

    # Calculate confidence-weighted scores by multiplying polarity signs by model probability distributions
    df['weighted_sentiment'] = df['numeric_sentiment'] * df['sentiment_score']

    # Resample unstructured publication times down to uniform daily calendar bars using a pooling mean
    daily_sentiment = df['weighted_sentiment'].resample('D').mean()

    # Cast final arrays into structured data frame formats matching system ingestion configurations
    daily_sentiment_df = daily_sentiment.to_frame(name='Sentiment')
    daily_sentiment_df.index.name = 'Date'

    logger.info(f"Generated sentiment for {len(daily_sentiment_df)} days")
    return daily_sentiment_df