"""
Project: Stock Price Predictor v2
Filename: config.py
Description: Centralized configuration management module. Handles environment
             variable loading, datatype casting, and default fallback parameters for 
             data pipelines, model architectures, training loops, and the web app.
Author: Kartik Kant (AI/ML Engineer)
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Ingest environment variables from local env configuration file profiles
load_dotenv()


# ==========================================
# ENVIRONMENT VARIABLE PARSING HELPERS
# ==========================================
def _get_env_str(key: str, default: str) -> str:
    """Parses environment keys as strings with a fallback default value."""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Parses environment keys as integers with default fallback conversions."""
    value = os.getenv(key, str(default))
    return int(value) if value else default


def _get_env_float(key: str, default: float) -> float:
    """Parses environment keys as floating-point numbers with configuration defaults."""
    value = os.getenv(key, str(default))
    return float(value) if value else default


def _get_env_list(key: str, default: List[str]) -> List[str]:
    """Splits comma-separated environment strings into list formats with whitespace stripping."""
    value = os.getenv(key)
    if value:
        return [item.strip() for item in value.split(',')]
    return default


# Establish absolute workspace root directory reference anchor
BASE_DIR: Path = Path(__file__).parent.absolute()


# ==========================================
# CONFIGURATION SUITES BY DOMAIN
# ==========================================
class DataConfig:
    """Encapsulates hyper-parameters for market ingestion and sequence building."""

    # Operational market ticker token and window bounds configurations
    TICKER: str = _get_env_str("TICKER", "AAPL")
    START_DATE: str = _get_env_str("START_DATE", "2015-01-01")
    END_DATE: str = _get_env_str("END_DATE", "2025-12-31")

    # Mathematical lookback interval steps required for LSTM sequence modeling
    TIME_STEP: int = _get_env_int("TIME_STEP", 60)

    # Feature schema list definitions tracking technical parameters and NLP inputs
    FEATURES: List[str] = _get_env_list(
        "FEATURES",
        [
        'Close',
        'High',
        'Low',
        'Open',
        'Volume',

        'RSI',
        'MA20',
        'MA50',

        'MACD',
        'MACD_SIGNAL',
        'MACD_HIST',

        'BB_UPPER',
        'BB_LOWER',
        'BB_WIDTH',
        'BB_MIDDLE',
        'BB_PERCENT',

        'ATR',
        'ADX',

        'Sentiment'
        ]
    )
    TARGET_FEATURE: str = _get_env_str("TARGET_FEATURE", "Close")


class ModelConfig:
    """Defines internal structural layer parameters and binary serialization paths."""

    # Dimensions and drop weights for the primary deep learning network
    LSTM_UNITS: int = _get_env_int("LSTM_UNITS", 128)
    DROPOUT_RATE: float = _get_env_float("DROPOUT_RATE", 0.3)
    DENSE_UNITS: int = _get_env_int("DENSE_UNITS", 32)
    LEARNING_RATE: float = _get_env_float("LEARNING_RATE", 0.0001)

    # Persistent on-disk absolute local file paths for binaries and metadata
    MODEL_PATH: str = _get_env_str("MODEL_PATH", str(BASE_DIR / "stock_predictor.keras"))
    SCALER_X_PATH: str = _get_env_str("SCALER_X_PATH", str(BASE_DIR / "scaler_x.joblib"))
    SCALER_Y_PATH: str = _get_env_str("SCALER_Y_PATH", str(BASE_DIR / "scaler_y.joblib"))
    SCALER_FEATURES_PATH: str = _get_env_str(
        "SCALER_FEATURES_PATH",
        str(BASE_DIR / "scaler_features.json")
    )

    @property
    def model_path(self) -> str:
        """Exposes absolute model file generation tracking location path."""
        return self.MODEL_PATH

    @property
    def scaler_x_path(self) -> str:
        """Exposes absolute multi-variable input feature scaler storage path."""
        return self.SCALER_X_PATH

    @property
    def scaler_y_path(self) -> str:
        """Exposes absolute single-variable targeted close scaler file location."""
        return self.SCALER_Y_PATH

    @property
    def scaler_features_path(self) -> str:
        """Exposes schema description storage path containing feature name lists."""
        return self.SCALER_FEATURES_PATH


class TrainingConfig:
    """Stores parameter sets regulating backend optimization runs."""

    # Hyperparameters managing training bounds, batch spacing, and validation constraints
    EPOCHS: int = _get_env_int("EPOCHS", 100)
    BATCH_SIZE: int = _get_env_int("BATCH_SIZE", 64)
    VALIDATION_SPLIT: float = _get_env_float("VALIDATION_SPLIT", 0.2)
    EARLY_STOPPING_PATIENCE: int = _get_env_int("EARLY_STOPPING_PATIENCE", 10)

    # Standard mathematical range definitions for normalization functions
    SCALER_FEATURE_RANGE: tuple = (0, 1)


class AppConfig:
    """Maintains dashboard presentation frameworks and layout configurations."""

    # Web application metadata headers and presentation rendering schemas
    PAGE_TITLE: str = _get_env_str("PAGE_TITLE", "AI Stock Price Predictor")
    PAGE_ICON: str = _get_env_str("PAGE_ICON", "📈")
    LAYOUT: str = _get_env_str("LAYOUT", "wide")

    # Time limits governing dashboard visualization loops and pre-calculation buffers
    HISTORY_DAYS: int = _get_env_int("HISTORY_DAYS", 90)
    DATA_BUFFER_DAYS: int = _get_env_int("DATA_BUFFER_DAYS", 100)

    # Identifier token path map points targeting remote transformer repositories
    SENTIMENT_MODEL_NAME: str = _get_env_str("SENTIMENT_MODEL_NAME", "ProsusAI/finbert")


# ==========================================
# AGGREGATED APPLICATION GLOBAL OBJECT
# ==========================================
class Settings:
    """Consolidates sub-config modules into a single global operational interface."""

    # Derived Data Configuration attributes mapping
    TICKER: str = DataConfig.TICKER
    START_DATE: str = DataConfig.START_DATE
    END_DATE: str = DataConfig.END_DATE
    TIME_STEP: int = DataConfig.TIME_STEP
    FEATURES: List[str] = DataConfig.FEATURES
    TARGET_FEATURE: str = DataConfig.TARGET_FEATURE

    # Derived Model Configuration attributes mapping
    MODEL_PATH: str = ModelConfig.MODEL_PATH
    SCALER_X_PATH: str = ModelConfig.SCALER_X_PATH
    SCALER_Y_PATH: str = ModelConfig.SCALER_Y_PATH
    SCALER_FEATURES_PATH: str = ModelConfig.SCALER_FEATURES_PATH
    LSTM_UNITS: int = ModelConfig.LSTM_UNITS
    DROPOUT_RATE: float = ModelConfig.DROPOUT_RATE
    DENSE_UNITS: int = ModelConfig.DENSE_UNITS
    LEARNING_RATE: float = ModelConfig.LEARNING_RATE

    # Derived Training Configuration attributes mapping
    EPOCHS: int = TrainingConfig.EPOCHS
    BATCH_SIZE: int = TrainingConfig.BATCH_SIZE
    VALIDATION_SPLIT: float = TrainingConfig.VALIDATION_SPLIT
    EARLY_STOPPING_PATIENCE: int = TrainingConfig.EARLY_STOPPING_PATIENCE

    # Derived App UI Configuration attributes mapping
    PAGE_TITLE: str = AppConfig.PAGE_TITLE
    PAGE_ICON: str = AppConfig.PAGE_ICON
    LAYOUT: str = AppConfig.LAYOUT
    HISTORY_DAYS: int = AppConfig.HISTORY_DAYS
    DATA_BUFFER_DAYS: int = AppConfig.DATA_BUFFER_DAYS
    SENTIMENT_MODEL_NAME: str = AppConfig.SENTIMENT_MODEL_NAME

    @property
    def model_path(self) -> str:
        """Centralized wrapper referencing active binary execution paths."""
        return self.MODEL_PATH

    @property
    def scaler_x_path(self) -> str:
        """Centralized wrapper referencing input sequence scaler parameters."""
        return self.SCALER_X_PATH

    @property
    def scaler_y_path(self) -> str:
        """Centralized wrapper referencing targeted value inverse matrices."""
        return self.SCALER_Y_PATH

    @property
    def scaler_features_path(self) -> str:
        """Centralized wrapper referencing system serialized JSON data features."""
        return self.SCALER_FEATURES_PATH


# Instantiate global settings reference profile for direct cross-module importing
settings: Settings = Settings()