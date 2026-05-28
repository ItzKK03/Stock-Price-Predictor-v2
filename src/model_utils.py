"""
Project: Stock Price Predictor v2
Package: src
Filename: model_utils.py
Description: Model architecture and serialization utility module. Establishes neural 
             network topologies using a hybrid Bidirectional LSTM, Attention, and GRU framework, 
             and provides robust disk persistence mechanisms for deep learning binaries and scalers.
Author: Kartik Kant (AI/ML Engineer)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Tuple, List, Optional, Any, Dict
from tensorflow.keras.models import (
    Model,
    Sequential,
    load_model as keras_load_model
)
from tensorflow.keras.layers import (
Input,
Bidirectional,
LSTM,
GRU,
Dense,
Dropout,
Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from src.logger import get_logger

# Initialize centralized logging context for model tracking telemetry
logger = get_logger(__name__)


# ==========================================
# SERIALIZATION PATH FILE CONSTANTS
# ==========================================
MODEL_PATH: str = 'stock_predictor.keras'
SCALER_X_PATH: str = 'scaler_x.joblib'
SCALER_Y_PATH: str = 'scaler_y.joblib'
SCALER_FEATURES_PATH: str = 'scaler_features.json'


# ==========================================
# TOPOLOGY DIMENSION SHAPE VALIDATION
# ==========================================
def validate_input_shape(input_shape: Tuple[int, int]) -> bool:
    """
    Verifies that the incoming matrix dimension configuration matches the requirements for a sequential model.

    Args:
        input_shape (Tuple[int, int]): Dimensions mapping to (sequence_time_steps, feature_count).

    Returns:
        bool: True if dimensions are non-zero and positive, False otherwise.
    """
    if not input_shape or len(input_shape) != 2:
        return False
    time_steps, n_features = input_shape
    return time_steps > 0 and n_features > 0


# ==========================================
# HYBRID DEEP LEARNING MODEL FACTORY
# ==========================================
def build_model(
    input_shape,
    lstm_units=128,
    dropout_rate=0.3,
    dense_units=32,
    learning_rate=0.0001
):
    """
    Constructs and compiles a hybrid deep neural network topology combining 
    stacked Bidirectional LSTMs, localized dot-product Attention layers, 
    and a recurrent GRU compression block.

    Args:
        input_shape (tuple): Feature shape mapping input matrices.
        lstm_units (int): Dimensional footprint of the structural LSTM layers.
        dropout_rate (float): Regularization weight controlling neuron deactivation boundaries.
        dense_units (int): Structural footprint for fully connected hidden networks.
        learning_rate (float): Magnitude parameter regulating Adam optimizer gradient updates.

    Returns:
        Model: Fully compiled Keras functional API neural network framework instance.
    """
    if not validate_input_shape(input_shape):
        raise ValueError("Invalid input shape")

    logger.info(f"Building improved BiLSTM model...")

    # Layer 1: Instantiate functional network tensor processing entry node points
    inputs = Input(shape=input_shape)

    # Layer 2: Stack a primary Bidirectional LSTM layer. 
    # Processes sequences both forward and backward to retain historical and future temporal contexts.
    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True
        )
    )(inputs)

    # Layer 3: Prevent overfitting by randomly dropping activation links during backward passes
    x = Dropout(dropout_rate)(x)

    # Layer 4: Stack a secondary Bidirectional LSTM compression block to capture higher-level features
    x = Bidirectional(
        LSTM(
            lstm_units // 2,
            return_sequences=True
        )
    )(x)

    # Layer 5: Secondary regularization checkpoint boundary
    x = Dropout(dropout_rate)(x)

    # Layer 6: Inject a Dot-Product Attention mechanism.
    # Mitigates vanishing gradient memory loss by computing rolling context weights over long-term temporal frames.
    attn = Attention()([x, x])

    # Layer 7: Route attention state representations into a GRU layer to condense sequential information
    x = GRU(32)(attn)

    # Layer 8: Dense layer with Rectified Linear Unit activations to extract structural trends
    x = Dense(
        dense_units,
        activation='relu'
    )(x)

    # Layer 9: Final single-node linear projection wrapper mapping to quantitative pricing outputs
    outputs = Dense(1)(x)

    # Consolidate computational paths into a formalized model structure
    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    # Compile the optimization layer. Uses Huber loss rather than MSE 
    # to provide robust protection against financial market anomalies and outlier noise.
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss=Huber()
    )

    logger.info("Improved model built successfully")

    return model


# ==========================================
# MODEL SAVING & LOADING OPERATIONS
# ==========================================
def save_model(model: Sequential, path: str) -> None:
    """
    Serializes a trained neural network structure and parameters directly onto local storage drives.

    Args:
        model (Sequential): Active pre-compiled model architecture instance.
        path (str): Destination disk file location identifier string.
    """
    if model is None:
        logger.error("Cannot save None model")
        raise ValueError("Model cannot be None")

    if not path:
        logger.error("Cannot save model to empty path")
        raise ValueError("Path cannot be empty")

    logger.info(f"Saving model to {path}")
    model.save(path)
    logger.info("Model saved successfully")


def load_model(path: str) -> Sequential:
    """
    Deserializes compiled model binaries from persistent storage back into operational memory.

    Args:
        path (str): Targeted input file location path string.

    Returns:
        Sequential: Reconstructed Keras neural network execution matrix.
    """
    if not path:
        logger.error("Cannot load model from empty path")
        raise ValueError("Path cannot be empty")

    if not os.path.exists(path):
        logger.error(f"Model file not found at {path}")
        raise FileNotFoundError(f"Model file not found at {path}")

    logger.info(f"Loading model from {path}")
    return keras_load_model(path)


# ==========================================
# SCALER PERSISTENCE PROCESSING LAYERS
# ==========================================
def save_scaler(
    scaler: Any,
    scaler_path: str,
    features: List[str]
) -> None:
    """
    Serializes data transformation scale constants along with their corresponding feature definitions.

    Args:
        scaler (Any): Fitted Scikit-Learn data normalization class instance.
        scaler_path (str): Persistent file path for storing serialization binaries.
        features (List[str]): Operational dataset feature labels matching the scaler schema.
    """
    if scaler is None:
        logger.error("Cannot save None scaler")
        raise ValueError("Scaler cannot be None")

    if not scaler_path:
        logger.error("Cannot save scaler to empty path")
        raise ValueError("Scaler path cannot be empty")

    if not features or not isinstance(features, list):
        logger.error(f"Invalid features: {features}")
        raise ValueError("Features must be a non-empty list")

    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    # Maintain clear structural alignment by saving field strings into a separate JSON file
    features_path = scaler_path.replace('.joblib', '_features.json')
    with open(features_path, 'w') as f:
        json.dump(features, f)
    logger.info(f"Features saved to {features_path}")


def load_scaler(scaler_path: str) -> Tuple[Any, Optional[List[str]]]:
    """
    Loads pre-calculated serialization matrices and field tracking descriptions.

    Args:
        scaler_path (str): Target data normalization file location path.

    Returns:
        Tuple[Any, Optional[List[str]]]: (Reconstructed scaler class instance, Associated feature lists)
    """
    if not scaler_path:
        logger.error("Cannot load scaler from empty path")
        raise ValueError("Scaler path cannot be empty")

    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at {scaler_path}")
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Locate and ingest associated column layout structures if available
    features_path = scaler_path.replace('.joblib', '_features.json')
    features: Optional[List[str]] = None
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
        logger.debug(f"Loaded {len(features)} features")
    else:
        logger.warning(f"Features file not found at {features_path}")

    return scaler, features


# ==========================================
# SLIDING-WINDOW TIMESERIES TRANSFORMATION
# ==========================================
def create_dataset(
    dataset_x: np.ndarray,
    dataset_y: np.ndarray,
    time_step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms flattened 2D chronological data frames into overlapping 3D sequence tensors 
    required for training and validating recurrent network topologies.

    Args:
        dataset_x (np.ndarray): Multi-variable inputs feature space array.
        dataset_y (np.ndarray): Target closing valuation prediction tracking array.
        time_step (int): Length parameter defining the sliding sequence lookback window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (3D Tensor Input X [Samples, Lookback Range, Features], 
                                       1D Array Target Vector y [Samples])
    """
    if dataset_x is None or dataset_y is None:
        logger.error("dataset_x and dataset_y cannot be None")
        raise ValueError("dataset_x and dataset_y cannot be None")

    if not isinstance(dataset_x, np.ndarray) or not isinstance(dataset_y, np.ndarray):
        logger.error("dataset_x and dataset_y must be numpy arrays")
        raise ValueError("dataset_x and dataset_y must be numpy arrays")

    if time_step <= 0:
        logger.error(f"Invalid time_step: {time_step}")
        raise ValueError("time_step must be positive")

    if len(dataset_x) != len(dataset_y):
        logger.error(f"Dataset length mismatch: X={len(dataset_x)}, y={len(dataset_y)}")
        raise ValueError("dataset_x and dataset_y must have the same length")

    if len(dataset_x) <= time_step:
        logger.error(
            f"Dataset too small: {len(dataset_x)} samples, need more than time_step={time_step}"
        )
        raise ValueError(f"Dataset must have more than time_step ({time_step}) samples")

    logger.debug(f"Creating dataset with time_step={time_step}, samples={len(dataset_x)}")

    dataX: List[np.ndarray] = []
    dataY: List[float] = []

    # Iterate over data records using a sliding window to compile overlapping historical sequences
    for i in range(len(dataset_x) - time_step):
        a = dataset_x[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset_y[i + time_step, 0])

    result_x = np.array(dataX)
    result_y = np.array(dataY)

    logger.info(f"Created dataset with {len(result_x)} samples")
    return result_x, result_y