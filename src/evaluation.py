"""
Project: Stock Price Predictor v2
Package: src
Filename: evaluation.py
Description: Model evaluation and performance metrics module. Computes core mathematical 
             regression errors (MAE, RMSE, MAPE, R²) alongside classification metrics 
             (Directional Accuracy) to evaluate deep learning stock price prediction networks.
Author: Kartik Kant (AI/ML Engineer)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

from src.logger import get_logger

# Initialize centralized logging context for telemetry validation
logger = get_logger(__name__)


# ==========================================
# REGRESSION ERROR METRICS LAYER
# ==========================================
def calculate_mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Computes the Mean Absolute Error to quantify absolute distance deviations.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.

    Returns:
        float: Calculated MAE value.

    Raises:
        ValueError: If input matrices have mismatched shapes or lengths are zero.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    mae = mean_absolute_error(y_true, y_pred)
    logger.debug(f"MAE calculated: {mae:.6f}")
    return mae


def calculate_rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Computes Root Mean Squared Error to disproportionately penalize large outlier prediction mistakes.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.

    Returns:
        float: Calculated RMSE value.

    Raises:
        ValueError: If array instances have mismatched layouts or are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    logger.debug(f"RMSE calculated: {rmse:.6f}")
    return rmse


def calculate_mape(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    epsilon: float = 1e-10
) -> float:
    """
    Computes the Mean Absolute Percentage Error to present error ratios relative to target magnitudes.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.
        epsilon (float): Floating-point denominator buffer to avoid mathematical division-by-zero errors.

    Returns:
        float: Calculated MAPE value represented as a percentage.

    Raises:
        ValueError: If arrays are missing vector points or have asymmetrical layouts.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter out values near zero to stabilize fractional computations
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        logger.warning("All true values are near zero, MAPE may be unreliable")
        return float('inf')

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    logger.debug(f"MAPE calculated: {mape:.4f}%")
    return mape


# ==========================================
# DIRECTIONAL CLASSIFICATION LAYER
# ==========================================
def calculate_directional_accuracy(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """
    Computes the Directional Accuracy ratio by modeling pricing series as movement classifications.
    Evaluates how effectively the system isolates daily up/down vector orientations.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.

    Returns:
        float: Directional tracking precision score represented as a percentage.

    Raises:
        ValueError: If arrays hold insufficient dimensions to compute standard differentials.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for directional accuracy")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Extract sign profiles (-1, 0, 1) across sequential discrete indices to track price movement direction
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    # Evaluate matches between actual market trends and model directional estimates
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    logger.debug(f"Directional accuracy calculated: {accuracy:.2f}%")
    return accuracy


# ==========================================
# VARIANCE CORRELATION LAYER
# ==========================================
def calculate_r2(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Computes the Coefficient of Determination (R²) to track variance coverage properties.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.

    Returns:
        float: Calculated R-squared value capped at an ideal value of 1.0.

    Raises:
        ValueError: If input sample bounds do not meet requirements.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for R² calculation")

    r2 = r2_score(y_true, y_pred)
    logger.debug(f"R² calculated: {r2:.6f}")
    return r2


# ==========================================
# INPUT STRUCTURAL PARSING ROUTINES
# ==========================================
def _validate_inputs(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes input array instances into clean, formatted NumPy vector arrays.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized NumPy array output structures.

    Raises:
        ValueError: If properties are missing or array dimensions don't match.
    """
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred cannot be None")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Input length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )

    return y_true, y_pred


# ==========================================
# CONSOLIDATED KPI BUNDLING METRICS
# ==========================================
def evaluate_predictions(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    Gathers and coordinates multiple analytical metrics to generate a unified evaluation dictionary.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.

    Returns:
        Dict[str, float]: Dictionary mapping string identifiers to evaluated numeric scores.
    """
    logger.info("Calculating evaluation metrics...")

    y_true, y_pred = _validate_inputs(y_true, y_pred)

    metrics: Dict[str, float] = {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }

    logger.info(
        f"Evaluation complete: MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%, "
        f"Directional Accuracy={metrics['directional_accuracy']:.2f}%, "
        f"R²={metrics['r2']:.4f}"
    )

    return metrics


# ==========================================
# REPORT FORMATTING LAYER
# ==========================================
def generate_evaluation_report(
    metrics: Dict[str, float],
    model_name: str = "Model"
) -> str:
    """
    Formats multi-dimensional evaluation results into structured, human-readable terminal blocks.

    Args:
        metrics (Dict[str, float]): Dictionary collection containing compiled pipeline score parameters.
        model_name (str): Context model label used for header identification.

    Returns:
        str: Multi-line string report format profile.
    """
    report = f"""
{'='*60}
{model_name} - Evaluation Report
{'='*60}

Regression Metrics:
  - Mean Absolute Error (MAE):     {metrics.get('mae', 'N/A'):.4f}
  - Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.4f}
  - Mean Absolute Percentage Error: {metrics.get('mape', 'N/A'):.2f}%
  - R-squared (R²):                {metrics.get('r2', 'N/A'):.4f}

Classification Metric:
  - Directional Accuracy:          {metrics.get('directional_accuracy', 'N/A'):.2f}%

{'='*60}
"""
    return report


def create_evaluation_dataframe(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    dates: Optional[pd.DatetimeIndex] = None
) -> pd.DataFrame:
    """
    Assembles real values alongside predictions inside standardized comparison data frames.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.
        dates (Optional[pd.DatetimeIndex]): Timeline configuration setting index limits.

    Returns:
        pd.DataFrame: Completed analytical data table with error columns.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })

    # Track numeric variances and percentage errors across rows
    df['Error'] = df['Predicted'] - df['Actual']
    df['Error_%'] = (df['Error'] / df['Actual'].replace(0, np.nan)) * 100

    if dates is not None:
        df.index = dates[:len(df)]

    return df


# ==========================================
# BASELINE COMPARISON EVALUATION ROUTINES
# ==========================================
def calculate_improvement_vs_baseline(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    baseline_pred: Optional[Union[np.ndarray, List[float]]] = None,
    metric: str = 'mae'
) -> Optional[float]:
    """
    Benchmarks model execution against a naive random-walk baseline model.

    Args:
        y_true (Union[np.ndarray, List[float]]): Target ground truth actual valuations.
        y_pred (Union[np.ndarray, List[float]]): Corresponding model output projections.
        baseline_pred (Optional[Union[np.ndarray, List[float]]]): Baseline array sequence (defaults to t-1 shift).
        metric (str): Targeted scoring routine parameter mapping ('mae', 'rmse', 'mape').

    Returns:
        Optional[float]: Gain margin ratio calculation. Positive indicates model outperformance.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if baseline_pred is None:
        # Form naive baseline by executing a historical index step-shift vector roll
        baseline_pred = np.roll(y_true, 1)
        baseline_pred[0] = y_true[0]  

    metric_func = {
        'mae': calculate_mae,
        'rmse': calculate_rmse,
        'mape': calculate_mape
    }.get(metric.lower())

    if metric_func is None:
        logger.error(f"Unknown metric: {metric}")
        return None

    model_score = metric_func(y_true, y_pred)
    baseline_score = metric_func(y_true, baseline_pred)

    if baseline_score == 0:
        return None

    # Compute percentage performance lift over baseline
    improvement = ((baseline_score - model_score) / baseline_score) * 100
    logger.info(
        f"Model {metric.upper()}={model_score:.4f} vs Baseline={baseline_score:.4f}, "
        f"Improvement={improvement:.2f}%"
    )

    return improvement