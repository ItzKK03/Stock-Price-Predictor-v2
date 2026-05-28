"""
Project: Stock Price Predictor v2
Filename: train_model.py
Description: Core machine learning training engine. Manages end-to-end data pipelines,
             feature scaling, walk-forward time-series validation splits, and optimization 
             of a deep learning LSTM architecture integrated with NLP sentiment data.
Author: Kartik Kant (AI/ML Engineer)
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from typing import List, Optional, Any, Dict, Tuple

# ==========================================
# REPOSITORY MODULE INGESTION LAYER
# ==========================================
from src.data_collection import get_stock_data, get_stock_news
from src.feature_engineering import add_technical_indicators
from src.sentiment_analysis import initialize_sentiment_model, get_daily_sentiment
from src.model_utils import build_model, create_dataset, save_model, save_scaler
from src.data_preprocessing import merge_stock_sentiment, prepare_features
from src.evaluation import (
    evaluate_predictions,
    generate_evaluation_report,
    create_evaluation_dataframe,
    calculate_improvement_vs_baseline
)
from src.logger import get_logger

# Global application configuration profiles
from config import settings

# Initialize centralized logging context for pipeline tracing
logger = get_logger(__name__)


def train() -> None:
    """
    Executes the comprehensive deep learning training pipeline.
    Orchestrates ingestion, feature processing, temporal cross-validation, 
    neural network optimization, and target evaluation tracking metrics.
    """
    logger.info(f"Starting training process for {settings.TICKER}...")

    # ==========================================
    # 1. NLP SENTIMENT MODEL INGESTION
    # ==========================================
    logger.info("Initializing sentiment model...")
    sentiment_pipeline = initialize_sentiment_model()

    # ==========================================
    # 2. TIME-SERIES MARKET DATA INGESTION
    # ==========================================
    logger.info(f"Fetching data for {settings.TICKER} from {settings.START_DATE} to {settings.END_DATE}...")
    stock_data = get_stock_data(settings.TICKER, settings.START_DATE, settings.END_DATE)

    # ==========================================
    # 3. TECHNICAL FEATURE ENGINEERING
    # ==========================================
    logger.info("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)

    # ==========================================
    # 4. NATURAL LANGUAGE SENTIMENT EXTRACTION
    # ==========================================
    logger.info("Fetching and analyzing news sentiment...")
    daily_sentiment = get_daily_sentiment(settings.TICKER, sentiment_pipeline)

    # ==========================================
    # 5. DATA FRAME MATRIX CONSOLIDATION
    # ==========================================
    logger.info("Combining stock data and sentiment...")
    data = merge_stock_sentiment(stock_data, daily_sentiment)

    # Define operational feature profiles using global settings profiles
    features: List[str] = settings.FEATURES
    features_for_y: List[str] = [settings.TARGET_FEATURE]  

    # ==========================================
    # 6. FEATURE SCHEMATIZATION
    # ==========================================
    logger.info("Preparing features for training...")
    data_x, data_y, _ = prepare_features(data, features, settings.TARGET_FEATURE)

    # ==========================================
    # 7. FEATURE SCALING & LEAKAGE MANAGEMENT
    # ==========================================
    logger.info("Scaling data...")
    # Safe validation check to fetch feature scaling range, defaulting to standard bounds
    scale_range = getattr(settings, 'SCALER_FEATURE_RANGE', (0, 1))
    
    # Isolate vector spaces using distinct transformation instances to prevent data leakage
    scaler_x = MinMaxScaler(feature_range=scale_range)
    scaler_y = MinMaxScaler(feature_range=scale_range)

    # Transform multi-dimensional matrices down to boundary target scales
    scaled_x = scaler_x.fit_transform(data_x)
    scaled_y = scaler_y.fit_transform(data_y)

    # Serialize pre-fitted normalization parameters and feature configurations
    save_scaler(scaler_x, settings.scaler_x_path, features)
    save_scaler(scaler_y, settings.scaler_y_path, features_for_y)
    with open(settings.scaler_features_path, 'w') as f:
        json.dump(features, f)
    logger.info(f"Feature list saved to {settings.scaler_features_path}")

    # ==========================================
    # 8. TENSOR SHAPING FOR SEQUENTIAL PROCESSING
    # ==========================================
    logger.info(f"Creating time-series dataset with time_step={settings.TIME_STEP}...")
    X_train, y_train = create_dataset(scaled_x, scaled_y, settings.TIME_STEP)

    # Reshape matrices into required 3D tensor format: [Samples, Sequence Time Steps, Features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))

    # Gracefully handle insufficient array volumes before compilation triggers
    if X_train.shape[0] == 0:
        logger.error(f"Not enough data to create training samples. Need at least {settings.TIME_STEP+1} days of data.")
        return

    logger.info(f"Training dataset shape: X={X_train.shape}, y={y_train.shape}")

    # ==========================================
    # 9. WALK-FORWARD VALIDATION FRAMEWORK
    # ==========================================
    # Construct distinct temporal window folds to simulate live trading environments.
    # Essential for preserving historical sequence causality and preventing data lookahead bias.
    n = len(X_train)

    walk_forward_splits = [

    (0,int(n*0.60),int(n*0.72)),

    (0,int(n*0.72),int(n*0.84)),

    (0,int(n*0.84),n)

    ]

    logger.info(
    f"Using {len(walk_forward_splits)} walk-forward folds"
    )

    # ==========================================
    # 10. NEURAL NETWORK ARCHITECTURE SETTINGS
    # ==========================================
    logger.info("Building LSTM model...")
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=settings.LSTM_UNITS,
        dropout_rate=settings.DROPOUT_RATE,
        dense_units=settings.DENSE_UNITS,
        learning_rate=settings.LEARNING_RATE
    )

    # Callback configuration 1: Auto-checkpoint optimal weight configurations
    checkpoint = ModelCheckpoint(
        settings.model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    # Callback configuration 2: Stop optimization threads if validation loss plateaus
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=settings.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    # Callback configuration 3: Step-down learning rates when learning trajectories flatten
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
    )

    logger.info("Starting model training...")
    logger.info(f"Training config: epochs={settings.EPOCHS}, batch_size={settings.BATCH_SIZE}, "
                f"validation_split={settings.VALIDATION_SPLIT}")

    fold_scores = []

    # Execute iterative training over sequential temporal cross-validation slices
    for i,(train_start,
        train_end,
        val_end) in enumerate(
        walk_forward_splits):

        logger.info(
        f"Walk-forward fold {i+1}"
        )

        # Slice training window properties
        X_train_fold = X_train[
            train_start:train_end
        ]

        y_train_fold = y_train[
            train_start:train_end
        ]

        # Slice out-of-sample validation windows sequentially following training ranges
        X_val_fold = X_train[
            train_end:val_end
        ]

        y_val_fold = y_train[
            train_end:val_end
        ]

        # Fit model configurations to active validation fold bounds
        history = model.fit(

            X_train_fold,

            y_train_fold,

            validation_data=(

            X_val_fold,

            y_val_fold

            ),

            epochs=settings.EPOCHS,

            batch_size=
            settings.BATCH_SIZE,

            callbacks=[
                checkpoint,
                early_stopping,
                reduce_lr
            ],

            verbose=1

        )

        # Evaluate model convergence patterns based on local validation targets
        val_loss = model.evaluate(

            X_val_fold,

            y_val_fold,

            verbose=0

        )

        fold_scores.append(
            val_loss
        )

    logger.info(
    f"Average Walk-Forward Loss: {
    sum(fold_scores)/len(fold_scores)
    }"
    )

    logger.info("Training complete. Evaluating model...")

    # ==========================================
    # 11. METRIC PARSING & SYSTEM PERFORMANCE EVALUATION
    # ==========================================
    logger.info("Generating predictions for evaluation...")

    # Run network predictions on the final validation sequence matrix
    y_val_pred_scaled = model.predict(X_val_fold)
    
    # Reverse output vector normalizations back to absolute currency scales
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
    y_val_actual = scaler_y.inverse_transform(
    y_val_fold.reshape(-1,1)
    )

    # Flatten tensors to 1D arrays for standard performance reporting
    y_val_pred_flat = y_val_pred.flatten()
    y_val_actual_flat = y_val_actual.flatten()

    # Calculate standard evaluation metrics (MAE, RMSE, MAPE)
    metrics = evaluate_predictions(y_val_actual_flat, y_val_pred_flat)

    # Compute operational improvement gains against a baseline naive strategy
    improvement = calculate_improvement_vs_baseline(
        y_val_actual_flat,
        y_val_pred_flat,
        metric='mae'
    )

    # Assemble formatting parameters into structured summary blocks
    report = generate_evaluation_report(metrics, model_name=f"LSTM Stock Predictor ({settings.TICKER})")

    if improvement is not None:
        report += f"\nImprovement over Naive Baseline (MAE): {improvement:.2f}%\n"

    # Log summary performance profiles out to deployment contexts
    logger.info(report)
    print(report)  

    # Align sequential calendar data bounds with execution metrics arrays
    eval_df = create_evaluation_dataframe(
        y_val_actual_flat,
        y_val_pred_flat,
        dates=data.index[-len(y_val_actual_flat):] if len(data) >= len(y_val_actual_flat) else None
    )

    # Save target validation predictions into clear CSV tables
    eval_csv_path = settings.model_path.replace('.keras', '_evaluation.csv')
    eval_df.to_csv(eval_csv_path)
    logger.info(f"Evaluation results saved to {eval_csv_path}")

    # Serialize calculated score parameters into clear JSON profiles
    metrics_json_path = settings.model_path.replace('.keras', '_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_json_path}")

    # Export complete training log sequences to track performance variance curves
    history_df = pd.DataFrame(history.history)
    history_csv_path = settings.model_path.replace('.keras', '_training_history.csv')
    history_df.to_csv(history_csv_path)
    logger.info(f"Training history saved to {history_csv_path}")

    logger.info(f"Model saved to {settings.model_path}")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Prior to code initialization, purge outdated tracking logs and binary profiles
    for path in [
        settings.model_path,
        settings.scaler_x_path,
        settings.scaler_y_path,
        settings.scaler_features_path
    ]:
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Removed existing file: {path}")

    train()