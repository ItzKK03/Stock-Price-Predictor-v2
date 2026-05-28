"""
Project: Stock Price Predictor v2
Package: src
Filename: backtesting.py
Description: Algorithmic backtesting engine. Computes quantitative evaluation metrics 
             including cumulative equity curves, annualized Sharpe ratios, and peak-to-trough 
             maximum drawdowns to benchmark machine learning strategy performance against 
             standard buy-and-hold market baselines.
Author: Kartik Kant (AI/ML Engineer)
"""

import numpy as np
import pandas as pd


def run_backtest(df,
                 prediction_col='Predicted_Close',
                 price_col='Close'):
    """
    Simulates strategy execution based on directional prediction outputs and compiles 
    key financial performance indicators.

    Args:
        df (pd.DataFrame): Time-series dataframe containing actual prices and model predictions.
        prediction_col (str): DataFrame column label for next-day target predictions.
        price_col (str): DataFrame column label for actual historical closing prices.

    Returns:
        Tuple[pd.DataFrame, dict]: (Dataframe appended with returns and equity metrics, 
                                   Dictionary holding calculated summary performance KPIs)
    """
    data = df.copy()

    # ==========================================
    # STRATEGY SIGNAL GENERATION
    # ==========================================
    # Generates binary execution signals (1 = Long, 0 = Cash) by evaluating if the predicted close 
    # outpaces the active actual closing price. Both vectors are shifted by 1 index period to 
    # guarantee that trading actions are executed using historical data, completely preventing lookahead bias.
    data['Signal'] = (
        data[prediction_col].shift(1)
        >
        data[price_col].shift(1)
    ).astype(int)

    # ==========================================
    # LOGMARKET RETURNS COMPUTATION
    # ==========================================
    # Computes simple percentage price changes between consecutive trading intervals
    data['Market_Return'] = (
        data[price_col].pct_change()
    )

    # Calculates realized strategy returns by mapping daily binary signals against market performance
    data['Strategy_Return'] = (
        data['Signal']
        *
        data['Market_Return']
    )

    # ==========================================
    # CUMULATIVE EQUITY TRAJECTORIES
    # ==========================================
    # Compiles compounded performance curves over time utilizing rolling cumulative products.
    # Empty entry frames (NaN values) are padded with zero to stabilize financial tracking paths.
    data['BuyHold_Equity'] = (
        1 +
        data['Market_Return'].fillna(0)
    ).cumprod()

    data['Strategy_Equity'] = (
        1 +
        data['Strategy_Return'].fillna(0)
    ).cumprod()

    # ==========================================
    # RISK-ADJUSTED PERFORMANCE (SHARPE RATIO)
    # ==========================================
    # Computes the annualized risk-adjusted outperformance index. 
    # Standardizes the mean daily asset return against its standard deviation volatility profile.
    # Incorporates a tiny epsilon floating-point offset (1e-9) to guarantee division-by-zero protection, 
    # then annualizes the outcome using the square root of standard yearly trading days (252).
    sharpe = (
        data['Strategy_Return'].mean()
        /
        (data['Strategy_Return'].std()+1e-9)
    ) * np.sqrt(252)

    # ==========================================
    # PEAK-TO-TROUGH RISK PROFILING (DRAWDOWN)
    # ==========================================
    # Locates historical capital degradation intervals by isolating running maximum equity peaks.
    roll_max = data[
        'Strategy_Equity'
    ].cummax()

    # Tracks active percentage capital variances from local trailing peak limits
    drawdown = (
        data['Strategy_Equity']
        /
        roll_max
        -1
    )

    # Extracts the single absolute worst-case downside valuation drop encountered during evaluation
    max_drawdown = drawdown.min()

    # ==========================================
    # SUMMARY STRATEGY KPI PACKAGING
    # ==========================================
    metrics = {

      'strategy_return_pct':
      (data[
      'Strategy_Equity'
      ].iloc[-1]-1)*100,

      'buyhold_return_pct':
      (data[
      'BuyHold_Equity'
      ].iloc[-1]-1)*100,

      'sharpe_ratio':
      sharpe,

      'max_drawdown_pct':
      max_drawdown*100

    }

    return data, metrics