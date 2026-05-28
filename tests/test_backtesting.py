"""
Project: Stock Price Predictor v2
Package: test
Filename: test_backtesting.py
Description: Unit test suite for the algorithmic backtesting engine. Validates that 
             the simulation pipeline properly ingests price histories, computes strategy 
             returns, and accurately structures performance metrics.
Author: Kartik Kant (AI/ML Engineer)
"""

import pandas as pd
from src.backtesting import run_backtest


def test_backtest_runs():
    """
    Validates the execution loop of the backtesting engine using a deterministic 
    synthetic time-series dataset.
    """

    # Create a minimal mock DataFrame with historical actual closes and model predictions
    df = pd.DataFrame({

      'Close':
      [100,102,104,103,106],

      'Predicted_Close':
      [101,103,105,104,107]

    })

    # Execute the algorithmic backtesting simulator module
    bt,metrics = run_backtest(df)

    # Assert that the strategy return tracking vector is appended to the data layout
    assert 'Strategy_Return' in bt.columns

    # Assert that risk-adjusted performance KPIs (Sharpe ratio) are calculated
    assert 'sharpe_ratio' in metrics

    # Assert that maximum drawdown downside risk boundaries are captured
    assert 'max_drawdown_pct' in metrics