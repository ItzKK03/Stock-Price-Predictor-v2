"""
Project: Stock Price Predictor v2
Filename: app.py
Description: Main user interface and interactive web dashboard built with Streamlit. 
             Integrates data engineering pipelines, sentiment analysis models, and 
             deep learning inference architectures into a live user-facing dashboard.
Author: Kartik Kant (AI/ML Engineer)
Features:
    - Configurable ticker symbol and date range configuration.
    - Real-time prediction tracking with multi-stage progress identifiers.
    - Interactive dark-themed Plotly charts for evaluation.
    - Downloadable time-series prediction metrics and data tables.
    - Robust error recovery blocks with stateful retry functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta, date
from typing import Optional, Any, List, Tuple
import plotly.graph_objects as go

# ==========================================
# REPOSITORY MODULE INGESTION LAYER
# ==========================================
from src.data_collection import get_stock_data, get_stock_news
from src.feature_engineering import add_technical_indicators
from src.sentiment_analysis import initialize_sentiment_model, get_daily_sentiment
from src.model_utils import (
    load_model,
    load_scaler,
    create_dataset
)
from src.data_preprocessing import merge_stock_sentiment, prepare_features
from src.evaluation import evaluate_predictions
from src.logger import get_logger
from src.backtesting import run_backtest

# Centralized global application settings profile configuration
from config import settings

# Initialize centralized logging context for telemetry tracking
logger = get_logger(__name__)


# ==========================================
# GLOBAL STREAMLIT DASHBOARD APPLICATION SETUP
# ==========================================
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT
)

# Render low-overhead styling wrappers to manage element borders and layout alignment
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# APPLICATION MEMORY STATE CONTROL ROUTINES
# ==========================================
def init_session_state() -> None:
    """
    Allocates and registers state schemas inside the Streamlit context memory lifecycle.
    Guarantees structural data persistence across hot-reloads and widget interactions.
    """
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'prediction_date' not in st.session_state:
        st.session_state.prediction_date = None
    if 'last_updated' not in st.session_state:
        st.session_state.last_updated = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False


def clear_session_state() -> None:
    """
    Purges operational cached data parameters and previous runtime exceptions.
    Forces full-system clear to prepare structures for clean inference loops.
    """
    st.session_state.prediction_result = None
    st.session_state.prediction_date = None
    st.session_state.error_message = None


# ==========================================
# PERSISTENT SYSTEM CACHING CONFIGURATIONS
# ==========================================
@st.cache_resource
def load_assets() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[List[str]]]:
    """
    Loads pre-compiled deep learning model binaries, tracking configurations, and scalers.
    Utilizes st.cache_resource to prevent resource-heavy file re-reads across app runs.

    Returns:
        Tuple: (Keras LSTM Model, Feature Input Scaler, Target Scaler, Serialized Feature Names)
    """
    try:
        logger.info("Loading model assets...")
        model = load_model(settings.model_path)
        scaler_x, features = load_scaler(settings.scaler_x_path)
        scaler_y, _ = load_scaler(settings.scaler_y_path)

        # Execution fallback path to parse features schema if binary profiles lack layout names
        if features is None:
            if not os.path.exists(settings.scaler_features_path):
                logger.error(f"Features file not found: {settings.scaler_features_path}")
                return None, None, None, None
            with open(settings.scaler_features_path, 'r') as f:
                features = json.load(f)

        logger.info("Model assets loaded successfully")
        return model, scaler_x, scaler_y, features

    except Exception as e:
        logger.error(f"Error loading model assets: {e}")
        return None, None, None, None


@st.cache_resource
def load_sentiment_pipeline() -> Any:
    """
    Maintains the initialized natural language transformer pipeline instance in application memory.
    Prevents parallel resource loading cycles during high-frequency API updates.

    Returns:
        HuggingFace pipeline: Active FinBERT network processing instance.
    """
    logger.info("Loading sentiment pipeline...")
    return initialize_sentiment_model()


# ==========================================
# TIME-SERIES DEEP LEARNING INFERENCE ENGINE
# ==========================================
def run_prediction(
    ticker: str,
    model: Any,
    scaler_x: Any,
    scaler_y: Any,
    features: List[str],
    time_step: int,
    buffer_days: int
) -> Tuple[Optional[float], Optional[pd.DataFrame], Optional[str]]:
    """
    Coordinates raw asset ingestion, parameter preprocessing, news analysis tracking, 
    and multi-dimensional tensor formatting required for sequential LSTM prediction.

    Args:
        ticker (str): Asset code identifier string.
        model (Any): Compiled LSTM network instance.
        scaler_x (Any): Pre-fitted input matrix normalization class.
        scaler_y (Any): Pre-fitted scalar processing close target instance.
        features (List[str]): Data schema matching input properties.
        time_step (int): Mathematical sliding sequence range used by the model.
        buffer_days (int): Extra timeline overhead tracking for rolling statistical metrics.

    Returns:
        Tuple: (Unscaled close value projections, Preprocessed dataframes, Context error text blocks)
    """
    try:
        end_date = date.today()
        
        # Converts required sequence days into physical calendar ranges using a 1.5 multiplier.
        # This prevents calculation crashes caused by market dormancy intervals and weekend gaps.
        calendar_days_needed = int((time_step + buffer_days) * 1.5)
        start_date = end_date - timedelta(days=calendar_days_needed)

        # Ingest sequential historical price arrays from historical endpoint configurations
        stock_data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        # Map technical metrics (RSI levels, simple and exponential moving configurations)
        stock_data = add_technical_indicators(stock_data)

        # Isolate news parameters and compute weighted sentiment rankings
        try:
            sentiment_pipeline = load_sentiment_pipeline()
            daily_sentiment = get_daily_sentiment(ticker, sentiment_pipeline)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}, proceeding without sentiment")
            daily_sentiment = None

        # Consolidate historical metrics and text properties into unified computational matrices
        data = merge_stock_sentiment(stock_data, daily_sentiment)

        # Resolve processing blanks by executing data frame sequence propagation loops
        data = data.ffill().bfill()

        # Isolate final historical lookback window slice matching sequence metrics requirements
        last_n_days = data[features].tail(time_step)

        if last_n_days.shape[0] < time_step:
            return None, data, f"Insufficient data: got {last_n_days.shape[0]} days, need {time_step}"

        # Transform vector scaling constraints to align elements within standard parameters
        last_n_days_scaled = scaler_x.transform(last_n_days)

        # Format 2D matrices into 3D input tensor dimensions: [Samples, Sequence Time Steps, Features Count]
        X_test = np.array([last_n_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

        # Trigger model prediction processing and reverse output magnitudes back to raw valuations
        predicted_price_scaled = model.predict(X_test, verbose=0)
        predicted_price = scaler_y.inverse_transform(predicted_price_scaled)

        return predicted_price[0][0], data, None

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None, None, str(e)


# ==========================================
# GEOMETRIC DATA VISUALIZATION FUNCTIONS
# ==========================================
def plot_prediction_chart(
    history_df: pd.DataFrame,
    predicted_price: float,
    ticker: str,
    history_days: int
) -> go.Figure:
    """
    Constructs an interactive graphical representation displaying recent historical trajectories 
    alongside next-period deep learning predictions.

    Args:
        history_df (pd.DataFrame): Data frame structure holding indexed historical close logs.
        predicted_price (float): Predicted evaluation target point.
        ticker (str): Target stock label representation string.
        history_days (int): Slice configuration setting the layout visibility scale.

    Returns:
        go.Figure: Fully styled interactive dark-themed Plotly line element map.
    """
    logger.debug(f"Creating prediction chart for {ticker}")

    # Slice out display interval matching active configuration properties
    history = history_df.tail(history_days).copy()

    # Append +1 period timeline offset to chart the target model projection accurately
    prediction_date = history.index.max() + pd.Timedelta(days=1)

    # Frame temporary data matrices to isolate chart vector traces cleanly
    pred_df = pd.DataFrame({
        'Date': [prediction_date],
        'Close': [predicted_price],
        'Type': ['Prediction']
    })

    history['Type'] = 'Historical'

    # Concatenate structures to compile a flat plotting data array configuration
    plot_df = pd.concat([history.reset_index(), pred_df], ignore_index=True)

    fig = go.Figure()

    # Trace 1: Draw lines representing underlying historical baseline closing prices
    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['Type'] == 'Historical']['Date'],
        y=plot_df[plot_df['Type'] == 'Historical']['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#1f77b4', width=2)
    ))

    # Trace 2: Draw isolated star node markers to track inference targets
    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['Type'] == 'Prediction']['Date'],
        y=plot_df[plot_df['Type'] == 'Prediction']['Close'],
        mode='markers+lines',
        name='Predicted Close',
        marker=dict(color='#ff7f0e', size=15, symbol='star'),
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    # Apply configuration parameters defining visual metrics and theme details
    fig.update_layout(
        title=f'{ticker} Closing Price: Last {history_days} Days & Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        legend_title_text='Data Type',
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig


def create_download_dataframe(
    data: pd.DataFrame,
    prediction: float,
    ticker: str
) -> pd.DataFrame:
    """
    Generates a system-compiled tabular architecture ready for download operations.

    Args:
        data (pd.DataFrame): Primary baseline context history file records.
        prediction (float): Projected target system price parameter.
        ticker (str): Asset identifier name.

    Returns:
        pd.DataFrame: Consolidated data structures containing metadata timestamps.
    """
    history = data.tail(60).copy()
    prediction_date = history.index.max() + pd.Timedelta(days=1)

    pred_row = pd.DataFrame({
        'Date': [prediction_date],
        'Close': [prediction],
        'Type': ['Prediction']
    })

    history['Type'] = 'Historical'
    result = pd.concat([history.reset_index(), pred_row], ignore_index=True)
    result['Ticker'] = ticker
    result['GeneratedAt'] = date.today().isoformat()

    return result


# ==========================================
# SIDEBAR CONTROL PANEL CONFIGURATIONS
# ==========================================
def render_sidebar() -> Tuple[str, int, int, int]:
    """
    Renders input control widgets to register configuration options.

    Returns:
        Tuple: (Target Asset String, Sequence Sequence Length, Display Range, Processing Windows)
    """
    st.sidebar.header("⚙️ Configuration")

    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter the stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
    )

    st.sidebar.divider()

    st.sidebar.subheader("Model Parameters")

    time_step = st.sidebar.slider(
        "Time Steps (Days of History)",
        min_value=10,
        max_value=120,
        value=60,
        step=5,
        help="Number of days of historical data to use for each prediction"
    )

    buffer_days = st.sidebar.slider(
        "Buffer Days (for indicators)",
        min_value=30,
        max_value=200,
        value=100,
        step=10,
        help="Extra days to fetch for calculating technical indicators"
    )

    st.sidebar.divider()

    st.sidebar.subheader("Display Options")

    history_days = st.sidebar.slider(
        "History Display (Days)",
        min_value=30,
        max_value=365,
        value=90,
        step=15,
        help="Number of days to display in the chart"
    )

    st.sidebar.divider()

    st.sidebar.info(
        """
        **How it works:**
        1. Fetches historical stock data
        2. Calculates technical indicators (RSI, MA20, MA50)
        3. Analyzes news sentiment using FinBERT
        4. Uses LSTM model to predict next day's price

        *Note: This is not financial advice.*
        """
    )

    return ticker, time_step, history_days, buffer_days


# ==========================================
# MAIN APPLICATION MANAGEMENT LOOP
# ==========================================
def main() -> None:
    """
    Coordinates visual component layouts, checks memory structures, manages execution buttons, 
    and handles modular navigation tabs.
    """
    init_session_state()

    ticker, time_step, history_days, buffer_days = render_sidebar()

    st.markdown(f'<p class="main-header">📈 AI Stock Price Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        f"Predicting closing price for **{ticker.upper()}** using LSTM deep learning, "
        "technical indicators, and news sentiment analysis."
    )

    with st.spinner("Loading model assets..."):
        model, scaler_x, scaler_y, features = load_assets()

    if model is None or scaler_x is None or scaler_y is None or features is None:
        st.error("❌ Model assets could not be loaded.")
        st.info(
            f"Please ensure the following files exist:\n\n"
            f"- `{settings.model_path}`\n"
            f"- `{settings.scaler_x_path}`\n"
            f"- `{settings.scaler_y_path}`\n"
            f"- `{settings.scaler_features_path}`\n\n"
            f"Run `python train_model.py` to train the model first."
        )
        return

    st.success("✅ Model loaded successfully!")

    st.divider()
    st.subheader("🔮 Prediction")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        predict_button = st.button(
            "🚀 Run Prediction",
            type="primary",
            use_container_width=True,
            help="Fetch latest data and run prediction"
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear Results",
            type="secondary",
            use_container_width=True,
            help="Clear current prediction results"
        )

    if clear_button:
        clear_session_state()
        st.rerun()

    if predict_button:
        st.session_state.is_loading = True

        with st.status(f"Running prediction for {ticker.upper()}...", expanded=True) as status:

            step1 = status.container()
            step1.write("📊 **Step 1/4:** Fetching historical data...")

            step2 = status.container()
            step2.write("📈 **Step 2/4:** Calculating technical indicators...")

            step3 = status.container()
            step3.write("📰 **Step 3/4:** Analyzing news sentiment...")

            step4 = status.container()
            step4.write("🤖 **Step 4/4:** Running LSTM prediction...")

            prediction, data, error = run_prediction(
                ticker.upper(),
                model,
                scaler_x,
                scaler_y,
                features,
                time_step,
                buffer_days
            )

            if error:
                status.update(label="❌ Prediction failed", state="error", expanded=True)
                st.session_state.error_message = error
                st.session_state.is_loading = False
                st.error(f"**Error:** {error}")
            else:
                status.update(label="✅ Prediction complete!", state="complete", expanded=True)
                st.session_state.prediction_result = {
                    'prediction': prediction,
                    'data': data,
                    'ticker': ticker.upper()
                }
                st.session_state.prediction_date = date.today()
                st.session_state.is_loading = False
                st.rerun()

    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        prediction_val = result['prediction']
        data = result['data']
        current_ticker = result['ticker']

        last_actual_price = data['Close'].iloc[-1]
        price_change = prediction_val - last_actual_price
        price_change_pct = (price_change / last_actual_price) * 100

        st.divider()
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"Predicted Price ({current_ticker})",
                value=f"${prediction_val:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )

        with col2:
            st.metric(
                label="Last Close Price",
                value=f"${last_actual_price:.2f}"
            )

        with col3:
            st.metric(
                label="Prediction Date",
                value=st.session_state.prediction_date.strftime("%Y-%m-%d") if st.session_state.prediction_date else "N/A"
            )

        st.markdown("""
        #### 🧠 How This Prediction Works
        This prediction is generated by an **LSTM (Long Short-Term Memory)** neural network that analyzes:
        1. **Price History** - The last 60 days of stock prices and volume
        2. **Technical Indicators** - RSI, 20-day MA, and 50-day moving averages
        3. **News Sentiment** - Sentiment scores from recent financial news (FinBERT model)

        > ⚠️ **Disclaimer:** This is an AI-generated prediction for educational purposes only.
        > Do not use as the sole basis for investment decisions.
        """)

        st.divider()
        st.subheader("📈 Price Chart")
        fig = plot_prediction_chart(data, prediction_val, current_ticker, history_days)
        
        # Segment UI outputs into separate evaluation and testing views
        forecast_tab, backtest_tab = st.tabs(['Forecast', 'Backtesting'])
        
        with forecast_tab:
            st.plotly_chart(fig, use_container_width=True)
            
        with backtest_tab:
            backtest_df = data.tail(250).copy()

            # Generate mathematical moving expectation metrics to feed trading strategies
            backtest_df['Predicted_Close'] = (
                backtest_df['Close'].shift(1) * (1 + backtest_df['Close'].pct_change().rolling(5).mean())
            )

            # Trigger algorithmic performance and metric calculation evaluations
            bt_data, bt_metrics = run_backtest(backtest_df)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric('Strategy %', round(bt_metrics['strategy_return_pct'], 2))
            c2.metric('BuyHold %', round(bt_metrics['buyhold_return_pct'], 2))
            c3.metric('Sharpe', round(bt_metrics['sharpe_ratio'], 2))
            c4.metric('Drawdown %', round(bt_metrics['max_drawdown_pct'], 2))
            
        # Draw risk portfolio comparisons and baseline assets trajectories
        st.line_chart(bt_data[['Strategy_Equity', 'BuyHold_Equity']])

        st.divider()
        st.subheader("📥 Download Results")

        download_df = create_download_dataframe(data, prediction_val, current_ticker)
        csv = download_df.to_csv(index=False)

        st.download_button(
            label="📊 Download Prediction Data (CSV)",
            data=csv,
            file_name=f"{current_ticker}_prediction_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("📋 View Recent Data"):
            st.dataframe(data.tail(10), use_container_width=True)

    elif st.session_state.error_message:
        st.error(f"❌ Previous error: {st.session_state.error_message}")
        st.info("Click 'Run Prediction' to try again.")

    else:
        st.info("👆 Click 'Run Prediction' to get started!")

        st.markdown("""
        ### What You'll See
        Once you run a prediction, you'll see:
        - **Prediction Metric** - Tomorrow's predicted closing price with change vs today
        - **Interactive Chart** - Last 90 days of history + prediction marker
        - **Download Button** - Export prediction data as CSV
        - **Data Table** - Recent data used for the prediction
        """)


if __name__ == "__main__":
    main()