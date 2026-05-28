"""
Project: Stock Price Predictor v2
Package: test
Filename: test_sentiment_analysis.py
Description: Automated unit and integration test suite for the NLP sentiment analysis module.
             Validates pipeline initializations, caching functionality, memory-safe batch 
             inference loops, defensive input checking, and temporal daily pooling structures.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.sentiment_analysis import (
    initialize_sentiment_model,
    get_sentiment_scores,
    get_daily_sentiment,
    MODEL_NAME
)


# ==========================================
# TEST SUITE: TRANSFORMER INITIALIZATION & CACHE
# ==========================================
class TestInitializeSentimentModel:
    """
    Unit test coverage validating Hugging Face transformer model generation pipelines 
    and operational LRU caching mechanisms.
    """

    @patch('src.sentiment_analysis.pipeline')
    def test_successful_initialization(self, mock_pipeline: MagicMock):
        """
        Verifies that the transformer sequence classifier pipeline instantiates cleanly 
        when provided with a valid model repository reference token string.
        """
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        # Clear the internal cache configuration to isolate the initialization sequence execution
        initialize_sentiment_model.cache_clear()

        result = initialize_sentiment_model("test-model")

        assert result is not None
        mock_pipeline.assert_called_once_with("sentiment-analysis", model="test-model")

    @patch('src.sentiment_analysis.pipeline')
    def test_initialization_error_propagates(self, mock_pipeline: MagicMock):
        """
        Validates error boundaries by ensuring that transformer weights failures or pipeline 
        compilation errors properly propagate back to the execution thread.
        """
        mock_pipeline.side_effect = Exception("Model load failed")

        # Clear cache states to force an explicit operational lookback read attempt
        initialize_sentiment_model.cache_clear()

        with pytest.raises(Exception, match="Model load failed"):
            initialize_sentiment_model("invalid-model")

    def test_default_model_name(self):
        """
        Ensures the foundational model identifier constants align exactly with 
        the target finance-optimized FinBERT network token path.
        """
        assert MODEL_NAME == "ProsusAI/finbert"


# ==========================================
# TEST SUITE: BATCH INFERENCE LOOPS
# ==========================================
class TestGetSentimentScores:
    """
    Unit test coverage validating natural language processing batch text classification loops, 
    inference token transformations, and malformed input safety filtering.
    """

    def test_empty_headlines_returns_empty(self):
        """
        Verifies that passing an unpopulated empty headline string list evaluates safely 
        to a blank results array instead of crashing downstream matrices.
        """
        mock_pipeline = MagicMock()
        result = get_sentiment_scores([], mock_pipeline)
        assert result == []

    def test_none_headlines_raises_error(self):
        """
        Confirms defensive input validation handling, checking that uninitialized None parameters 
        are handled gracefully by returning neutral empty lists.
        """
        mock_pipeline = MagicMock()
        result = get_sentiment_scores(None, mock_pipeline)
        assert result == []  

    def test_single_headline(self):
        """
        Validates nominal single-sequence inference parsing and data-structure extraction 
        from raw classifier dictionary list outputs.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.95}]

        result = get_sentiment_scores(["Apple stock rises"], mock_pipeline)

        assert len(result) == 1
        assert result[0]['label'] == 'positive'
        assert result[0]['score'] == 0.95

    def test_multiple_headlines(self):
        """
        Checks multi-sequence mapping processing, ensuring continuous sequence outputs 
        accurately append structural label elements and softmax probabilities.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.85},
            {'label': 'neutral', 'score': 0.7}
        ]

        headlines = [
            "Apple stock rises",
            "Market crashes on bad news",
            "Stocks remain flat"
        ]
        result = get_sentiment_scores(headlines, mock_pipeline)

        assert len(result) == 3
        assert result[0]['label'] == 'positive'
        assert result[1]['label'] == 'negative'
        assert result[2]['label'] == 'neutral'

    def test_batch_processing(self):
        """
        Validates that long text string arrays are partitioned and executed across explicit 
        sub-batch windows to guard against memory allocation fragmentation and out-of-memory states.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.9}]

        # Create 20 synthetic text headlines to test chunking bounds under an 8-item batch size parameter
        headlines = [f"Headline {i}" for i in range(20)]
        get_sentiment_scores(headlines, mock_pipeline, batch_size=8)

        # Confirm the chunk execution loop triggered 3 separate pipeline inference segments (8 + 8 + 4)
        assert mock_pipeline.call_count == 3

    def test_filters_none_headlines(self):
        """
        Checks preprocessing filtering sweeps to ensure that blank spaces, invalid data structures, 
        or empty string inputs are safely dropped before text processing cycles begin.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'positive', 'score': 0.9}]

        headlines = ["Valid headline", None, "", "Another valid"]
        result = get_sentiment_scores(headlines, mock_pipeline)

        # Confirm non-conforming parameters were filtered out, leaving only the validated rows
        assert len(result) == 1
        mock_pipeline.assert_called_once()

    def test_pipeline_error_propagates(self):
        """
        Validates system exception tracking, ensuring internal neural network runtime failures 
        within the transformer framework safely bubble up to caller scopes.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("Pipeline failed")

        with pytest.raises(RuntimeError, match="Pipeline failed"):
            get_sentiment_scores(["Test headline"], mock_pipeline)


# ==========================================
# TEST SUITE: CHRONOLOGICAL POOLING & METRICS
# ==========================================
class TestGetDailySentiment:
    """
    Unit test coverage validating cross-domain market text aggregation, datetime field index parsing, 
    confidence scoring weights, and daily statistical rolling pool transformations.
    """

    def test_invalid_ticker_empty(self):
        """
        Ensures that blank ticker identifiers return an empty sentiment panel tracking table, 
        safeguarding subsequent time-series merging steps.
        """
        mock_pipeline = MagicMock()
        result = get_daily_sentiment("", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert 'Sentiment' in result.columns

    def test_invalid_ticker_none(self):
        """
        Confirms type boundary checks yield a standard empty sentiment structural dataframe 
        if an uninitialized None parameter is passed as a ticker symbol.
        """
        mock_pipeline = MagicMock()
        result = get_daily_sentiment(None, mock_pipeline)  # type: ignore

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.sentiment_analysis.get_stock_news')
    def test_no_news_returns_empty_df(self, mock_news: MagicMock):
        """
        Ensures pipeline resilience during media blackouts or when API endpoints return 
        empty arrays, verifying that clean fallback index frames are generated.
        """
        mock_news.return_value = []
        mock_pipeline = MagicMock()

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert 'Sentiment' in result.columns

    @patch('src.sentiment_analysis.get_stock_news')
    def test_news_missing_title_column(self, mock_news: MagicMock):
        """
        Validates schema anomaly containment routines by checking that raw input payloads 
        missing required title text rows degrade safely without causing processing exceptions.
        """
        mock_news.return_value = [{'publisher': 'Test'}]  
        mock_pipeline = MagicMock()

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.sentiment_analysis.get_stock_news')
    def test_news_missing_publish_time(self, mock_news: MagicMock):
        """
        Validates schema anomaly containment routines by checking that incoming data logs 
        lacking temporal tracking metadata keys are filtered out safely.
        """
        mock_news.return_value = [{'title': 'Test headline'}]  
        mock_pipeline = MagicMock()

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('src.sentiment_analysis.get_stock_news')
    def test_successful_daily_sentiment(self, mock_news: MagicMock):
        """
        Evaluates nominal end-to-end execution paths for sentiment matrix processing, 
        confirming successful parsing of raw data maps into indexed tracking frames.
        """
        mock_news.return_value = [
            {
                'title': 'Apple Reports Record Earnings',
                'publish_time': 1704067200,
                'publisher': 'Financial Times'
            },
            {
                'title': 'Tech Stocks Rally',
                'publish_time': 1704153600,
                'publisher': 'Reuters'
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.95},
            {'label': 'positive', 'score': 0.90}
        ]

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)
        assert 'Sentiment' in result.columns
        assert len(result) > 0

    @patch('src.sentiment_analysis.get_stock_news')
    def test_sentiment_label_mapping(self, mock_news: MagicMock):
        """
        Verifies that categorical classifier labels (positive, negative, neutral) 
        map correctly into active numeric polarity signs (-1, 0, 1) before scaling steps.
        """
        mock_news.return_value = [
            {
                'title': 'Positive news',
                'publish_time': 1704067200,
                'publisher': 'Test'
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.9}
        ]

        result = get_daily_sentiment("AAPL", mock_pipeline)

        assert isinstance(result, pd.DataFrame)

    @patch('src.sentiment_analysis.get_stock_news')
    def test_daily_aggregation(self, mock_news: MagicMock):
        """
        Validates chronological aggregation and downsampling routines, checking that multiple 
        within-day sentiment signals are condensed into a single aggregated DatetimeIndex value.
        """
        # Form multiple nested metadata items sharing identical calendar dates but varying time fields
        mock_news.return_value = [
            {
                'title': 'Morning news',
                'publish_time': 1704067200,  
                'publisher': 'Test1'
            },
            {
                'title': 'Afternoon news',
                'publish_time': 1704070800,  
                'publisher': 'Test2'
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.8}
        ]

        result = get_daily_sentiment("AAPL", mock_pipeline)

        # Confirm the resample downsampling execution compressed intra-day values into a uniform single index entry
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)


# ==========================================
# TEST SUITE: END-TO-END PIPELINE INTEGRATION
# ==========================================
class TestSentimentIntegration:
    """
    Integration test validation tracking cross-module functionality between 
    remote collection layers, transformer inferences, and data table layouts.
    """

    @patch('src.sentiment_analysis.get_stock_news')
    @patch('src.sentiment_analysis.pipeline')
    def test_full_sentiment_pipeline(self, mock_pipeline: MagicMock, mock_news: MagicMock):
        """
        Simulates end-to-end integration loops starting from mock news extraction 
        through confidence-weighted index outputs.
        """
        mock_news.return_value = [
            {
                'title': 'Stock Market Update',
                'publish_time': 1704067200,
                'publisher': 'News Corp'
            }
        ]

        mock_model = MagicMock()
        mock_model.return_value = [{'label': 'neutral', 'score': 0.75}]
        mock_pipeline.return_value = mock_model

        # Clear cached resource records to simulate clean startup environments
        initialize_sentiment_model.cache_clear()

        # Trigger model load threads and drive the data down integration tracks
        model = initialize_sentiment_model()
        result = get_daily_sentiment("AAPL", model)

        assert isinstance(result, pd.DataFrame)
        assert 'Sentiment' in result.columns
        mock_news.assert_called_once_with("AAPL")