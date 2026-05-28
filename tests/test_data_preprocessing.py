"""
Project: Stock Price Predictor v2
Package: test
Filename: test_data_preprocessing.py
Description: Automated unit test suite for the data preprocessing module. Validates 
             structural dataframe validation rules, left-join time-series sentiment alignment, 
             feature matrix matrix slicing, time-series missing data imputation loops, 
             and chronological index mapping routines.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    merge_stock_sentiment,
    validate_features,
    prepare_features,
    fill_missing_values,
    create_time_index,
    validate_dataframe
)


# ==========================================
# TEST SUITE: DATAFRAME STRUCTURAL VALIDATION
# ==========================================
class TestValidateDataFrame:
    """
    Unit test boundaries evaluating core structural data type checks and validation rules 
    applied to input matrices.
    """

    def test_none_dataframe_raises_error(self):
        """
        Verifies that an uninitialized None object passed to the structural validator 
        properly flags a descriptive ValueError exception.
        """
        with pytest.raises(ValueError, match="cannot be None"):
            validate_dataframe(None, "test_df")

    def test_non_dataframe_raises_error(self):
        """
        Ensures that invalid data types (e.g., native python lists) are blocked 
        by type checking and trigger a clean validation failure.
        """
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            validate_dataframe([1, 2, 3], "test_df")  # type: ignore

    def test_empty_dataframe_raises_error(self):
        """
        Confirms that dataframes missing rows and columns fail shape analysis 
        constraints and trigger a structural error.
        """
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dataframe(df, "test_df")

    def test_valid_dataframe_passes(self, sample_stock_data: pd.DataFrame):
        """
        Validates the positive control execution path where a fully populated, standard 
        OHLCV dataframe cleanly clears the validation layers without exceptions.
        """
        validate_dataframe(sample_stock_data, "test_df")


# ==========================================
# TEST SUITE: MULTI-SOURCE PIPELINE MERGING
# ==========================================
class TestMergeStockSentiment:
    """
    Unit test coverage checking chronological cross-domain data consolidation, left-join 
    feature alignment loops, and null entry fallback allocations.
    """

    def test_none_stock_data_raises_error(self):
        """
        Guarantees that providing a missing stock timeline to the merging pipeline 
        gracefully halts processing with a clear defensive initialization error.
        """
        with pytest.raises(ValueError, match="stock_data cannot be None"):
            merge_stock_sentiment(None, None)

    def test_empty_stock_data_raises_error(self):
        """
        Ensures that blank trading records fail initial alignment routines 
        and throw an explicit tabular empty-state validation error.
        """
        with pytest.raises(ValueError, match="stock_data cannot be empty"):
            merge_stock_sentiment(pd.DataFrame(), None)

    def test_non_dataframe_stock_data_raises_error(self):
        """
        Validates type isolation barriers by ensuring malformed base pricing objects 
        are filtered out before merge processing loops trigger.
        """
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            merge_stock_sentiment([1, 2, 3], None)  # type: ignore

    def test_merge_with_none_sentiment(self, sample_stock_data: pd.DataFrame):
        """
        Tests system behavior when news features are completely unavailable. Verifies that 
        the pipeline safely falls back to filling the data column with standard neutral constants.
        """
        result = merge_stock_sentiment(sample_stock_data, None, fill_sentiment=0.5)

        assert 'Sentiment' in result.columns
        assert result['Sentiment'].iloc[0] == 0.5

    def test_merge_with_empty_sentiment(self, sample_stock_data: pd.DataFrame):
        """
        Confirms that passing a schema-only sentiment dataframe with zero records degrades 
        safely into standard neutral column allocation routines.
        """
        empty_sentiment = pd.DataFrame(columns=['Sentiment'])
        result = merge_stock_sentiment(sample_stock_data, empty_sentiment)

        assert 'Sentiment' in result.columns

    def test_merge_successful(self, sample_stock_data: pd.DataFrame, sample_sentiment_data: pd.DataFrame):
        """
        Evaluates the nominal end-to-end execution path for left joins. Confirms the final index 
        structures align with standard DatetimeIndex parameters and features map correctly.
        """
        sentiment = sample_sentiment_data.iloc[:50]

        result = merge_stock_sentiment(sample_stock_data, sentiment)

        assert 'Sentiment' in result.columns
        assert len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_merge_drops_nan(self, sample_stock_data: pd.DataFrame, sample_sentiment_data: pd.DataFrame):
        """
        Verifies row truncation behavior during merging. Confirms that rows containing residual 
        NaN attributes are successfully dropped to maintain continuous matrices for model ingestion.
        """
        sentiment = sample_sentiment_data.iloc[:50]

        initial_count = len(sample_stock_data)
        result = merge_stock_sentiment(sample_stock_data, sentiment)

        assert len(result) <= initial_count


# ==========================================
# TEST SUITE: SCHEMA ALIGNMENT VALIDATION
# ==========================================
class TestValidateFeatures:
    """
    Unit test coverage checking tracking schemas and verifying column arrays 
    match configuration settings specifications.
    """

    def test_empty_data_raises_error(self):
        """
        Ensures that executing column schema validations against unpopulated, empty 
        data tables throws a clear descriptive ValueError.
        """
        with pytest.raises(ValueError, match="data cannot be empty"):
            validate_features(pd.DataFrame(), ['feature1'])

    def test_empty_features_raises_error(self, sample_stock_data: pd.DataFrame):
        """
        Validates baseline defense checks by ensuring passing a completely empty column 
        target list throws a clear structural error.
        """
        with pytest.raises(ValueError, match="required_features cannot be empty"):
            validate_features(sample_stock_data, [])

    def test_non_list_features_raises_error(self, sample_stock_data: pd.DataFrame):
        """
        Checks datatype enforcement routines to confirm feature column parameters are 
        strictly formatted as iterable string lists.
        """
        with pytest.raises(ValueError, match="must be a list"):
            validate_features(sample_stock_data, "feature1")  # type: ignore

    def test_all_features_present(self, sample_stock_data: pd.DataFrame):
        """
        Tests positive condition handling by verifying that when all requested columns exist, 
        the validation flag returns True with an empty array of missing items.
        """
        is_valid, missing = validate_features(sample_stock_data, ['Close', 'Open', 'High'])

        assert is_valid is True
        assert missing == []

    def test_missing_features(self, sample_stock_data: pd.DataFrame):
        """
        Evaluates negative condition handling. Confirms that missing data coordinates 
        properly lower the validation flag and isolate the exact missing column strings.
        """
        is_valid, missing = validate_features(sample_stock_data, ['Close', 'NonExistent'])

        assert is_valid is False
        assert 'NonExistent' in missing


# ==========================================
# TEST SUITE: MATRIX MATRIX SLICING
# ==========================================
class TestPrepareFeatures:
    """
    Unit test coverage validating split boundaries partitioning arrays down into 
    independent feature vector spaces X and dependent sequence matrices y.
    """

    def test_missing_features_raises_error(self, sample_stock_data: pd.DataFrame):
        """
        Verifies that attempting to slice independent variable dimensions from non-existent 
        columns successfully catches the fault and terminates execution.
        """
        with pytest.raises(ValueError, match="Missing features"):
            prepare_features(sample_stock_data, ['Close', 'NonExistent'], 'Close')

    def test_missing_target_raises_error(self, sample_stock_data: pd.DataFrame):
        """
        Ensures that targeting a missing dependent pricing column label catches the 
        fault and throws a clear target-not-found ValueError.
        """
        with pytest.raises(ValueError, match="not found in data"):
            prepare_features(sample_stock_data, ['Close'], 'NonExistent')

    def test_returns_correct_structure(self, sample_stock_data: pd.DataFrame):
        """
        Checks that feature matrix generation accurately outputs aligned pandas data structure 
        types alongside structural tracking string schemas.
        """
        X, y, features = prepare_features(sample_stock_data, ['Close', 'Volume'], 'Close')

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(features, list)

    def test_feature_columns_correct(self, sample_stock_data: pd.DataFrame):
        """
        Confirms that the sliced independent matrix X maps exactly to the requested feature dimensions, 
        ensuring zero leakage from out-of-bounds parameters.
        """
        X, y, features = prepare_features(sample_stock_data, ['Close', 'Volume'], 'Close')

        assert list(X.columns) == ['Close', 'Volume']

    def test_target_column_correct(self, sample_stock_data: pd.DataFrame):
        """
        Confirms that the sliced target variable y matches the explicit objective variable settings, 
        maintaining exact tracking coordinates for model optimization.
        """
        X, y, features = prepare_features(sample_stock_data, ['Close'], 'Close')

        assert list(y.columns) == ['Close']


# ==========================================
# TEST SUITE: MISSING DATA IMPUTATION LAYER
# ==========================================
class TestFillMissingValues:
    """
    Unit test coverage benchmarking sequence forward/backward propagation fills and data 
    truncation routines used to resolve matrix holes.
    """

    def test_none_data_raises_error(self):
        """
        Verifies that passing uninitialized objects to data imputation processors throws 
        a clear descriptive initialization error.
        """
        with pytest.raises(ValueError, match="data cannot be None"):
            fill_missing_values(None)

    def test_empty_dataframe_returns_as_is(self):
        """
        Ensures that blank data matrices pass through the filling operations untouched 
        without causing structural logic crashes.
        """
        df = pd.DataFrame()
        result = fill_missing_values(df)
        assert result.empty

    def test_invalid_method_uses_default(self, sample_stock_data: pd.DataFrame):
        """
        Confirms system fault-containment behavior: passing an unrecognized string option 
        automatically defaults to standard forward-then-backward tracking strategies.
        """
        df = sample_stock_data.copy()
        df.iloc[0, 0] = np.nan

        result = fill_missing_values(df, fill_method='invalid_method')

        assert not pd.isna(result.iloc[0, 0])

    def test_ffill_method(self):
        """
        Validates the strict forward-fill ('ffill') algorithm track, ensuring that historical values 
        are properly propagated forward into subsequent empty array slots.
        """
        df = pd.DataFrame({'value': [1, np.nan, np.nan, 4, np.nan]})

        result = fill_missing_values(df, fill_method='ffill')

        assert result['value'].iloc[1] == 1
        assert result['value'].iloc[2] == 1
        assert result['value'].iloc[3] == 4

    def test_bfill_method(self):
        """
        Validates the strict backward-fill ('bfill') algorithm track, ensuring that future values 
        are properly back-propagated into preceding empty array slots.
        """
        df = pd.DataFrame({'value': [np.nan, 1, np.nan, np.nan, 4]})

        result = fill_missing_values(df, fill_method='bfill')

        assert result['value'].iloc[0] == 1
        assert result['value'].iloc[2] == 4

    def test_drop_method(self):
        """
        Validates row truncation dropping routines, verifying that all rows containing null entries 
        are successfully purged from the resulting data frame structure.
        """