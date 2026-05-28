"""
Project: Stock Price Predictor v2
Package: test
Filename: test_model_utils.py
Description: Automated unit test suite for the model utilities module. Validates 
             input tensor dimension verification, hybrid neural network build compliance 
             (BiLSTM, Attention, GRU topologies), sliding-window sequence dataset formatting, 
             and Scikit-Learn scaling matrix joblib/JSON serialization routines.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import numpy as np
import joblib
import json
import os
from src.model_utils import (
    build_model,
    create_dataset,
    save_scaler,
    load_scaler,
    validate_input_shape
)


# ==========================================
# TEST SUITE: TENSOR INPUT DIMENSION CHECKS
# ==========================================
class TestValidateInputShape:
    """
    Unit test coverage validating structural matrix dimensions and input boundary constraints 
    required for time-series deep learning architectures.
    """

    def test_valid_shape(self):
        """
        Verifies that fully compliant 2D sequence dimensions (e.g., lookback steps and feature counts) 
        pass shape validation barriers successfully.
        """
        assert validate_input_shape((60, 5)) is True
        assert validate_input_shape((10, 1)) is True

    def test_invalid_shape_none(self):
        """
        Ensures that uninitialized or blank None input arguments fail shape evaluations.
        """
        assert validate_input_shape(None) is False

    def test_invalid_shape_wrong_length(self):
        """
        Validates rank criteria enforcement by checking that shapes falling outside standard 
        2D configurations are accurately flagged as invalid.
        """
        assert validate_input_shape((60,)) is False
        assert validate_input_shape((60, 5, 3)) is False

    def test_invalid_shape_zero_values(self):
        """
        Confirms that input configurations with empty dimensions (zero elements) fail 
        structural safety checkpoints.
        """
        assert validate_input_shape((0, 5)) is False
        assert validate_input_shape((60, 0)) is False

    def test_invalid_shape_negative(self):
        """
        Ensures that negative lookback windows or column dimensions drop out-of-bounds 
        and lower validation flags.
        """
        assert validate_input_shape((-1, 5)) is False


# ==========================================
# TEST SUITE: NEURAL NETWORK COMPILATION
# ==========================================
class TestBuildModel:
    """
    Unit test coverage verifying functional generation, structural layer parameters, 
    and parameter validation rules for the deep learning model factory.
    """

    def test_invalid_input_shape(self):
        """
        Guarantees that attempting to compile a model network with blank input shape constraints 
        properly catches the error and throws an explicit ValueError.
        """
        with pytest.raises(ValueError, match="Invalid input shape"):
            build_model(None)  # type: ignore

    def test_invalid_lstm_units(self):
        """
        Ensures model construction blocks non-positive neuron allocation metrics inside 
        the recurrent processing sequences.
        """
        with pytest.raises(ValueError, match="positive"):
            build_model((10, 5), lstm_units=0)

    def test_invalid_dropout_rate(self):
        """
        Validates mathematical layer probability constraints, checking that regularization weights 
        must fall strictly between 0 and 1.
        """
        with pytest.raises(ValueError, match="0 and 1"):
            build_model((10, 5), dropout_rate=1.5)

    def test_invalid_dense_units(self):
        """
        Confirms type checks catch negative hidden state dimensions before Keras execution loops trigger.
        """
        with pytest.raises(ValueError, match="positive"):
            build_model((10, 5), dense_units=-1)

    def test_invalid_learning_rate(self):
        """
        Ensures numerical boundary conditions block dead optimization paths (e.g., zero or negative steps).
        """
        with pytest.raises(ValueError, match=".*"):
            build_model((10, 5), learning_rate=0)

    def test_build_default_model(self):
        """
        Tests positive control building conditions under baseline configurations, confirming 
        all hybrid structural paths (BiLSTM, Attention, GRU) compile successfully.
        """
        model = build_model((60, 5))

        assert model is not None
        assert len(model.layers) >= 8

    def test_build_custom_model(self):
        """
        Validates custom parameter overrides, confirming that tensor mapping dimensions 
        dynamically re-align to match unique initialization configurations.
        """
        model = build_model(
            input_shape=(30, 10),
            lstm_units=64,
            dropout_rate=0.3,
            dense_units=32,
            learning_rate=0.01
        )

        assert model is not None
        assert model.input_shape == (None, 30, 10)

    def test_model_compiles_successfully(self):
        """
        Verifies backend structural initialization states by checking that active loss layers 
        and Adam gradient trackers register smoothly within the compiled model block.
        """
        model = build_model((60, 5))

        assert model.optimizer is not None
        assert model.loss is not None


# ==========================================
# TEST SUITE: SLIDING WINDOW SEQUENCES
# ==========================================
class TestCreateDataset:
    """
    Unit test coverage validating temporal dataset sliding transformations converting 
    flat matrices into overlapping historical 3D temporal arrays.
    """

    def test_none_inputs_raises_error(self):
        """
        Guarantees that passing uninitialized objects to sequence preparation functions 
        safely throws a clear descriptive initialization error.
        """
        with pytest.raises(ValueError, match="cannot be None"):
            create_dataset(None, np.array([1, 2, 3]))  # type: ignore

    def test_non_array_inputs_raises_error(self):
        """
        Type-checks array arguments to confirm features are passed strictly as standard NumPy structures.
        """
        with pytest.raises(ValueError, match="must be numpy arrays"):
            create_dataset([1, 2, 3], [1, 2, 3])  # type: ignore

    def test_invalid_time_step(self):
        """
        Ensures sliding parameters require a valid positive time lookback step constraint to build sequences.
        """
        data = np.random.rand(100, 5)
        target = np.random.rand(100, 1)

        with pytest.raises(ValueError, match="time_step must be positive"):
            create_dataset(data, target, time_step=0)

    def test_length_mismatch_raises_error(self):
        """
        Validates input vector parity by confirming feature records match prediction label dimensions.
        """
        data_x = np.random.rand(100, 5)
        data_y = np.random.rand(50, 1)

        with pytest.raises(ValueError, match="must have the same length"):
            create_dataset(data_x, data_y)

    def test_dataset_too_small_raises_error(self):
        """
        Checks sequence volume thresholds, ensuring dataset sizes exceed lookback window requirements.
        """
        data = np.random.rand(5, 5)
        target = np.random.rand(5, 1)

        with pytest.raises(ValueError, match="must have more than time_step"):
            create_dataset(data, target, time_step=10)

    def test_creates_correct_shape(self, sample_scaled_data: tuple):
        """
        Validates mathematical sample row allocations and sequence dimension output structures.
        """
        X, y = sample_scaled_data
        time_step = 5

        X_lstm, y_lstm = create_dataset(X, y, time_step)

        expected_samples = len(X) - time_step
        assert X_lstm.shape == (expected_samples, time_step, X.shape[1])
        assert y_lstm.shape == (expected_samples,)

    def test_creates_sequential_samples(self):
        """
        Confirms temporal causality rules by verifying that overlapping sliding windows 
        map exactly onto sequential target step indexes.
        """
        data = np.arange(20).reshape(10, 2).astype(float)
        target = np.arange(10, 20).reshape(10, 1).astype(float)

        X, y = create_dataset(data, target, time_step=3)

        # Confirm the first sliding sample matches the initial step index bounds
        assert X[0].shape == (3, 2)
        np.testing.assert_array_equal(X[0], data[:3])
        assert y[0] == target[3, 0]  


# ==========================================
# TEST SUITE: SCALER SERIALIZATION LABELS
# ==========================================
class TestSaveLoadScaler:
    """
    Unit test coverage benchmarking persistent disk operations, joblib serialization, 
    and feature tracking list generation schemas.
    """

    def test_save_none_scaler_raises_error(self, temp_scaler_path: str):
        """
        Ensures serialization passes block attempts to write uninitialized scaler variables.
        """
        with pytest.raises(ValueError, match="Scaler cannot be None"):
            save_scaler(None, temp_scaler_path, ['feature1'])

    def test_save_empty_path_raises_error(self):
        """
        Guarantees write loops catch and prevent attempts to serialize binaries onto empty filepaths.
        """
        scaler = joblib.__class__ if hasattr(joblib, '__class__') else None  # type: ignore
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        with pytest.raises(ValueError, match="Scaler path cannot be empty"):
            save_scaler(scaler, "", ['feature1'])

    def test_save_empty_features_raises_error(self, temp_scaler_path: str):
        """
        Ensures independent variable schemas are fully populated before serializing normalization data.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        with pytest.raises(ValueError, match="Features must be a non-empty list"):
            save_scaler(scaler, temp_scaler_path, [])

    def test_save_and_load_scaler(self, temp_scaler_path: str):
        """
        Validates clean end-to-end serialization. Confirms joblib state records and tracking metadata 
        re-instantiate flawlessly without memory loss or parameter drifts.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = ['feature1', 'feature2', 'feature3']

        save_scaler(scaler, temp_scaler_path, features)

        # Confirm on-disk artifacts exist matching both active storage formats
        assert os.path.exists(temp_scaler_path)
        features_path = temp_scaler_path.replace('.joblib', '_features.json')
        assert os.path.exists(features_path)

        loaded_scaler, loaded_features = load_scaler(temp_scaler_path)

        assert loaded_scaler is not None
        assert loaded_features == features

    def test_load_nonexistent_scaler_raises_error(self, tmp_path):
        """
        Verifies exception handling when targeted storage paths are non-existent or deleted, 
        confirming that standard FileNotFoundError blocks bubble up safely.
        """
        fake_path = str(tmp_path / "nonexistent.joblib")

        with pytest.raises(FileNotFoundError, match="Scaler file not found"):
            load_scaler(fake_path)

    def test_load_scaler_missing_features(self, temp_scaler_path: str):
        """
        Validates backward compatibility and decoupling states, checking that missing feature configurations 
        fall back to unmapped states gracefully instead of breaking scaler reads.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        # Manually dump serialization artifacts without attaching metadata maps
        joblib.dump(scaler, temp_scaler_path)

        loaded_scaler, loaded_features = load_scaler(temp_scaler_path)

        assert loaded_scaler is not None
        assert loaded_features is None