"""
Project: Stock Price Predictor v2
Package: test
Filename: test_evaluation.py
Description: Automated unit test suite for the model evaluation module. Validates 
             mathematical regression metrics (MAE, RMSE, MAPE, R²), classification bounds 
             (Directional Accuracy Tracking), and baseline naive comparison models.
Author: Kartik Kant (AI/ML Engineer)
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_directional_accuracy,
    calculate_r2,
    evaluate_predictions,
    generate_evaluation_report,
    create_evaluation_dataframe,
    calculate_improvement_vs_baseline,
    _validate_inputs
)


# ==========================================
# TEST SUITE: VECTOR INPUT VALIDATION
# ==========================================
class TestValidateInputs:
    """
    Unit test boundaries evaluating core vector checking and type handling layers.
    """

    def test_none_inputs_raises_error(self):
        """
        Verifies that passing uninitialized None inputs triggers validation failures.
        """
        with pytest.raises(ValueError, match="cannot be None"):
            _validate_inputs(None, [1, 2, 3])

        with pytest.raises(ValueError, match="cannot be None"):
            _validate_inputs([1, 2, 3], None)

    def test_length_mismatch_raises_error(self):
        """
        Ensures that arrays with asymmetric dimensions fail validation shapes.
        """
        with pytest.raises(ValueError, match="Input length mismatch"):
            _validate_inputs([1, 2, 3], [1, 2])

    def test_valid_inputs_returned_as_arrays(self):
        """
        Confirms that compliant inputs are correctly returned as formatted NumPy arrays.
        """
        y_true, y_pred = _validate_inputs([1, 2, 3], [1.1, 2.1, 3.1])

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        np.testing.assert_array_equal(y_true, [1, 2, 3])


# ==========================================
# TEST SUITE: MEAN ABSOLUTE ERROR (MAE)
# ==========================================
class TestCalculateMAE:
    """
    Unit test coverage checking Mean Absolute Error (MAE) statistical bounds.
    """

    def test_empty_inputs_raises_error(self):
        """
        Verifies that empty sequence inputs raise explicit validation errors.
        """
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_mae([], [])

    def test_perfect_predictions(self):
        """
        Checks that a perfect model match evaluates to an absolute zero error level.
        """
        mae = calculate_mae([1, 2, 3], [1, 2, 3])
        assert mae == 0.0

    def test_constant_error(self):
        """
        Confirms linear error tracking under steady constant variances.
        """
        mae = calculate_mae([1, 2, 3], [2, 3, 4])
        assert mae == 1.0

    def test_negative_values(self):
        """
        Verifies the correct handling of signed negative metrics within vector spaces.
        """
        mae = calculate_mae([-1, -2, -3], [-1, -2, -3])
        assert mae == 0.0


# ==========================================
# TEST SUITE: ROOT MEAN SQUARED ERROR (RMSE)
# ==========================================
class TestCalculateRMSE:
    """
    Unit test coverage checking Root Mean Squared Error (RMSE) constraints.
    """

    def test_empty_inputs_raises_error(self):
        """
        Ensures empty vectors fail validation controls during RMSE extraction steps.
        """
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_rmse([], [])

    def test_perfect_predictions(self):
        """
        Confirms error outputs evaluate to absolute zero on perfect predictions.
        """
        rmse = calculate_rmse([1, 2, 3], [1, 2, 3])
        assert rmse == 0.0

    def test_constant_error(self):
        """
        Checks RMSE outputs remain stable under uniform unit scaling steps.
        """
        rmse = calculate_rmse([1, 2, 3], [2, 3, 4])
        assert rmse == 1.0

    def test_penalizes_large_errors(self):
        """
        Verifies that RMSE disproportionately penalizes outlier variances more than MAE.
        """
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 10]  

        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)

        assert rmse > mae  


# ==========================================
# TEST SUITE: MEAN ABSOLUTE PCT ERROR (MAPE)
# ==========================================
class TestCalculateMAPE:
    """
    Unit test coverage checking Mean Absolute Percentage Error (MAPE) outputs.
    """

    def test_empty_inputs_raises_error(self):
        """
        Ensures blank sequences break early inside error normalization loops.
        """
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_mape([], [])

    def test_perfect_predictions(self):
        """
        Confirms percentage variance resolves to exactly zero for zero-error runs.
        """
        mape = calculate_mape([100, 200, 300], [100, 200, 300])
        assert mape == 0.0

    def test_ten_percent_error(self):
        """
        Validates fractional tracking scaling metrics under fixed 10% errors.
        """
        mape = calculate_mape([100, 100], [110, 110])
        assert mape == pytest.approx(10.0, rel=0.01)

    def test_returns_percentage(self):
        """
        Ensures the calculated metric is output as a standard percentage scalar.
        """
        mape = calculate_mape([100, 200], [105, 210])
        assert mape > 0
        assert mape < 100  


# ==========================================
# TEST SUITE: DIRECTIONAL ACCURACY
# ==========================================
class TestCalculateDirectionalAccuracy:
    """
    Unit test coverage checking binary sequence directional trajectory evaluations.
    """

    def test_insufficient_samples_raises_error(self):
        """
        Confirms dimension thresholds require at least two steps to run differences.
        """
        with pytest.raises(ValueError, match="at least 2 samples"):
            calculate_directional_accuracy([1], [1])

    def test_perfect_direction_prediction(self):
        """
        Validates 100% directional consistency when trends map exactly.
        """
        y_true = [1, 2, 3, 4, 5]  
        y_pred = [1, 2, 3, 4, 5]  

        accuracy = calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 100.0

    def test_wrong_direction(self):
        """
        Validates a complete 0% directional match when sign orientations are inverted.
        """
        y_true = [1, 2, 3, 4, 5]  
        y_pred = [1, 0, -1, -2, -3]  

        accuracy = calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0

    def test_mixed_accuracy(self):
        """
        Evaluates system stability on variable, non-uniform trend movements.
        """
        y_true = [1, 2, 1, 2, 1]  
        y_pred = [1, 2, 1, 0, 1]  

        accuracy = calculate_directional_accuracy(y_true, y_pred)
        assert 0 <= accuracy <= 100


# ==========================================
# TEST SUITE: COEFFICIENT OF DETERMINATION
# ==========================================
class TestCalculateR2:
    """
    Unit test coverage tracking Coefficient of Determination (R²) variance checks.
    """

    def test_insufficient_samples_raises_error(self):
        """
        Ensures sample size conditions are met before computing squares.
        """
        with pytest.raises(ValueError, match="at least 2 samples"):
            calculate_r2([1], [1])

    def test_perfect_fit(self):
        """
        Confirms ideal variance tracking converges exactly to unity.
        """
        r2 = calculate_r2([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert r2 == pytest.approx(1.0, abs=0.001)

    def test_reasonable_fit(self):
        """
        Verifies correlation mappings stay above baseline threshold constraints.
        """
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]

        r2 = calculate_r2(y_true, y_pred)
        assert r2 > 0.9  


# ==========================================
# TEST SUITE: PIPELINE EVALUATION BUNDLER
# ==========================================
class TestEvaluatePredictions:
    """
    Unit test coverage checking the aggregated statistical calculation block.
    """

    def test_returns_all_metrics(self, sample_predictions: tuple):
        """
        Ensures the summary routine packages every registered performance index.
        """
        y_true, y_pred = sample_predictions

        metrics = evaluate_predictions(y_true, y_pred)

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'directional_accuracy' in metrics
        assert 'r2' in metrics

    def test_metrics_are_numeric(self, sample_predictions: tuple):
        """
        Confirms all compiled dictionary outputs are formatted as floating points.
        """
        y_true, y_pred = sample_predictions

        metrics = evaluate_predictions(y_true, y_pred)

        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mape'], float)
        assert isinstance(metrics['directional_accuracy'], float)
        assert isinstance(metrics['r2'], float)


# ==========================================
# TEST SUITE: REPORT GENERATOR FORMATTING
# ==========================================
class TestGenerateEvaluationReport:
    """
    Unit test coverage validating human-readable terminal report formatting blocks.
    """

    def test_report_contains_model_name(self):
        """
        Verifies header metadata parameters map correctly into string report formats.
        """
        metrics = {'mae': 1.0, 'rmse': 2.0, 'mape': 5.0, 'directional_accuracy': 60.0, 'r2': 0.9}

        report = generate_evaluation_report(metrics, model_name="Test Model")

        assert "Test Model" in report

    def test_report_contains_all_metrics(self):
        """
        Ensures every calculated regression and classification field is printed cleanly.
        """
        metrics = {'mae': 1.0, 'rmse': 2.0, 'mape': 5.0, 'directional_accuracy': 60.0, 'r2': 0.9}

        report = generate_evaluation_report(metrics, model_name="Test")

        assert "MAE" in report
        assert "RMSE" in report
        assert "%" in report
        assert (
        "Directional Accuracy" in report
        or "directional_accuracy" in report.lower()
        )
        assert "R-squared" in report or "R²" in report


# ==========================================
# TEST SUITE: TABULAR EVALUATION FRAMING
# ==========================================
class TestCreateEvaluationDataframe:
    """
    Unit test coverage validating tabular error analysis structures.
    """

    def test_creates_correct_columns(self):
        """
        Checks that output comparison dataframes contain the full tracking schema.
        """
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]

        df = create_evaluation_dataframe(y_true, y_pred)

        assert 'Actual' in df.columns
        assert 'Predicted' in df.columns
        assert 'Error' in df.columns
        assert 'Error_%' in df.columns

    def test_error_calculation(self):
        """
        Validates the raw rows calculation arithmetic tracking algebraic offsets.
        """
        y_true = [10, 20, 30]
        y_pred = [12, 22, 32]

        df = create_evaluation_dataframe(y_true, y_pred)

        assert df['Error'].iloc[0] == 2
        assert df['Error'].iloc[1] == 2
        assert df['Error'].iloc[2] == 2

    def test_with_dates_index(self):
        """
        Confirms chronological DatetimeIndex attributes attach correctly onto error rows.
        """
        y_true = [1, 2, 3]
        y_pred = [1.1, 2.1, 3.1]
        dates = pd.date_range('2024-01-01', periods=3)

        df = create_evaluation_dataframe(y_true, y_pred, dates)

        assert isinstance(df.index, pd.DatetimeIndex)


# ==========================================
# TEST SUITE: PERFORMANCE BASELINE LIFT
# ==========================================
class TestCalculateImprovementVsBaseline:
    """
    Unit test coverage benchmarking performance lift ratios against standard baseline loops.
    """

    def test_model_better_than_baseline(self):
        """
        Validates positive lift scaling calculation flags when model errors fall below baselines.
        """
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]  
        baseline = [1, 1, 1, 1, 1]  

        improvement = calculate_improvement_vs_baseline(y_true, y_pred, baseline, 'mae')

        assert improvement is not None
        assert improvement > 0  

    def test_model_worse_than_baseline(self):
        """
        Validates negative lift scaling flags when model metrics underperform targets.
        """
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 1, 1, 1, 1]  
        baseline = [1.1, 2.1, 3.1, 4.1, 5.1]  

        improvement = calculate_improvement_vs_baseline(y_true, y_pred, baseline, 'mae')

        assert improvement is not None
        assert improvement < 0  

    def test_naive_baseline(self):
        """
        Ensures execution threads fall back to shifting true data paths by t-1 step if no baseline is given.
        """
        y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_pred = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  

        improvement = calculate_improvement_vs_baseline(y_true, y_pred, metric='mae')

        assert improvement is not None