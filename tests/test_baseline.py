import numpy as np
import pytest

from stock_predictor.baseline import evaluate_price_predictions, naive_log_return_prediction


def test_naive_log_return_prediction_is_all_zeros():
    result = naive_log_return_prediction(5)
    assert result.shape == (5, 1)
    assert np.all(result == 0.0)


def test_evaluate_price_predictions_perfect_forecast_has_zero_error():
    prev_close = np.array([100.0, 105.0], dtype=np.float32)
    true_close = np.array([105.0, 110.25], dtype=np.float32)
    perfect_log_returns = np.log(true_close / prev_close).reshape(-1, 1).astype(np.float32)

    metrics = evaluate_price_predictions(prev_close, true_close, perfect_log_returns)

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["mape"] == pytest.approx(0.0, abs=1e-3)


def test_evaluate_price_predictions_detects_error():
    prev_close = np.array([100.0], dtype=np.float32)
    true_close = np.array([110.0], dtype=np.float32)
    wrong_log_returns = np.array([[0.0]], dtype=np.float32)  # prevê "sem mudança"

    metrics = evaluate_price_predictions(prev_close, true_close, wrong_log_returns)

    assert metrics["mae"] == pytest.approx(10.0, abs=1e-2)
    assert metrics["mape"] == pytest.approx(100 * 10.0 / 110.0, abs=1e-1)
