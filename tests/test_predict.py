import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from stock_predictor.model import LSTMRegressor
from stock_predictor.predict import predict_future_prices


def _build_scaler() -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.array([[0.0], [0.01], [-0.01], [0.02]], dtype=np.float32))
    return scaler


def test_predict_future_prices_returns_expected_length():
    model = LSTMRegressor(input_size=1, hidden_size=4, num_layers=1, learning_rate=1e-3)
    scaler = _build_scaler()
    sequence_length = 5
    close_prices = [100.0 + i for i in range(sequence_length + 1)]

    predictions = predict_future_prices(
        close_prices, days_ahead=3, model=model, scaler=scaler, sequence_length=sequence_length
    )

    assert len(predictions) == 3
    assert all(p > 0 for p in predictions)


def test_predict_future_prices_raises_with_insufficient_history():
    model = LSTMRegressor(input_size=1, hidden_size=4, num_layers=1, learning_rate=1e-3)
    scaler = _build_scaler()

    with pytest.raises(ValueError, match="pelo menos"):
        predict_future_prices(
            [100.0, 101.0], days_ahead=1, model=model, scaler=scaler, sequence_length=10
        )
