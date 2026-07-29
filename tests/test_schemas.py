import pytest
from pydantic import ValidationError

from stock_predictor.api.schemas import PredictRequest


def test_predict_request_accepts_valid_payload():
    request = PredictRequest(closes=[10.0, 10.5, 11.0], days_ahead=3)
    assert request.days_ahead == 3
    assert len(request.closes) == 3


def test_predict_request_defaults_days_ahead_to_one():
    request = PredictRequest(closes=[10.0, 11.0])
    assert request.days_ahead == 1


def test_predict_request_rejects_negative_prices():
    with pytest.raises(ValidationError):
        PredictRequest(closes=[10.0, -1.0], days_ahead=1)


def test_predict_request_rejects_too_few_prices():
    with pytest.raises(ValidationError):
        PredictRequest(closes=[10.0], days_ahead=1)
