# tests/test_api.py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from stock_predictor.api import main
from stock_predictor.model import LSTMRegressor


class _DummyModel(LSTMRegressor):
    """Modelo determinístico (repete o último valor da janela) só para testar a API sem pesos treinados."""

    def forward(self, x):
        return x[:, -1, :]


def _fake_load_artifacts():
    scaler = StandardScaler()
    scaler.fit(np.array([[0.0], [0.01], [-0.01]], dtype=np.float32))
    model = _DummyModel(input_size=1, hidden_size=2, num_layers=1, learning_rate=1e-3)
    return model, scaler, 5


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_artifacts", _fake_load_artifacts)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_requested_number_of_days(client):
    closes = [100.0 + i for i in range(10)]
    response = client.post("/predict", json={"closes": closes, "days_ahead": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["days_ahead"] == 2
    assert len(body["predictions"]) == 2
    assert body["ticker"] == "PETR4.SA"


def test_predict_rejects_insufficient_history(client):
    response = client.post("/predict", json={"closes": [100.0, 101.0], "days_ahead": 1})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "app_cpu_usage_percent" in response.text


def _failing_load_artifacts():
    raise FileNotFoundError("artefatos não encontrados")


def test_health_reports_model_not_loaded_when_artifacts_missing(monkeypatch):
    monkeypatch.setattr(main, "load_artifacts", _failing_load_artifacts)
    with TestClient(main.app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "load_artifacts", _failing_load_artifacts)
    with TestClient(main.app) as test_client:
        response = test_client.post(
            "/predict", json={"closes": [100.0 + i for i in range(10)], "days_ahead": 1}
        )
        assert response.status_code == 503
