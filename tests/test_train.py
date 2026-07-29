import json

import numpy as np
import pytest

from stock_predictor import config, train
from stock_predictor.data import Datasets, SplitData


def _synthetic_split(n: int, sequence_length: int) -> SplitData:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, sequence_length, 1)).astype(np.float32)
    y = (rng.normal(size=(n, 1)) * 0.01).astype(np.float32)
    prev_close = np.full(n, 30.0, dtype=np.float32)
    true_close = prev_close * np.exp(y.flatten())
    return SplitData(X=X, y=y, prev_close=prev_close, true_close=true_close)


def _fake_build_datasets(sequence_length: int = config.SEQUENCE_LENGTH) -> Datasets:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(1).normal(size=(50, 1)).astype(np.float32) * 0.01)
    return Datasets(
        train=_synthetic_split(40, sequence_length),
        val=_synthetic_split(10, sequence_length),
        test=_synthetic_split(10, sequence_length),
        scaler=scaler,
    )


def test_train_pipeline_produces_artifacts_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "build_datasets", _fake_build_datasets)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "lstm_model.joblib")
    monkeypatch.setattr(config, "SCALER_PATH", tmp_path / "scaler.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "MAX_EPOCHS", 1)
    monkeypatch.setattr(config, "EARLY_STOPPING_PATIENCE", 1)
    monkeypatch.setattr(config, "SEQUENCE_LENGTH", 5)
    monkeypatch.chdir(tmp_path)

    result = train.train()

    assert (tmp_path / "lstm_model.joblib").exists()
    assert (tmp_path / "scaler.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert set(result.keys()) == {"lstm", "naive_baseline"}
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(saved_metrics["lstm"].keys()) == {"mae", "rmse", "mape"}
    assert set(saved_metrics["naive_baseline"].keys()) == {"mae", "rmse", "mape"}
