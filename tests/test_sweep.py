import numpy as np

from stock_predictor import config, sweep
from stock_predictor.data import Datasets, SplitData


def _synthetic_split(n: int, sequence_length: int) -> SplitData:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, sequence_length, 1)).astype(np.float32)
    y = (rng.normal(size=(n, 1)) * 0.01).astype(np.float32)
    prev_close = np.full(n, 30.0, dtype=np.float32)
    true_close = prev_close * np.exp(y.flatten())
    return SplitData(X=X, y=y, prev_close=prev_close, true_close=true_close)


def _fake_build_datasets(sequence_length: int) -> Datasets:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(1).normal(size=(50, 1)).astype(np.float32) * 0.01)
    return Datasets(
        train=_synthetic_split(40, sequence_length),
        val=_synthetic_split(10, sequence_length),
        test=_synthetic_split(10, sequence_length),
        scaler=scaler,
    )


def test_run_sweep_picks_best_candidate_by_validation_mae(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "build_datasets", _fake_build_datasets)
    monkeypatch.setattr(config, "MAX_EPOCHS", 1)
    monkeypatch.setattr(config, "EARLY_STOPPING_PATIENCE", 1)
    monkeypatch.chdir(tmp_path)

    candidates = [
        dict(sequence_length=5, hidden_size=4, num_layers=1, learning_rate=1e-3, dropout=0.0),
        dict(sequence_length=5, hidden_size=4, num_layers=1, learning_rate=1e-3, dropout=0.2),
    ]

    outcome = sweep.run_sweep(candidates=candidates)

    assert len(outcome["results"]) == 2
    for result in outcome["results"]:
        assert set(result.keys()) >= {"val_mae", "val_rmse", "val_mape", "naive_val_mae"}
    assert outcome["best"] == min(outcome["results"], key=lambda r: r["val_mae"])


def test_run_sweep_reuses_datasets_for_shared_sequence_length(tmp_path, monkeypatch):
    calls = []

    def _counting_build_datasets(sequence_length: int) -> Datasets:
        calls.append(sequence_length)
        return _fake_build_datasets(sequence_length)

    monkeypatch.setattr(sweep, "build_datasets", _counting_build_datasets)
    monkeypatch.setattr(config, "MAX_EPOCHS", 1)
    monkeypatch.setattr(config, "EARLY_STOPPING_PATIENCE", 1)
    monkeypatch.chdir(tmp_path)

    candidates = [
        dict(sequence_length=5, hidden_size=4, num_layers=1, learning_rate=1e-3, dropout=0.0),
        dict(sequence_length=5, hidden_size=8, num_layers=1, learning_rate=1e-3, dropout=0.0),
    ]

    sweep.run_sweep(candidates=candidates)

    assert calls == [5]
