import numpy as np
import pandas as pd
import pytest

from stock_predictor.data import chronological_split, compute_log_returns, make_windows


def test_compute_log_returns_matches_manual_calculation():
    close = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-01", periods=3))
    result = compute_log_returns(close)
    expected_first = np.log(110.0 / 100.0)
    assert result.iloc[0] == pytest.approx(expected_first)
    assert len(result) == 2


def test_make_windows_produces_correct_shapes():
    values = np.arange(10, dtype=np.float32)
    X, y = make_windows(values, sequence_length=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7, 1)
    assert X[0].flatten().tolist() == [0.0, 1.0, 2.0]
    assert y[0].item() == 3.0


def test_chronological_split_respects_fractions():
    train_slice, val_slice, test_slice = chronological_split(100, train_fraction=0.7, val_fraction=0.15)
    assert train_slice == slice(0, 70)
    assert val_slice == slice(70, 85)
    assert test_slice == slice(85, 100)
