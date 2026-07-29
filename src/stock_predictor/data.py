"""Coleta (yfinance) e pré-processamento dos dados para o LSTM: log-retornos, janelas e scaler."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from stock_predictor import config


def download_close_prices(ticker: str = config.TICKER, start: str = config.START_DATE) -> pd.Series:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(
            f"Não foi possível obter dados de '{ticker}' via yfinance (resposta vazia ou sem coluna 'Close')."
        )
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "close"
    return close.dropna()


def compute_log_returns(close: pd.Series) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    log_returns.name = "log_return"
    return log_returns.dropna()


def chronological_split(
    length: int,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> tuple[slice, slice, slice]:
    train_end = int(length * train_fraction)
    val_end = int(length * (train_fraction + val_fraction))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, length)


def make_windows(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length])
    X = np.asarray(X, dtype=np.float32).reshape(-1, sequence_length, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    prev_close: np.ndarray
    true_close: np.ndarray


@dataclass
class Datasets:
    train: SplitData
    val: SplitData
    test: SplitData
    scaler: StandardScaler


def build_datasets(sequence_length: int = config.SEQUENCE_LENGTH) -> Datasets:
    close = download_close_prices()
    log_returns = compute_log_returns(close)

    returns_values = log_returns.to_numpy(dtype=np.float32)
    X_raw, y_raw = make_windows(returns_values, sequence_length)
    # y_raw[i] corresponde ao alvo em log_returns.index[i + sequence_length]
    target_index = log_returns.index[sequence_length:]

    prev_close = close.shift(1).loc[target_index].to_numpy(dtype=np.float32)
    true_close = close.loc[target_index].to_numpy(dtype=np.float32)

    train_slice, val_slice, test_slice = chronological_split(len(target_index))

    scaler = StandardScaler()
    scaler.fit(y_raw[train_slice])

    def _scale_X(X: np.ndarray) -> np.ndarray:
        shape = X.shape
        scaled = scaler.transform(X.reshape(-1, 1))
        return scaled.reshape(shape).astype(np.float32)

    def _split_data(sl: slice) -> SplitData:
        return SplitData(
            X=_scale_X(X_raw[sl]),
            y=scaler.transform(y_raw[sl]).astype(np.float32),
            prev_close=prev_close[sl],
            true_close=true_close[sl],
        )

    return Datasets(
        train=_split_data(train_slice),
        val=_split_data(val_slice),
        test=_split_data(test_slice),
        scaler=scaler,
    )
