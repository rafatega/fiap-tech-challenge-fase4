"""Carrega os artefatos treinados (.joblib) e roda inferência multi-step recursiva."""
from __future__ import annotations

import joblib
import numpy as np
import torch

from stock_predictor import config
from stock_predictor.model import LSTMRegressor


def load_artifacts(
    model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH
) -> tuple[LSTMRegressor, object, int]:
    payload = joblib.load(model_path)
    model = LSTMRegressor(**payload["hyperparameters"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = joblib.load(scaler_path)
    sequence_length = payload["sequence_length"]
    return model, scaler, sequence_length


def predict_future_prices(
    close_prices: list[float],
    days_ahead: int,
    model: LSTMRegressor,
    scaler,
    sequence_length: int,
) -> list[float]:
    if len(close_prices) < sequence_length + 1:
        raise ValueError(
            f"É necessário pelo menos {sequence_length + 1} preços históricos, "
            f"recebido {len(close_prices)}."
        )

    prices = list(close_prices)
    log_returns = list(np.diff(np.log(prices)))[-sequence_length:]

    predictions: list[float] = []
    last_price = prices[-1]

    with torch.no_grad():
        for _ in range(days_ahead):
            scaled_window = scaler.transform(np.array(log_returns).reshape(-1, 1)).flatten()
            x = torch.tensor(scaled_window, dtype=torch.float32).reshape(1, sequence_length, 1)
            pred_scaled = model(x).numpy()
            pred_log_return = float(scaler.inverse_transform(pred_scaled).item())

            next_price = last_price * float(np.exp(pred_log_return))
            predictions.append(next_price)

            log_returns = log_returns[1:] + [pred_log_return]
            last_price = next_price

    return predictions
