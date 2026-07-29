"""Baseline naive (persistence) e avaliação de previsões em escala de preço."""
from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def naive_log_return_prediction(n: int) -> np.ndarray:
    """Baseline ingênua: assume que o log-retorno de amanhã é zero (preço não muda)."""
    return np.zeros((n, 1), dtype=np.float32)


def evaluate_price_predictions(
    prev_close: np.ndarray, true_close: np.ndarray, pred_log_returns: np.ndarray
) -> dict[str, float]:
    """Reconstrói o preço a partir do log-retorno previsto e calcula MAE/RMSE/MAPE em R$."""
    pred_close = prev_close * np.exp(pred_log_returns.flatten())
    true_t = torch.tensor(true_close, dtype=torch.float32)
    pred_t = torch.tensor(pred_close, dtype=torch.float32)
    return {
        "mae": mean_absolute_error(pred_t, true_t).item(),
        "rmse": mean_squared_error(pred_t, true_t, squared=False).item(),
        "mape": mean_absolute_percentage_error(pred_t, true_t).item() * 100,
    }
