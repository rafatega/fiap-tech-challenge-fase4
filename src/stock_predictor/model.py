"""Arquitetura LSTM para previsão de log-retornos, como LightningModule."""
from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError


class LSTMRegressor(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        # Avaliação real do modelo é feita em escala de preço via
        # baseline.evaluate_price_predictions (ver train.py), não aqui —
        # por isso não há test_step/test_mae/test_rmse: trainer.test() não é usado.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = self.dropout(out[:, -1, :])
        return self.head(last_step)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch)
        self.train_mae(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch)
        self.val_mae(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_mae", self.val_mae, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
