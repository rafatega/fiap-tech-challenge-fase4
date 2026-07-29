"""Busca pequena de hiperparâmetros para o LSTM.

Metodologia deliberada: cada candidato é avaliado em MAE/RMSE/MAPE na
VALIDAÇÃO, nunca no teste, para não contaminar o conjunto de teste como
critério de seleção de modelo. O teste só deve ser tocado uma única vez,
depois que a configuração vencedora já estiver decidida (fora deste script,
em train.py). Todas as execuções são registradas no MLflow, num experimento
separado do treino de produção, para comparação.
"""
from __future__ import annotations

import json

import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from stock_predictor import baseline, config
from stock_predictor.data import Datasets, build_datasets
from stock_predictor.model import LSTMRegressor

# Configuração atual de produção listada primeiro, como referência; as demais
# variam um eixo por vez a partir dela (contexto temporal, capacidade do
# modelo, regularização e taxa de aprendizado).
CANDIDATES: list[dict] = [
    dict(sequence_length=30, hidden_size=16, num_layers=1, learning_rate=1e-3, dropout=0.0),
    dict(sequence_length=20, hidden_size=16, num_layers=1, learning_rate=1e-3, dropout=0.0),
    dict(sequence_length=45, hidden_size=16, num_layers=1, learning_rate=1e-3, dropout=0.0),
    dict(sequence_length=60, hidden_size=16, num_layers=1, learning_rate=1e-3, dropout=0.0),
    dict(sequence_length=30, hidden_size=8, num_layers=1, learning_rate=1e-3, dropout=0.0),
    dict(sequence_length=30, hidden_size=32, num_layers=1, learning_rate=1e-3, dropout=0.0),
    dict(sequence_length=30, hidden_size=16, num_layers=1, learning_rate=1e-3, dropout=0.2),
    dict(sequence_length=30, hidden_size=16, num_layers=2, learning_rate=1e-3, dropout=0.2),
    dict(sequence_length=30, hidden_size=16, num_layers=1, learning_rate=5e-4, dropout=0.0),
]


def _to_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_one(cand: dict, datasets: Datasets) -> dict:
    pl.seed_everything(config.RANDOM_SEED, workers=True)

    train_loader = _to_loader(datasets.train.X, datasets.train.y, config.BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(datasets.val.X, datasets.val.y, config.BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(
        hidden_size=cand["hidden_size"],
        num_layers=cand["num_layers"],
        learning_rate=cand["learning_rate"],
        dropout=cand["dropout"],
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE
    )
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping],
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    if not checkpoint_callback.best_model_path:
        raise RuntimeError(f"Candidato {cand} não produziu checkpoint (treino abortou cedo demais).")

    best_model = LSTMRegressor.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()

    with torch.no_grad():
        val_pred_scaled = best_model(torch.from_numpy(datasets.val.X)).numpy()
    val_pred_log_return = datasets.scaler.inverse_transform(val_pred_scaled)

    return baseline.evaluate_price_predictions(
        datasets.val.prev_close, datasets.val.true_close, val_pred_log_return
    )


def run_sweep(candidates: list[dict] = CANDIDATES) -> dict:
    mlflow.set_experiment("lstm-hp-sweep")
    mlflow.pytorch.autolog()

    datasets_by_seq_len: dict[int, Datasets] = {}
    results = []

    for cand in candidates:
        seq_len = cand["sequence_length"]
        if seq_len not in datasets_by_seq_len:
            datasets_by_seq_len[seq_len] = build_datasets(sequence_length=seq_len)
        datasets = datasets_by_seq_len[seq_len]

        run_name = (
            f"seq{seq_len}_hid{cand['hidden_size']}_layers{cand['num_layers']}"
            f"_lr{cand['learning_rate']}_drop{cand['dropout']}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cand)
            val_metrics = _train_one(cand, datasets)
            mlflow.log_metrics({f"val_price_{k}": v for k, v in val_metrics.items()})

            naive_val_metrics = baseline.evaluate_price_predictions(
                datasets.val.prev_close,
                datasets.val.true_close,
                baseline.naive_log_return_prediction(len(datasets.val.true_close)),
            )
            mlflow.log_metrics({f"val_naive_{k}": v for k, v in naive_val_metrics.items()})

            results.append(
                {
                    **cand,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mape": val_metrics["mape"],
                    "naive_val_mae": naive_val_metrics["mae"],
                }
            )

    best = min(results, key=lambda r: r["val_mae"])
    return {"results": results, "best": best}


if __name__ == "__main__":
    outcome = run_sweep()
    print(json.dumps(outcome, indent=2))
