"""Pipeline de treino: monta os dados, treina o LSTM (Lightning), avalia e exporta os artefatos."""
from __future__ import annotations

import json

import joblib
import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from stock_predictor import baseline, config
from stock_predictor.data import build_datasets
from stock_predictor.model import LSTMRegressor


def _to_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train() -> dict:
    pl.seed_everything(config.RANDOM_SEED, workers=True)

    datasets = build_datasets(sequence_length=config.SEQUENCE_LENGTH)

    train_loader = _to_loader(datasets.train.X, datasets.train.y, config.BATCH_SIZE, shuffle=True)
    val_loader = _to_loader(datasets.val.X, datasets.val.y, config.BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        learning_rate=config.LEARNING_RATE,
        dropout=config.DROPOUT,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.PROJECT_ROOT / "checkpoints"), monitor="val_loss", mode="min", save_top_k=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=config.EARLY_STOPPING_PATIENCE
    )

    mlflow.pytorch.autolog()
    with mlflow.start_run():
        trainer = pl.Trainer(
            max_epochs=config.MAX_EPOCHS,
            callbacks=[checkpoint_callback, early_stopping],
            logger=False,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)

        if not checkpoint_callback.best_model_path:
            raise RuntimeError(
                "Nenhum checkpoint foi salvo — o treino pode ter abortado antes da primeira validação."
            )

        best_model = LSTMRegressor.load_from_checkpoint(checkpoint_callback.best_model_path)
        best_model.eval()

        with torch.no_grad():
            test_pred_scaled = best_model(torch.from_numpy(datasets.test.X)).numpy()

        test_pred_log_return = datasets.scaler.inverse_transform(test_pred_scaled)

        lstm_metrics = baseline.evaluate_price_predictions(
            datasets.test.prev_close, datasets.test.true_close, test_pred_log_return
        )
        naive_pred = baseline.naive_log_return_prediction(len(datasets.test.true_close))
        naive_metrics = baseline.evaluate_price_predictions(
            datasets.test.prev_close, datasets.test.true_close, naive_pred
        )

        mlflow.log_metrics({f"test_price_{k}": v for k, v in lstm_metrics.items()})
        mlflow.log_metrics({f"naive_price_{k}": v for k, v in naive_metrics.items()})

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "state_dict": best_model.state_dict(),
            "hyperparameters": dict(best_model.hparams),
            "sequence_length": config.SEQUENCE_LENGTH,
        },
        config.MODEL_PATH,
    )
    joblib.dump(datasets.scaler, config.SCALER_PATH)

    metrics_payload = {"lstm": lstm_metrics, "naive_baseline": naive_metrics}
    config.METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))

    return metrics_payload


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
