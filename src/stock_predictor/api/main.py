# src/stock_predictor/api/main.py
"""Aplicação FastAPI que serve o modelo LSTM de previsão de preços de PETR4.SA."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from stock_predictor import config
from stock_predictor.api.monitoring import setup_monitoring
from stock_predictor.api.schemas import HealthResponse, PredictedPrice, PredictRequest, PredictResponse
from stock_predictor.predict import load_artifacts, predict_future_prices

logger = logging.getLogger(__name__)

_state: dict = {"model": None, "scaler": None, "sequence_length": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, scaler, sequence_length = load_artifacts()
        _state.update(model=model, scaler=scaler, sequence_length=sequence_length)
    except Exception:
        logger.exception("Falha ao carregar artefatos do modelo; API sobe em modo degradado.")
        _state.update(model=None, scaler=None, sequence_length=None)
    yield
    _state.update(model=None, scaler=None, sequence_length=None)


app = FastAPI(title="LSTM Stock Predictor API", lifespan=lifespan)
setup_monitoring(app)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        predictions = predict_future_prices(
            close_prices=request.closes,
            days_ahead=request.days_ahead,
            model=_state["model"],
            scaler=_state["scaler"],
            sequence_length=_state["sequence_length"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        ticker=config.TICKER,
        days_ahead=request.days_ahead,
        predictions=[
            PredictedPrice(day=i + 1, predicted_close=price) for i, price in enumerate(predictions)
        ],
    )
