"""Contratos Pydantic de entrada e saída da API."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    closes: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        ...,
        min_length=2,
        max_length=5000,
        description="Preços históricos de fechamento, em ordem cronológica.",
    )
    days_ahead: int = Field(1, ge=1, le=30, description="Quantos dias úteis futuros prever.")

    @field_validator("closes")
    @classmethod
    def validate_closes(cls, value: list[float]) -> list[float]:
        if any(price <= 0 for price in value):
            raise ValueError("Todos os preços em 'closes' devem ser positivos.")
        return value


class PredictedPrice(BaseModel):
    day: int
    predicted_close: float = Field(..., allow_inf_nan=False)


class PredictResponse(BaseModel):
    ticker: str
    days_ahead: int
    predictions: list[PredictedPrice]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
