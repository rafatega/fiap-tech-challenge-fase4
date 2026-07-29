# dashboard/app.py
"""Dashboard Streamlit: histórico de preços, previsão do LSTM e métricas do modelo."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TICKER = "PETR4.SA"
METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "metrics.json"

# Cores validadas via skill `dataviz` (references/palette.md, categorical slots 1-2).
# node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> ALL CHECKS PASS
# (lightness band, chroma floor, CVD/normal-vision ΔE, contraste vs. superfície).
COLOR_HISTORY = "#2a78d6"  # slot 1 (blue) - série "Histórico"
COLOR_FORECAST = "#eb6834"  # slot 2 (orange) - série "Previsão"
SURFACE_COLOR = "#fcfcfb"  # superfície do gráfico (light)
GRIDLINE_COLOR = "#e1e0d9"  # hairline, recessivo
AXIS_COLOR = "#c3c2b7"  # baseline/eixo
TEXT_PRIMARY = "#0b0b0b"  # tinta primária


def fetch_history(days: int = 180) -> pd.DataFrame:
    df = yf.download(TICKER, period=f"{days}d", auto_adjust=True, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError("Não foi possível obter o histórico de preços via yfinance.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("close").to_frame()


def call_predict_api(closes: list[float], days_ahead: int) -> list[float]:
    # timeout=60: no plano free do Render a API pode estar "dormindo" e o
    # cold-start leva de 30 a 60s para responder à primeira requisição.
    response = requests.post(
        f"{API_URL}/predict", json={"closes": closes, "days_ahead": days_ahead}, timeout=60
    )
    response.raise_for_status()
    return [item["predicted_close"] for item in response.json()["predictions"]]


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


def render_chart(history: pd.DataFrame, forecast_df: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["close"],
            name="Histórico",
            mode="lines",
            line=dict(color=COLOR_HISTORY, width=2),
        )
    )
    if forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["close"],
                name="Previsão",
                mode="lines+markers",
                line=dict(color=COLOR_FORECAST, width=2, dash="dash"),
                marker=dict(color=COLOR_FORECAST, size=8, line=dict(color=SURFACE_COLOR, width=2)),
            )
        )
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        legend_title="Série",
        hovermode="x unified",
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_PRIMARY),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRIDLINE_COLOR, gridwidth=1, linecolor=AXIS_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE_COLOR, gridwidth=1, linecolor=AXIS_COLOR)
    return fig


def main() -> None:
    st.set_page_config(page_title="LSTM Stock Predictor - PETR4.SA", layout="wide")
    st.title("Previsão de Preços - PETR4.SA (LSTM)")

    try:
        history = fetch_history()
    except Exception as exc:  # yfinance pode falhar de várias formas (rede, rate limit, resposta vazia)
        st.error(f"Não foi possível carregar o histórico de preços: {exc}")
        st.stop()

    days_ahead = st.slider("Dias úteis para prever", min_value=1, max_value=10, value=5)

    forecast_df = None
    if st.button("Gerar previsão"):
        closes = history["close"].tolist()
        try:
            predictions = call_predict_api(closes, days_ahead)
            future_dates = pd.bdate_range(history.index[-1], periods=days_ahead + 1)[1:]
            forecast_df = pd.DataFrame({"close": predictions}, index=future_dates)
        except requests.RequestException as exc:
            st.error(f"Não foi possível gerar a previsão: {exc}")

    st.plotly_chart(render_chart(history.tail(90), forecast_df), use_container_width=True)

    metrics = load_metrics()
    if metrics:
        st.subheader("Desempenho do modelo (conjunto de teste)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**LSTM**")
            st.json(metrics["lstm"])
        with col2:
            st.markdown("**Baseline naive (persistência)**")
            st.json(metrics["naive_baseline"])
    else:
        st.info("Métricas do modelo ainda não disponíveis (rode o treino primeiro).")


if __name__ == "__main__":
    main()
