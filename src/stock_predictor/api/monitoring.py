# src/stock_predictor/api/monitoring.py
"""Instrumentação Prometheus (via biblioteca, sem servidor Prometheus real) + uso de recursos."""
from __future__ import annotations

import psutil
from fastapi import FastAPI
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

CPU_USAGE = Gauge("app_cpu_usage_percent", "Uso de CPU do processo (%)")
MEMORY_USAGE = Gauge("app_memory_usage_mb", "Uso de memória RSS do processo (MB)")

_process = psutil.Process()


def _update_resource_gauges() -> None:
    # psutil.Process.cpu_percent(interval=None) é não-bloqueante e sempre retorna 0.0
    # na primeira chamada (não há uma janela anterior para comparar); comportamento
    # documentado do psutil, não um bug — a partir da segunda chamada o valor é real.
    CPU_USAGE.set(_process.cpu_percent(interval=None))
    MEMORY_USAGE.set(_process.memory_info().rss / (1024 * 1024))


def setup_monitoring(app: FastAPI) -> None:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")

    @app.middleware("http")
    async def resource_metrics_middleware(request, call_next):
        _update_resource_gauges()
        return await call_next(request)
