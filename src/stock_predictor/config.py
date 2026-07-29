"""Constantes de configuração: ticker, datas, hiperparâmetros e caminhos de artefatos."""
from pathlib import Path

TICKER = "PETR4.SA"
START_DATE = "2018-01-01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lstm_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Sequence length de 30 dias úteis (~1 mês de pregão): dentro da faixa comum
# na literatura de LSTM para séries financeiras (tipicamente 20-60 dias),
# equilibrando contexto histórico suficiente sem diluir demais o sinal com
# um passado muito distante.
SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 16
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
DROPOUT = 0.0
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
# fração restante (0.15) é o conjunto de teste

RANDOM_SEED = 42
