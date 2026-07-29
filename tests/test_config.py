# tests/test_config.py
from stock_predictor import config


def test_split_fractions_leave_room_for_test_set():
    assert config.TRAIN_FRACTION + config.VAL_FRACTION < 1.0


def test_paths_are_under_project_root():
    assert config.MODEL_PATH.parent == config.MODELS_DIR
    assert config.SCALER_PATH.parent == config.MODELS_DIR
    assert config.METRICS_PATH.parent == config.MODELS_DIR
