import torch

from stock_predictor.model import LSTMRegressor


def test_forward_pass_output_shape():
    model = LSTMRegressor(input_size=1, hidden_size=8, num_layers=1, learning_rate=1e-3)
    x = torch.randn(4, 10, 1)
    output = model(x)
    assert output.shape == (4, 1)


def test_training_step_returns_scalar_loss():
    model = LSTMRegressor(input_size=1, hidden_size=8, num_layers=1, learning_rate=1e-3)
    x = torch.randn(4, 10, 1)
    y = torch.randn(4, 1)
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0


def test_hyperparameters_are_saved_for_checkpoint_reload():
    model = LSTMRegressor(input_size=1, hidden_size=8, num_layers=1, learning_rate=1e-3)
    assert dict(model.hparams) == {
        "input_size": 1,
        "hidden_size": 8,
        "num_layers": 1,
        "learning_rate": 1e-3,
        "dropout": 0.0,
    }


def test_dropout_defaults_to_zero_and_is_configurable():
    model = LSTMRegressor(input_size=1, hidden_size=8, num_layers=1, learning_rate=1e-3)
    assert model.dropout.p == 0.0

    model_with_dropout = LSTMRegressor(
        input_size=1, hidden_size=8, num_layers=1, learning_rate=1e-3, dropout=0.3
    )
    assert model_with_dropout.dropout.p == 0.3

    x = torch.randn(4, 10, 1)
    output = model_with_dropout(x)
    assert output.shape == (4, 1)
