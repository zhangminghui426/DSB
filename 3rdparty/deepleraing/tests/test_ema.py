"""Tests for Model EMA."""

import torch

from models.ema import ModelEMA


def test_ema_update():
    model = torch.nn.Linear(10, 10)
    ema = ModelEMA(model, decay=0.999)

    # Initial parameters
    initial_weight = model.weight.data.clone()

    # Modify model parameters
    with torch.no_grad():
        model.weight.add_(torch.randn_like(model.weight))

    # Update EMA
    ema.update(model)

    # EMA should be between initial and new
    assert not torch.allclose(ema.module.weight.data, initial_weight)
    assert not torch.allclose(ema.module.weight.data, model.weight.data)


def test_ema_decay():
    model = torch.nn.Linear(10, 10)
    ema = ModelEMA(model, decay=0.0)

    initial_weight = model.weight.data.clone()

    with torch.no_grad():
        model.weight.add_(torch.randn_like(model.weight))

    ema.update(model)

    # decay=0 means EMA = current model
    assert torch.allclose(ema.module.weight.data, model.weight.data)
