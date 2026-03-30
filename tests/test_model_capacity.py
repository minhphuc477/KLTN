from __future__ import annotations

import logging

from torch import nn

from src.utils.model_capacity import count_parameters, log_capacity_guardrails


def test_count_parameters_respects_trainable_flag():
    model = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 2))
    for param in model[1].parameters():
        param.requires_grad = False

    total = (4 * 5 + 5) + (5 * 2 + 2)
    trainable = 4 * 5 + 5

    assert count_parameters(model, trainable_only=False) == total
    assert count_parameters(model, trainable_only=True) == trainable


def test_log_capacity_guardrails_warns_on_small_data_large_model(caplog):
    logger = logging.getLogger("tests.model_capacity")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_capacity_guardrails(
            logger,
            stage_name="Diffusion trainer",
            dataset_size=128,
            param_groups={
                "diffusion_plus_guidance": 100_000_000,
                "condition_encoder": 6_000_000,
            },
            recommended_config="configs/zelda_hmolqd.yaml",
            capacity_knobs="diffusion.model_channels, diffusion.condition_hidden_dim",
        )

    assert "trainable parameters" in caplog.text
    assert "high risk for overfitting" in caplog.text
    assert "small-data danger zone" in caplog.text
