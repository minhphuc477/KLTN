from types import SimpleNamespace

from src.utils.checkpoint import MetricsLogger


def test_metrics_logger_forwards_numeric_metrics_to_opt_in_wandb(monkeypatch, tmp_path):
    logged = []
    run = SimpleNamespace(log=lambda metrics, step=None: logged.append((metrics, step)))
    fake_wandb = SimpleNamespace(init=lambda **kwargs: run)

    monkeypatch.setenv("HMOLQD_WANDB_ENABLED", "1")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    logger = MetricsLogger(str(tmp_path), "wandb_test")
    logger.log({"loss": 0.5, "epoch": 2, "note": "ignored"}, step=7)

    assert logged == [({"loss": 0.5, "epoch": 2, "step": 7}, 7)]
