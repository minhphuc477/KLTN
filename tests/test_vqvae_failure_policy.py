"""Tokenizer numerical failure tests using tiny synthetic modules only."""

from types import SimpleNamespace

import pytest
import torch

from src.config_system import merge_config
from src.core.vqvae import VQVAETrainer
from src.train_vqvae import evaluate_vqvae_loader, vqvae_training_kwargs_from_resolved_config
from src.publication.protocol import build_leave_one_dungeon_out_folds
from experiments.paper_v1.run_protocol import build_run_plan


class _ToyTokenizer(torch.nn.Module):
    def __init__(self, nonfinite_loss=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.nonfinite_loss = nonfinite_loss

    def compute_loss(self, batch):
        value = self.weight * (float("nan") if self.nonfinite_loss else batch.mean())
        return {key: value for key in (
            "total_loss", "recon_loss", "vq_loss", "illegal_adjacency_penalty", "perplexity"
        )}

    def forward_with_losses(self, batch):
        return batch, None, self.compute_loss(batch)


@pytest.mark.parametrize("phase", ["loss", "gradient", "clip", "evaluation"])
def test_publication_tokenizer_aborts_before_nonfinite_update(phase, monkeypatch):
    model = _ToyTokenizer(nonfinite_loss=phase in {"loss", "evaluation"})
    trainer = VQVAETrainer(model, nonfinite_policy="error")
    if phase == "gradient":
        model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    if phase == "clip":
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args: torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError, match="non-finite"):
        if phase == "evaluation":
            trainer.eval_step(torch.ones(1, 2, 2, 2))
        else:
            trainer.train_step(torch.ones(1, 2, 2, 2))
    assert model.weight.item() == 1.0
    assert not trainer.optimizer.state


def test_tokenizer_compatibility_policy_still_reports_skipped_batches():
    trainer = VQVAETrainer(_ToyTokenizer(nonfinite_loss=True))
    loss, metrics = trainer.train_step(torch.ones(1, 2, 2, 2), return_metrics=True)
    assert loss == 0.0
    assert metrics["skipped_nonfinite_batch"] == 1.0


def test_tokenizer_validation_weights_partial_batch_by_room_count():
    values = iter([1.0, 4.0])
    trainer = SimpleNamespace(eval_step=lambda batch: {"loss": next(values)})
    metrics = evaluate_vqvae_loader(
        torch.nn.Identity(), trainer,
        [torch.zeros(2, 1, 2, 2), torch.zeros(1, 1, 2, 2)],
        num_classes=44, device=torch.device("cpu"),
    )
    assert metrics["loss"] == pytest.approx(2.0)
    assert metrics["samples"] == 3


def test_publication_tokenizer_config_propagates_strict_policy(tmp_path):
    _manifest, configs = build_run_plan(
        base_config=merge_config(), output_root=tmp_path,
        folds=(build_leave_one_dungeon_out_folds()[0],), training_seeds=(42,),
    )
    tokenizer_configs = [config for _path, config in configs if config["training"]["stage"] == "vqvae"]
    assert len(tokenizer_configs) == 1
    kwargs = vqvae_training_kwargs_from_resolved_config(tokenizer_configs[0])
    assert kwargs["nonfinite_policy"] == "error"
    assert kwargs["best_checkpoint_metric"] == "val_loss"


def test_publication_tokenizer_requires_held_out_validation(tmp_path):
    config = merge_config()
    config["vqvae"]["validation_fraction"] = 0.0
    with pytest.raises(ValueError, match="held-out validation"):
        build_run_plan(base_config=config, output_root=tmp_path,
                       folds=(build_leave_one_dungeon_out_folds()[0],),
                       training_seeds=(42,))
