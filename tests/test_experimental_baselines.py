from __future__ import annotations

import json

import numpy as np
import torch

from experiments.baselines.common import (
    BaselineEvalConfig,
    canonicalize_generated_grid,
    evaluate_generated_grids,
    flatten_grids_to_tokens,
)
from experiments.baselines.run_wfc_baseline import (
    build_overlapping_pattern_model,
    generate_wfc_samples,
)
from experiments.baselines.train_dcgan_baseline import RoomGenerator, grids_to_onehot
from experiments.baselines.train_llm_baseline import generate_tokens
from src.core.definitions import TileID


def _simple_room() -> np.ndarray:
    grid = np.full((6, 6), int(TileID.FLOOR), dtype=np.int32)
    grid[0, :] = int(TileID.WALL)
    grid[-1, :] = int(TileID.WALL)
    grid[:, 0] = int(TileID.WALL)
    grid[:, -1] = int(TileID.WALL)
    grid[1, 1] = int(TileID.START)
    grid[4, 4] = int(TileID.TRIFORCE)
    return grid


def test_flatten_grids_to_tokens_preserves_row_major_shape_contract() -> None:
    grid = _simple_room()
    tokens = flatten_grids_to_tokens([grid])

    assert tokens.shape == (1, grid.shape[0] * grid.shape[1])
    assert tokens[0, 0] == int(TileID.WALL)
    assert tokens[0, 7] == int(TileID.START)


def test_canonicalize_generated_grid_injects_terminals_and_json_evaluates() -> None:
    raw = np.full((6, 6), int(TileID.FLOOR), dtype=np.int32)
    raw[2, 2] = 999
    canonical, diagnostics = canonicalize_generated_grid(raw)

    assert int(np.sum(canonical == int(TileID.START))) == 1
    assert int(np.sum(canonical == int(TileID.TRIFORCE))) == 1
    assert diagnostics["invalid_tile_count"] == 1
    assert diagnostics["start_injected"] is True
    assert diagnostics["goal_injected"] is True
    assert diagnostics["duplicate_start_tiles_removed"] == 0
    assert diagnostics["duplicate_goal_tiles_removed"] == 0

    report = evaluate_generated_grids(
        [canonical],
        [_simple_room()],
        BaselineEvalConfig(name="unit_test", run_pcbs=False, timeout_astar=200),
    )
    assert report["aggregate"]["num_samples"] == 1
    assert "terminal_normalization_rate" in report["aggregate"]
    json.dumps(report)


def test_wfc_baseline_generates_locally_valid_shapes() -> None:
    reference = [_simple_room(), np.flipud(_simple_room()).copy()]
    model = build_overlapping_pattern_model(reference, pattern_size=3)
    samples = generate_wfc_samples(model, output_shape=reference[0].shape, num_samples=2, seed=3)

    assert model.num_patterns > 0
    assert len(samples) == 2
    assert all(sample.shape == reference[0].shape for sample in samples)
    assert all(np.isin(sample, [int(v) for v in TileID]).all() for sample in samples)


def test_dcgan_generator_shape_and_onehot_contract() -> None:
    grid = _simple_room()
    one_hot = grids_to_onehot([grid])
    generator = RoomGenerator(z_dim=8, num_tiles=44, hidden=16, room_shape=grid.shape)
    logits = generator(torch.randn(2, 8))

    assert one_hot.shape == (1, 44, grid.shape[0], grid.shape[1])
    assert torch.allclose(one_hot.sum(dim=1), torch.ones(1, grid.shape[0], grid.shape[1]))
    assert logits.shape == (2, 44, grid.shape[0], grid.shape[1])


def test_llm_generation_runs_under_no_grad_and_returns_tokens() -> None:
    class TinyModel(torch.nn.Module):
        def forward(self, tokens):
            batch, seq = tokens.shape
            logits = torch.zeros(batch, seq, 44, device=tokens.device)
            logits[..., int(TileID.FLOOR)] = 1.0
            return type("Out", (), {"logits": logits})

    tokens = generate_tokens(
        TinyModel(),
        num_samples=3,
        seq_len=10,
        vocab_size=44,
        device=torch.device("cpu"),
        seed=5,
    )

    assert tokens.shape == (3, 10)
    assert int(tokens.max()) < 44
