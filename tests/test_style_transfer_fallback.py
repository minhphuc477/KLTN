"""Tests for style transfer fallback behavior."""

import numpy as np
import torch


def test_transfer_style_neural_falls_back_to_theme_mapping():
    from src.generation.style_transfer import StyleTransferEngine

    engine = StyleTransferEngine(model_path=None)
    semantic_grid = np.array(
        [
            [0, 1, 2],
            [2, 1, 0],
        ],
        dtype=np.int32,
    )

    styled = engine.transfer_style_neural(
        semantic_grid=semantic_grid,
        theme_embedding=np.zeros((8,), dtype=np.float32),
        preserve_edges=True,
    )

    assert styled.shape == (2, 3, 3)
    assert styled.dtype == np.uint8
    assert int(styled.sum()) > 0
    assert engine.last_transfer_metadata["requested_method"] == "neural_style_transfer"
    assert engine.last_transfer_metadata["method"] == "theme_palette_mapping"
    assert engine.last_transfer_metadata["neural_model_used"] is False
    assert engine.last_transfer_metadata["fallback_reason"] == "style_model_unavailable"


def test_style_engine_missing_model_path_is_safe():
    from src.generation.style_transfer import StyleTransferEngine

    engine = StyleTransferEngine(model_path="checkpoints/does_not_exist_style.pt")
    assert engine.style_model is None


def test_apply_theme_records_non_neural_mapping_metadata():
    from src.generation.style_transfer import StyleTransferEngine

    engine = StyleTransferEngine(model_path=None)
    theme = engine._fallback_theme_manager.get_current_theme()

    styled = engine.apply_theme(np.ones((2, 2), dtype=np.int32), theme)

    assert styled.shape == (2, 2, 3)
    assert engine.last_transfer_metadata["requested_method"] == "theme_mapping"
    assert engine.last_transfer_metadata["method"] == "theme_palette_mapping"
    assert engine.last_transfer_metadata["theme_name"] == theme.theme_name


def test_transfer_style_neural_accepts_numpy_style_embedding_when_model_loaded():
    from src.generation.style_transfer import StyleTransferEngine

    class _DummyStyleModel(torch.nn.Module):
        def forward(self, *, semantic, edges, style):
            assert tuple(style.shape) == (1, 8)
            assert edges is not None
            h, w = semantic.shape[-2:]
            return torch.ones((1, 3, h, w), dtype=torch.float32, device=semantic.device) * 0.5

    engine = StyleTransferEngine(model_path=None)
    engine.style_model = _DummyStyleModel()

    styled = engine.transfer_style_neural(
        semantic_grid=np.ones((2, 3), dtype=np.int32),
        theme_embedding=np.zeros((8,), dtype=np.float32),
        preserve_edges=True,
    )

    assert styled.shape == (2, 3, 3)
    assert styled.dtype == np.uint8
    assert engine.last_transfer_metadata["method"] == "neural_style_model"
    assert engine.last_transfer_metadata["neural_model_used"] is True
