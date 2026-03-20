"""Tests for style transfer fallback behavior."""

import numpy as np


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


def test_style_engine_missing_model_path_is_safe():
    from src.generation.style_transfer import StyleTransferEngine

    engine = StyleTransferEngine(model_path="checkpoints/does_not_exist_style.pt")
    assert engine.style_model is None
