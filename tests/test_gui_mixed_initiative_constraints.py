import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.gui.ai.generation_pipeline import apply_mixed_initiative_constraints


class _Logger:
    def info(self, *_args, **_kwargs):
        return None


def test_apply_mixed_initiative_constraints_applies_key_anchor():
    grid = np.full((6, 6), SEMANTIC_PALETTE["FLOOR"], dtype=np.int32)
    grid[0, 0] = SEMANTIC_PALETTE["START"]
    grid[-1, -1] = SEMANTIC_PALETTE["TRIFORCE"]

    updated, applied = apply_mixed_initiative_constraints(
        tile_grid=grid,
        constraints={"key_norm": (0.5, 0.5)},
        np_module=np,
        logger=_Logger(),
    )

    assert bool(applied.get("key_applied")) is True
    assert int((updated == SEMANTIC_PALETTE["KEY_SMALL"]).sum()) >= 1


def test_apply_mixed_initiative_constraints_reports_all_flags():
    grid = np.full((4, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int32)

    _updated, applied = apply_mixed_initiative_constraints(
        tile_grid=grid,
        constraints={},
        np_module=np,
        logger=_Logger(),
    )

    assert set(applied.keys()) == {"boss_applied", "lock_applied", "key_applied"}
