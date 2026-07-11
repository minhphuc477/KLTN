"""Regression coverage for the checkpoint-backed controllability protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.core.definitions import SEMANTIC_PALETTE


def _load_controllability_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_controllability.py"
    spec = importlib.util.spec_from_file_location("validate_controllability_for_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the controllability script for contract testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controllability_observation_never_reuses_requested_target_curve(tmp_path):
    module = _load_controllability_module()
    test = module.ControllabilityTest(output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="No generated room grids"):
        test._extract_actual_tension_curve(
            {"tension_curve": [0.0, 1.0], "actual_curve_source": "room_semantic_proxy"}
        )


def test_controllability_semantic_proxy_uses_generated_room_grids(tmp_path):
    module = _load_controllability_module()
    test = module.ControllabilityTest(output_dir=tmp_path)
    floor_room = np.full((4, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int32)
    enemy_room = floor_room.copy()
    enemy_room[1, 1] = SEMANTIC_PALETTE["ENEMY"]

    observed = test._extract_actual_tension_curve(
        {
            "tension_curve": [1.0, 0.0],
            "actual_curve_source": "room_semantic_proxy",
            "room_order": ["quiet", "combat"],
            "rooms": [
                {"id": "quiet", "grid": floor_room},
                {"id": "combat", "grid": enemy_room},
            ],
        }
    )

    assert observed.tolist() == [0.0, 1.0]
