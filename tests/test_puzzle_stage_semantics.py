import numpy as np
import torch

from src.core import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.puzzle_stage_semantics import (
    DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
    PuzzleStageSemanticsHead,
    build_puzzle_stage_semantic_targets,
)
from src.pipeline.room_topology_conditioning import (
    PUZZLE_STAGE_GATE_FAMILY_IDS,
    PUZZLE_STAGE_KIND_IDS,
    build_puzzle_stage_condition_metadata,
)


def test_build_puzzle_stage_semantic_targets_encodes_ordered_sequence():
    targets = build_puzzle_stage_semantic_targets(
        [
            {
                "gate_family": "switch",
                "sequence_required": True,
                "stage_sequence": [
                    {"kind": "collect_key"},
                    {"kind": "push_block_to_switch"},
                    {"kind": "reach_exit"},
                ],
            },
            {},
        ],
        max_sequence_length=4,
    )

    assert tuple(targets["stage_slots"].shape) == (2, 4)
    assert int(targets["gate_family"][0].item()) == PUZZLE_STAGE_GATE_FAMILY_IDS["switch"]
    assert float(targets["sequence_required"][0].item()) == 1.0
    assert int(targets["stage_count"][0].item()) == 3
    assert int(targets["stage_slots"][0, 0].item()) == PUZZLE_STAGE_KIND_IDS["collect_key"]
    assert int(targets["stage_slots"][0, 1].item()) == PUZZLE_STAGE_KIND_IDS["push_block_to_switch"]
    assert int(targets["stage_slots"][0, 2].item()) == PUZZLE_STAGE_KIND_IDS["reach_exit"]
    assert float(targets["stage_slot_mask"][0, 0].item()) == 1.0
    assert float(targets["stage_slot_mask"][1].sum().item()) == 0.0
    assert int(targets["gate_family"][1].item()) == PUZZLE_STAGE_GATE_FAMILY_IDS["generic"]


def test_build_puzzle_stage_semantic_targets_clamps_to_max_length():
    targets = build_puzzle_stage_semantic_targets(
        [
            {
                "gate_family": "bombable",
                "sequence_required": True,
                "stage_sequence": [
                    {"kind": "collect_key"},
                    {"kind": "collect_item"},
                    {"kind": "step_on_puzzle"},
                ],
            }
        ],
        max_sequence_length=2,
    )

    assert int(targets["gate_family"][0].item()) == PUZZLE_STAGE_GATE_FAMILY_IDS["bombable"]
    assert int(targets["stage_count"][0].item()) == 2
    assert tuple(targets["stage_slots"].shape) == (1, 2)
    assert float(targets["stage_slot_mask"][0].sum().item()) == 2.0


def test_puzzle_stage_semantics_head_returns_finite_loss_and_gradients():
    head = PuzzleStageSemanticsHead(
        num_tile_classes=44,
        hidden_dim=32,
        max_sequence_length=DEFAULT_PUZZLE_STAGE_MAX_SEQUENCE_LENGTH,
    )
    tile_logits = torch.randn(2, 44, 16, 11, requires_grad=True)
    stage_conditions = [
        {
            "gate_family": "key",
            "sequence_required": True,
            "stage_sequence": [{"kind": "collect_key"}, {"kind": "reach_exit"}],
        },
        {
            "gate_family": "switch",
            "sequence_required": True,
            "stage_sequence": [{"kind": "push_block_to_switch"}],
        },
    ]

    loss, metrics = head.compute_loss(tile_logits, stage_conditions)

    assert torch.isfinite(loss)
    assert metrics["puzzle_stage_semantic_loss"] >= 0.0
    assert 0.0 <= metrics["puzzle_stage_gate_acc"] <= 1.0
    assert 0.0 <= metrics["puzzle_stage_slot_acc"] <= 1.0

    loss.backward()

    assert tile_logits.grad is not None
    assert torch.isfinite(tile_logits.grad).all()


def test_puzzle_stage_semantics_head_handles_batches_without_sequence_slots():
    head = PuzzleStageSemanticsHead(num_tile_classes=44, hidden_dim=16, max_sequence_length=4)
    tile_logits = torch.randn(2, 44, 16, 11)

    loss, metrics = head.compute_loss(tile_logits, [{}, {}])

    assert torch.isfinite(loss)
    assert metrics["puzzle_stage_slot_loss"] == 0.0
    assert metrics["puzzle_stage_slot_acc"] == 1.0


def test_switch_stage_trace_requires_a_real_block_push():
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    wall = int(SEMANTIC_PALETTE["WALL"])
    switch = (8, 5)
    start = (10, 5)
    goal = (1, 5)

    def _metadata(grid: np.ndarray):
        return build_puzzle_stage_condition_metadata(
            room_grid=grid,
            start=start,
            goal=goal,
            required_doors={"N": True, "S": False, "E": False, "W": False},
            incoming_dirs=set(),
            outgoing_dirs={"N"},
            edge_constraint_tokens={"N": {"switch_locked"}},
            room_role_flags={"has_puzzle": True},
            anchors={"start": start, "puzzle": switch, "goal": goal},
        )

    push_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), wall, dtype=np.int32)
    push_grid[1:-1, 1:-1] = floor
    push_grid[switch] = int(SEMANTIC_PALETTE["PUZZLE"])
    push_grid[9, 5] = int(SEMANTIC_PALETTE["BLOCK"])
    push_metadata = _metadata(push_grid)

    assert push_metadata["stage_trace_complete"] is True
    assert push_metadata["stage_trace_completed_count"] == 2
    assert push_metadata["stage_trace_failed_stage_index"] is None
    assert float(push_metadata["stage_trace_mask"][switch]) > 0.0

    walk_only_grid = push_grid.copy()
    walk_only_grid[9, 5] = floor
    walk_only_metadata = _metadata(walk_only_grid)

    assert walk_only_metadata["stage_trace_complete"] is False
    assert walk_only_metadata["stage_trace_completed_count"] == 0
    assert walk_only_metadata["stage_trace_failed_stage_index"] == 0
