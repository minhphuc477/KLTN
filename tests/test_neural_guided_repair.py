import numpy as np
import torch

from src.core.neural_guided_repair import NeuralGuidedRepair


class _FakeLogicNet:
    def __init__(self):
        self.training = True
        self.received_eval = False

    def eval(self):
        self.training = False
        self.received_eval = True
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, tile_logits, graph_data=None):
        walkability = torch.zeros((1, 1, 5, 5), dtype=tile_logits.dtype, device=tile_logits.device)
        walkability[:, :, :, 0] = 1.0
        walkability[:, :, 4, :] = 1.0
        info = {
            "walkability": walkability,
            "grid_distances": torch.zeros_like(walkability),
        }
        return torch.tensor(0.25, dtype=tile_logits.dtype, device=tile_logits.device), info

    def _resolve_room_logic_targets(self, graph_data, batch_size, device, dtype):
        trace = torch.zeros((batch_size, 1, 5, 5), dtype=dtype, device=device)
        trace[:, :, 2, 2] = 1.0
        anchor = torch.zeros_like(trace)
        anchor[:, :, 0, 0] = 1.0
        return {"trace_target": trace, "anchor_target": anchor}


class _RecordingRefiner:
    def __init__(self):
        self.kwargs = None

    def repair_room_with_feedback(self, **kwargs):
        self.kwargs = kwargs
        diagnostics = {
            "cost_guidance_used": kwargs.get("cost_map") is not None,
            "required_floor_pixels": int(np.sum(kwargs.get("required_floor_mask"))),
        }
        return np.asarray(kwargs["grid"]).copy(), True, diagnostics


def test_neural_guided_repair_passes_logicnet_cost_and_floor_mask():
    logic_net = _FakeLogicNet()
    refiner = _RecordingRefiner()
    repair = NeuralGuidedRepair(logic_net, refiner, obstacle_weight=5.0)
    grid = np.zeros((5, 5), dtype=np.int64)
    tile_logits = torch.randn(1, 4, 5, 5)

    repaired, success, diagnostics = repair.repair_room_with_neural_guidance(
        grid,
        start=(0, 0),
        goal=(4, 4),
        tile_logits=tile_logits,
        graph_data={},
    )

    assert success is True
    assert repaired.shape == grid.shape
    assert logic_net.received_eval is True
    assert logic_net.training is True
    assert diagnostics["neural_guidance_used"] is True
    assert diagnostics["logicnet_cost_used"] is True
    assert diagnostics["logicnet_floor_mask_pixels"] == 2
    assert refiner.kwargs["cost_map"][0, 0] == 1.0
    assert refiner.kwargs["cost_map"][0, 1] == 6.0
    assert bool(refiner.kwargs["required_floor_mask"][2, 2]) is True


def test_neural_guided_repair_merges_external_mask_and_feedback_callback():
    logic_net = _FakeLogicNet()
    refiner = _RecordingRefiner()
    repair = NeuralGuidedRepair(logic_net, refiner)
    grid = np.zeros((5, 5), dtype=np.int64)
    tile_logits = torch.randn(1, 4, 5, 5)
    external_mask = np.zeros((5, 5), dtype=bool)
    external_mask[3, 3] = True

    def _feedback(*args, **kwargs):
        return grid

    _repaired, success, diagnostics = repair.repair_room_with_neural_guidance(
        grid,
        start=(0, 0),
        goal=(4, 4),
        tile_logits=tile_logits,
        graph_data={},
        required_floor_mask=external_mask,
        feedback_callback=_feedback,
        max_feedback_rounds=2,
    )

    assert success is True
    assert diagnostics["logicnet_floor_mask_pixels"] == 3
    assert refiner.kwargs["feedback_callback"] is _feedback
    assert refiner.kwargs["max_feedback_rounds"] == 2
    assert bool(refiner.kwargs["required_floor_mask"][2, 2]) is True
    assert bool(refiner.kwargs["required_floor_mask"][3, 3]) is True
