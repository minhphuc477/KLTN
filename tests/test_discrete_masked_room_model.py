import networkx as nx
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.discrete_masked_model import create_discrete_masked_model
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline


def test_discrete_masked_model_respects_fixed_tokens():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=64,
        model_channels=32,
        context_dim=256,
        num_steps=3,
    )
    context = torch.zeros(1, 1, 256)
    fixed_tokens = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.long)
    fixed_mask = torch.zeros(1, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.bool)
    fixed_tokens[0, 0, 5] = int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    fixed_tokens[0, 8, 5] = int(SEMANTIC_PALETTE["START"])
    fixed_mask[0, 0, 5] = True
    fixed_mask[0, 8, 5] = True

    tokens, _logits, _hidden = model.sample(
        context=context,
        fixed_tokens=fixed_tokens,
        fixed_mask=fixed_mask,
        num_steps=2,
        seed=123,
    )

    assert int(tokens[0, 0, 5]) == int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    assert int(tokens[0, 8, 5]) == int(SEMANTIC_PALETTE["START"])


def test_pipeline_generate_room_uses_discrete_masked_mode(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="discrete_masked",
    )
    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))

    graph = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    graph_context = pipeline._build_room_graph_context(
        graph_data=graph,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((8, 0), (8, 10)),
    )

    called = {"sample": 0}

    def _sample(**kwargs):
        called["sample"] += 1
        fixed_tokens = kwargs.get("fixed_tokens")
        fixed_mask = kwargs.get("fixed_mask")
        tokens = torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), fill_value=int(SEMANTIC_PALETTE["FLOOR"]), dtype=torch.long)
        if isinstance(fixed_tokens, torch.Tensor) and isinstance(fixed_mask, torch.Tensor):
            tokens[fixed_mask] = fixed_tokens[fixed_mask]
        logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
        hidden = torch.zeros(1, 64, ROOM_HEIGHT, ROOM_WIDTH, dtype=torch.float32)
        return tokens, logits, hidden

    monkeypatch.setattr(pipeline.masked_room_model, "sample", _sample)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=0,
        apply_repair=False,
        num_diffusion_steps=4,
        seed=7,
        start_goal_coords=((8, 0), (8, 10)),
    )

    assert called["sample"] == 1
    assert result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert int(result.room_grid[8, 0]) == int(SEMANTIC_PALETTE["START"])
    assert int(result.room_grid[8, 10]) == int(SEMANTIC_PALETTE["TRIFORCE"])
