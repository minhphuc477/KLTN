import networkx as nx
import numpy as np
import torch

from src.core.definitions import DOOR_POSITIONS, ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.latent_diffusion import create_latent_diffusion
from src.core.symbolic_refiner import DEFAULT_ADJACENCY, TileType
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline


def test_inpaint_schedule_starts_at_noise_level_and_preserves_previous_timestep(monkeypatch):
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=10,
    )
    model.guidance.logic_net = None
    model.guidance.guidance_scale = 0.0

    recorded_q = []
    recorded_denoise = []

    def _fake_q_sample(x_0, t, noise=None):
        recorded_q.append(int(t[0].item()))
        return torch.zeros_like(x_0)

    def _fake_predict_noise_cfg(x_t, t, context, **kwargs):
        recorded_denoise.append(int(t[0].item()))
        return torch.zeros_like(x_t)

    def _fake_convert_prediction(prediction, x_t, t):
        return torch.zeros_like(x_t), torch.zeros_like(x_t)

    monkeypatch.setattr(model, "q_sample", _fake_q_sample)
    monkeypatch.setattr(model, "_predict_noise_cfg", _fake_predict_noise_cfg)
    monkeypatch.setattr(model, "_convert_prediction", _fake_convert_prediction)

    x_0 = torch.zeros(1, 4, 2, 2)
    mask = torch.ones(1, 1, 2, 2)
    context = torch.zeros(1, 8)

    model.inpaint(
        x_0=x_0,
        mask=mask,
        context=context,
        num_steps=3,
        noise_strength=0.5,
    )

    # start_t = int(10 * 0.5) = 5, and the reverse schedule must include both 5 and 0.
    assert recorded_denoise == [5, 2, 0]
    # q_sample is called once for initialization at start_t, then for known-region
    # reinjection at the aligned previous timestep of each reverse step.
    assert recorded_q == [5, 2, 0]


def test_p_sample_applies_logic_guidance_as_gradient_descent(monkeypatch):
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=10,
    )
    model.guidance.logic_net = object()
    model.guidance.guidance_scale = 1.0

    def _fake_p_mean_variance(x_t, t, context, **kwargs):
        mean = torch.zeros_like(x_t)
        variance = torch.full_like(x_t, 2.0)
        log_variance = torch.zeros_like(x_t)
        return mean, variance, log_variance

    def _fake_compute_guidance(x_t, graph_data=None, **kwargs):
        return torch.ones_like(x_t)

    monkeypatch.setattr(model, "p_mean_variance", _fake_p_mean_variance)
    monkeypatch.setattr(model.guidance, "compute_guidance", _fake_compute_guidance)

    x_t = torch.randn(1, 4, 2, 2)
    context = torch.zeros(1, 8)

    out = model.p_sample(x_t, t=0, context=context)

    # With zero sampler noise at t=0, the guided step should be:
    # mean - variance * grad = 0 - 2 * 1 = -2.
    assert torch.allclose(out, torch.full_like(out, -2.0))


def test_generate_room_constrained_decode_uses_exact_door_type():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="key_locked")

    graph_data = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
    logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
    latent = torch.zeros(1, int(pipeline.diffusion.latent_dim), 4, 3, dtype=torch.float32)

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=room_graph_context,
        room_id=0,
        apply_repair=False,
        logic_guidance_scale=0.0,
        num_diffusion_steps=4,
        start_goal_coords=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
        precomputed_latent=latent,
        precomputed_logits=logits,
    )

    spec = DOOR_POSITIONS["E"]
    col = int(spec["col"])
    row_start = int(spec["row_start"])
    row_end = int(spec["row_end"]) + 1

    assert np.all(result.neural_grid[row_start:row_end, col] == int(SEMANTIC_PALETTE["DOOR_LOCKED"]))


def test_default_wfc_adjacency_allows_multi_tile_doors():
    door_open = int(TileType.DOOR_OPEN.value)
    door_locked = int(TileType.DOOR_LOCKED.value)

    assert door_open in DEFAULT_ADJACENCY[door_open]
    assert door_locked in DEFAULT_ADJACENCY[door_open]
    assert int(TileType.FLOOR.value) in DEFAULT_ADJACENCY[door_open]


def test_pipeline_refiner_can_refresh_into_learned_rules():
    refiner = NeuralSymbolicDungeonPipeline._create_refiner(use_learned_rules=True)
    assert refiner.learned_stats is not None

    floor = int(TileType.FLOOR.value)
    wall = int(TileType.WALL.value)
    before = set(refiner.adjacency[floor])

    observed = np.array(
        [
            [floor, wall],
            [wall, floor],
        ],
        dtype=np.int32,
    )
    refiner.learned_stats.observe(observed)
    refiner.refresh_learned_rules()

    after = set(refiner.adjacency[floor])
    assert after == {floor, wall}
    assert after != before
