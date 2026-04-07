import json

import networkx as nx
import numpy as np
import pytest
import torch

from src.core.definitions import DOOR_POSITIONS, ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.condition_encoder import create_condition_encoder
from src.core.latent_diffusion import create_latent_diffusion
from src.core.logic_net import LogicNet
from src.core.vqvae import create_vqvae
from src.core.symbolic_refiner import DEFAULT_ADJACENCY, TileType
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNEL_COUNT


def _generate_precomputed_room(
    pipeline: NeuralSymbolicDungeonPipeline,
    mission_graph: nx.DiGraph,
    *,
    room_id: int = 0,
    start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
):
    graph_data = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=mission_graph,
        room_id=room_id,
        start_goal=start_goal,
    )

    logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
    logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
    latent = torch.zeros(1, int(pipeline.diffusion.latent_dim), 4, 3, dtype=torch.float32)

    return pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=room_graph_context,
        room_id=room_id,
        apply_repair=False,
        logic_guidance_scale=0.0,
        num_diffusion_steps=4,
        start_goal_coords=start_goal,
        precomputed_latent=latent,
        precomputed_logits=logits,
    )


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


def test_generate_room_semantic_constrained_decode_biases_graph_marker_inside_decoder():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_semantic_constrained_decoding_enabled=True,
        default_semantic_marker_logit_bias=12.0,
        default_semantic_marker_suppression_bias=3.0,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
    logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0

    structural_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    placements = pipeline._plan_room_graph_marker_layout(
        structural_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    assert placements
    tile_id, slot = placements[0]

    stats = pipeline._apply_semantic_constrained_decoding(
        logits,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    decoded = logits.argmax(dim=1).detach().cpu().numpy()[0]

    assert stats["planned_markers"] == 1
    assert stats["biased_slots"] == 1
    assert int(decoded[int(slot[0]), int(slot[1])]) == int(tile_id)


def test_generate_room_re_salvages_graph_marker_after_boundary_enforcement():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_semantic_constrained_decoding_enabled=True,
        default_semantic_marker_logit_bias=10000.0,
        default_semantic_marker_suppression_bias=100.0,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

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

    assert result.metrics["neural_post_boundary_graph_semantic_hints_salvaged"] == pytest.approx(1.0)
    assert result.metrics["neural_graph_marker_exact_match_rate"] == pytest.approx(1.0)


def test_generate_room_can_disable_deterministic_graph_marker_overlay():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_semantic_constrained_decoding_enabled=False,
        default_deterministic_graph_marker_overlay_enabled=False,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

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

    assert int(np.sum(result.room_grid == int(SEMANTIC_PALETTE["START"]))) == 0
    assert result.metrics["final_graph_markers_placed"] == pytest.approx(0.0)


def test_generate_room_puzzle_scaffold_adds_structure_to_underfilled_puzzle_rooms():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_puzzle_room_scaffold_enabled=True,
        default_puzzle_room_scaffold_min_structure_tiles=10,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, node_type="puzzle", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    result = _generate_precomputed_room(pipeline, mission_graph)

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    puzzle_id = int(SEMANTIC_PALETTE["PUZZLE"])
    assert result.metrics["final_puzzle_scaffold_applied"] == pytest.approx(1.0)
    assert result.metrics["final_puzzle_scaffold_tiles_added"] > 0
    assert result.metrics["final_puzzle_scaffold_segments_added"] > 0
    assert int(np.sum(result.room_grid == block_id)) > 0
    assert int(np.sum(result.room_grid == puzzle_id)) == 1


def test_generate_room_puzzle_scaffold_skips_non_puzzle_rooms():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_puzzle_room_scaffold_enabled=True,
        default_puzzle_room_scaffold_min_structure_tiles=10,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    result = _generate_precomputed_room(pipeline, mission_graph)

    assert result.metrics["final_puzzle_scaffold_applied"] == pytest.approx(0.0)
    assert result.metrics["final_puzzle_scaffold_tiles_added"] == pytest.approx(0.0)
    assert result.metrics["final_puzzle_scaffold_segments_added"] == pytest.approx(0.0)


def test_generate_room_puzzle_scaffold_preserves_planned_route_cells():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_puzzle_room_scaffold_enabled=True,
        default_puzzle_room_scaffold_min_structure_tiles=10,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, node_type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    start_goal = ((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2))
    result = _generate_precomputed_room(pipeline, mission_graph, start_goal=start_goal)

    assert isinstance(result.room_plan_mask, np.ndarray)
    route_mask = np.asarray(result.room_plan_mask, dtype=np.float32) > 0.0
    route_tiles = result.room_grid[route_mask]
    disallowed = {int(SEMANTIC_PALETTE["WALL"]), int(SEMANTIC_PALETTE["BLOCK"])}
    assert route_tiles.size > 0
    assert not any(int(tile) in disallowed for tile in route_tiles.tolist())


def test_generate_room_enforces_boundary_shell_except_required_doors():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, pos=(0, 0))
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

    boundary_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[ROOM_HEIGHT - 1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, ROOM_WIDTH - 1] = True
    allowed_door_mask = pipeline._required_room_door_slots_mask(graph=mission_graph, room_id=0)

    wall_id = int(SEMANTIC_PALETTE["WALL"])
    assert np.all(result.room_grid[boundary_mask & ~allowed_door_mask] == wall_id)

    spec = DOOR_POSITIONS["E"]
    col = int(spec["col"])
    row_start = int(spec["row_start"])
    row_end = int(spec["row_end"]) + 1
    assert np.all(result.room_grid[row_start:row_end, col] == int(SEMANTIC_PALETTE["DOOR_LOCKED"]))
    assert np.all(result.room_grid[row_start:row_end, ROOM_WIDTH - 2] == int(SEMANTIC_PALETTE["FLOOR"]))


def test_start_marker_overlay_stays_inside_room_boundary():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    base_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int32)
    base_grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])

    overlaid, marker_count, marker_ids = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    start_positions = np.argwhere(overlaid == int(SEMANTIC_PALETTE["START"]))
    assert marker_count == 1
    assert marker_ids == [int(SEMANTIC_PALETTE["START"])]
    assert start_positions.shape[0] == 1
    assert int(start_positions[0, 0]) not in {0, ROOM_HEIGHT - 1}
    assert int(start_positions[0, 1]) not in {0, ROOM_WIDTH - 1}


def test_strip_volatile_room_semantics_salvages_nearby_graph_owned_marker():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    structural_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int32)
    structural_grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])
    placements = pipeline._plan_room_graph_marker_layout(
        structural_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    assert placements
    tile_id, slot = placements[0]

    noisy_grid = structural_grid.copy()
    nearby_col = int(slot[1]) + 1 if int(slot[1]) < ROOM_WIDTH - 2 else int(slot[1]) - 1
    nearby_slot = (int(slot[0]), int(nearby_col))
    noisy_grid[nearby_slot[0], nearby_slot[1]] = int(tile_id)

    cleaned, stripped_count, stripped_ids, preserved_count, preserved_ids = pipeline._strip_volatile_room_semantics(
        noisy_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    assert cleaned[int(slot[0]), int(slot[1])] == int(tile_id)
    assert cleaned[nearby_slot[0], nearby_slot[1]] == int(SEMANTIC_PALETTE["FLOOR"])
    assert stripped_count == 0
    assert stripped_ids == []
    assert preserved_count == 1
    assert preserved_ids == [int(tile_id)]


def test_graph_marker_alignment_metrics_detect_missing_neural_semantics():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    base_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int32)
    base_grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])

    placements = pipeline._plan_room_graph_marker_layout(
        base_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    metrics = pipeline._measure_room_graph_marker_alignment(
        base_grid,
        placements=placements,
        prefix="neural_",
    )

    assert metrics["neural_graph_marker_expected"] == pytest.approx(1.0)
    assert metrics["neural_graph_marker_exact_matches"] == pytest.approx(0.0)
    assert metrics["neural_graph_marker_exact_match_rate"] == pytest.approx(0.0)
    assert metrics["neural_semantic_anchor_avg_manhattan_error"] > 0.0


def test_graph_marker_alignment_metrics_reach_exact_match_after_overlay():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    base_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int32)
    base_grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])

    placements = pipeline._plan_room_graph_marker_layout(
        base_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    overlaid, _, _ = pipeline._overlay_room_graph_markers(
        base_grid,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    metrics = pipeline._measure_room_graph_marker_alignment(
        overlaid,
        placements=placements,
        prefix="final_post_overlay_",
    )

    assert metrics["final_post_overlay_graph_marker_expected"] == pytest.approx(1.0)
    assert metrics["final_post_overlay_graph_marker_exact_matches"] == pytest.approx(1.0)
    assert metrics["final_post_overlay_graph_marker_exact_match_rate"] == pytest.approx(1.0)
    assert metrics["final_post_overlay_semantic_anchor_avg_manhattan_error"] == pytest.approx(0.0)


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


def test_compute_room_condition_reuses_global_tokens_without_second_encoder_pass(monkeypatch):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        use_graph_node_cross_attention=True,
    )

    graph_context = {
        "node_features": torch.randn(3, 12),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_features": torch.randn(2, 14),
        "tpe": torch.randn(3, 8),
        "current_node_idx": 0,
    }

    def _fake_forward(**kwargs):
        assert kwargs.get("return_global_tokens") is True
        return torch.zeros(1, 256), torch.zeros(1, 3, 256)

    def _fail_encode_global_only(*args, **kwargs):
        raise AssertionError("encode_global_only should not be called when forward returns global tokens")

    monkeypatch.setattr(pipeline.condition_encoder, "forward", _fake_forward)
    monkeypatch.setattr(pipeline.condition_encoder, "encode_global_only", _fail_encode_global_only)

    condition = pipeline._compute_room_condition(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        boundary_constraints=torch.zeros(1, 8),
        position=torch.zeros(1, 2),
    )

    assert tuple(condition.shape) == (1, 4, 256)


def test_room_graph_context_includes_current_node_distance_features():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_node(2, is_goal=True, pos=(0, 2))
    mission_graph.add_edge(0, 1)
    mission_graph.add_edge(1, 2)

    graph_data = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=mission_graph,
        room_id=1,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    current_node_distance = room_graph_context["current_node_distance"]
    current_node_idx = int(room_graph_context["current_node_idx"])

    assert tuple(current_node_distance.shape) == (3, 4)
    assert float(current_node_distance[current_node_idx, 0]) == 0.0
    assert float(current_node_distance[current_node_idx, 1]) == 0.0
    assert float(current_node_distance[current_node_idx, 2]) == 0.0
    assert float(current_node_distance[current_node_idx, 3]) == 1.0


def test_room_graph_context_preserves_explicit_numeric_style_id():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.graph["style_id"] = 2
    mission_graph.add_node(0, is_start=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1), style_id=5)
    mission_graph.add_edge(0, 1)

    graph_data = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=mission_graph,
        room_id=1,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    assert room_graph_context["style_id"] == 5


def test_room_graph_context_resolves_canonical_sector_theme_labels():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, is_start=True, pos=(0, 0), sector_theme="fire-temple")
    mission_graph.add_node(1, pos=(0, 1), sector_theme="shadow_dungeon")
    mission_graph.add_edge(0, 1)

    graph_data = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=mission_graph,
        room_id=1,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    assert room_graph_context["style_id"] == 4


def test_condition_encoder_return_global_tokens_keeps_full_graph_sequence():
    encoder = create_condition_encoder(latent_dim=32, output_dim=128)
    encoder.eval()

    neighbor_latents = {"N": None, "S": None, "E": None, "W": None}
    boundary_constraints = torch.zeros(1, 8)
    position = torch.zeros(1, 2)
    node_features = torch.randn(3, encoder.global_encoder.node_feature_dim)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_features = torch.randn(2, encoder.global_encoder.edge_feature_dim)
    tpe = torch.randn(3, 8)

    condition, node_tokens = encoder(
        neighbor_latents=neighbor_latents,
        boundary_constraints=boundary_constraints,
        position=position,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        tpe=tpe,
        current_node_idx=1,
        return_global_tokens=True,
    )

    expected_tokens = encoder.encode_global_only(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        tpe=tpe,
    ).unsqueeze(0)

    assert tuple(condition.shape) == (1, 128)
    assert tuple(node_tokens.shape) == (1, 3, 128)
    assert torch.allclose(node_tokens, expected_tokens, atol=1e-5)


def test_condition_encoder_reference_room_maps_change_conditioning_signal():
    encoder = create_condition_encoder(
        latent_dim=32,
        output_dim=64,
        use_reference_room_maps=True,
        reference_num_tile_types=44,
        reference_embedding_dim=16,
        reference_hidden_dim=32,
    )
    encoder.eval()

    kwargs = {
        "neighbor_latents": {"N": None, "S": None, "E": None, "W": None},
        "boundary_constraints": torch.zeros(1, 8),
        "position": torch.zeros(1, 2),
        "node_features": torch.randn(3, encoder.global_encoder.node_feature_dim),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_features": torch.randn(2, encoder.global_encoder.edge_feature_dim),
        "tpe": torch.randn(3, 8),
        "current_node_idx": 1,
    }

    baseline = encoder(**kwargs)
    conditioned = encoder(
        **kwargs,
        reference_room_maps={
            "N": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.25, dtype=torch.float32),
            "S": None,
            "E": torch.full((1, ROOM_HEIGHT, ROOM_WIDTH), 0.75, dtype=torch.float32),
            "W": None,
        },
    )

    assert tuple(conditioned.shape) == (1, 64)
    assert not torch.allclose(baseline, conditioned)


def test_pipeline_loaders_accept_composite_diffusion_checkpoint_metadata(tmp_path):
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device="cpu",
        enable_logging=False,
    )

    condition_encoder = create_condition_encoder(
        latent_dim=32,
        hidden_dim=128,
        output_dim=96,
        num_gnn_layers=2,
        gnn_type="sage",
        num_attention_heads=4,
        dropout=0.05,
        use_current_node_distance_features=False,
    )
    diffusion = create_latent_diffusion(
        latent_dim=32,
        context_dim=96,
        num_timesteps=17,
        prediction_type="v",
        cfg_scale=2.5,
        cfg_schedule_mode="cosine_decay",
        cfg_schedule_min_scale=1.5,
        cfg_schedule_power=2.0,
        min_snr_gamma=2.5,
        model_channels=48,
        topology_refinement_mode="lightweight",
        attention_mode="linear_hedgehog",
        topology_conditioning_mode="spade",
        hedgehog_feature_dim=16,
        unet_channel_mult=(1, 2),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(1,),
        unet_num_heads=4,
        unet_dropout=0.05,
        graph_auto_linear_attention_nodes=32,
        spatial_graph_gate_init=-1.25,
        spatial_topology_gate_init=-0.75,
        room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
    )
    logic_net = LogicNet(
        latent_dim=32,
        num_classes=44,
        num_iterations=7,
        topology_trace_weight=0.6,
        topology_anchor_weight=0.4,
    )

    ckpt_path = tmp_path / "diffusion_bundle.pth"
    torch.save(
        {
            "diffusion_state_dict": diffusion.state_dict(),
            "condition_encoder_state_dict": condition_encoder.state_dict(),
            "logic_net_state_dict": logic_net.state_dict(),
            "config": {
                "latent_dim": 32,
                "context_dim": 96,
                "condition_hidden_dim": 128,
                "condition_num_gnn_layers": 2,
                "condition_num_attention_heads": 4,
                "condition_dropout": 0.05,
                "condition_gnn_type": "sage",
                "use_current_node_distance_features": False,
                "num_timesteps": 17,
                "prediction_type": "v",
                "cfg_scale": 2.5,
                "cfg_schedule_mode": "cosine_decay",
                "cfg_schedule_min_scale": 1.5,
                "cfg_schedule_power": 2.0,
                "min_snr_gamma": 2.5,
                "model_channels": 48,
                "topology_refinement_mode": "lightweight",
                "attention_mode": "linear_hedgehog",
                "topology_conditioning_mode": "spade",
                "hedgehog_feature_dim": 16,
                "unet_channel_mult": [1, 2],
                "unet_num_res_blocks": 1,
                "unet_attention_resolutions": [1],
                "unet_num_heads": 4,
                "unet_dropout": 0.05,
                "graph_auto_linear_attention_nodes": 32,
                "spatial_graph_gate_init": -1.25,
                "spatial_topology_gate_init": -0.75,
                "room_topology_channels": ROOM_TOPOLOGY_CHANNEL_COUNT,
                "num_logic_iterations": 7,
                "logic_topology_trace_weight": 0.6,
                "logic_topology_anchor_weight": 0.4,
                "num_classes": 44,
            },
        },
        ckpt_path,
    )
    ckpt_path.with_suffix(".pth.meta.json").write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "model_type": "diffusion",
                "architecture": {
                    "latent_dim": 32,
                    "num_classes": 44,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded_condition_encoder = pipeline._load_condition_encoder(str(ckpt_path))
    loaded_diffusion = pipeline._load_diffusion(str(ckpt_path))
    loaded_logic_net = pipeline._load_logic_net(str(ckpt_path))

    assert loaded_condition_encoder.latent_dim == 32
    assert loaded_condition_encoder.output_dim == 96
    assert loaded_condition_encoder.global_encoder.hidden_dim == 128
    assert loaded_condition_encoder.global_encoder.gnn_type == "sage"
    assert loaded_condition_encoder.global_encoder.use_current_node_distance_features is False

    assert loaded_diffusion.latent_dim == 32
    assert loaded_diffusion.context_dim == 96
    assert loaded_diffusion.num_timesteps == 17
    assert loaded_diffusion.prediction_type == "v"
    assert loaded_diffusion.cfg_schedule_mode == "cosine_decay"
    assert loaded_diffusion.topology_conditioning_mode == "spade"
    assert loaded_diffusion.denoiser.model_channels == 48

    assert loaded_logic_net.latent_dim == 32
    assert loaded_logic_net.num_classes == 44
    assert loaded_logic_net.graph_pathfinder.num_iterations == 7
    assert loaded_logic_net.topology_trace_weight == pytest.approx(0.6)
    assert loaded_logic_net.topology_anchor_weight == pytest.approx(0.4)


def test_pipeline_vqvae_loader_accepts_embedded_vqvae_from_composite_checkpoint(tmp_path):
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device="cpu",
        enable_logging=False,
    )

    vqvae = create_vqvae(
        num_classes=44,
        codebook_size=32,
        latent_dim=16,
        hidden_dim=32,
        use_coordconv=False,
    )

    ckpt_path = tmp_path / "diffusion_bundle_with_vqvae.pth"
    torch.save(
        {
            "vqvae_state_dict": vqvae.state_dict(),
            "diffusion_state_dict": {"dummy": torch.tensor(1.0)},
            "config": {
                "num_classes": 44,
                "latent_dim": 16,
                "codebook_size": 32,
                "use_coordconv": False,
            },
        },
        ckpt_path,
    )
    ckpt_path.with_suffix(".pth.meta.json").write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "model_type": "diffusion",
                "architecture": {
                    "num_classes": 44,
                    "latent_dim": 16,
                    "codebook_size": 32,
                    "use_coordconv": False,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded_vqvae = pipeline._load_vqvae(str(ckpt_path))

    assert loaded_vqvae.num_classes == 44
    assert loaded_vqvae.latent_dim == 16
    assert loaded_vqvae.codebook_size == 32
    assert loaded_vqvae.encoder.conv_in.__class__.__name__ == "Conv2d"


def test_pipeline_diffusion_loader_rejects_checkpoint_without_diffusion_state_dict(tmp_path):
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device="cpu",
        enable_logging=False,
    )

    ckpt_path = tmp_path / "broken_diffusion_bundle.pth"
    torch.save({"config": {"latent_dim": 32, "context_dim": 96}}, ckpt_path)
    ckpt_path.with_suffix(".pth.meta.json").write_text(
        json.dumps({"format_version": "1.0", "model_type": "diffusion"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not contain a loadable state_dict"):
        pipeline._load_diffusion(str(ckpt_path))


def test_pipeline_diffusion_loader_prefers_ema_weights_for_inference(tmp_path):
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device="cpu",
        enable_logging=False,
    )

    model = create_latent_diffusion(
        latent_dim=4,
        context_dim=8,
        model_channels=8,
        num_timesteps=10,
        cfg_scale=3.0,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_num_heads=1,
        room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
    )
    raw_state = model.state_dict()
    probe_key = next(
        key for key, value in raw_state.items()
        if torch.is_tensor(value) and value.dtype.is_floating_point
    )
    ema_state = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in raw_state.items()
    }
    ema_state[probe_key] = ema_state[probe_key] + 1.0

    ckpt_path = tmp_path / "diffusion_ema_bundle.pth"
    torch.save(
        {
            "diffusion_state_dict": raw_state,
            "ema_diffusion_state_dict": ema_state,
            "config": {
                "latent_dim": 4,
                "context_dim": 8,
                "num_timesteps": 10,
                "cfg_scale": 3.0,
                "model_channels": 8,
                "unet_channel_mult": [1],
                "unet_num_res_blocks": 1,
                "unet_attention_resolutions": [],
                "unet_num_heads": 1,
                "room_topology_channels": ROOM_TOPOLOGY_CHANNEL_COUNT,
            },
        },
        ckpt_path,
    )

    loaded = pipeline._load_diffusion(str(ckpt_path))

    assert getattr(loaded, "inference_checkpoint_state_key") == "ema_diffusion_state_dict"
    assert torch.allclose(loaded.state_dict()[probe_key], ema_state[probe_key])


def test_pipeline_random_init_loaders_follow_bound_component_dimensions():
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device="cpu",
        enable_logging=False,
    )
    pipeline.vqvae = create_vqvae(
        num_classes=31,
        codebook_size=32,
        latent_dim=24,
        hidden_dim=32,
        use_coordconv=False,
    )
    pipeline.condition_encoder = create_condition_encoder(
        latent_dim=24,
        hidden_dim=80,
        output_dim=72,
        num_gnn_layers=2,
        gnn_type="sage",
        num_attention_heads=4,
    )

    diffusion = pipeline._load_diffusion(None)
    pipeline.diffusion = diffusion
    logic_net = pipeline._load_logic_net(None)
    masked_room = pipeline._load_masked_room_model(None)

    assert diffusion.latent_dim == 24
    assert diffusion.context_dim == 72
    assert logic_net.latent_dim == 24
    assert logic_net.num_classes == 31
    assert masked_room is None
