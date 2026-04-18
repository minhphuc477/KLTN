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
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline, RoomGenerationResult
from src.pipeline.room_stitching import StitchedRoomLayout
from src.pipeline.room_topology_conditioning import (
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    build_room_semantic_anchor_points,
)


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


def test_validator_plan_budget_scales_up_for_semantically_complex_puzzle_rooms():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_validator_plan_max_states=512,
    )

    budget = pipeline._resolve_validator_plan_state_budget(
        attrs={
            "has_puzzle": True,
            "has_enemy": True,
            "has_key": True,
        },
        semantics={
            "required_doors": {"N": True, "S": True, "E": True, "W": False},
            "edge_constraints": {
                "N": {"switch_locked"},
                "S": {"key_locked"},
                "E": {"item_gate", "combat"},
            },
        },
    )

    assert budget > 512
    assert budget <= 2048


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


def test_generate_room_no_puzzle_structure_cleanup_strips_block_clutter():
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        default_puzzle_room_scaffold_enabled=False,
        default_puzzle_room_structure_enabled=False,
    )

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, node_type="boss_door", pos=(0, 0))
    mission_graph.add_node(1, node_type="boss", pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="boss")

    graph_data = pipeline._prepare_graph_context(mission_graph, use_tpe=True)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )
    logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
    logits[:, int(SEMANTIC_PALETTE["FLOOR"]), :, :] = 4.0
    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    logits[:, block_id, 9:14, 7] = 9.0
    logits[:, block_id, 12, 6:9] = 9.0
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

    assert int(np.sum(result.room_grid == block_id)) == 0
    assert result.metrics["final_no_puzzle_structure_cleanup_applied"] == pytest.approx(1.0)
    assert result.metrics["final_no_puzzle_block_tiles_removed"] > 0


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
    semantics = pipeline._extract_room_topology_semantics(mission_graph, 0)
    role_flags = pipeline._room_role_flags(dict(mission_graph.nodes[0]))
    semantic_anchors = build_room_semantic_anchor_points(
        room_shape=(ROOM_HEIGHT, ROOM_WIDTH),
        start=start_goal[0],
        goal=start_goal[1],
        required_doors=semantics["required_doors"],
        incoming_dirs=semantics["incoming_dirs"],
        outgoing_dirs=semantics["outgoing_dirs"],
        room_role_flags=role_flags,
        semantic_puzzle_offset=pipeline.default_semantic_puzzle_offset,
    )
    profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs=dict(mission_graph.nodes[0]),
        role_flags=role_flags,
        semantics=semantics,
        node_type="switch",
    )
    template_mask = pipeline._build_puzzle_room_route_template(
        archetype=profile["archetype"],
        gate_family=profile["gate_family"],
        variant_spec={
            "name": str(result.metrics.get("final_puzzle_scaffold_variant_name", "") or "baseline"),
            "style": str(result.metrics.get("final_puzzle_scaffold_variant_style", "") or "baseline"),
            "side_bias": int(result.metrics.get("final_puzzle_scaffold_variant_side_bias", 0) or 0),
        },
        stateful_anchor=semantic_anchors.get("puzzle"),
        flow_is_horizontal=False,
        source_anchor=semantic_anchors.get("start", pipeline._clamp_room_coord(start_goal[0])),
        destination_anchor=semantic_anchors.get("goal", pipeline._clamp_room_coord(start_goal[1])),
        puzzle_anchor=semantic_anchors.get("puzzle", (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)),
        role_flags=role_flags,
        semantics=semantics,
    )
    route_tiles = result.room_grid[np.asarray(template_mask, dtype=bool)]
    disallowed = {int(SEMANTIC_PALETTE["WALL"]), int(SEMANTIC_PALETTE["BLOCK"])}
    assert route_tiles.size > 0
    assert result.metrics["final_puzzle_scaffold_route_template_used"] == pytest.approx(1.0)
    assert not any(int(tile) in disallowed for tile in route_tiles.tolist())


def test_puzzle_scaffold_profile_adapts_to_complex_puzzle_topology():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0

    attrs = {
        "type": "COMPLEX_PUZZLE",
        "has_puzzle": True,
        "difficulty_rating": "HARD",
    }
    role_flags = pipeline._room_role_flags(attrs)
    semantics = {
        "required_doors": {"N": True, "S": True, "E": True, "W": False},
        "edge_constraints": {"N": set(), "S": set(), "E": set(), "W": set()},
        "incoming_dirs": {"N"},
        "outgoing_dirs": {"S", "E"},
    }

    profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs=attrs,
        role_flags=role_flags,
        semantics=semantics,
        node_type="complex_puzzle",
    )

    assert profile["archetype"] in {"hub", "serpentine"}
    assert profile["branch_density"] >= 0.7
    assert profile["block_budget"] >= 26


def test_room_role_flags_treat_switch_and_complex_puzzle_types_as_puzzles():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)

    switch_flags = pipeline._room_role_flags({"type": "switch"})
    complex_flags = pipeline._room_role_flags({"type": "complex_puzzle"})

    assert switch_flags["has_puzzle"] is True
    assert switch_flags["is_switch_puzzle"] is True
    assert complex_flags["has_puzzle"] is True
    assert complex_flags["is_complex_puzzle"] is True


def test_puzzle_scaffold_profile_prefers_combat_archetype_for_combat_puzzle():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0

    attrs = {
        "type": "COMBAT_PUZZLE",
        "has_puzzle": True,
        "has_enemy": True,
        "difficulty_rating": "MODERATE",
    }
    role_flags = pipeline._room_role_flags(attrs)
    semantics = {
        "required_doors": {"N": False, "S": True, "E": True, "W": False},
        "edge_constraints": {"N": set(), "S": {"path"}, "E": {"path"}, "W": set()},
        "incoming_dirs": {"S"},
        "outgoing_dirs": {"E"},
    }

    profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs=attrs,
        role_flags=role_flags,
        semantics=semantics,
        node_type="combat_puzzle",
    )

    assert profile["archetype"] == "combat"
    assert profile["branch_density"] <= 0.45
    assert profile["block_budget"] <= 18


def test_puzzle_scaffold_profile_classifies_stateful_gate_families():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0

    common = {
        "required_doors": {"N": False, "S": False, "E": True, "W": True},
        "incoming_dirs": {"W"},
        "outgoing_dirs": {"E"},
    }
    role_flags = pipeline._room_role_flags({"type": "puzzle", "has_puzzle": True})

    bombable_profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs={"type": "puzzle", "has_puzzle": True},
        role_flags=role_flags,
        semantics={**common, "edge_constraints": {"N": set(), "S": set(), "E": {"bombable"}, "W": set()}},
        node_type="puzzle",
    )
    item_profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs={"type": "puzzle", "has_puzzle": True},
        role_flags=role_flags,
        semantics={**common, "edge_constraints": {"N": set(), "S": set(), "E": {"item_gate"}, "W": set()}},
        node_type="puzzle",
    )
    key_profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs={"type": "puzzle", "has_puzzle": True},
        role_flags=role_flags,
        semantics={**common, "edge_constraints": {"N": set(), "S": set(), "E": {"key_locked"}, "W": set()}},
        node_type="puzzle",
    )
    switch_profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs={"type": "switch", "has_puzzle": True},
        role_flags=pipeline._room_role_flags({"type": "switch", "has_puzzle": True}),
        semantics={**common, "edge_constraints": {"N": set(), "S": set(), "E": {"switch_locked"}, "W": set()}},
        node_type="switch",
    )
    toggle_profile = pipeline._resolve_puzzle_room_scaffold_profile(
        attrs={"type": "puzzle", "has_puzzle": True},
        role_flags=role_flags,
        semantics={**common, "edge_constraints": {"N": set(), "S": set(), "E": {"on_off_gate"}, "W": set()}},
        node_type="puzzle",
    )

    assert bombable_profile["gate_family"] == "bombable"
    assert bombable_profile["archetype"] == "gate"
    assert item_profile["gate_family"] == "item_unlock"
    assert item_profile["archetype"] == "gate"
    assert key_profile["gate_family"] == "key"
    assert key_profile["archetype"] == "gate"
    assert switch_profile["gate_family"] == "switch"
    assert switch_profile["archetype"] == "gate"
    assert toggle_profile["gate_family"] == "toggle"
    assert toggle_profile["archetype"] == "gate"


def test_puzzle_scaffold_cleans_small_interior_noise_components_before_layout():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])
    room[5, 5] = int(SEMANTIC_PALETTE["BLOCK"])
    room[10, 7] = int(SEMANTIC_PALETTE["BLOCK"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    cleaned, cleanup_stats = pipeline._strip_small_interior_structure_components(
        room,
        graph=mission_graph,
        room_id=0,
    )
    assert cleanup_stats["removed_components"] == 2
    assert cleanup_stats["removed_tiles"] == 2
    assert int(cleaned[5, 5]) != int(SEMANTIC_PALETTE["BLOCK"])
    assert int(cleaned[10, 7]) != int(SEMANTIC_PALETTE["BLOCK"])

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    assert stats["noise_components_removed"] == 2
    assert stats["noise_tiles_removed"] == 2
    assert int(np.sum(out == int(SEMANTIC_PALETTE["BLOCK"]))) >= 3


def test_gate_puzzle_scaffold_uses_route_template_and_builds_transverse_gate():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    row_block_counts = np.sum(out == block_id, axis=1)

    assert stats["route_template_used"] == 1
    assert stats["archetype"] == "gate"
    assert stats["gate_family"] == "switch"
    assert int(np.max(row_block_counts)) >= 3


def test_gate_puzzle_scaffold_adds_readable_push_block_prop_near_interaction_zone():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[:, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    route_cells = {(int(r), ROOM_WIDTH // 2) for r in range(ROOM_HEIGHT)}
    isolated_blocks = []
    for row, col in np.argwhere(out == block_id):
        neighbors = 0
        for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_r = int(row) + d_r
            next_c = int(col) + d_c
            if 0 <= next_r < ROOM_HEIGHT and 0 <= next_c < ROOM_WIDTH and int(out[next_r, next_c]) == block_id:
                neighbors += 1
        if neighbors <= 1 and (int(row), int(col)) not in route_cells:
            isolated_blocks.append((int(row), int(col)))

    assert stats["push_block_props_added"] >= 1
    assert isolated_blocks


def test_switch_gate_puzzle_scaffold_exposes_push_interaction_geometry():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[:, ROOM_WIDTH // 2] = 1.0
    _out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    assert stats["route_template_used"] == 1
    assert stats["interaction_valid"] == 1
    assert stats["interaction_push_slot_count"] >= 1


def test_empty_switch_room_is_interaction_invalid_without_scaffold_geometry():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])
    route_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    route_mask[:, ROOM_WIDTH // 2] = True
    source = (1, ROOM_WIDTH // 2)
    destination = (ROOM_HEIGHT - 2, ROOM_WIDTH // 2)
    stateful = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2 - 2)

    route_quality = pipeline._evaluate_puzzle_candidate_route_quality(
        grid=room,
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_mask=route_mask,
        gate_family="switch",
        baseline_path_length=None,
    )
    interaction = pipeline._evaluate_puzzle_candidate_interaction_geometry(
        grid=room,
        gate_family="switch",
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_mask=route_mask,
        route_quality=route_quality,
    )

    assert interaction["valid"] == 0
    assert "missing_push_interaction" in interaction["failure_reasons"]


def test_complex_puzzle_sequence_resolves_multiple_local_interactions():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)

    semantic_anchors = {
        "key": (ROOM_HEIGHT // 2, 2),
        "item": (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2),
        "puzzle": (ROOM_HEIGHT // 2, ROOM_WIDTH // 2),
        "enemy": (ROOM_HEIGHT // 2 + 2, ROOM_WIDTH - 3),
    }
    sequence = pipeline._resolve_puzzle_interaction_sequence(
        archetype="hub",
        gate_family="generic",
        role_flags={
            "has_key": True,
            "has_item": True,
            "has_puzzle": True,
            "has_enemy": True,
            "is_complex_puzzle": True,
        },
        semantic_anchors=semantic_anchors,
    )

    assert [name for name, _anchor in sequence] == ["key", "item", "puzzle", "enemy"]


def test_complex_puzzle_route_template_covers_interaction_sequence():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)

    source = (1, ROOM_WIDTH // 2)
    destination = (ROOM_HEIGHT - 2, ROOM_WIDTH // 2)
    puzzle_anchor = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)
    sequence = [
        ("key", (ROOM_HEIGHT // 2, 2)),
        ("item", (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2)),
        ("puzzle", puzzle_anchor),
    ]
    route_mask = pipeline._build_puzzle_room_route_template(
        archetype="hub",
        gate_family="generic",
        variant_spec={"name": "baseline", "style": "baseline"},
        stateful_anchor=puzzle_anchor,
        interaction_sequence=sequence,
        flow_is_horizontal=False,
        source_anchor=source,
        destination_anchor=destination,
        puzzle_anchor=puzzle_anchor,
        role_flags={"has_puzzle": True, "is_complex_puzzle": True},
        semantics={"required_doors": {}, "incoming_dirs": set(), "outgoing_dirs": set()},
    )
    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    seq_eval = pipeline._evaluate_puzzle_candidate_interaction_sequence(
        grid=room,
        route_mask=route_mask,
        source_anchor=source,
        destination_anchor=destination,
        interaction_sequence=sequence,
    )

    assert seq_eval["required"] == 1
    assert seq_eval["valid"] == 1
    assert seq_eval["sequence_length"] == 3
    assert seq_eval["route_anchor_coverage"] == pytest.approx(1.0)


def test_bombable_gate_puzzle_scaffold_builds_offset_bypass_instead_of_center_gap():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.default_puzzle_room_resource_bypass_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="puzzle", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="bombable")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    center_row = ROOM_HEIGHT // 2
    row_block_counts = np.sum(out == block_id, axis=1)

    assert stats["gate_family"] == "bombable"
    assert stats["stateful_anchor_name"] == "puzzle"
    assert stats["route_template_used"] == 1
    assert int(np.max(row_block_counts)) >= 3
    assert max(
        int(np.sum(row_block_counts[:center_row] >= 2)),
        int(np.sum(row_block_counts[center_row + 1 :] >= 2)),
    ) >= 1
    assert stats["interaction_valid"] == 1
    assert stats["interaction_route_divergence"] >= 0.1
    assert stats["interaction_barrier_axis_tiles"] >= 1


def test_item_unlock_puzzle_scaffold_prefers_item_anchor_when_present():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.default_puzzle_room_item_slot_depth = 3

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="puzzle", has_puzzle=True, has_item=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="item_gate")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    row_block_counts = np.sum(out == block_id, axis=1)

    assert stats["gate_family"] == "item_unlock"
    assert stats["stateful_anchor_name"] == "item"
    assert stats["route_template_used"] == 1
    assert int(np.max(row_block_counts)) >= 3


def test_toggle_gate_puzzle_scaffold_builds_state_corridor():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.default_puzzle_room_toggle_corridor_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="puzzle", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="on_off_gate")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    row_block_counts = np.sum(out == block_id, axis=1)

    assert stats["gate_family"] == "toggle"
    assert stats["stateful_anchor_name"] == "puzzle"
    assert stats["route_template_used"] == 1
    assert int(np.sum(row_block_counts >= 2)) >= 2


def test_serpentine_puzzle_scaffold_forms_multiple_baffles_in_complex_rooms():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="complex_puzzle", has_puzzle=True, difficulty_rating="HARD", pos=(0, 0))
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="path")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[ROOM_HEIGHT // 2, 1:ROOM_WIDTH - 1] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    row_block_counts = np.sum(out == block_id, axis=1)

    assert stats["route_template_used"] == 1
    assert stats["archetype"] == "serpentine"
    assert int(np.sum(row_block_counts >= 3)) >= 2


def test_hub_puzzle_scaffold_builds_meaningful_ring_for_four_door_complex_room():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(
        0,
        type="complex_puzzle",
        has_puzzle=True,
        difficulty_rating="HARD",
        pos=(1, 1),
    )
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_node(2, pos=(2, 1))
    mission_graph.add_node(3, pos=(1, 0))
    mission_graph.add_node(4, pos=(1, 2))
    mission_graph.add_edge(1, 0, edge_type="key_locked")
    mission_graph.add_edge(3, 0, edge_type="bombable")
    mission_graph.add_edge(0, 2, edge_type="path")
    mission_graph.add_edge(0, 4, edge_type="switch_locked")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[ROOM_HEIGHT // 2, 1:ROOM_WIDTH - 1] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    block_id = int(SEMANTIC_PALETTE["BLOCK"])
    block_count = int(np.sum(out == block_id))
    row_block_counts = np.sum(out == block_id, axis=1)

    assert stats["route_template_used"] == 1
    assert stats["archetype"] == "hub"
    assert block_count >= 8
    assert int(np.sum(row_block_counts >= 2)) >= 2


def test_puzzle_scaffold_novelty_diversifies_switch_variants_across_rooms():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.default_puzzle_room_switch_pocket_depth = 3
    pipeline.default_puzzle_room_resource_bypass_offset = 2
    pipeline.default_puzzle_room_key_pocket_depth = 3
    pipeline.default_puzzle_room_item_slot_depth = 3
    pipeline.default_puzzle_room_toggle_corridor_offset = 2
    pipeline.default_puzzle_room_novelty_enabled = True
    pipeline.default_puzzle_room_candidate_count = 4
    pipeline.default_puzzle_room_novelty_weight = 0.45
    pipeline._puzzle_novelty_history = []
    pipeline._puzzle_variant_cache = {}
    pipeline._puzzle_novelty_committed = set()

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])
    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[:, ROOM_WIDTH // 2] = 1.0

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, type="switch", has_puzzle=True, pos=(1, 0))
    mission_graph.add_node(2, pos=(0, 1))
    mission_graph.add_node(3, pos=(1, 1))
    mission_graph.add_edge(0, 2, edge_type="switch_locked")
    mission_graph.add_edge(1, 3, edge_type="switch_locked")

    _room_a, stats_a = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )
    pipeline._commit_puzzle_novelty_choice(room_id=0, scaffold_stats=stats_a)

    _room_b, stats_b = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=1,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    assert stats_a["gate_family"] == "switch"
    assert stats_b["gate_family"] == "switch"
    assert stats_a["variant_name"] != stats_b["variant_name"]


def test_puzzle_route_quality_rewards_stateful_readability():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    route_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    route_mask[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = True
    source = (1, ROOM_WIDTH // 2)
    destination = (ROOM_HEIGHT - 2, ROOM_WIDTH // 2)
    stateful = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)

    readable = room.copy()
    readable[ROOM_HEIGHT // 2, ROOM_WIDTH // 2 - 3:ROOM_WIDTH // 2] = int(SEMANTIC_PALETTE["BLOCK"])

    noisy = room.copy()
    noisy[2:ROOM_HEIGHT - 2, ROOM_WIDTH // 2 - 3] = int(SEMANTIC_PALETTE["BLOCK"])

    readable_quality = pipeline._evaluate_puzzle_candidate_route_quality(
        grid=readable,
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_mask=route_mask,
        gate_family="switch",
        baseline_path_length=None,
    )
    noisy_quality = pipeline._evaluate_puzzle_candidate_route_quality(
        grid=noisy,
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=(ROOM_HEIGHT // 2, 2),
        route_mask=route_mask,
        gate_family="switch",
        baseline_path_length=None,
    )

    assert readable_quality["path_exists"] == 1
    assert noisy_quality["path_exists"] == 1
    assert readable_quality["score"] > noisy_quality["score"]


def test_puzzle_candidate_scoring_uses_route_quality_not_only_novelty():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_novelty_weight = 0.0
    pipeline._puzzle_novelty_history = []

    base_descriptor = {
        "variant_name": "candidate",
        "gate_family": "switch",
        "archetype": "gate",
        "tiles_added": 10,
        "segments_added": 3,
        "row_coverage": 4,
        "col_coverage": 3,
        "center_row": float(ROOM_HEIGHT // 2),
        "center_col": float(ROOM_WIDTH // 2),
        "quadrants": [2, 2, 2, 2],
    }
    low_quality_stats = {
        "profile_block_budget": 28,
        "tiles_added": 10,
        "segments_added": 3,
        "optional_segments_applied": 1,
        "route_quality_score": 0.2,
    }
    high_quality_stats = {
        "profile_block_budget": 28,
        "tiles_added": 10,
        "segments_added": 3,
        "optional_segments_applied": 1,
        "route_quality_score": 1.6,
    }

    low_score = pipeline._score_puzzle_candidate(
        descriptor=base_descriptor,
        stats=low_quality_stats,
        room_id=0,
    )
    high_score = pipeline._score_puzzle_candidate(
        descriptor=base_descriptor,
        stats=high_quality_stats,
        room_id=0,
    )

    assert high_score > low_score


def test_puzzle_contract_rewards_readable_stateful_anchor_pocket():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])
    source = (1, ROOM_WIDTH // 2)
    destination = (ROOM_HEIGHT - 2, ROOM_WIDTH // 2)
    stateful = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2 - 2)
    route_mask = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    route_mask[1:ROOM_HEIGHT - 1, ROOM_WIDTH // 2] = True

    readable = room.copy()
    readable[ROOM_HEIGHT // 2 - 2:ROOM_HEIGHT // 2 + 3, ROOM_WIDTH // 2 - 4] = int(SEMANTIC_PALETTE["BLOCK"])
    readable[ROOM_HEIGHT // 2 - 2:ROOM_HEIGHT // 2 + 3, ROOM_WIDTH // 2] = int(SEMANTIC_PALETTE["BLOCK"])

    noisy = room.copy()

    readable_quality = pipeline._evaluate_puzzle_candidate_route_quality(
        grid=readable,
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_mask=route_mask,
        gate_family="key",
        baseline_path_length=None,
    )
    noisy_quality = pipeline._evaluate_puzzle_candidate_route_quality(
        grid=noisy,
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_mask=route_mask,
        gate_family="key",
        baseline_path_length=None,
    )
    readable_contract = pipeline._evaluate_puzzle_candidate_contract(
        grid=readable,
        gate_family="key",
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_quality=readable_quality,
    )
    noisy_contract = pipeline._evaluate_puzzle_candidate_contract(
        grid=noisy,
        gate_family="key",
        source_anchor=source,
        destination_anchor=destination,
        stateful_anchor=stateful,
        route_quality=noisy_quality,
    )

    assert readable_contract["valid"] == 1
    assert noisy_contract["valid"] == 0
    assert readable_contract["score"] > noisy_contract["score"]


def test_puzzle_candidate_scoring_penalizes_contract_failures():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_novelty_weight = 0.0
    pipeline._puzzle_novelty_history = []

    descriptor = {
        "variant_name": "candidate",
        "gate_family": "key",
        "archetype": "gate",
        "tiles_added": 10,
        "segments_added": 3,
        "row_coverage": 4,
        "col_coverage": 3,
        "center_row": float(ROOM_HEIGHT // 2),
        "center_col": float(ROOM_WIDTH // 2),
        "quadrants": [2, 2, 2, 2],
    }
    good_stats = {
        "profile_block_budget": 28,
        "tiles_added": 10,
        "segments_added": 3,
        "optional_segments_applied": 1,
        "route_quality_score": 1.0,
        "contract_score": 0.8,
        "contract_valid": 1,
    }
    bad_stats = {
        "profile_block_budget": 28,
        "tiles_added": 10,
        "segments_added": 3,
        "optional_segments_applied": 1,
        "route_quality_score": 1.0,
        "contract_score": -0.4,
        "contract_valid": 0,
    }

    good_score = pipeline._score_puzzle_candidate(
        descriptor=descriptor,
        stats=good_stats,
        room_id=0,
    )
    bad_score = pipeline._score_puzzle_candidate(
        descriptor=descriptor,
        stats=bad_stats,
        room_id=0,
    )

    assert good_score > bad_score


def test_puzzle_scaffold_skips_when_candidate_does_not_clear_quality_gate():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.default_puzzle_room_switch_pocket_depth = 3
    pipeline.default_puzzle_room_resource_bypass_offset = 2
    pipeline.default_puzzle_room_key_pocket_depth = 3
    pipeline.default_puzzle_room_item_slot_depth = 3
    pipeline.default_puzzle_room_toggle_corridor_offset = 2
    pipeline.default_puzzle_room_novelty_enabled = True
    pipeline.default_puzzle_room_candidate_count = 4
    pipeline.default_puzzle_room_novelty_weight = 0.45
    pipeline.default_puzzle_room_min_quality_gain = 20.0
    pipeline._puzzle_novelty_history = []
    pipeline._puzzle_variant_cache = {}
    pipeline._puzzle_novelty_committed = set()

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0, type="switch", has_puzzle=True, pos=(0, 0))
    mission_graph.add_node(1, pos=(1, 0))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[:, ROOM_WIDTH // 2] = 1.0
    out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((0, ROOM_WIDTH // 2), (ROOM_HEIGHT - 1, ROOM_WIDTH // 2)),
    )

    assert stats["quality_gate_skipped"] == 1
    assert stats["applied"] == 0
    assert np.array_equal(out, room)


def test_complex_puzzle_scaffold_reports_multi_step_sequence_metrics():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_puzzle_room_scaffold_enabled = True
    pipeline.default_puzzle_room_scaffold_min_structure_tiles = 10
    pipeline.default_puzzle_room_archetype_mode = "auto"
    pipeline.default_puzzle_room_branch_density = 0.75
    pipeline.default_puzzle_room_block_budget = 28
    pipeline.default_puzzle_room_preserve_route_margin = 0
    pipeline.default_semantic_puzzle_offset = 2
    pipeline.default_puzzle_room_resource_bypass_offset = 2
    pipeline.default_puzzle_room_key_pocket_depth = 3
    pipeline.default_puzzle_room_item_slot_depth = 3
    pipeline.default_puzzle_room_toggle_corridor_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(
        0,
        type="complex_puzzle",
        has_puzzle=True,
        has_key=True,
        has_item=True,
        has_enemy=True,
        difficulty_rating="HARD",
        pos=(1, 1),
    )
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_node(2, pos=(2, 1))
    mission_graph.add_node(3, pos=(1, 0))
    mission_graph.add_node(4, pos=(1, 2))
    mission_graph.add_edge(1, 0, edge_type="key_locked")
    mission_graph.add_edge(3, 0, edge_type="bombable")
    mission_graph.add_edge(0, 2, edge_type="item_gate")
    mission_graph.add_edge(0, 4, edge_type="switch_locked")

    route = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)
    route[ROOM_HEIGHT // 2, 1:ROOM_WIDTH - 1] = 1.0
    _out, stats = pipeline._apply_puzzle_room_scaffold(
        room,
        graph=mission_graph,
        room_id=0,
        room_plan_mask=route,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    assert stats["route_template_used"] == 1
    assert stats["interaction_sequence_length"] >= 2
    assert stats["interaction_sequence_valid"] == 1
    assert stats["interaction_sequence_route_anchor_coverage"] == pytest.approx(1.0)


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


def test_compute_room_condition_keeps_batch_dim_stable_when_puzzle_control_runs_in_cross_attention_fallback(
    monkeypatch,
):
    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        room_generator_mode="latent_diffusion",
        use_graph_node_cross_attention=True,
        diffusion_fallback_config={"puzzle_structure_dropout_prob": 0.35},
    )

    graph_context = {
        "node_features": torch.randn(3, 12),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_features": torch.randn(2, 14),
        "tpe": torch.randn(3, 8),
        "current_node_idx": 0,
        "puzzle_room_structure_enabled": False,
    }

    def _fail_forward(**kwargs):
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(pipeline.condition_encoder, "forward", _fail_forward)

    condition = pipeline._compute_room_condition(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        boundary_constraints=torch.zeros(1, 8),
        position=torch.zeros(1, 2),
    )

    assert tuple(condition.shape) == (1, 5, 256)


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


def test_room_puzzle_metadata_builds_ordered_stateful_plan():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    pipeline.default_semantic_puzzle_offset = 2

    room = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32)
    room[0, :] = int(SEMANTIC_PALETTE["WALL"])
    room[-1, :] = int(SEMANTIC_PALETTE["WALL"])
    room[:, 0] = int(SEMANTIC_PALETTE["WALL"])
    room[:, -1] = int(SEMANTIC_PALETTE["WALL"])

    spec = DOOR_POSITIONS["E"]
    room[int(spec["row_start"]): int(spec["row_end"]) + 1, int(spec["col"])] = int(SEMANTIC_PALETTE["DOOR_PUZZLE"])

    mission_graph = nx.DiGraph()
    mission_graph.add_node(
        0,
        type="complex_puzzle",
        has_puzzle=True,
        has_key=True,
        has_item=True,
        pos=(0, 0),
    )
    mission_graph.add_node(1, pos=(0, 1))
    mission_graph.add_edge(0, 1, edge_type="switch_locked")

    marker_plan = [
        (int(SEMANTIC_PALETTE["KEY_SMALL"]), (ROOM_HEIGHT // 2, 2)),
        (int(SEMANTIC_PALETTE["KEY_ITEM"]), (ROOM_HEIGHT // 2 - 2, ROOM_WIDTH // 2)),
        (int(SEMANTIC_PALETTE["PUZZLE"]), (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)),
    ]
    metadata = pipeline._build_room_puzzle_metadata(
        grid=room,
        graph=mission_graph,
        room_id=0,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
        marker_plan=marker_plan,
        scaffold_stats={
            "gate_family": "switch",
            "archetype": "hub",
            "interaction_valid": 1,
            "contract_valid": 1,
            "interaction_sequence_valid": 1,
            "interaction_sequence_required": 1,
        },
    )

    assert metadata["gate_family"] == "switch"
    assert [stage["kind"] for stage in metadata["stage_sequence"]] == [
        "collect_key",
        "collect_item",
        "push_block_to_switch",
    ]
    assert len(metadata["controlled_doors_local"]) >= 1


def test_globalize_room_puzzle_metadata_offsets_stage_and_door_coordinates():
    pipeline = NeuralSymbolicDungeonPipeline.__new__(NeuralSymbolicDungeonPipeline)
    room_result = RoomGenerationResult(
        room_id=7,
        room_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32),
        latent=torch.empty(0),
        neural_grid=np.full((ROOM_HEIGHT, ROOM_WIDTH), int(SEMANTIC_PALETTE["FLOOR"]), dtype=np.int32),
        was_repaired=False,
        puzzle_metadata={
            "plan_id": "room_7",
            "controlled_doors_local": [[1, ROOM_WIDTH - 1]],
            "stage_sequence": [
                {
                    "stage_index": 0,
                    "name": "puzzle",
                    "kind": "push_block_to_switch",
                    "local_anchor": [3, 4],
                }
            ],
        },
        metrics={},
    )
    stitched_layout = StitchedRoomLayout(
        dungeon_grid=np.zeros((ROOM_HEIGHT * 2, ROOM_WIDTH * 2), dtype=np.int32),
        slot_positions={7: (0, 0)},
        room_offsets={7: (20, 30)},
        layout_map={7: (30, 20, 30 + ROOM_WIDTH - 1, 20 + ROOM_HEIGHT - 1)},
    )

    payload = pipeline._globalize_room_puzzle_metadata(
        rooms={7: room_result},
        stitched_layout=stitched_layout,
    )

    plan = payload["plans"]["room_7"]
    assert plan["room_offset"] == [20, 30]
    assert plan["stage_sequence"][0]["global_anchor"] == [23, 34]
    assert plan["controlled_doors_global"][0] == [21, 30 + ROOM_WIDTH - 1]
