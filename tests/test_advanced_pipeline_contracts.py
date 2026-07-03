import numpy as np
import networkx as nx
from types import SimpleNamespace

from src.generation.style_transfer import ThemeType
from src.core.definitions import SEMANTIC_PALETTE
from src.generation.graph_constraint_enforcer import GraphConstraintEnforcer, enforce_all_rooms
from src.simulation.edge_logic import can_traverse_edge_type, edge_type_from_data
from src.simulation.validator import GameState, StateSpaceAStar, ZeldaLogicEnv, ZeldaValidator
import src.pipeline.advanced_pipeline as advanced_pipeline_module
from src.pipeline.advanced_pipeline import (
    AdvancedNeuralSymbolicPipeline,
    AdvancedPipelineConfig,
)


def _make_test_config() -> AdvancedPipelineConfig:
    return AdvancedPipelineConfig(
        use_lcm_lora=True,
        strict_checkpoint_mode=False,
        enable_seam_smoothing=False,
        enable_collision_validation=False,
        enable_big_rooms=False,
        enable_global_state=False,
        calculate_fun_metrics=False,
        enable_diversity_analysis=False,
        record_demo=False,
        enable_explainability=False,
    )


def test_hazard_edges_compile_to_element_only_when_protection_is_required():
    enforcer = GraphConstraintEnforcer(
        {
            "wall": SEMANTIC_PALETTE["WALL"],
            "floor": SEMANTIC_PALETTE["FLOOR"],
            "door": SEMANTIC_PALETTE["DOOR_OPEN"],
            "hazard": SEMANTIC_PALETTE["ELEMENT"],
        }
    )

    assert enforcer._door_tile_for_edge(
        {"edge_type": "HAZARD", "protection_item_id": "FIRE_TUNIC"}
    ) == SEMANTIC_PALETTE["ELEMENT"]
    assert enforcer._door_tile_for_edge(
        {"edge_type": "HAZARD"}
    ) == SEMANTIC_PALETTE["DOOR_OPEN"]


def test_protected_hazard_metadata_requires_generic_traversal_item():
    edge_type = edge_type_from_data(
        {"edge_type": "HAZARD", "protection_item_id": "FIRE_TUNIC"}
    )
    callbacks = {
        "strict_original_mode": False,
        "get_room_for_position": lambda _position: None,
        "is_room_cleared": lambda _room, _state: False,
    }

    assert edge_type == "hazard_protected"
    assert not can_traverse_edge_type(
        edge_type,
        GameState(position=(0, 0), has_item=False),
        **callbacks,
    )
    assert can_traverse_edge_type(
        edge_type,
        GameState(position=(0, 0), has_item=True),
        **callbacks,
    )
    assert edge_type_from_data({"edge_type": "HAZARD"}) == "hazard"


def test_graph_transition_does_not_bypass_protected_hazard():
    grid = np.full((16, 22), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
    grid[1, 1] = SEMANTIC_PALETTE["START"]
    grid[1, 12] = SEMANTIC_PALETTE["TRIFORCE"]
    graph = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        edge_type="HAZARD",
        protection_item_id="FIRE_TUNIC",
    )
    room_positions = {(0, 0): (0, 0), (0, 1): (0, 11)}
    room_to_node = {(0, 0): 0, (0, 1): 1}
    env = ZeldaLogicEnv(
        grid,
        graph=graph,
        room_positions=room_positions,
        room_to_node=room_to_node,
        node_to_room={0: (0, 0), 1: (0, 1)},
    )
    solver = StateSpaceAStar(env)
    edge_type = solver._edge_type_from_data(graph.get_edge_data(0, 1))

    blocked, _ = solver.apply_graph_edge_transition(
        GameState(position=(1, 1), has_item=False),
        (1, 1),
        (1, 12),
        edge_type,
    )
    allowed, _ = solver.apply_graph_edge_transition(
        GameState(position=(1, 1), has_item=True),
        (1, 1),
        (1, 12),
        edge_type,
    )

    assert edge_type == "hazard_protected"
    assert blocked is False
    assert allowed is True


def test_spatial_pipeline_rejects_multiple_unrepresentable_protection_identities():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"edge_type": "HAZARD", "protection_item_id": "FIRE_TUNIC"}),
            (1, 2, {"edge_type": "HAZARD", "protection_item_id": "SPIKE_BOOTS"}),
        ]
    )
    pipeline = object.__new__(AdvancedNeuralSymbolicPipeline)

    with np.testing.assert_raises_regex(ValueError, "multiple named hazard protections"):
        pipeline._validate_spatial_mechanics(graph)


def test_advanced_pipeline_disables_requested_lcm_without_real_backend():
    """Requested LCM-LoRA should not activate when only the experimental path exists."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())

    assert pipeline.fast_sampling_active is False
    assert "no distilled consistency-LoRA checkpoint" in pipeline.fast_sampling_reason


def test_advanced_pipeline_uses_standard_diffusion_steps_without_real_lcm(monkeypatch):
    """Advanced pipeline should keep standard DDIM steps when no real LCM backend is active."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())
    captured = {}

    class _RoomResult:
        def __init__(self):
            self.room_grid = np.zeros((16, 11), dtype=np.int32)

    def fake_generate_room(**kwargs):
        captured["num_diffusion_steps"] = kwargs["num_diffusion_steps"]
        return _RoomResult()

    monkeypatch.setattr(pipeline.neural_pipeline, "generate_room", fake_generate_room)

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0)
    room = pipeline._generate_single_room_with_ml(
        node_id=0,
        mission_graph=mission_graph,
        graph_context={},
        neighbor_latents={},
        theme=ThemeType.ZELDA_CLASSIC,
    )

    assert room.shape == (16, 11)
    assert captured["num_diffusion_steps"] == 50


def test_advanced_pipeline_activates_compatible_consistency_lora_backend(monkeypatch, tmp_path):
    """A validated repo fast-sampler adapter should flow into the core generation pipeline."""
    adapter = tmp_path / "fast_sampler_best.pth"
    adapter.write_bytes(b"adapter")
    base = tmp_path / "best_model.pth"
    base.write_bytes(b"base")
    captured = {}

    def fake_load_fast_sampler_checkpoint(path):
        assert str(path) == str(adapter)
        return {}, SimpleNamespace(
            distillation_type="consistency_lora",
            base_diffusion_checkpoint=str(base),
            num_inference_steps=4,
            lora_rank=8,
            lora_alpha=8.0,
            target_modules=(),
        )

    class _FakeDiffusion:
        def supports_fast_sampling(self):
            return True

    class _FakePipeline:
        def __init__(self, **kwargs):
            captured["constructor_kwargs"] = kwargs
            self.diffusion = _FakeDiffusion()
            self.default_guidance_scale = 3.0
            self.default_logic_guidance_scale = 0.0
            self.default_apply_repair = False
            self.default_start_goal_coords = None

        def generate_room(self, **kwargs):
            captured["generate_room_kwargs"] = kwargs

            class _RoomResult:
                room_grid = np.zeros((16, 11), dtype=np.int32)

            return _RoomResult()

    monkeypatch.setattr(advanced_pipeline_module, "load_fast_sampler_checkpoint", fake_load_fast_sampler_checkpoint)
    monkeypatch.setattr(advanced_pipeline_module, "NeuralSymbolicDungeonPipeline", _FakePipeline)

    pipeline = AdvancedNeuralSymbolicPipeline(
        AdvancedPipelineConfig(
            use_lcm_lora=True,
            lcm_lora_checkpoint=adapter,
            lcm_steps=4,
            enable_seam_smoothing=False,
            enable_collision_validation=False,
            enable_big_rooms=False,
            enable_global_state=False,
            calculate_fun_metrics=False,
            enable_diversity_analysis=False,
            record_demo=False,
            enable_explainability=False,
        )
    )

    assert pipeline.fast_sampling_active is True
    constructor_kwargs = captured["constructor_kwargs"]
    assert constructor_kwargs["fast_sampling_checkpoint"] == str(adapter)
    assert constructor_kwargs["diffusion_checkpoint"] == str(base)
    assert constructor_kwargs["default_use_fast_sampling"] is True

    mission_graph = nx.DiGraph()
    mission_graph.add_node(0)
    pipeline._generate_single_room_with_ml(
        node_id=0,
        mission_graph=mission_graph,
        graph_context={},
        neighbor_latents={},
        theme=ThemeType.ZELDA_CLASSIC,
    )

    generate_kwargs = captured["generate_room_kwargs"]
    assert generate_kwargs["use_fast_sampling"] is True
    assert generate_kwargs["num_diffusion_steps"] == 4


def test_advanced_pipeline_reports_no_lcm_speedup_without_real_backend():
    """Speedup is undefined without both a real fast backend and paired baseline."""
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())

    assert pipeline._compute_reported_lcm_speedup(room_count=8, gen_time=12.0) is None


def test_advanced_pipeline_reports_only_explicit_paired_speedup():
    config = _make_test_config()
    config.paired_baseline_generation_time_sec = 24.0
    pipeline = AdvancedNeuralSymbolicPipeline(config)
    pipeline.fast_sampling_active = True

    assert pipeline._compute_reported_lcm_speedup(room_count=8, gen_time=12.0) == 2.0


def test_advanced_pipeline_uses_canonical_graph_conditioning_schema():
    pipeline = AdvancedNeuralSymbolicPipeline(_make_test_config())
    graph = nx.DiGraph()
    graph.add_node(0, type="START", position=(0, 0))
    graph.add_node(1, type="GOAL", position=(1, 0))
    graph.add_edge(0, 1, edge_type="key_locked")

    context = pipeline._prepare_graph_context(graph)
    expected_node_dim = pipeline.neural_pipeline.condition_encoder.global_encoder.node_feature_dim
    expected_edge_dim = pipeline.neural_pipeline.condition_encoder.global_encoder.edge_feature_dim

    assert tuple(context["node_features"].shape) == (2, expected_node_dim)
    assert tuple(context["edge_features"].shape) == (1, expected_edge_dim)
    assert tuple(context["tpe"].shape) == (2, 8)
    assert context["node_to_idx"] == {0: 0, 1: 1}


def test_weighted_wfc_priors_are_loaded_from_empirical_grids(tmp_path):
    source = tmp_path / "prior_grids.npz"
    np.savez(
        source,
        rooms=np.stack(
            [
                np.full((4, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64),
                np.full((4, 4), SEMANTIC_PALETTE["WALL"], dtype=np.int64),
                np.tile(
                    np.array(
                        [SEMANTIC_PALETTE["WALL"], SEMANTIC_PALETTE["FLOOR"]],
                        dtype=np.int64,
                    ),
                    (4, 2),
                ),
            ]
        ),
    )

    pipeline = object.__new__(AdvancedNeuralSymbolicPipeline)
    pipeline.config = AdvancedPipelineConfig(
        wfc_prior_grids_path=source,
        wfc_min_prior_grids=3,
    )

    priors = pipeline._load_wfc_tile_priors()

    assert SEMANTIC_PALETTE["FLOOR"] in priors
    assert SEMANTIC_PALETTE["WALL"] in priors
    assert pipeline._wfc_prior_source == str(source.resolve())
    assert pipeline._wfc_prior_grid_count == 3
    assert len(pipeline._wfc_prior_sha256) == 64


def test_advanced_pipeline_fun_evaluation_resolves_graph_route_not_insertion_order():
    """Pacing should follow the mission path even when nodes were inserted out of order."""
    graph = nx.DiGraph()
    graph.add_node(2, is_boss=True, is_triforce=True)
    graph.add_node(0, is_start=True)
    graph.add_node(1, type="BIG_KEY")
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="boss_locked")

    assert AdvancedNeuralSymbolicPipeline._resolve_mission_solution_path(graph) == [0, 1, 2]


def test_graph_to_grid_compiler_preserves_one_lock_as_one_consumable_gate():
    """A physical room connection must not duplicate one graph lock into two locks."""
    palette = SEMANTIC_PALETTE
    tile_config = {
        "wall": int(palette["WALL"]),
        "floor": int(palette["FLOOR"]),
        "door": int(palette["DOOR_OPEN"]),
        "door_locked": int(palette["DOOR_LOCKED"]),
        "door_bomb": int(palette["DOOR_BOMB"]),
        "door_puzzle": int(palette["DOOR_PUZZLE"]),
        "door_boss": int(palette["DOOR_BOSS"]),
        "door_soft": int(palette["DOOR_SOFT"]),
        "start": int(palette["START"]),
        "goal": int(palette["TRIFORCE"]),
    }
    grid = np.full((7, 14), int(palette["FLOOR"]), dtype=np.int64)
    mission_graph = {
        "nodes": {0: {"type": "START"}, 1: {"type": "GOAL"}},
        "edges": [(0, 1, {"edge_type": "LOCKED"})],
    }
    layout = {0: (0, 0, 6, 6), 1: (7, 0, 13, 6)}

    compiled = enforce_all_rooms(grid, mission_graph, layout, tile_config)
    compiled[2, 3] = int(palette["KEY_SMALL"])

    assert sorted((int(compiled[3, 6]), int(compiled[3, 7]))) == sorted(
        (int(palette["DOOR_LOCKED"]), int(palette["DOOR_OPEN"]))
    )
    assert ZeldaValidator().validate_single(compiled).is_solvable is True
    capped = ZeldaValidator().validate_single(compiled, solver_timeout=1)
    assert capped.termination_status == "budget_exhausted"
    assert capped.proven_unsolvable is False


def test_advanced_pipeline_fun_contents_preserve_graph_and_entity_semantics():
    """Analyzer inputs should retain boss, goal, puzzle, lock, health, and treasure signals."""
    graph = nx.DiGraph()
    graph.add_node(0, type="start", is_start=True)
    graph.add_node(1, type="puzzle", has_puzzle=True)
    graph.add_node(2, type="boss", is_boss=True, is_triforce=True)
    graph.add_edge(0, 1, edge_type="open")
    graph.add_edge(1, 2, edge_type="boss_locked")
    entities = [
        SimpleNamespace(room_id=2, entity_type=SimpleNamespace(value="enemy_boss")),
        SimpleNamespace(room_id=2, entity_type=SimpleNamespace(value="health_potion")),
        SimpleNamespace(room_id=2, entity_type=SimpleNamespace(value="chest")),
    ]

    contents = AdvancedNeuralSymbolicPipeline._build_fun_room_contents(graph, entities)

    assert contents[1]["puzzles"] == 1
    assert contents[2]["boss"] is True
    assert contents[2]["goal"] is True
    assert contents[2]["locks"] == 1
    assert contents[2]["enemies"] == 1
    assert contents[2]["health_pickups"] == 1
    assert contents[2]["treasures"] == 1
