"""
Integration Tests for Neural-Symbolic Dungeon Pipeline
=======================================================

Tests the complete 7-block pipeline end-to-end.

Run:
    pytest tests/test_neural_pipeline.py -v
    
    # Quick test
    pytest tests/test_neural_pipeline.py -v -k "test_pipeline_initialization"
"""

import pytest
import torch
import numpy as np
import networkx as nx
import src.pipeline.dungeon_pipeline as dungeon_pipeline_module

from src.pipeline import (
    NeuralSymbolicDungeonPipeline,
    MissingPipelineComponentError,
    PipelineComponents,
    SymbolicGenerationComponents,
    create_pipeline,
)
from src.pipeline.dungeon_pipeline import RoomGenerationResult
from src.core import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for tests to avoid GPU memory issues."""
    return 'cpu'


@pytest.fixture
def pipeline(device):
    """Create pipeline without checkpoints (random init)."""
    return NeuralSymbolicDungeonPipeline(
        device=device,
        enable_logging=False,
    )


@pytest.fixture
def simple_graph():
    """Create a simple 3-room mission graph."""
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2)])
    return G


@pytest.fixture
def neighbor_latents(device):
    """Create dummy neighbor latents."""
    return {
        'N': torch.randn(1, 64, 4, 3, device=device),
        'S': None,
        'E': torch.randn(1, 64, 4, 3, device=device),
        'W': None,
    }


@pytest.fixture
def graph_context(device):
    """Create dummy graph context."""
    return {
        'node_features': torch.randn(3, 6, device=device),
        'edge_index': torch.tensor([[0, 1], [1, 2]], device=device).t(),
        'tpe': torch.randn(3, 8, device=device),
        'current_node_idx': 0,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

def test_pipeline_initialization(pipeline):
    """Test that pipeline initializes all components."""
    assert pipeline.vqvae is not None
    assert pipeline.condition_encoder is not None
    assert pipeline.diffusion is not None
    assert pipeline.logic_net is not None
    assert pipeline.refiner is not None
    assert pipeline.map_elites is not None
    
    print("✓ All 7 blocks initialized successfully")


def test_pipeline_device(device):
    """Test that pipeline respects device setting."""
    pipeline = NeuralSymbolicDungeonPipeline(device=device, enable_logging=False)
    assert str(pipeline.device) == device
    
    # Check model devices
    assert next(pipeline.vqvae.parameters()).device.type == device
    assert next(pipeline.diffusion.parameters()).device.type == device
    
    print(f"✓ Pipeline correctly initialized on {device}")


def test_create_pipeline_convenience():
    """Test convenience function for pipeline creation."""
    # Should work even without checkpoint directory
    pipeline = create_pipeline(
        checkpoint_dir="./nonexistent",
        device='cpu'
    )
    
    assert isinstance(pipeline, NeuralSymbolicDungeonPipeline)
    print("✓ Convenience function works correctly")


def test_symbolic_only_pipeline_skips_neural_stack_initialization():
    """Symbolic-only constructor should avoid building the neural generation stack."""
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device='cpu',
        enable_logging=False,
    )

    status = pipeline.component_status()
    assert status['vqvae'] is False
    assert status['condition_encoder'] is False
    assert status['diffusion'] is False
    assert status['logic_net'] is False
    assert status['refiner'] is True
    assert pipeline.supports_room_generation() is False
    assert pipeline.supports_symbolic_repair() is True


def test_symbolic_only_pipeline_repair_room_public_api():
    """Symbolic-only pipelines should still expose a working repair entry point."""
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device='cpu',
        enable_logging=False,
    )
    floor = int(SEMANTIC_PALETTE['FLOOR'])
    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32)

    repaired, success, diagnostics = pipeline.repair_room(
        room_grid,
        start=(ROOM_HEIGHT // 2, 0),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
    )

    assert success is True
    assert repaired.shape == room_grid.shape
    assert isinstance(diagnostics, dict)
    assert diagnostics.get('final_failure_count', 1) == 0


def test_symbolic_only_pipeline_generate_room_fails_fast():
    """Missing neural components should raise a targeted error instead of deep attribute failures."""
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device='cpu',
        enable_logging=False,
    )

    with pytest.raises(MissingPipelineComponentError, match="generate_room requires neural generation components"):
        pipeline.generate_room(
            neighbor_latents={'N': None, 'S': None, 'E': None, 'W': None},
            graph_context={},
            room_id=0,
            apply_repair=False,
            logic_guidance_scale=0.0,
            num_diffusion_steps=1,
            seed=42,
        )


def test_symbolic_only_pipeline_repair_and_stitch_dungeon_public_api():
    """Symbolic-only pipelines should expose repair-and-stitch without neural components."""
    pipeline = NeuralSymbolicDungeonPipeline.create_symbolic_repair_pipeline(
        device='cpu',
        enable_logging=False,
    )
    floor = int(SEMANTIC_PALETTE['FLOOR'])
    room_grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32)
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True, pos=(0, 0))
    graph.add_node(1, pos=(0, 1))
    graph.add_edge(0, 1, edge_type="open")

    result = pipeline.repair_and_stitch_dungeon(
        rooms={0: room_grid, 1: room_grid.copy()},
        mission_graph=graph,
        apply_repair=True,
        enable_map_elites=False,
    )

    assert result.dungeon_grid.ndim == 2
    assert set(result.rooms.keys()) == {0, 1}
    assert result.metrics["symbolic_only"] is True
    assert result.map_elites_score is None


def test_injected_stitcher_is_used_for_public_stitch_rooms():
    """Room stitching should delegate to an injected stitcher when one is provided."""

    class _StubStitcher:
        def __init__(self):
            self.calls = 0

        def stitch_rooms(self, *, rooms, graph):
            self.calls += 1
            return np.full((ROOM_HEIGHT, ROOM_WIDTH), 7, dtype=np.int32)

    stitcher = _StubStitcher()
    pipeline = NeuralSymbolicDungeonPipeline.from_components(
        components=PipelineComponents(
            symbolic=SymbolicGenerationComponents(stitcher=stitcher),
        ),
        device='cpu',
        enable_logging=False,
    )

    rooms = {
        0: RoomGenerationResult(
            room_id=0,
            room_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
            latent=torch.zeros(1, 64, 4, 3),
            neural_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={},
        )
    }
    graph = nx.Graph()
    graph.add_node(0)

    stitched = pipeline.stitch_rooms(rooms, graph)

    assert stitcher.calls == 1
    assert stitched.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert int(stitched[0, 0]) == 7


def test_public_stitch_rooms_supports_positional_only_stitchers():
    """Public stitch API should fall back to positional arguments for legacy stitchers."""

    class _PositionalStitcher:
        def __init__(self):
            self.calls = 0

        def stitch_rooms(self, rooms, graph):
            self.calls += 1
            return np.full((ROOM_HEIGHT, ROOM_WIDTH), 9, dtype=np.int32)

    stitcher = _PositionalStitcher()
    pipeline = NeuralSymbolicDungeonPipeline.from_components(
        components=PipelineComponents(
            symbolic=SymbolicGenerationComponents(stitcher=stitcher),
        ),
        device='cpu',
        enable_logging=False,
    )

    rooms = {
        0: RoomGenerationResult(
            room_id=0,
            room_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
            latent=torch.zeros(1, 64, 4, 3),
            neural_grid=np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={},
        )
    }
    graph = nx.Graph()
    graph.add_node(0)

    stitched = pipeline.stitch_rooms(rooms, graph)

    assert stitcher.calls == 1
    assert stitched.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    assert int(stitched[0, 0]) == 9


def test_prepare_graph_context_and_room_graph_context_include_spatial_topology(pipeline):
    """Prepared graph context should expose node positions and per-room topology maps."""
    graph = nx.DiGraph()
    graph.add_node(0, is_start=True, pos=(0, 0))
    graph.add_node(1, has_boss=True, pos=(0, 1))
    graph.add_edge(0, 1, edge_type="key_locked")

    graph_data = pipeline._prepare_graph_context(graph)
    room_graph_context = pipeline._build_room_graph_context(
        graph_data=graph_data,
        mission_graph=graph,
        room_id=1,
        start_goal=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
    )

    assert "node_positions" in graph_data
    assert tuple(graph_data["node_positions"].shape) == (2, 2)
    assert room_graph_context["has_room_anchor"] is True
    assert "room_topology_map" in room_graph_context
    assert "puzzle_stage_condition" in room_graph_context
    topo = room_graph_context["room_topology_map"]
    assert tuple(topo.shape) == (
        1,
        int(dungeon_pipeline_module.ROOM_TOPOLOGY_CHANNEL_COUNT),
        ROOM_HEIGHT,
        ROOM_WIDTH,
    )
    assert float(topo[:, 5:11].sum().item()) > 0.0
    assert room_graph_context["puzzle_stage_condition"]["sequence_required"] is True


def test_validate_dungeon_without_map_elites_returns_none():
    """Validation should no-op cleanly when MAP-Elites is not configured."""
    pipeline = NeuralSymbolicDungeonPipeline.from_components(
        components=PipelineComponents(
            symbolic=SymbolicGenerationComponents(),
        ),
        device='cpu',
        enable_logging=False,
    )

    dungeon_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
    assert pipeline._validate_dungeon(dungeon_grid) is None


def test_public_repair_room_clamps_row_col_coordinates():
    """Public repair API should normalize room-local coordinates as (row, col)."""

    class _CaptureRefiner:
        def __init__(self):
            self.start = None
            self.goal = None

        def repair_room_with_feedback(self, grid, start, goal, **_kwargs):
            self.start = start
            self.goal = goal
            return np.asarray(grid, dtype=np.int32), True, {"captured": True}

    refiner = _CaptureRefiner()
    pipeline = NeuralSymbolicDungeonPipeline.from_components(
        components=PipelineComponents(
            symbolic=SymbolicGenerationComponents(refiner=refiner),
        ),
        device='cpu',
        enable_logging=False,
    )

    grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
    repaired, success, diagnostics = pipeline.repair_room(
        grid,
        start=(-5, ROOM_WIDTH + 3),
        goal=(ROOM_HEIGHT + 4, -2),
    )

    assert success is True
    assert diagnostics["captured"] is True
    assert repaired.shape == grid.shape
    assert refiner.start == (0, ROOM_WIDTH - 1)
    assert refiner.goal == (ROOM_HEIGHT - 1, 0)


def test_public_repair_room_forwards_required_floor_mask():
    """Public repair API should forward an optional traversability prior to the refiner."""

    class _CaptureRefiner:
        def __init__(self):
            self.required_floor_mask = None

        def repair_room_with_feedback(self, grid, start, goal, required_floor_mask=None, **_kwargs):
            self.required_floor_mask = required_floor_mask
            return np.asarray(grid, dtype=np.int32), True, {"captured": True}

    refiner = _CaptureRefiner()
    pipeline = NeuralSymbolicDungeonPipeline.from_components(
        components=PipelineComponents(
            symbolic=SymbolicGenerationComponents(refiner=refiner),
        ),
        device='cpu',
        enable_logging=False,
    )

    grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
    required = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=bool)
    required[ROOM_HEIGHT // 2, :] = True

    _repaired, success, diagnostics = pipeline.repair_room(
        grid,
        start=(ROOM_HEIGHT // 2, 0),
        goal=(ROOM_HEIGHT // 2, ROOM_WIDTH - 1),
        required_floor_mask=required,
    )

    assert success is True
    assert diagnostics["captured"] is True
    assert isinstance(refiner.required_floor_mask, np.ndarray)
    assert bool(refiner.required_floor_mask[ROOM_HEIGHT // 2, ROOM_WIDTH // 2]) is True


def test_prepare_dungeon_generation_returns_graph_bundle(pipeline, simple_graph):
    """Graph preparation should expose the original graph, physical graph, and conditioning tensors."""
    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=simple_graph,
        use_topological_positional_encoding=True,
    )

    assert prepared.mission_graph is simple_graph
    assert isinstance(prepared.mission_graph_physical, nx.DiGraph)
    assert set(prepared.mission_graph_physical.nodes()) == set(simple_graph.nodes())
    assert 'node_features' in prepared.graph_data
    assert 'edge_index' in prepared.graph_data
    assert 'node_to_idx' in prepared.graph_data


def test_prepare_dungeon_generation_uses_pipeline_topology_defaults(monkeypatch):
    captured = {}

    class _StubTopologyGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def evolve(self, directed_output=False):
            captured["directed_output"] = bool(directed_output)
            graph = nx.DiGraph()
            graph.add_nodes_from([0, 1])
            graph.add_edge(0, 1)
            return graph

    monkeypatch.setattr(dungeon_pipeline_module, "EvolutionaryTopologyGenerator", _StubTopologyGenerator)
    monkeypatch.setattr(dungeon_pipeline_module, "validate_graph_topology", lambda graph: (True, []))

    pipeline = NeuralSymbolicDungeonPipeline(
        device="cpu",
        enable_logging=False,
        topology_default_target_curve=[0.1, 0.25, 0.5, 0.8],
        topology_num_rooms=11,
        topology_population_size=24,
        topology_generations=12,
        topology_mutation_rate=0.22,
        topology_crossover_rate=0.61,
        topology_genome_length=17,
        topology_rule_space="full",
        topology_transition_mix=0.5,
        topology_search_strategy="cvt_emitter",
        topology_qd_archive_cells=160,
        topology_qd_init_random_fraction=0.25,
        topology_qd_emitter_mutation_rate=0.27,
        topology_qd_archive_path="results/topology_qd.pkl",
        topology_qd_load_archive=True,
        topology_qd_autosave_archive=True,
        topology_max_lock_key_rules=2,
        topology_enable_rule_credit_assignment=True,
        topology_enforce_generation_constraints=True,
        topology_allow_candidate_repairs=True,
    )

    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=None,
        generate_topology=True,
        use_topological_positional_encoding=False,
        seed=123,
    )

    assert captured["target_curve"] == [0.1, 0.25, 0.5, 0.8]
    assert captured["population_size"] == 24
    assert captured["generations"] == 12
    assert captured["mutation_rate"] == pytest.approx(0.22)
    assert captured["crossover_rate"] == pytest.approx(0.61)
    assert captured["genome_length"] == 17
    assert captured["max_nodes"] == 11
    assert captured["rule_space"] == "full"
    assert captured["transition_mix"] == pytest.approx(0.5)
    assert captured["search_strategy"] == "cvt_emitter"
    assert captured["qd_archive_cells"] == 160
    assert captured["qd_init_random_fraction"] == pytest.approx(0.25)
    assert captured["qd_emitter_mutation_rate"] == pytest.approx(0.27)
    assert captured["qd_archive_path"] == "results/topology_qd.pkl"
    assert captured["qd_load_archive"] is True
    assert captured["qd_autosave_archive"] is True
    assert captured["max_lock_key_rules"] == 2
    assert captured["enable_rule_credit_assignment"] is True
    assert captured["enforce_generation_constraints"] is True
    assert captured["allow_candidate_repairs"] is True
    assert captured["directed_output"] is True
    assert prepared.mission_graph.number_of_nodes() == 2
    assert prepared.mission_graph.is_directed()


def test_generate_rooms_for_graph_partial_api(monkeypatch, pipeline, simple_graph):
    """Room-only phase API should generate all rooms without requiring full dungeon assembly."""
    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=simple_graph,
        use_topological_positional_encoding=False,
    )

    def fake_generate_room(**kwargs):
        room_id = kwargs['room_id']
        room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        return RoomGenerationResult(
            room_id=int(room_id),
            room_grid=room_grid,
            latent=torch.zeros(1, 64, 4, 3),
            neural_grid=room_grid.copy(),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={'room_id': int(room_id)},
        )

    monkeypatch.setattr(pipeline, 'generate_room', fake_generate_room)

    room_set = pipeline.generate_rooms_for_graph(
        prepared,
        num_diffusion_steps=1,
        logic_guidance_scale=0.0,
        apply_repair=False,
        batch_independent_rooms=False,
        seed=42,
    )

    assert set(room_set.rooms.keys()) == set(simple_graph.nodes())
    assert set(room_set.room_latents.keys()) == set(simple_graph.nodes())
    assert isinstance(room_set.batch_runtime_diagnostics, list)


# =============================================================================
# DIMENSION TESTS
# =============================================================================

def test_room_dimensions(pipeline, neighbor_latents, graph_context):
    """Test that generated rooms have correct dimensions."""
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=2,  # Quick test
        apply_repair=False,
        seed=42
    )
    
    # Check room grid dimensions
    assert result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH), \
        f"Expected (16, 11), got {result.room_grid.shape}"
    
    # Check latent dimensions
    assert isinstance(result.latent, torch.Tensor)
    assert result.latent.device.type == 'cpu'
    assert result.latent.shape == (1, 64, 4, 3), \
        f"Expected (1, 64, 4, 3), got {result.latent.shape}"
    
    # Check neural grid dimensions
    assert result.neural_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
    
    print(f"✓ Room dimensions correct: {result.room_grid.shape}")


def test_latent_space_consistency(pipeline):
    """Test VQ-VAE encode/decode consistency."""
    # Create dummy one-hot room
    room = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH, device=pipeline.device)
    room[:, 1, :, :] = 1.0  # All floor tiles
    
    # Encode
    z_q, indices = pipeline.vqvae.encode(room)
    
    # Check latent dimensions
    assert z_q.shape == (1, 64, 4, 3), f"Latent shape mismatch: {z_q.shape}"
    assert indices.shape == (1, 4, 3), f"Indices shape mismatch: {indices.shape}"
    
    # Decode
    recon = pipeline.vqvae.decode(z_q)
    assert recon.shape == (1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    
    print(f"✓ VQ-VAE latent space consistent: {z_q.shape} → {recon.shape}")


# =============================================================================
# SINGLE ROOM GENERATION TESTS
# =============================================================================

def test_pipeline_threads_cfg_schedule_into_diffusion():
    """Pipeline init should propagate CFG scheduling into the diffusion model."""
    pipeline = NeuralSymbolicDungeonPipeline(
        device='cpu',
        enable_logging=False,
        diffusion_cfg_schedule_mode="cosine_decay",
        diffusion_cfg_schedule_min_scale=1.25,
        diffusion_cfg_schedule_power=2.0,
    )

    assert pipeline.diffusion is not None
    assert pipeline.diffusion.cfg_schedule_mode == "cosine_decay"
    assert pipeline.diffusion.cfg_schedule_min_scale == pytest.approx(1.25)
    assert pipeline.diffusion.cfg_schedule_power == pytest.approx(2.0)

def test_single_room_generation_basic(pipeline, neighbor_latents, graph_context):
    """Test basic single room generation without repair."""
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=42
    )
    
    # Check result structure
    assert result.room_id == 0
    assert result.room_grid is not None
    assert result.latent is not None
    assert isinstance(result.latent, torch.Tensor)
    assert result.neural_grid is not None
    assert not result.was_repaired
    assert result.repair_mask is None
    
    # Check metrics
    assert 'room_id' in result.metrics
    assert 'neural_grid_entropy' in result.metrics
    
    print(f"✓ Basic room generation successful (entropy={result.metrics['neural_grid_entropy']:.3f})")


def test_latent_diffusion_room_cleanup_strips_invalid_doors_and_tiny_obstacles():
    """Latent-room postprocessing should remove obvious decode noise without touching required doors."""
    pipeline = NeuralSymbolicDungeonPipeline(
        device='cpu',
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

    floor_id = int(SEMANTIC_PALETTE["FLOOR"])
    door_open_id = int(SEMANTIC_PALETTE["DOOR_OPEN"])
    wall_id = int(SEMANTIC_PALETTE["WALL"])

    logits = torch.full((1, 44, ROOM_HEIGHT, ROOM_WIDTH), fill_value=-4.0, dtype=torch.float32)
    logits[:, floor_id, :, :] = 4.0
    logits[:, door_open_id, 8, 5] = 8.0
    logits[:, floor_id, 8, 5] = -8.0
    logits[:, wall_id, 6, 4] = 8.0
    logits[:, floor_id, 6, 4] = -8.0
    logits[:, wall_id, 6, 5] = 8.0
    logits[:, floor_id, 6, 5] = -8.0
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

    assert int(result.neural_grid[8, 5]) == floor_id
    assert int(result.neural_grid[6, 4]) == floor_id
    assert int(result.neural_grid[6, 5]) == floor_id
    assert int(result.neural_grid[8, ROOM_WIDTH - 1]) == int(SEMANTIC_PALETTE["DOOR_LOCKED"])
    assert result.metrics["neural_invalid_door_tiles_removed"] == 1
    assert result.metrics["neural_interior_obstacle_tiles_removed"] == 2
    assert result.metrics["neural_interior_obstacle_components_removed"] == 1


def test_single_room_generation_with_repair(pipeline, neighbor_latents, graph_context):
    """Test room generation with symbolic repair."""
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=1,
        num_diffusion_steps=5,
        apply_repair=True,
        start_goal_coords=((8, 0), (8, 10)),
        seed=42
    )
    
    # Check that repair was attempted
    assert isinstance(result.was_repaired, bool)
    
    if result.was_repaired:
        assert result.repair_mask is not None
        assert result.repair_mask.shape == (ROOM_HEIGHT, ROOM_WIDTH)
        assert 'tiles_changed' in result.metrics
        print(f"✓ Repair successful ({result.metrics['tiles_changed']} tiles changed)")
    else:
        print("✓ Repair not needed (neural output already valid)")


def test_reproducibility_with_seed(pipeline, neighbor_latents, graph_context):
    """Test that seeded generation is reproducible."""
    result1 = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=12345
    )
    
    result2 = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=12345
    )
    
    # Check that results are identical
    assert np.array_equal(result1.neural_grid, result2.neural_grid), \
        "Seeded generation should be reproducible"
    
    print("✓ Reproducibility verified with seed=12345")


# =============================================================================
# MULTI-ROOM DUNGEON GENERATION TESTS
# =============================================================================

def test_dungeon_generation_simple(pipeline, simple_graph):
    """Test complete dungeon generation with simple graph."""
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=3,
        apply_repair=False,
        seed=42,
        enable_map_elites=False,
    )
    
    # Check result structure
    assert result.dungeon_grid is not None
    assert len(result.rooms) == 3
    assert result.mission_graph is simple_graph
    
    # Check metrics
    assert result.metrics['num_rooms'] == 3
    assert 'generation_time_sec' in result.metrics
    assert result.generation_time > 0
    
    print(f"✓ 3-room dungeon generated in {result.generation_time:.2f}s")


def test_dungeon_generation_with_repair(pipeline, simple_graph):
    """Test dungeon generation with repair enabled."""
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=3,
        apply_repair=True,
        seed=42,
        enable_map_elites=False,
    )
    
    # Check repair metrics
    assert 'repair_rate' in result.metrics
    assert 'total_tiles_repaired' in result.metrics
    
    repair_rate = result.metrics['repair_rate']
    assert 0.0 <= repair_rate <= 1.0
    
    print(f"✓ Dungeon with repair (rate={repair_rate:.1%}, "
          f"tiles={result.metrics['total_tiles_repaired']})")


def test_room_order_preservation(pipeline, simple_graph):
    """Test that rooms are generated in graph order."""
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=2,
        seed=42
    )
    
    # Check all rooms present
    expected_ids = set(simple_graph.nodes())
    actual_ids = set(result.rooms.keys())
    assert expected_ids == actual_ids, f"Missing rooms: {expected_ids - actual_ids}"
    
    print(f"✓ All {len(expected_ids)} rooms generated correctly")


def test_generate_dungeon_passes_boundary_and_position(monkeypatch, pipeline, simple_graph):
    """Dungeon loop must pass graph-derived boundary constraints and positions into room generation."""
    captured = []

    def fake_generate_room(**kwargs):
        captured.append(
            {
                'room_id': kwargs.get('room_id'),
                'boundary_constraints': kwargs.get('boundary_constraints'),
                'position': kwargs.get('position'),
            }
        )
        rid = int(kwargs.get('room_id', 0))
        room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        latent = torch.zeros(1, 64, 4, 3)
        return RoomGenerationResult(
            room_id=rid,
            room_grid=room_grid,
            latent=latent,
            neural_grid=room_grid.copy(),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={},
        )

    monkeypatch.setattr(pipeline, 'generate_room', fake_generate_room)

    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=1,
        apply_repair=False,
        enable_map_elites=False,
        seed=42,
    )

    assert result.metrics['num_rooms'] == len(simple_graph.nodes())
    assert len(captured) == len(simple_graph.nodes())
    for entry in captured:
        bc = entry['boundary_constraints']
        pos = entry['position']
        assert bc is not None
        assert tuple(bc.shape) == (1, 8)
        assert pos is not None
        assert tuple(pos.shape) == (1, 2)


def test_encode_room_grid_to_latent_returns_tensor(pipeline):
    room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
    latent = pipeline._encode_room_grid_to_latent(room_grid)
    assert isinstance(latent, torch.Tensor)
    assert latent.dim() == 4
    assert tuple(latent.shape[1:]) == (64, 4, 3)


def test_generate_dungeon_emits_batch_diagnostics(monkeypatch, pipeline, simple_graph):
    """Batch generation mode should record planner/chunk diagnostics in metrics."""
    def fake_generate_room_batch(**kwargs):
        out = {}
        for room_id in kwargs.get('room_ids', []):
            room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
            out[room_id] = RoomGenerationResult(
                room_id=int(room_id),
                room_grid=room_grid,
                latent=torch.zeros(1, 64, 4, 3),
                neural_grid=room_grid.copy(),
                was_repaired=False,
                repair_mask=None,
                neural_probs=None,
                metrics={},
            )
        return out

    monkeypatch.setattr(pipeline, '_generate_room_batch', fake_generate_room_batch)
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=1,
        apply_repair=False,
        enable_map_elites=False,
        batch_independent_rooms=True,
        max_batch_size=2,
        seed=42,
    )

    diagnostics = result.metrics.get('batch_generation_diagnostics')
    assert isinstance(diagnostics, list)
    assert len(diagnostics) > 0


def test_generate_room_batch_stacks_current_node_distance_per_room(monkeypatch, pipeline, simple_graph):
    """Batched diffusion guidance should receive current-node distance as [B, N, 4]."""
    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=simple_graph,
        generate_topology=False,
        use_topological_positional_encoding=True,
    )

    captured = {}

    def fake_ddim_sample(*, context, shape, num_steps, graph_data, **kwargs):
        current_node_distance = graph_data.get("current_node_distance")
        assert isinstance(current_node_distance, torch.Tensor)
        assert tuple(current_node_distance.shape) == (
            int(shape[0]),
            int(prepared.graph_data["node_features"].shape[0]),
            4,
        )
        assert graph_data.get("edge_rrwp") is prepared.graph_data["edge_rrwp"]
        captured["shape"] = tuple(current_node_distance.shape)
        return torch.zeros(shape, device=pipeline.device)

    def fake_decode(z_batch):
        batch = int(z_batch.shape[0])
        num_classes = int(getattr(pipeline.vqvae, "num_classes", 44))
        return torch.zeros((batch, num_classes, ROOM_HEIGHT, ROOM_WIDTH), device=z_batch.device)

    def fake_generate_room(**kwargs):
        room_id = int(kwargs["room_id"])
        room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        latent = kwargs.get("precomputed_latent")
        if not isinstance(latent, torch.Tensor):
            latent = torch.zeros(1, 64, 4, 3)
        return RoomGenerationResult(
            room_id=room_id,
            room_grid=room_grid,
            latent=latent.detach().cpu(),
            neural_grid=room_grid.copy(),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={},
        )

    monkeypatch.setattr(pipeline.diffusion, "ddim_sample", fake_ddim_sample)
    monkeypatch.setattr(pipeline.vqvae, "decode", fake_decode)
    monkeypatch.setattr(pipeline, "generate_room", fake_generate_room)

    room_set = pipeline.generate_rooms_for_graph(
        prepared,
        num_diffusion_steps=1,
        apply_repair=False,
        batch_independent_rooms=True,
        max_batch_size=8,
        seed=42,
    )

    assert len(room_set.rooms) == len(prepared.mission_graph_physical.nodes())
    assert captured["shape"][1] == len(prepared.graph_data["node_features"])
    assert captured["shape"][2] == 4


def test_generate_room_batch_categorical_sampler_uses_shared_latent_shape(monkeypatch, pipeline, simple_graph):
    """Categorical batch sampling should not depend on diffusion-only branch locals."""
    pipeline.diffusion = None
    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=simple_graph,
        generate_topology=False,
        use_topological_positional_encoding=True,
    )
    captured_latents = {}

    def fake_generate_room(**kwargs):
        room_id = int(kwargs["room_id"])
        latent = kwargs.get("precomputed_latent")
        assert isinstance(latent, torch.Tensor)
        captured_latents[room_id] = tuple(latent.shape)
        room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        return RoomGenerationResult(
            room_id=room_id,
            room_grid=room_grid,
            latent=latent.detach().cpu(),
            neural_grid=room_grid.copy(),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={},
        )

    monkeypatch.setattr(pipeline, "generate_room", fake_generate_room)
    monkeypatch.setattr(
        pipeline,
        "_decode_latent_with_vqvae",
        lambda _latent: (_ for _ in ()).throw(AssertionError("categorical batch should use decode_indices")),
    )

    room_set = pipeline.generate_rooms_for_graph(
        prepared,
        num_diffusion_steps=1,
        apply_repair=False,
        batch_independent_rooms=True,
        max_batch_size=8,
        latent_sampler="categorical",
        categorical_codebook_size=8,
        seed=123,
    )

    assert len(room_set.rooms) == len(prepared.mission_graph_physical.nodes())
    assert set(captured_latents) == set(prepared.mission_graph_physical.nodes())
    assert all(shape[0] == 1 for shape in captured_latents.values())
    assert all(shape[-2:] == (4, 3) for shape in captured_latents.values())


def test_generate_room_categorical_sampler_does_not_require_diffusion_or_redecode(monkeypatch, pipeline, simple_graph):
    """Categorical room sampling should use exact code-index decode without diffusion weights."""
    pipeline.diffusion = None
    pipeline.default_latent_sampler = "categorical"
    pipeline.default_fast_sampler_teacher_fallback_enabled = False
    assert pipeline.supports_room_generation() is True

    prepared = pipeline.prepare_dungeon_generation(
        mission_graph=simple_graph,
        generate_topology=False,
        use_topological_positional_encoding=True,
    )
    room_id = 0
    start_goal = pipeline._extract_room_start_goal(prepared.mission_graph_physical, room_id)
    graph_context = pipeline._build_room_graph_context(
        graph_data=prepared.graph_data,
        mission_graph=prepared.mission_graph_physical,
        room_id=room_id,
        start_goal=start_goal,
    )
    floor = int(SEMANTIC_PALETTE["FLOOR"])

    def fake_decode_indices(indices):
        batch = int(indices.shape[0])
        logits = torch.full((batch, 44, ROOM_HEIGHT, ROOM_WIDTH), -8.0)
        logits[:, floor, :, :] = 8.0
        return logits

    monkeypatch.setattr(pipeline.vqvae, "decode_indices", fake_decode_indices)
    monkeypatch.setattr(
        pipeline,
        "_decode_latent_with_vqvae",
        lambda _latent: (_ for _ in ()).throw(AssertionError("categorical sampler should not re-decode latents")),
    )

    result = pipeline.generate_room(
        neighbor_latents={"N": None, "S": None, "E": None, "W": None},
        graph_context=graph_context,
        room_id=room_id,
        boundary_constraints=pipeline._build_room_boundary_constraints(prepared.mission_graph_physical, room_id),
        position=pipeline._build_room_position_tensor(prepared.mission_graph_physical, room_id, 0),
        apply_repair=False,
        logic_guidance_scale=0.0,
        num_diffusion_steps=1,
        latent_sampler="categorical",
        categorical_codebook_size=8,
        allow_teacher_fallback=False,
        start_goal_coords=start_goal,
        seed=123,
    )

    assert tuple(result.latent.shape) == (1, int(pipeline.vqvae.latent_dim), 4, 3)
    assert result.neural_probs.shape == (44, ROOM_HEIGHT, ROOM_WIDTH)


def test_strict_adjacency_placement_preserves_all_edges(pipeline):
    """Strict placement must embed all graph edges as room-adjacent relationships."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

    placement = pipeline._compute_strict_room_placement(G, room_ids=list(G.nodes()))
    assert set(placement.keys()) == set(G.nodes())

    for u, v in G.edges():
        manhattan = abs(placement[u][0] - placement[v][0]) + abs(placement[u][1] - placement[v][1])
        assert manhattan == 1, f"Edge {u}->{v} is non-adjacent under strict placement"


def test_strict_adjacency_placement_falls_back_on_degree_over_four(pipeline):
    """When strict adjacency is impossible, placement must gracefully fall back."""
    G = nx.Graph()
    G.add_node(0)
    for nid in [1, 2, 3, 4, 5]:
        G.add_edge(0, nid)

    placement = pipeline._compute_strict_room_placement(G, room_ids=list(G.nodes()))
    assert set(placement.keys()) == set(G.nodes())
    assert len(set(placement.values())) == len(G.nodes())


def test_stitch_rooms_public_api_handles_impossible_strict_graph(pipeline):
    """Public stitch API must succeed for topologies that cannot be strict-embedded."""
    G = nx.Graph()
    G.add_node(0)
    for nid in [1, 2, 3, 4, 5]:
        G.add_edge(0, nid)

    rooms = {}
    for room_id in G.nodes():
        room_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH), dtype=np.int32)
        rooms[room_id] = RoomGenerationResult(
            room_id=int(room_id),
            room_grid=room_grid,
            latent=torch.zeros(1, 64, 4, 3),
            neural_grid=room_grid.copy(),
            was_repaired=False,
            repair_mask=None,
            neural_probs=None,
            metrics={},
        )

    stitched = pipeline.stitch_rooms(rooms, G)
    assert stitched.ndim == 2
    assert stitched.shape[0] >= ROOM_HEIGHT
    assert stitched.shape[1] >= ROOM_WIDTH


def test_stitch_room_layout_supports_variable_room_sizes(pipeline):
    """Shared stitch core should return bbox metadata for non-uniform room grids."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1)])

    rooms = {
        0: np.full((8, 6), 1, dtype=np.int32),
        1: np.full((6, 10), 2, dtype=np.int32),
    }

    stitched = pipeline.stitch_room_layout(
        rooms,
        G,
        enforce_room_dimensions=None,
        carve_connections=True,
    )

    assert stitched.dungeon_grid.ndim == 2
    assert set(stitched.layout_map.keys()) == {0, 1}

    x0_min, y0_min, x0_max, y0_max = stitched.layout_map[0]
    x1_min, y1_min, x1_max, y1_max = stitched.layout_map[1]
    assert (y0_max - y0_min + 1, x0_max - x0_min + 1) == rooms[0].shape
    assert (y1_max - y1_min + 1, x1_max - x1_min + 1) == rooms[1].shape
    assert np.any(stitched.dungeon_grid[y0_min:y0_max + 1, x0_min:x0_max + 1] == 1)
    assert np.any(stitched.dungeon_grid[y1_min:y1_max + 1, x1_min:x1_max + 1] == 2)


# =============================================================================
# GUIDANCE TESTS
# =============================================================================

def test_logic_guidance_effect(pipeline, neighbor_latents, graph_context):
    """Randomly initialized LogicNet must not steer runtime generation."""
    # Without guidance
    result_no_guidance = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        logic_guidance_scale=0.0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=42
    )
    
    pipeline.runtime_diagnostics = {}

    # Guidance is requested, but this test pipeline has no LogicNet checkpoint.
    result_with_guidance = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        logic_guidance_scale=1.0,
        num_diffusion_steps=5,
        apply_repair=False,
        seed=42
    )
    
    assert np.array_equal(result_no_guidance.neural_grid, result_with_guidance.neural_grid)
    assert pipeline.runtime_diagnostics["logic_guidance_disabled_untrained_logic_net"] >= 1
    assert pipeline.diffusion.guidance.logic_net is None


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_missing_neighbor_handling(pipeline, graph_context):
    """Test that pipeline handles missing neighbors gracefully."""
    # All neighbors None
    neighbor_latents = {'N': None, 'S': None, 'E': None, 'W': None}
    
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=3,
        apply_repair=False,
        seed=42
    )
    
    assert result.room_grid is not None
    print("✓ Missing neighbors handled correctly")


def test_invalid_graph_context_fallback(pipeline, neighbor_latents):
    """Test fallback when graph context is invalid."""
    # Empty graph context
    graph_context = {}
    
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        num_diffusion_steps=3,
        apply_repair=False,
        seed=42
    )
    
    assert result.room_grid is not None
    print("✓ Invalid graph context handled with fallback")


def test_refiner_tile_vocab_uses_canonical_semantic_ids(pipeline):
    """Symbolic refiner should not emit legacy out-of-schema tile IDs."""
    tile_ids = {int(v) for v in getattr(pipeline.refiner, 'tile_types', [])}
    assert 50 not in tile_ids


def test_sanitize_semantic_grid_replaces_invalid_ids_with_fallback(pipeline):
    """Invalid semantic IDs should be clamped to canonical palette."""
    grid = np.array([[1, 50, 2], [999, 0, 3]], dtype=np.int32)
    fallback = np.array([[1, 2, 2], [1, 0, 3]], dtype=np.int32)
    sanitized, invalid_count, invalid_ids = pipeline._sanitize_semantic_grid(
        grid,
        fallback_grid=fallback,
    )
    assert int(invalid_count) == 2
    assert sorted(int(v) for v in invalid_ids) == [50, 999]
    assert sanitized.shape == grid.shape
    assert int(sanitized[0, 1]) == int(fallback[0, 1])
    assert int(sanitized[1, 0]) == int(fallback[1, 0])


def test_sanitize_semantic_grid_can_strip_void_tiles(pipeline):
    """Generated room grids should not preserve interior VOID tiles."""
    floor_id = int(SEMANTIC_PALETTE["FLOOR"])
    void_id = int(SEMANTIC_PALETTE["VOID"])
    grid = np.array([[floor_id, void_id, floor_id], [void_id, floor_id, void_id]], dtype=np.int32)
    fallback = np.array([[floor_id, floor_id, floor_id], [floor_id, floor_id, floor_id]], dtype=np.int32)
    sanitized, invalid_count, invalid_ids = pipeline._sanitize_semantic_grid(
        grid,
        fallback_grid=fallback,
        strip_void=True,
    )
    assert int(invalid_count) == 3
    assert int(void_id) in {int(v) for v in invalid_ids}
    assert np.all(sanitized == floor_id)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.slow
def test_generation_performance(pipeline, simple_graph):
    """Test generation performance (marked as slow)."""
    import time
    
    start = time.time()
    result = pipeline.generate_dungeon(
        mission_graph=simple_graph,
        num_diffusion_steps=50,
        apply_repair=True,
        seed=42
    )
    duration = time.time() - start
    
    # Should complete in reasonable time (adjust based on hardware)
    assert duration < 60.0, f"Generation too slow: {duration:.1f}s"
    
    rooms_per_sec = len(result.rooms) / duration
    print(f"✓ Performance: {rooms_per_sec:.2f} rooms/sec ({duration:.2f}s total)")


# =============================================================================
# INTEGRATION SMOKE TEST
# =============================================================================

def test_complete_pipeline_smoke():
    """Comprehensive smoke test of the entire pipeline."""
    print("\n" + "="*70)
    print("COMPLETE PIPELINE SMOKE TEST")
    print("="*70)
    
    # Initialize pipeline
    pipeline = NeuralSymbolicDungeonPipeline(device='cpu', enable_logging=True)
    
    # Create test graph
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # Generate dungeon
    result = pipeline.generate_dungeon(
        mission_graph=G,
        guidance_scale=7.5,
        logic_guidance_scale=0.5,
        num_diffusion_steps=10,
        apply_repair=True,
        seed=42,
        enable_map_elites=True,
    )
    
    # Verify all components
    assert len(result.rooms) == 4
    assert result.dungeon_grid.shape[1] == ROOM_WIDTH
    assert result.metrics['num_rooms'] == 4
    assert 'generation_time_sec' in result.metrics
    
    # Check individual rooms
    for _room_id, room_result in result.rooms.items():
        assert room_result.room_grid.shape == (ROOM_HEIGHT, ROOM_WIDTH)
        assert isinstance(room_result.latent, torch.Tensor)
        assert room_result.latent.device.type == 'cpu'
        assert room_result.latent.shape == (1, 64, 4, 3)
        assert 'neural_grid_entropy' in room_result.metrics
    
    print("\n" + "="*70)
    print("SMOKE TEST RESULTS:")
    print(f"  Rooms generated: {len(result.rooms)}")
    print(f"  Dungeon shape: {result.dungeon_grid.shape}")
    print(f"  Repair rate: {result.metrics['repair_rate']:.1%}")
    print(f"  Generation time: {result.generation_time:.2f}s")
    if result.map_elites_score:
        print(f"  MAP-Elites linearity: {result.map_elites_score['linearity']:.3f}")
        print(f"  MAP-Elites leniency: {result.map_elites_score['leniency']:.3f}")
    print("="*70)
    print("✓ COMPLETE PIPELINE SMOKE TEST PASSED")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Run smoke test directly
    test_complete_pipeline_smoke()
