import threading
import json
from pathlib import Path

import numpy as np
import torch

from src import generate as generation_cli
from src.gui.ai.ai_generation_controls import generate_level, start_ai_dungeon_generation
from src.gui.app.run_completion_handlers import handle_ai_generation_completion
from src.gui import ai_generation_worker
from src.gui.ai import generation_pipeline


class _DummyGUI:
    def __init__(self):
        self.messages = []
        self.ai_gen_thread = None
        self.ai_gen_result = object()
        self.ai_gen_done = True

    def _set_message(self, message, duration=3.0):
        self.messages.append((message, duration))

    def _generate_ai_dungeon_worker(self):
        return None


class _CompletionGUI:
    def __init__(self, result):
        self.ai_gen_done = True
        self.ai_gen_result = result
        self.maps = []
        self.map_names = []
        self.current_map_idx = 0
        self.effects = []
        self.step_count = 5
        self.auto_path = [(1, 1)]
        self.auto_mode = True
        self.ai_constraint_boss_norm = (0.5, 0.5)
        self.ai_constraint_lock_norm = (0.2, 0.2)
        self.ai_constraint_key_norm = (0.8, 0.8)
        self.ai_mission_graph_draft = None
        self.loaded = 0
        self.centered = False
        self.messages = []

    def _load_current_map(self):
        self.loaded += 1

    def _center_view(self):
        self.centered = True

    def _set_message(self, message, duration=3.0):
        self.messages.append((message, duration))


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def exception(self, *_args, **_kwargs):
        return None


def test_start_ai_generation_sets_thread_and_message():
    gui = _DummyGUI()
    gui.ai_gen_done = False
    gui.ai_gen_result = None
    start_ai_dungeon_generation(gui, threading)

    assert gui.ai_gen_thread is not None
    gui.ai_gen_thread.join(timeout=1.0)
    assert gui.ai_gen_result is None
    assert gui.ai_gen_done is False
    assert gui.messages[-1][0] == "AI generation started (background)"


def test_start_ai_generation_preserves_pending_result():
    gui = _DummyGUI()

    start_ai_dungeon_generation(gui, threading)

    assert gui.ai_gen_thread is None
    assert gui.ai_gen_result is not None
    assert gui.ai_gen_done is True
    assert gui.messages[-1][0] == "AI generation result pending"


def test_generate_level_uses_loaded_checkpoint(tmp_path):
    gui = _DummyGUI()
    gui.ai_gen_done = False
    gui.ai_gen_result = None
    checkpoint = tmp_path / "gui_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    gui.ai_checkpoint_path = str(checkpoint)

    generate_level(gui, threading, _Logger())
    gui.ai_gen_thread.join(timeout=1.0)

    assert gui.ai_gen_thread is not None
    assert gui.messages[-1][0] == "AI generation started (background)"


def test_generate_level_reports_stale_loaded_checkpoint(tmp_path):
    gui = _DummyGUI()
    gui.ai_checkpoint_path = str(tmp_path / "missing_model.pth")

    generate_level(gui, threading, _Logger())

    assert gui.ai_gen_thread is None
    assert "Loaded AI model not found" in gui.messages[-1][0]


def test_generate_level_falls_back_to_procedural_without_checkpoint():
    gui = _DummyGUI()
    gui.generated_procedural = False
    gui._generate_dungeon = lambda: setattr(gui, "generated_procedural", True)

    generate_level(gui, threading, _Logger())

    assert gui.generated_procedural is True


def test_handle_ai_generation_completion_applies_worker_payload_on_main_thread():
    grid = np.zeros((16, 11), dtype=np.int32)
    graph = object()
    gui = _CompletionGUI(
        {
            "success": True,
            "grid": grid,
            "name": "AI Test",
            "message": "AI done",
            "clear_mixed_constraints": True,
            "mission_graph_draft": graph,
        }
    )

    handle_ai_generation_completion(gui)

    assert gui.ai_gen_done is False
    assert gui.ai_gen_result is None
    assert len(gui.maps) == 1
    assert gui.maps[0] is grid
    assert gui.map_names == ["AI Test"]
    assert gui.loaded == 1
    assert gui.centered is True
    assert gui.step_count == 0
    assert gui.auto_path == []
    assert gui.auto_mode is False
    assert gui.ai_constraint_boss_norm is None
    assert gui.ai_constraint_lock_norm is None
    assert gui.ai_constraint_key_norm is None
    assert gui.ai_mission_graph_draft is graph
    assert gui.messages[-1][0] == "AI done"


def test_handle_ai_generation_completion_auto_imports_generated_txt(tmp_path):
    grid = np.arange(12, dtype=np.int32).reshape(3, 4)
    txt_path = tmp_path / "generated_level.txt"
    np.savetxt(txt_path, grid, fmt="%d")
    graph = object()
    gui = _CompletionGUI(
        {
            "success": True,
            "grid": np.zeros((3, 4), dtype=np.int32),
            "name": "AI TXT",
            "message": "AI txt done",
            "generated_txt_path": str(txt_path),
            "clear_mixed_constraints": True,
            "mission_graph_draft": graph,
        }
    )

    handle_ai_generation_completion(gui)

    assert gui.ai_gen_done is False
    assert gui.ai_gen_result is None
    assert len(gui.maps) == 1
    assert np.array_equal(gui.maps[0], grid)
    assert gui.map_names == ["AI TXT"]
    assert gui.loaded == 1
    assert gui.centered is True
    assert gui.auto_path == []
    assert gui.auto_mode is False
    assert gui.ai_constraint_boss_norm is None
    assert gui.ai_constraint_lock_norm is None
    assert gui.ai_constraint_key_norm is None
    assert gui.ai_mission_graph_draft is graph
    assert "loaded generated_level.txt" in gui.messages[-1][0]


def test_worker_reports_missing_checkpoint(monkeypatch):
    gui = _DummyGUI()
    gui.ai_checkpoint_path = str(Path("__definitely_missing_checkpoint__.pth"))

    ai_generation_worker.run_ai_generation_worker(gui, _Logger())

    assert gui.messages
    assert gui.messages[-1][0] == "No AI checkpoint found - train first!"


def test_save_generated_dungeon_txt_writes_unique_and_latest_files(tmp_path):
    grid = np.arange(20, dtype=np.int32).reshape(4, 5)

    written = generation_pipeline.save_generated_dungeon_txt(
        tile_grid=grid,
        seed=123,
        num_nodes=2,
        num_edges=1,
        checkpoint_path="checkpoint.pth",
        export_dir=tmp_path,
        np_module=np,
    )

    assert written["txt_path"].exists()
    assert written["latest_txt_path"].exists()
    assert written["png_path"].exists()
    assert written["latest_png_path"].exists()
    assert written["metadata_path"].exists()
    assert np.array_equal(np.loadtxt(written["txt_path"], dtype=np.int32), grid)
    assert np.array_equal(np.loadtxt(written["latest_txt_path"], dtype=np.int32), grid)
    metadata = json.loads(written["metadata_path"].read_text(encoding="utf-8"))
    assert Path(metadata["png_path"]).exists()
    assert Path(metadata["latest_png_path"]).exists()


def test_load_or_reuse_gui_generation_pipeline_uses_cache(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class _GUI:
        pass

    def _fake_loader(**kwargs):
        calls.append(kwargs)
        return {"pipeline": len(calls)}

    monkeypatch.setattr("src.gui.ai.generation_worker.load_canonical_generation_pipeline", _fake_loader)
    gui = _GUI()

    first = ai_generation_worker.load_or_reuse_gui_generation_pipeline(
        gui=gui,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
        logger=_Logger(),
        strict_checkpoint_mode=False,
    )
    second = ai_generation_worker.load_or_reuse_gui_generation_pipeline(
        gui=gui,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
        logger=_Logger(),
        strict_checkpoint_mode=False,
    )

    assert first is second
    assert len(calls) == 1
    assert calls[0]["gui_fast_mode"] is False


def test_load_or_reuse_gui_generation_pipeline_can_opt_into_fast_sampler(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class _GUI:
        ai_use_fast_sampler = True

    def _fake_loader(**kwargs):
        calls.append(kwargs)
        return {"pipeline": len(calls)}

    monkeypatch.setattr("src.gui.ai.generation_worker.load_canonical_generation_pipeline", _fake_loader)
    gui = _GUI()

    ai_generation_worker.load_or_reuse_gui_generation_pipeline(
        gui=gui,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
        logger=_Logger(),
        strict_checkpoint_mode=False,
    )

    assert calls[0]["gui_fast_mode"] is True


def test_resolve_checkpoint_path_honors_env_override(monkeypatch):
    monkeypatch.setenv("KLTN_CHECKPOINT_PATH", "checkpoints/custom_model.pth")
    resolved = generation_pipeline.resolve_checkpoint_path()

    assert resolved.name == "custom_model.pth"


def test_resolve_checkpoint_path_honors_explicit_path(monkeypatch):
    monkeypatch.setenv("KLTN_CHECKPOINT_PATH", "checkpoints/env_model.pth")
    resolved = generation_pipeline.resolve_checkpoint_path("checkpoints/gui_model.pth")

    assert resolved.name == "gui_model.pth"


def test_resolve_checkpoint_path_defaults_to_repo_checkpoints_when_discovery_disabled(monkeypatch):
    monkeypatch.delenv("KLTN_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("KLTN_DISABLE_OUTPUT_CHECKPOINT_DISCOVERY", "1")
    resolved = generation_pipeline.resolve_checkpoint_path()
    expected = Path(generation_pipeline.__file__).resolve().parents[3] / "checkpoints" / "final_model.pth"

    assert resolved == expected


def test_resolve_checkpoint_path_prefers_discovered_output_checkpoint(monkeypatch):
    monkeypatch.delenv("KLTN_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("KLTN_DISABLE_OUTPUT_CHECKPOINT_DISCOVERY", raising=False)

    discovered = generation_pipeline.discover_best_output_checkpoint()
    resolved = generation_pipeline.resolve_checkpoint_path()

    if discovered is None:
        expected = Path(generation_pipeline.__file__).resolve().parents[3] / "checkpoints" / "final_model.pth"
        assert resolved == expected
    else:
        assert resolved == discovered


def test_generate_mission_graph_is_deterministic_with_seed():
    import random

    fixed_seed = 314159
    data_a = generation_pipeline.generate_mission_graph(random, seed=fixed_seed)
    data_b = generation_pipeline.generate_mission_graph(random, seed=fixed_seed)

    assert data_a["seed"] == fixed_seed
    assert data_b["seed"] == fixed_seed
    assert data_a["num_nodes"] == data_b["num_nodes"]
    assert data_a["num_edges"] == data_b["num_edges"]


def test_generate_mission_graph_honors_requested_node_count_when_possible():
    import random

    data = generation_pipeline.generate_mission_graph(
        random,
        seed=112,
        num_rooms=12,
        difficulty="HARD",
        max_keys=3,
    )

    assert data["requested_num_rooms"] == 12
    assert data["num_nodes"] == 12
    assert data["difficulty"] == "HARD"
    assert data["max_keys"] == 3


def test_compute_editor_layout_preserves_string_node_ids():
    from src.generation.grammar import EdgeType, MissionGraph, MissionNode, NodeType

    graph = MissionGraph()
    graph.add_node(MissionNode(id="start", node_type=NodeType.START))
    graph.add_node(MissionNode(id="boss", node_type=NodeType.GOAL))
    graph.add_edge("start", "boss", edge_type=EdgeType.PATH)

    layout = generation_pipeline._compute_editor_layout(graph)

    assert set(layout.keys()) == {"start", "boss"}
    for x, y in layout.values():
        assert 0.0 <= float(x) <= 1.0
        assert 0.0 <= float(y) <= 1.0


def test_compute_editor_layout_supports_mixed_hashable_node_ids():
    from src.generation.grammar import EdgeType, MissionGraph, MissionNode, NodeType

    graph = MissionGraph()
    graph.add_node(MissionNode(id=0, node_type=NodeType.START))
    graph.add_node(MissionNode(id="boss", node_type=NodeType.GOAL))
    graph.add_edge(0, "boss", edge_type=EdgeType.PATH)

    layout = generation_pipeline._compute_editor_layout(graph)

    assert set(layout.keys()) == {0, "boss"}


def test_apply_mission_graph_constraints_preserves_string_node_ids():
    from src.generation.grammar import EdgeType, MissionGraph, MissionNode, NodeType

    class _Logger:
        def __init__(self):
            self.info_calls = []

        def info(self, *args, **kwargs):
            self.info_calls.append((args, kwargs))

        def warning(self, *_args, **_kwargs):
            return None

        def exception(self, *_args, **_kwargs):
            return None

    graph = MissionGraph()
    graph.add_node(MissionNode(id="start", node_type=NodeType.START))
    graph.add_node(MissionNode(id="hall", node_type=NodeType.EMPTY))
    graph.add_node(MissionNode(id="boss", node_type=NodeType.GOAL))
    graph.add_edge("start", "hall", edge_type=EdgeType.PATH)

    logger = _Logger()
    updated, applied = generation_pipeline.apply_mission_graph_constraints(
        graph,
        {
            "boss_node": "boss",
            "locked_edges": [("start", "hall"), ("hall", "boss"), (3.5, "boss"), ("boss", "boss")],
        },
        logger,
    )

    assert updated.nodes["boss"].node_type == NodeType.BOSS
    assert applied == {"boss_applied": True, "locked_edges_applied": 2}
    assert any(
        edge.source == "start" and edge.target == "hall" and edge.edge_type == EdgeType.LOCKED
        for edge in updated.edges
    )
    assert any(
        edge.source == "hall" and edge.target == "boss" and edge.edge_type == EdgeType.LOCKED
        for edge in updated.edges
    )
    assert logger.info_calls


def test_resolve_vqvae_checkpoint_prefers_sibling_pretrain_for_composite_diffusion(tmp_path):
    composite = tmp_path / "final_model.pth"
    sibling_vqvae = tmp_path / "vqvae_pretrained.pth"

    torch.save({"diffusion_state_dict": {"weight": torch.tensor(1.0)}}, composite)
    torch.save({"model_state_dict": {"weight": torch.tensor(2.0)}}, sibling_vqvae)

    resolved = generation_pipeline._resolve_vqvae_checkpoint_for_generation(composite)

    assert resolved == sibling_vqvae


def test_resolve_vqvae_checkpoint_prefers_stage_sibling_pretrain_for_diffusion_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    diffusion_dir = checkpoint_dir / "diffusion"
    vqvae_dir = checkpoint_dir / "vqvae"
    diffusion_dir.mkdir(parents=True)
    vqvae_dir.mkdir(parents=True)

    diffusion_ckpt = diffusion_dir / "best_model.pth"
    stage_vqvae = vqvae_dir / "vqvae_pretrained.pth"

    torch.save({"diffusion_state_dict": {"weight": torch.tensor(1.0)}}, diffusion_ckpt)
    torch.save({"model_state_dict": {"weight": torch.tensor(2.0)}}, stage_vqvae)

    resolved = generation_pipeline._resolve_vqvae_checkpoint_for_generation(diffusion_ckpt)

    assert resolved == stage_vqvae


def test_resolve_vqvae_checkpoint_prefers_metadata_declared_pretrain(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoints" / "diffusion"
    checkpoint_dir.mkdir(parents=True)
    diffusion_ckpt = checkpoint_dir / "best_model.pth"
    declared_vqvae = tmp_path / "tokenizer" / "vqvae_pretrained.pth"
    declared_vqvae.parent.mkdir(parents=True)

    torch.save({"diffusion_state_dict": {"weight": torch.tensor(1.0)}}, diffusion_ckpt)
    torch.save({"model_state_dict": {"weight": torch.tensor(2.0)}}, declared_vqvae)
    diffusion_ckpt.with_name(f"{diffusion_ckpt.name}.meta.json").write_text(
        json.dumps(
            {
                "model_type": "diffusion",
                "architecture": {},
                "extra": {"vqvae_checkpoint": str(declared_vqvae)},
            }
        ),
        encoding="utf-8",
    )

    resolved = generation_pipeline._resolve_vqvae_checkpoint_for_generation(diffusion_ckpt)

    assert resolved == declared_vqvae.resolve()


def test_generate_dungeon_with_pipeline_uses_canonical_roomwise_generation():
    import networkx as nx

    from src.generation.grammar import EdgeType, MissionGraph, MissionNode, NodeType

    class _Logger:
        def __init__(self):
            self.info_calls = []

        def info(self, *args, **kwargs):
            self.info_calls.append((args, kwargs))

    class _Guidance:
        guidance_scale = 0.75

    class _Diffusion:
        cfg_scale = 2.5
        guidance = _Guidance()

    class _Pipeline:
        def __init__(self):
            self.diffusion = _Diffusion()
            self.calls = []

        def generate_dungeon(self, **kwargs):
            self.calls.append(kwargs)

            class _Result:
                dungeon_grid = np.zeros((16, 11), dtype=np.int32)
                metrics = {"num_rooms": 2, "repair_rate": 0.0}

            return _Result()

    mission_graph = MissionGraph()
    mission_graph.add_node(MissionNode(id=0, node_type=NodeType.START))
    mission_graph.add_node(MissionNode(id=1, node_type=NodeType.GOAL))
    mission_graph.add_edge(0, 1, edge_type=EdgeType.PATH)

    pipeline = _Pipeline()
    logger = _Logger()

    result = generation_pipeline.generate_dungeon_with_pipeline(
        pipeline,
        mission_graph,
        seed=123,
        logger=logger,
    )

    assert tuple(result.dungeon_grid.shape) == (16, 11)
    assert pipeline.calls
    kwargs = pipeline.calls[0]
    assert isinstance(kwargs["mission_graph"], nx.DiGraph)
    assert kwargs["seed"] == 123
    assert kwargs["enable_map_elites"] is False
    assert kwargs["apply_repair"] is True
    assert logger.info_calls


def test_generation_cli_canonical_wrapper_samples_through_pipeline(monkeypatch):
    pipeline = object()
    captured = {}

    def _fake_generate_mission_graph(random_module, *, seed=None, num_rooms=None):
        captured["mission_seed"] = seed
        return {"mission_graph": {"seed": seed}}

    def _fake_generate_dungeon_with_pipeline(active_pipeline, mission_graph, *, seed, logger):
        captured["pipeline"] = active_pipeline
        captured["mission_graph"] = mission_graph
        captured["sample_seed"] = seed

        class _Result:
            dungeon_grid = np.zeros((16, 11), dtype=np.int32)

        return _Result()

    monkeypatch.setattr(generation_cli, "generate_mission_graph", _fake_generate_mission_graph)
    monkeypatch.setattr(generation_cli, "generate_dungeon_with_pipeline", _fake_generate_dungeon_with_pipeline)

    generator = generation_cli.CanonicalDungeonGenerator(pipeline, seed=77)
    sample = generator.sample(device=torch.device("cpu"))

    assert tuple(sample.shape) == (1, 1, 16, 11)
    assert captured["pipeline"] is pipeline
    assert captured["mission_graph"] == {"seed": 77}
    assert captured["mission_seed"] == 77
    assert captured["sample_seed"] == 77


def test_dungeon_validator_uses_grid_bfs_for_semantic_dungeon_maps():
    validator = generation_cli.DungeonValidator(use_external=False)
    grid = np.full((6, 6), generation_cli.SEMANTIC_PALETTE["WALL"], dtype=np.int32)
    grid[1:5, 1:5] = generation_cli.SEMANTIC_PALETTE["FLOOR"]
    grid[1, 1] = generation_cli.SEMANTIC_PALETTE["START"]
    grid[4, 4] = generation_cli.SEMANTIC_PALETTE["TRIFORCE"]

    dungeon_map = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    assert validator.check_solvability(dungeon_map) is True


def test_dungeon_validator_does_not_use_random_logicnet_for_latent_maps():
    validator = generation_cli.DungeonValidator(use_external=False)
    latent_like = torch.zeros((1, 64, 4, 3), dtype=torch.float32)

    assert validator.check_solvability(latent_like) is False
    assert validator.logic_net_checkpoint_loaded is False
