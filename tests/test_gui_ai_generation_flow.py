import threading
from pathlib import Path

from src.gui.ai.ai_generation_controls import start_ai_dungeon_generation
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


def test_start_ai_generation_sets_thread_and_message():
    gui = _DummyGUI()
    start_ai_dungeon_generation(gui, threading)

    assert gui.ai_gen_thread is not None
    gui.ai_gen_thread.join(timeout=1.0)
    assert gui.ai_gen_result is None
    assert gui.ai_gen_done is False
    assert gui.messages[-1][0] == "AI generation started (background)"


def test_worker_reports_missing_checkpoint(monkeypatch):
    gui = _DummyGUI()

    def _missing_checkpoint():
        return Path("__definitely_missing_checkpoint__.pth")

    monkeypatch.setattr(ai_generation_worker, "resolve_checkpoint_path", _missing_checkpoint)

    class _Logger:
        def warning(self, *_args, **_kwargs):
            return None

        def exception(self, *_args, **_kwargs):
            return None

    ai_generation_worker.run_ai_generation_worker(gui, _Logger())

    assert gui.messages
    assert gui.messages[-1][0] == "No AI checkpoint found - train first!"


def test_resolve_checkpoint_path_honors_env_override(monkeypatch):
    monkeypatch.setenv("KLTN_CHECKPOINT_PATH", "checkpoints/custom_model.pth")
    resolved = generation_pipeline.resolve_checkpoint_path()

    assert resolved.name == "custom_model.pth"


def test_generate_mission_graph_is_deterministic_with_seed():
    import random

    fixed_seed = 314159
    data_a = generation_pipeline.generate_mission_graph(random, seed=fixed_seed)
    data_b = generation_pipeline.generate_mission_graph(random, seed=fixed_seed)

    assert data_a["seed"] == fixed_seed
    assert data_b["seed"] == fixed_seed
    assert data_a["num_nodes"] == data_b["num_nodes"]
    assert data_a["num_edges"] == data_b["num_edges"]


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

