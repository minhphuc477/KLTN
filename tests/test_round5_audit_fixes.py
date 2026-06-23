import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from src.core.definitions import ROOM_HEIGHT, ROOM_TOPOLOGY_CHANNELS, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_WIDTH, TileID
from src.core.discrete_masked_model import create_discrete_masked_model
from src.data_processing.data_adapter import IntelligentDataAdapter
from src.evaluation.map_elites import LinearityLeniencyExtractor
from src.evaluation.search_benchmark_utils import confusion_ratio_vs_oracle, normalized_confusion_ratio
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.ml.logic_net import DifferentiableTortuosity
from src.pipeline.room_stitching import carve_room_connection_between_bboxes
from src.zelda_data.parsers.core_parsers import VGLCParser


def test_output_node_cap_keeps_required_item_provider():
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.6, 1.0],
        population_size=2,
        generations=1,
        max_nodes=2,
        seed=1,
    )
    graph = nx.DiGraph()
    graph.add_node("start", type="START")
    graph.add_node("key", type="ITEM", item_type="HOOKSHOT")
    graph.add_node("gate", type="ENEMY")
    graph.add_node("goal", type="GOAL")
    graph.add_node("extra", type="TREASURE")
    graph.add_edges_from(
        [
            ("start", "key", {"edge_type": "PATH"}),
            ("key", "gate", {"edge_type": "ITEM_GATE", "item_required": "HOOKSHOT"}),
            ("gate", "goal", {"edge_type": "PATH"}),
            ("start", "extra", {"edge_type": "PATH"}),
        ]
    )

    capped = gen._enforce_output_node_cap(graph)

    assert "key" in capped.nodes
    assert "gate" in capped.nodes
    assert "goal" in capped.nodes


def test_output_connectivity_repairs_goal_only_component():
    graph = nx.DiGraph()
    graph.add_node("start", type="START", position=(0, 0, 0))
    graph.add_node("middle", type="ENEMY", position=(1, 0, 0))
    graph.add_node("goal", type="GOAL", position=(3, 0, 0))
    graph.add_edge("start", "middle", edge_type="PATH")

    repaired = EvolutionaryTopologyGenerator._repair_output_connectivity(graph)

    assert nx.is_connected(repaired.to_undirected())
    assert any("goal" in edge for edge in repaired.edges())


def test_leniency_counts_boss_key_and_boss_lock():
    graph = nx.DiGraph()
    graph.add_node("s", type="START")
    graph.add_node("g", type="GOAL")
    graph.add_edge("s", "g", edge_type="BOSS_LOCKED")
    extractor = LinearityLeniencyExtractor()

    assert extractor._compute_leniency(graph) == pytest.approx(0.0)

    graph.add_node("bk", type="BIG_KEY")
    assert extractor._compute_leniency(graph) == pytest.approx(1.0)


def test_confusion_ratios_handle_zero_length_oracle_paths():
    assert confusion_ratio_vs_oracle(0, 0, oracle_status="solved", candidate_success=True) == pytest.approx(0.0)
    assert math.isfinite(confusion_ratio_vs_oracle(0, 3, oracle_status="solved", candidate_success=True))
    assert normalized_confusion_ratio(0, 0, 0, oracle_status="solved", candidate_success=True) == pytest.approx(0.0)
    assert normalized_confusion_ratio(0, 3, 0, oracle_status="solved", candidate_success=True) == pytest.approx(1.0)


def test_differentiable_tortuosity_accepts_winding_paths_longer_than_h_plus_w():
    probability = torch.full((1, 1, 5, 5), 1e-4)
    path = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (3, 2),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 3),
        (0, 4),
    ]
    for row, col in path:
        probability[0, 0, row, col] = 1.0
    module = DifferentiableTortuosity(num_iterations=20)

    path_lengths = module.compute_soft_path_length(probability, [(0, 0)], [(0, 4)])

    assert float(path_lengths.item()) > 10.0
    assert float(path_lengths.item()) < 26.0


def test_edge_aware_logit_bias_accumulates_corner_evidence():
    model = create_discrete_masked_model(
        num_classes=44,
        hidden_dim=16,
        context_dim=16,
        num_steps=2,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_num_heads=4,
    )
    logits = torch.zeros(1, 44, ROOM_HEIGHT, ROOM_WIDTH)
    topology = torch.zeros(1, ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH)
    topology[:, ROOM_TOPOLOGY_CHANNELS["door_n"], 0, :] = 1.0
    topology[:, ROOM_TOPOLOGY_CHANNELS["door_w"], :, 0] = 1.0

    biased = model._apply_edge_aware_logit_bias(logits, {"room_topology_map": topology}, bias_strength=2.0)

    corner_door_delta = float(biased[0, int(TileID.DOOR_OPEN), 0, 0].item())
    corner_wall_delta = float(biased[0, int(TileID.WALL), 0, 0].item())
    assert corner_door_delta == pytest.approx(4.0)
    assert corner_wall_delta == pytest.approx(-4.0)


def test_quest2_graph_lookup_prefers_vglc_loz2_naming(tmp_path: Path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    canonical = graph_dir / "LoZ2_3.dot"
    legacy = graph_dir / "LoZ_3_q2.dot"
    canonical.write_text("digraph {}", encoding="utf-8")
    legacy.write_text("digraph {}", encoding="utf-8")

    adapter = object.__new__(IntelligentDataAdapter)
    adapter.graph_dir = graph_dir

    assert adapter._find_graph_file("tloz3_2") == canonical


def test_core_parser_detects_boundary_doors_not_inner_wall_doors():
    parser = VGLCParser(room_cls=object)
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), "W", dtype="<U1")
    grid[0, 4:7] = "D"
    grid[1, 4:7] = "W"
    grid[14, 4:7] = "D"
    grid[15, 4:7] = "W"

    doors = parser._detect_doors(grid)

    assert doors["N"] is True
    assert doors["S"] is False


def test_stitcher_uses_canonical_door_widths_for_adjacent_rooms():
    grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH * 2), dtype=np.int32)
    carve_room_connection_between_bboxes(
        grid,
        (0, 0, ROOM_WIDTH - 1, ROOM_HEIGHT - 1),
        (ROOM_WIDTH, 0, ROOM_WIDTH * 2 - 1, ROOM_HEIGHT - 1),
    )

    open_rows = np.flatnonzero(grid[:, ROOM_WIDTH - 1] != 0)
    assert open_rows.tolist() == [7, 8]
