import numpy as np
import torch

from src.config_system import merge_config
from src.core.condition_encoder import CrossAttentionFusion
from src.core.definitions import GRAPH_EDGE_FEATURE_DIM
from src.core.vqvae import VectorQuantizer
from src.generation.grammar import EdgeType, MissionGraph, MissionGrammar, MissionNode, NodeType
from src.generation.weighted_bayesian_wfc import TilePrior, WeightedBayesianWFC
from src.generation.wfc_refiner import CausalWFC, ZeldaTileSet
from src.train_masked_room import MaskedRoomTrainer


def _self_compatible_priors():
    return {
        1: TilePrior(
            tile_id=1,
            frequency=0.5,
            adjacency_counts={(1, direction): 1 for direction in ("N", "S", "E", "W")},
        ),
        2: TilePrior(
            tile_id=2,
            frequency=0.5,
            adjacency_counts={(2, direction): 1 for direction in ("N", "S", "E", "W")},
        ),
    }


def test_weighted_wfc_recursively_propagates_support_reductions():
    wfc = WeightedBayesianWFC(width=3, height=1, tile_priors=_self_compatible_priors())
    wfc._collapse_cell(0, 0, 1)

    assert wfc._propagate_constraints(0, 0, 1)
    assert np.array_equal(wfc.superposition[0, 1, :], np.array([1.0, 0.0]))
    assert np.array_equal(wfc.superposition[0, 2, :], np.array([1.0, 0.0]))


def test_weighted_wfc_zero_support_is_a_contradiction_not_a_prior_reset():
    priors = {1: TilePrior(tile_id=1, frequency=1.0, adjacency_counts={})}
    wfc = WeightedBayesianWFC(width=2, height=1, tile_priors=priors)
    wfc._collapse_cell(0, 0, 1)

    assert not wfc._propagate_constraints(0, 0, 1)
    assert wfc.get_diagnostics()["zero_prob_resets"] == 0


def test_causal_wfc_backtrack_rebuilds_lock_state_and_bans_failed_tile():
    wfc = CausalWFC(ZeldaTileSet(), width=3, height=1, seed=1)
    wfc.initialize()
    wfc._collapse_cell(0, 0, 5)
    wfc._update_game_state(0, 0, 5)
    wfc._collapse_cell(0, 1, 4)
    wfc._update_game_state(0, 1, 4)

    assert wfc.game_state.keys_collected == 1
    assert wfc.game_state.lock_positions == [(0, 1)]
    assert wfc._backtrack()
    assert wfc.game_state.keys_collected == 1
    assert wfc.game_state.lock_positions == []
    assert 4 not in wfc.grid[0][1].possibilities


def test_cross_attention_fusion_projects_mismatched_local_residual():
    fusion = CrossAttentionFusion(local_dim=5, global_dim=7, output_dim=8, num_heads=2)

    result = fusion(torch.randn(2, 5), torch.randn(2, 3, 7))

    assert result.shape == (2, 8)


def test_masked_room_edge_attr_fallback_uses_schema_width():
    result = MaskedRoomTrainer._encode_edge_features(
        {"edge_attr": torch.tensor([0, 3, GRAPH_EDGE_FEATURE_DIM + 1])},
        torch.device("cpu"),
    )

    assert result is not None
    assert result.shape == (3, GRAPH_EDGE_FEATURE_DIM)
    assert result[2, -1].item() == 1.0


def test_masked_room_config_accepts_puzzle_semantic_checkpoint_metric():
    config = merge_config(
        cli_overrides={
            "masked_room": {"best_checkpoint_metric": "val_puzzle_stage_semantic_loss"}
        }
    )

    assert config["masked_room"]["best_checkpoint_metric"] == "val_puzzle_stage_semantic_loss"


def test_vqvae_eval_reports_commitment_loss_without_mutating_ema():
    quantizer = VectorQuantizer(num_embeddings=4, embedding_dim=3, use_ema=True)
    quantizer.eval()
    before = quantizer.ema_cluster_size.clone()

    _, losses, _ = quantizer(torch.randn(1, 3, 2, 2))

    assert losses.item() > 0.0
    assert torch.equal(quantizer.ema_cluster_size, before)


def test_grammar_reachability_does_not_walk_backwards_through_path_edges():
    graph = MissionGraph()
    graph.add_node(MissionNode(id=0, node_type=NodeType.START))
    graph.add_node(MissionNode(id=1, node_type=NodeType.LOCK, key_id=2))
    graph.add_node(MissionNode(id=2, node_type=NodeType.KEY, key_id=2))
    graph.add_edge(0, 1, EdgeType.LOCKED, key_required=2)
    graph.add_edge(2, 1, EdgeType.PATH)

    grammar = MissionGrammar(seed=1)

    assert not grammar._is_reachable_without(graph, 0, 2, exclude={1})
