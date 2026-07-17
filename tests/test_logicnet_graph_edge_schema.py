import torch

from src.core.definitions import GRAPH_EDGE_FEATURE_DIM
from src.core.logic_net import (
    GRAPH_EDGE_STAIR_FEATURE_INDEX,
    LogicNet,
)


def test_stair_feature_is_traversable_without_lock_or_gate_penalty():
    logic_net = LogicNet(latent_dim=4, num_tile_classes=5)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_features = torch.zeros(1, GRAPH_EDGE_FEATURE_DIM)
    edge_features[0, GRAPH_EDGE_STAIR_FEATURE_INDEX] = 1.0

    penalty = logic_net._edge_feature_penalty(edge_features, None, num_edges=1)
    locked = logic_net._locked_edge_mask(
        node_count=2,
        edge_index=edge_index,
        adjacency=None,
        edge_features=edge_features,
        edge_attr=None,
        device=torch.device("cpu"),
    )
    adjacency, weights = logic_net._build_adjacency_and_weights(
        node_count=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        edge_index=edge_index,
        edge_features=edge_features,
    )

    assert penalty is not None
    assert float(penalty[0]) == 0.0
    assert not bool(locked[0, 1])
    assert adjacency is not None and float(adjacency[0, 1]) == 1.0
    assert weights is not None and float(weights[0, 1]) == 1.0
