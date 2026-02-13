"""
Utility Module for KLTN PCG Training
====================================

Common utilities for training, checkpointing, logging, and graph operations.

Components:
    - CheckpointManager: Save/load model checkpoints
    - EarlyStopping: Early stopping callback
    - MetricsLogger: Training metrics tracking
    - Graph utilities: VGLC-compliant graph operations
"""

from .checkpoint import CheckpointManager, EarlyStopping, MetricsLogger
from .graph_utils import (
    filter_virtual_nodes,
    get_physical_start_node,
    get_node_types,
    has_node_type,
    find_nodes_by_type,
    get_edge_type,
    validate_goal_subgraph,
    validate_graph_topology,
)

__all__ = [
    # Checkpointing/training
    'CheckpointManager', 
    'EarlyStopping', 
    'MetricsLogger',
    # Graph utilities
    'filter_virtual_nodes',
    'get_physical_start_node',
    'get_node_types',
    'has_node_type',
    'find_nodes_by_type',
    'get_edge_type',
    'validate_goal_subgraph',
    'validate_graph_topology',
]
