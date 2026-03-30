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
from .distributed import (
    DistributedContext,
    average_gradients,
    build_torchrun_command,
    destroy_distributed,
    get_env_local_rank,
    get_env_rank,
    get_env_world_size,
    initialize_distributed,
    is_torchrun_environment,
    make_distributed_sampler,
    maybe_barrier,
    maybe_launch_with_torchrun,
    reduce_scalar_metrics,
    resolve_device,
)
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
from .playtest_telemetry import (
    PlaytestEvent,
    PlaytestSession,
    PlaytestTelemetryCollector,
)

__all__ = [
    # Checkpointing/training
    'CheckpointManager', 
    'EarlyStopping', 
    'MetricsLogger',
    'DistributedContext',
    'average_gradients',
    'build_torchrun_command',
    'destroy_distributed',
    'get_env_local_rank',
    'get_env_rank',
    'get_env_world_size',
    'initialize_distributed',
    'is_torchrun_environment',
    'make_distributed_sampler',
    'maybe_barrier',
    'maybe_launch_with_torchrun',
    'reduce_scalar_metrics',
    'resolve_device',
    # Graph utilities
    'filter_virtual_nodes',
    'get_physical_start_node',
    'get_node_types',
    'has_node_type',
    'find_nodes_by_type',
    'get_edge_type',
    'validate_goal_subgraph',
    'validate_graph_topology',
    # Playtest telemetry
    'PlaytestEvent',
    'PlaytestSession',
    'PlaytestTelemetryCollector',
]
