"""
Machine Learning Module for KLTN PCG
====================================

Neural network components for procedural dungeon generation.

Components:
    - LogicNet: Differentiable solvability approximation via Soft-Bellman-Ford
    - HeuristicNetwork: ML-based heuristic learning for A* search
    
The LogicNet provides a differentiable proxy for dungeon solvability,
enabling end-to-end training with solvability constraints.
"""

from .heuristic_learning import HeuristicNetwork, TrainingExample

# Import LogicNet if available
try:
    from .logic_net import LogicNet, SoftBellmanFord
    __all__ = ['LogicNet', 'SoftBellmanFord', 'HeuristicNetwork', 'TrainingExample']
except ImportError:
    __all__ = ['HeuristicNetwork', 'TrainingExample']
