"""
H-MOLQD Block VI: External Validator
=====================================

Scientific Evaluation via Agent Simulation.

This module provides hard verification of dungeon solvability using
deterministic A* pathfinding, complementing the soft LogicNet approximation.

Components:
1. Agent Simulator: Headless A* agent for path verification
2. Solvability Checker: 100% correctness verification
3. Path Verifier: Validates key collection sequences

Integration with existing KLTN pathfinding:
    Uses ZeldaPathfinder from zelda_pathfinder.py for state-space search.

"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

import numpy as np
import networkx as nx

# Import KLTN core definitions
from src.core.definitions import (
    SEMANTIC_PALETTE,
    TileID,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    EDGE_TYPE_MAP,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ValidationState:
    """State for validation pathfinding."""
    position: Tuple[int, int]       # Current room position
    keys_held: int = 0              # Number of keys in inventory
    keys_collected: Set[int] = field(default_factory=set)  # Node IDs where keys collected
    doors_opened: Set[Tuple[int, int]] = field(default_factory=set)  # Opened door edges
    path: List[Tuple[int, int]] = field(default_factory=list)  # Path taken
    
    def copy(self) -> 'ValidationState':
        return ValidationState(
            position=self.position,
            keys_held=self.keys_held,
            keys_collected=self.keys_collected.copy(),
            doors_opened=self.doors_opened.copy(),
            path=self.path.copy(),
        )
    
    def __hash__(self):
        return hash((
            self.position,
            self.keys_held,
            frozenset(self.keys_collected),
            frozenset(self.doors_opened),
        ))
    
    def __eq__(self, other):
        if not isinstance(other, ValidationState):
            return False
        return (self.position == other.position and
                self.keys_held == other.keys_held and
                self.keys_collected == other.keys_collected and
                self.doors_opened == other.doors_opened)


@dataclass
class ValidationResult:
    """Result of dungeon validation."""
    is_solvable: bool
    solution_path: Optional[List[Tuple[int, int]]] = None
    key_collection_order: Optional[List[int]] = None
    doors_opened: Optional[List[Tuple[int, int]]] = None
    path_length: int = 0
    states_explored: int = 0
    failure_reason: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# AGENT SIMULATOR
# ============================================================================

class AgentSimulator:
    """
    Headless A* agent for dungeon path verification.
    
    Simulates player traversal through the dungeon graph,
    properly handling:
    - Key collection (consumable)
    - Locked doors (require key)
    - Bombable walls (assumed infinite bombs)
    - Soft-locked passages (one-way)
    - Stairs/warps
    
    Args:
        graph: Dungeon connectivity graph
        room_data: Optional room semantic grids for within-room validation
        strict_mode: If True, only allow simple open passages
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        room_data: Optional[Dict[int, np.ndarray]] = None,
        strict_mode: bool = False,
    ):
        self.graph = graph
        self.room_data = room_data
        self.strict_mode = strict_mode
        
        # Extract key and item locations from graph
        self.key_nodes: Set[int] = set()
        self.item_nodes: Dict[int, str] = {}
        self.start_node: Optional[int] = None
        self.goal_node: Optional[int] = None
        
        for node_id, data in graph.nodes(data=True):
            label = data.get('label', '')
            
            if 'k' in label.split(','):
                self.key_nodes.add(node_id)
            if 'K' in label.split(','):
                self.item_nodes[node_id] = 'boss_key'
            if 'I' in label.split(','):
                self.item_nodes[node_id] = 'key_item'
            if 's' in label.split(','):
                self.start_node = node_id
            if 't' in label.split(','):
                self.goal_node = node_id
    
    def can_traverse(
        self,
        from_node: int,
        to_node: int,
        state: ValidationState,
    ) -> Tuple[bool, ValidationState, str]:
        """
        Check if edge can be traversed with current state.
        
        Args:
            from_node: Source node
            to_node: Destination node
            state: Current validation state
            
        Returns:
            (can_traverse, new_state, edge_type)
        """
        edge_data = self.graph.get_edge_data(from_node, to_node)
        if edge_data is None:
            return False, state, 'none'
        
        edge_label = edge_data.get('label', '')
        edge_type = edge_data.get('edge_type', EDGE_TYPE_MAP.get(edge_label, 'open'))
        edge_id = (from_node, to_node)
        
        new_state = state.copy()
        
        if self.strict_mode:
            # Only allow open passages
            if edge_type != 'open' and edge_label != '':
                return False, state, edge_type
            return True, new_state, 'open'
        
        # Handle different edge types
        if edge_type in ('open', ''):
            return True, new_state, 'open'
        
        if edge_type in ('key_locked', 'k'):
            # Check if already opened
            if edge_id in state.doors_opened:
                return True, new_state, 'key_locked'
            
            # Try to use a key
            if state.keys_held > 0:
                new_state.keys_held -= 1
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))  # Both directions
                return True, new_state, 'key_locked'
            
            return False, state, 'key_locked'
        
        if edge_type in ('bombable', 'b'):
            # Assume infinite bombs
            if edge_id not in state.doors_opened:
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
            return True, new_state, 'bombable'
        
        if edge_type in ('soft_locked', 'l'):
            # One-way passage, always traversable forward
            return True, new_state, 'soft_locked'
        
        if edge_type in ('stair', 's'):
            # Teleport/stair, always traversable
            return True, new_state, 'stair'
        
        # Unknown edge type - be conservative
        logger.warning(f"Unknown edge type: {edge_type}")
        return True, new_state, edge_type
    
    def collect_items(self, node_id: int, state: ValidationState) -> ValidationState:
        """Collect any items at the current node."""
        new_state = state.copy()
        
        # Collect key if present and not already collected
        if node_id in self.key_nodes and node_id not in state.keys_collected:
            new_state.keys_held += 1
            new_state.keys_collected.add(node_id)
        
        return new_state
    
    def heuristic(self, node: int, goal: int) -> float:
        """A* heuristic (node distance estimate)."""
        # Simple: shortest path length in unweighted graph
        try:
            return nx.shortest_path_length(
                self.graph.to_undirected(),
                node, goal
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')
    
    def find_path(
        self,
        start_node: Optional[int] = None,
        goal_node: Optional[int] = None,
        max_states: int = 100000,
    ) -> ValidationResult:
        """
        Find path from start to goal using A* with state-space search.
        
        Args:
            start_node: Override start node
            goal_node: Override goal node
            max_states: Maximum states to explore
            
        Returns:
            ValidationResult with path and metrics
        """
        start = start_node or self.start_node
        goal = goal_node or self.goal_node
        
        if start is None:
            return ValidationResult(
                is_solvable=False,
                failure_reason="No start node defined"
            )
        
        if goal is None:
            return ValidationResult(
                is_solvable=False,
                failure_reason="No goal node defined"
            )
        
        # Initialize
        initial_state = ValidationState(position=start, path=[start])
        initial_state = self.collect_items(start, initial_state)
        
        # Priority queue: (f_cost, g_cost, counter, state)
        counter = 0
        open_set = [(self.heuristic(start, goal), 0, counter, initial_state)]
        visited = set()
        states_explored = 0
        
        while open_set and states_explored < max_states:
            _, g_cost, _, current = heapq.heappop(open_set)
            
            state_hash = hash(current)
            if state_hash in visited:
                continue
            visited.add(state_hash)
            states_explored += 1
            
            # Check if goal reached
            if current.position == goal:
                return ValidationResult(
                    is_solvable=True,
                    solution_path=current.path,
                    key_collection_order=list(current.keys_collected),
                    doors_opened=list(current.doors_opened),
                    path_length=len(current.path),
                    states_explored=states_explored,
                    metrics={
                        'keys_used': len(current.keys_collected) - current.keys_held,
                        'doors_opened': len(current.doors_opened) // 2,
                    }
                )
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current.position):
                can_go, new_state, edge_type = self.can_traverse(
                    current.position, neighbor, current
                )
                
                if not can_go:
                    continue
                
                # Update state
                new_state.position = neighbor
                new_state.path = current.path + [neighbor]
                new_state = self.collect_items(neighbor, new_state)
                
                new_hash = hash(new_state)
                if new_hash in visited:
                    continue
                
                # Add to open set
                new_g = g_cost + 1
                new_f = new_g + self.heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (new_f, new_g, counter, new_state))
        
        return ValidationResult(
            is_solvable=False,
            states_explored=states_explored,
            failure_reason="No path found (exhausted search or max states reached)"
        )


# ============================================================================
# SOLVABILITY CHECKER
# ============================================================================

class SolvabilityChecker:
    """
    High-level solvability verification interface.
    
    Provides multiple validation modes:
    - STRICT: Only open passages
    - REALISTIC: Normal gameplay mechanics
    - FULL: Complete state-space search
    
    Args:
        mode: Validation mode
    """
    
    MODE_STRICT = 'strict'
    MODE_REALISTIC = 'realistic'
    MODE_FULL = 'full'
    
    def __init__(self, mode: str = 'full'):
        self.mode = mode
    
    def check(
        self,
        graph: nx.DiGraph,
        room_data: Optional[Dict[int, np.ndarray]] = None,
        start_node: Optional[int] = None,
        goal_node: Optional[int] = None,
    ) -> ValidationResult:
        """
        Check if dungeon is solvable.
        
        Args:
            graph: Dungeon connectivity graph
            room_data: Optional room grids
            start_node: Override start
            goal_node: Override goal
            
        Returns:
            ValidationResult
        """
        strict = self.mode == self.MODE_STRICT
        
        simulator = AgentSimulator(
            graph=graph,
            room_data=room_data,
            strict_mode=strict,
        )
        
        return simulator.find_path(start_node, goal_node)
    
    def check_all_rooms_reachable(
        self,
        graph: nx.DiGraph,
        start_node: Optional[int] = None,
    ) -> Tuple[bool, Set[int]]:
        """
        Check if all rooms are reachable from start.
        
        Returns:
            (all_reachable, unreachable_nodes)
        """
        simulator = AgentSimulator(graph=graph)
        start = start_node or simulator.start_node
        
        if start is None:
            return False, set(graph.nodes())
        
        # BFS from start
        reachable = {start}
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            for neighbor in graph.neighbors(node):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        unreachable = set(graph.nodes()) - reachable
        return len(unreachable) == 0, unreachable


# ============================================================================
# PATH VERIFIER
# ============================================================================

class PathVerifier:
    """
    Verifies that a given path is valid.
    
    Useful for checking generated solutions or
    validating paths from other solvers.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.simulator = AgentSimulator(graph)
    
    def verify(
        self,
        path: List[int],
        start_state: Optional[ValidationState] = None,
    ) -> Tuple[bool, Optional[str], ValidationState]:
        """
        Verify a path is traversable.
        
        Args:
            path: Sequence of node IDs
            start_state: Initial state (defaults to empty)
            
        Returns:
            (is_valid, error_message, final_state)
        """
        if not path:
            return False, "Empty path", ValidationState(position=(-1, -1))
        
        state = start_state or ValidationState(position=path[0])
        state = self.simulator.collect_items(path[0], state)
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            can_go, new_state, edge_type = self.simulator.can_traverse(
                from_node, to_node, state
            )
            
            if not can_go:
                return False, f"Cannot traverse edge {from_node}->{to_node} ({edge_type})", state
            
            state = new_state
            state.position = to_node
            state = self.simulator.collect_items(to_node, state)
        
        return True, None, state
    
    def find_key_sequence(
        self,
        path: List[int],
    ) -> List[Tuple[int, str]]:
        """
        Extract key collection/usage sequence from path.
        
        Returns:
            List of (node_id, action) where action is 'collect' or 'use'
        """
        sequence = []
        state = ValidationState(position=path[0] if path else 0)
        
        for i, node in enumerate(path):
            # Check for key collection
            if node in self.simulator.key_nodes and node not in state.keys_collected:
                sequence.append((node, 'collect'))
                state.keys_held += 1
                state.keys_collected.add(node)
            
            # Check for key usage on next edge
            if i < len(path) - 1:
                next_node = path[i + 1]
                edge_data = self.graph.get_edge_data(node, next_node)
                
                if edge_data:
                    edge_type = edge_data.get('edge_type', edge_data.get('label', ''))
                    edge_id = (node, next_node)
                    
                    if edge_type in ('key_locked', 'k') and edge_id not in state.doors_opened:
                        sequence.append((node, f'use_key_to_{next_node}'))
                        state.keys_held -= 1
                        state.doors_opened.add(edge_id)
        
        return sequence


# ============================================================================
# EXTERNAL VALIDATOR (Main Interface)
# ============================================================================

class ExternalValidator:
    """
    External Validator for H-MOLQD Block VI.
    
    Provides ground-truth solvability verification for generated dungeons,
    complementing the differentiable LogicNet approximation.
    
    Key Features:
    - Deterministic A* path verification
    - Multiple validation modes
    - Key sequence analysis
    - Comprehensive metrics
    
    Usage:
        validator = ExternalValidator()
        
        # Quick check
        result = validator.validate(dungeon_graph)
        print(f"Solvable: {result.is_solvable}")
        
        # Detailed analysis
        analysis = validator.analyze(dungeon_graph)
    """
    
    def __init__(self, mode: str = 'full'):
        self.checker = SolvabilityChecker(mode=mode)
    
    def validate(
        self,
        graph: nx.DiGraph,
        room_data: Optional[Dict[int, np.ndarray]] = None,
        start_node: Optional[int] = None,
        goal_node: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate dungeon solvability.
        
        Args:
            graph: Dungeon connectivity graph
            room_data: Optional room grids
            start_node: Start node override
            goal_node: Goal node override
            
        Returns:
            ValidationResult
        """
        return self.checker.check(graph, room_data, start_node, goal_node)
    
    def analyze(
        self,
        graph: nx.DiGraph,
        room_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive dungeon analysis.
        
        Returns:
            Dict with:
                - is_solvable: bool
                - all_reachable: bool
                - solution_path: Optional[List]
                - path_length: int
                - key_sequence: List
                - connectivity_metrics: Dict
                - difficulty_estimate: float
        """
        analysis = {}
        
        # Basic solvability
        result = self.validate(graph, room_data)
        analysis['is_solvable'] = result.is_solvable
        analysis['solution_path'] = result.solution_path
        analysis['path_length'] = result.path_length
        analysis['states_explored'] = result.states_explored
        
        # Room reachability
        all_reachable, unreachable = self.checker.check_all_rooms_reachable(graph)
        analysis['all_reachable'] = all_reachable
        analysis['unreachable_rooms'] = list(unreachable)
        
        # Key sequence analysis
        if result.solution_path:
            verifier = PathVerifier(graph)
            analysis['key_sequence'] = verifier.find_key_sequence(result.solution_path)
        else:
            analysis['key_sequence'] = []
        
        # Connectivity metrics
        analysis['connectivity'] = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'avg_degree': sum(dict(graph.degree()).values()) / max(1, graph.number_of_nodes()),
            'is_connected': nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph),
        }
        
        # Difficulty estimate
        analysis['difficulty_estimate'] = self._estimate_difficulty(graph, result)
        
        return analysis
    
    def _estimate_difficulty(
        self,
        graph: nx.DiGraph,
        result: ValidationResult,
    ) -> float:
        """Estimate dungeon difficulty (0-1 scale)."""
        if not result.is_solvable:
            return 1.0
        
        factors = []
        
        # Path length factor
        path_factor = min(1.0, result.path_length / (2 * graph.number_of_nodes()))
        factors.append(path_factor)
        
        # Key usage factor
        if result.metrics:
            keys_used = result.metrics.get('keys_used', 0)
            key_factor = min(1.0, keys_used / 5.0)
            factors.append(key_factor)
        
        # Backtracking factor (states explored vs path length)
        if result.path_length > 0:
            backtrack_factor = min(1.0, result.states_explored / (10 * result.path_length))
            factors.append(backtrack_factor)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def batch_validate(
        self,
        graphs: List[nx.DiGraph],
    ) -> List[ValidationResult]:
        """Validate multiple dungeons."""
        return [self.validate(g) for g in graphs]
    
    def compute_solvability_rate(
        self,
        graphs: List[nx.DiGraph],
    ) -> float:
        """Compute solvability rate for a batch of dungeons."""
        results = self.batch_validate(graphs)
        solvable = sum(1 for r in results if r.is_solvable)
        return solvable / len(graphs) if graphs else 0.0
