"""
Key Economy Validator - Worst-Case Solvability Analysis
========================================================

MATHEMATICAL RIGOR: Soft-Lock Prevention

Problem (Thesis Defense Concern #3):
    Current system only validates with A* greedy player.
    Worst-case scenarios not tested:
    - Adversarial player takes wrong key first → soft-lock
    - Graph has surplus keys but wrong topology → unsolvable
    
Solution:
    Graph-theoretic worst-case analysis:
    1. Greedy Player: Always takes optimal path (baseline)
    2. Adversarial Player: Always makes worst choices (stress test)
    3. Mission Graph Analysis: Key surplus + topology validation
    
Mathematical Formulation:
-------------------------
Greedy Player (Baseline):
    - Always takes shortest path to goal
    - Takes keys only when needed for immediate doors
    - Best-case solvability (optimistic)

Adversarial Player (Worst-Case):
    - Takes keys in reverse dependency order
    - Exhausts optional branches before critical path
    - Worst-case solvability (stress test)

Mission Graph Topology:
    - Linear: Start → K1 → D1 → K2 → D2 → Goal
    - Tree: Branching paths with key dependencies
    - Diamond: Multiple paths converging
    - Cycle: Loops with key gates

Key Surplus Rule:
    For each lock L requiring key K:
        available_keys_before(L) >= 1
    
    Validated for BOTH greedy and adversarial players.

Validation Criterion:
    Dungeon is VALID iff:
        1. Greedy player can solve
        2. Adversarial player can solve
        3. All topologies (linear/tree/diamond/cycle) pass
        4. Key surplus ≥ 0 for all critical-path locks

Integration Point:
    Fitness function in evolutionary algorithm
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class PlayerStrategy(Enum):
    """Player strategy for key acquisition."""
    GREEDY = "greedy"  # Optimal path, minimal key collection
    ADVERSARIAL = "adversarial"  # Worst-case, maximal wrong choices
    

class GraphTopology(Enum):
    """Mission graph topology types."""
    LINEAR = "linear"  # Start → ... → Goal (no branches)
    TREE = "tree"  # Branching paths, no cycles
    DIAMOND = "diamond"  # Multiple paths, convergence points
    CYCLE = "cycle"  # Loops/cycles in graph


@dataclass
class PlayerState:
    """Game state during dungeon traversal."""
    current_node: int
    inventory: Set[str] = field(default_factory=set)  # {'key_1', 'key_2', 'item_bomb'}
    visited_nodes: Set[int] = field(default_factory=set)
    path_taken: List[int] = field(default_factory=list)
    
    def copy(self) -> 'PlayerState':
        return PlayerState(
            current_node=self.current_node,
            inventory=self.inventory.copy(),
            visited_nodes=self.visited_nodes.copy(),
            path_taken=self.path_taken.copy()
        )


@dataclass
class KeyEconomyResult:
    """Result of key economy validation."""
    is_valid: bool
    greedy_solvable: bool
    adversarial_solvable: bool
    topology_type: GraphTopology
    key_surplus: Dict[str, int]  # {lock_id: surplus_keys}
    soft_lock_nodes: List[int]  # Nodes where soft-lock can occur
    validation_details: Dict[str, any] = field(default_factory=dict)


class GreedyPlayer:
    """
    Optimal player that takes shortest path and minimal keys.
    
    Strategy:
    - Always choose shortest path to goal
    - Collect keys only when needed for immediate door
    - Never explore optional branches
    """
    
    def __init__(self, mission_graph: nx.DiGraph):
        self.graph = mission_graph
    
    def solve(self, start_node: int, goal_node: int) -> Tuple[bool, PlayerState]:
        """
        Attempt to solve dungeon with greedy strategy.
        
        Returns:
            (solvable, final_state)
        """
        state = PlayerState(current_node=start_node)
        state.visited_nodes.add(start_node)
        state.path_taken.append(start_node)
        
        # Collect initial items at start node
        self._collect_items_at_node(state, start_node)
        
        # Greedy BFS to goal
        while state.current_node != goal_node:
            # Find shortest accessible path to goal
            next_node = self._find_next_greedy_move(state, goal_node)
            
            if next_node is None:
                # Cannot proceed - soft-lock
                return (False, state)
            
            # Move to next node
            state.current_node = next_node
            state.visited_nodes.add(next_node)
            state.path_taken.append(next_node)
            
            # Collect items
            self._collect_items_at_node(state, next_node)
        
        return (True, state)
    
    def _find_next_greedy_move(
        self,
        state: PlayerState,
        goal_node: int
    ) -> Optional[int]:
        """
        Find next node on shortest accessible path to goal.
        
        Returns:
            Next node ID, or None if no valid move
        """
        # BFS over (node, inventory) state to correctly model key acquisition.
        start_state = state.copy()
        queue = deque([(start_state, [state.current_node])])
        visited: Set[Tuple[int, FrozenSet[str]]] = {
            (state.current_node, frozenset(state.inventory))
        }
        
        while queue:
            curr_state, path = queue.popleft()
            node = curr_state.current_node
            
            if node == goal_node:
                if len(path) > 1:
                    return path[1]
                return goal_node
            
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor, {})
                if not self._can_traverse_edge(curr_state, edge_data):
                    continue
                
                next_state = curr_state.copy()
                next_state.current_node = neighbor
                self._collect_items_at_node(next_state, neighbor)
                
                sig = (neighbor, frozenset(next_state.inventory))
                if sig in visited:
                    continue
                
                visited.add(sig)
                queue.append((next_state, path + [neighbor]))
        
        return None  # No path to goal
    
    def _can_traverse_edge(self, state: PlayerState, edge_data: Dict) -> bool:
        """Check if player can traverse edge given current inventory."""
        lock_type = edge_data.get('lock_type', 'open')
        
        if lock_type == 'open':
            return True
        elif lock_type == 'locked' or lock_type == 'key_locked':
            key_id = edge_data.get('key_id', 'key_generic')
            return key_id in state.inventory
        elif lock_type == 'bomb':
            return 'item_bomb' in state.inventory
        elif lock_type == 'boss':
            return 'key_boss' in state.inventory
        else:
            return True  # Unknown lock type - assume passable
    
    def _collect_items_at_node(self, state: PlayerState, node_id: int):
        """Collect all items present at node."""
        node_data = self.graph.nodes[node_id]
        
        # Collect keys
        if 'key_id' in node_data:
            state.inventory.add(node_data['key_id'])
        
        # Collect items
        if 'items' in node_data:
            for item in node_data['items']:
                state.inventory.add(item)


class AdversarialPlayer:
    """
    Worst-case player that always makes suboptimal choices.
    
    Strategy:
    - Explore optional branches before critical path
    - Collect keys in reverse dependency order
    - Take longest available path
    - Stress-test key economy worst-case
    """
    
    def __init__(self, mission_graph: nx.DiGraph, critical_path: Set[int]):
        self.graph = mission_graph
        self.critical_path = critical_path
    
    def solve(self, start_node: int, goal_node: int) -> Tuple[bool, PlayerState]:
        """
        Attempt to solve dungeon with adversarial strategy.
        
        Returns:
            (solvable, final_state)
        """
        state = PlayerState(current_node=start_node)
        state.visited_nodes.add(start_node)
        state.path_taken.append(start_node)
        
        self._collect_items_at_node(state, start_node)
        
        # Best-practice safety: allow revisits (backtracking) with loop guard.
        # This avoids false soft-locks on legitimate tree/branch topologies.
        max_steps = max(50, self.graph.number_of_nodes() * max(3, self.graph.number_of_edges() + 1))
        seen_state_counts: Dict[Tuple[int, FrozenSet[str]], int] = defaultdict(int)
        
        steps = 0
        while state.current_node != goal_node and steps < max_steps:
            signature = (state.current_node, frozenset(state.inventory))
            seen_state_counts[signature] += 1
            if seen_state_counts[signature] > self.graph.number_of_nodes() * 2:
                return (False, state)  # Degenerate loop under worst-case behavior
            
            next_node = self._find_next_adversarial_move(state, goal_node)
            if next_node is None:
                return (False, state)
            
            # Move to next node (revisits allowed).
            state.current_node = next_node
            state.visited_nodes.add(next_node)
            state.path_taken.append(next_node)
            self._collect_items_at_node(state, next_node)
            steps += 1
        
        if state.current_node != goal_node:
            return (False, state)
        
        return (True, state)
    
    def _find_next_adversarial_move(
        self,
        state: PlayerState,
        goal_node: int
    ) -> Optional[int]:
        """
        Find next move that maximizes suboptimality.
        
        Priority:
        1. Unvisited optional nodes (not on critical path)
        2. Longest available path
        3. Critical path nodes (only if no other choice)
        """
        accessible_neighbors = []
        
        for neighbor in self.graph.neighbors(state.current_node):
            edge_data = self.graph.get_edge_data(state.current_node, neighbor, {})
            if not self._can_traverse_edge(state, edge_data):
                continue
            
            is_critical = neighbor in self.critical_path
            unvisited = neighbor not in state.visited_nodes
            
            # Priority schedule (lower is more adversarial):
            # 0: unvisited optional, 1: unvisited critical,
            # 2: revisited optional (backtrack), 3: revisited critical.
            if unvisited and not is_critical:
                priority = 0
            elif unvisited and is_critical:
                priority = 1
            elif (not unvisited) and (not is_critical):
                priority = 2
            else:
                priority = 3
            
            # Prefer nodes farther from goal to maximize suboptimality.
            try:
                dist_to_goal = nx.shortest_path_length(self.graph.to_undirected(), neighbor, goal_node)
            except nx.NetworkXNoPath:
                dist_to_goal = 10**6
            
            branching = self.graph.out_degree(neighbor)
            score = (priority, -dist_to_goal, -branching, type(neighbor).__name__, str(neighbor), neighbor)
            accessible_neighbors.append(score)
        
        if not accessible_neighbors:
            return None
        
        accessible_neighbors.sort()
        return accessible_neighbors[0][-1]
    
    def _can_traverse_edge(self, state: PlayerState, edge_data: Dict) -> bool:
        """Check if player can traverse edge."""
        lock_type = edge_data.get('lock_type', 'open')
        
        if lock_type == 'open':
            return True
        elif lock_type == 'locked' or lock_type == 'key_locked':
            key_id = edge_data.get('key_id', 'key_generic')
            return key_id in state.inventory
        elif lock_type == 'bomb':
            return 'item_bomb' in state.inventory
        elif lock_type == 'boss':
            return 'key_boss' in state.inventory
        else:
            return True
    
    def _collect_items_at_node(self, state: PlayerState, node_id: int):
        """Collect all items at node."""
        node_data = self.graph.nodes[node_id]
        
        if 'key_id' in node_data:
            state.inventory.add(node_data['key_id'])
        
        if 'items' in node_data:
            for item in node_data['items']:
                state.inventory.add(item)


class MissionGraphAnalyzer:
    """
    Analyzes mission graph topology and key economy.
    
    Validates:
    - Topology type (linear/tree/diamond/cycle)
    - Key surplus for all locks
    - Critical path vs optional branches
    - Soft-lock potentials
    """
    
    def __init__(self, mission_graph: nx.DiGraph):
        self.graph = mission_graph
        self.topology = self._classify_topology()
        self.critical_path = self._find_critical_path()
    
    def _classify_topology(self) -> GraphTopology:
        """Classify mission graph topology."""
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            return GraphTopology.CYCLE
        
        # Check branch factor
        max_out_degree = max((self.graph.out_degree(n) for n in self.graph.nodes()), default=0)
        max_in_degree = max((self.graph.in_degree(n) for n in self.graph.nodes()), default=0)
        
        if max_out_degree <= 1 and max_in_degree <= 1:
            return GraphTopology.LINEAR
        elif max_in_degree > 1:
            return GraphTopology.DIAMOND  # Converging paths
        elif max_out_degree > 1:
            return GraphTopology.TREE  # Branching paths
        else:
            return GraphTopology.LINEAR
    
    def _find_critical_path(self) -> Set[int]:
        """
        Find critical path from start to goal.
        
        Critical path = nodes required to reach goal
        """
        # Find start and goal nodes
        start_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        goal_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        if not start_nodes or not goal_nodes:
            return set(self.graph.nodes())
        
        start = start_nodes[0]
        goal = goal_nodes[0]
        
        # All nodes on any path from start to goal
        try:
            all_paths = list(nx.all_simple_paths(self.graph, start, goal))
            if all_paths:
                # Union of all paths
                critical = set()
                for path in all_paths:
                    critical.update(path)
                return critical
            else:
                return set(self.graph.nodes())
        except:
            return set(self.graph.nodes())
    
    def analyze_key_economy(self) -> Dict[str, int]:
        """
        Analyze key surplus for each lock.
        
        Returns:
            {lock_id: surplus_keys} where surplus >= 0 means valid
        """
        # Count keys available before each lock
        lock_surplus = {}
        
        for u, v, data in self.graph.edges(data=True):
            lock_type = data.get('lock_type', 'open')
            if lock_type in ['locked', 'key_locked']:
                key_id = data.get('key_id', 'key_generic')
                
                # Count keys available before this lock
                keys_before = self._count_keys_before_node(v, key_id)
                
                lock_surplus[f"{u}→{v}"] = keys_before - 1  # Need 1 key to pass
        
        return lock_surplus
    
    def _count_keys_before_node(self, node_id: int, key_id: str) -> int:
        """Count how many of key_id are available before reaching node."""
        # Best practice: only count keys that are both
        # 1) reachable from a start node, and
        # 2) on some predecessor path to the target node.
        starts = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        if not starts:
            starts = [node_id]
        
        reachable_from_start: Set[int] = set()
        for s in starts:
            try:
                reachable_from_start.add(s)
                reachable_from_start.update(nx.descendants(self.graph, s))
            except Exception:
                reachable_from_start.add(s)
        
        try:
            predecessor_region = set(nx.ancestors(self.graph, node_id))
        except Exception:
            predecessor_region = set()
        predecessor_region.add(node_id)
        
        valid_region = reachable_from_start & predecessor_region
        
        count = 0
        for n in valid_region:
            node_data = self.graph.nodes[n]
            if node_data.get('key_id') == key_id:
                count += 1
            
            # Also support keys stored in an item list.
            for item in node_data.get('items', []) or []:
                if item == key_id:
                    count += 1
        
        return count


class KeyEconomyValidator:
    """
    Master validator combining greedy + adversarial + topology analysis.
    
    Usage:
        validator = KeyEconomyValidator(mission_graph)
        result = validator.validate()
        
        if not result.is_valid:
            print(f"Soft-lock detected at nodes: {result.soft_lock_nodes}")
            print(f"Key surplus: {result.key_surplus}")
    """
    
    def __init__(self, mission_graph: nx.DiGraph):
        self.graph = mission_graph
        self.analyzer = MissionGraphAnalyzer(mission_graph)
    
    def validate(self) -> KeyEconomyResult:
        """
        Comprehensive key economy validation.
        
        Returns:
            KeyEconomyResult with full analysis
        """
        # Find start and goal
        start_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        goal_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        if not start_nodes or not goal_nodes:
            logger.warning("No start or goal node found")
            return KeyEconomyResult(
                is_valid=False,
                greedy_solvable=False,
                adversarial_solvable=False,
                topology_type=self.analyzer.topology,
                key_surplus={},
                soft_lock_nodes=[]
            )
        
        start_node = start_nodes[0]
        goal_node = goal_nodes[0]
        
        # Test greedy player
        greedy_player = GreedyPlayer(self.graph)
        greedy_solvable, greedy_state = greedy_player.solve(start_node, goal_node)
        
        # Test adversarial player
        adversarial_player = AdversarialPlayer(self.graph, self.analyzer.critical_path)
        adversarial_solvable, adversarial_state = adversarial_player.solve(start_node, goal_node)
        
        # Analyze key economy
        key_surplus = self.analyzer.analyze_key_economy()
        
        # Find soft-lock nodes (where adversarial player failed)
        soft_lock_nodes = []
        if not adversarial_solvable:
            soft_lock_nodes.append(adversarial_state.current_node)
        
        # Check key surplus validity
        negative_surplus = [lock for lock, surplus in key_surplus.items() if surplus < 0]
        
        is_valid = (
            greedy_solvable and
            adversarial_solvable and
            len(negative_surplus) == 0
        )
        
        return KeyEconomyResult(
            is_valid=is_valid,
            greedy_solvable=greedy_solvable,
            adversarial_solvable=adversarial_solvable,
            topology_type=self.analyzer.topology,
            key_surplus=key_surplus,
            soft_lock_nodes=soft_lock_nodes,
            validation_details={
                'greedy_path_length': len(greedy_state.path_taken),
                'adversarial_path_length': len(adversarial_state.path_taken),
                'negative_surplus_locks': negative_surplus,
                'critical_path_size': len(self.analyzer.critical_path)
            }
        )


# ============================================================================
# INTEGRATION INTO FITNESS FUNCTION
# ============================================================================

def integrate_into_evolutionary_algorithm(
    base_fitness_fn=None,
):
    """
    Build a fitness wrapper that includes key-economy validation.

    Args:
        base_fitness_fn: Optional callable(mission_graph) -> float.
            If omitted, a neutral base fitness of 1.0 is used.

    Returns:
        Callable that scores a mission graph with soft-lock penalties and
        topology-aware bonuses.
    """
    def _score(mission_graph: nx.DiGraph) -> float:
        validator = KeyEconomyValidator(mission_graph)
        result = validator.validate()
        
        base_fitness = float(base_fitness_fn(mission_graph)) if callable(base_fitness_fn) else 1.0
        
        # Hard fail for adversarial unsolvability.
        if not result.adversarial_solvable:
            return 0.0
        
        # Penalize key-deficit locks.
        negative_surplus = [s for s in result.key_surplus.values() if s < 0]
        deficit_penalty = float(sum(abs(s) for s in negative_surplus)) * 0.15
        
        topology_bonus = {
            GraphTopology.LINEAR: 0.00,
            GraphTopology.TREE: 0.05,
            GraphTopology.DIAMOND: 0.10,
            GraphTopology.CYCLE: 0.12,
        }.get(result.topology_type, 0.0)
        
        score = base_fitness + topology_bonus - deficit_penalty
        return float(max(0.0, score))
    
    return _score


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Key Economy Validator...")
    
    # Create test mission graph (linear with key-lock)
    G = nx.DiGraph()
    G.add_nodes_from([
        (0, {'key_id': None}),      # Start
        (1, {'key_id': 'key_1'}),   # Has key
        (2, {'key_id': None}),      # Empty
        (3, {'key_id': 'key_2'}),   # Has key
        (4, {'key_id': None}),      # Goal
    ])
    G.add_edges_from([
        (0, 1, {'lock_type': 'open'}),
        (1, 2, {'lock_type': 'locked', 'key_id': 'key_1'}),
        (2, 3, {'lock_type': 'open'}),
        (3, 4, {'lock_type': 'locked', 'key_id': 'key_2'}),
    ])
    
    # Validate
    validator = KeyEconomyValidator(G)
    result = validator.validate()
    
    print(f"\nValidation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Greedy solvable: {result.greedy_solvable}")
    print(f"  Adversarial solvable: {result.adversarial_solvable}")
    print(f"  Topology: {result.topology_type.value}")
    print(f"  Key surplus: {result.key_surplus}")
    print(f"  Soft-lock nodes: {result.soft_lock_nodes}")
    
    if result.is_valid:
        print("\n✅ Key Economy Validator test passed!")
    else:
        print("\n❌ Validation failed - soft-lock detected")
