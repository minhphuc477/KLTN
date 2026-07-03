"""
H-MOLQD Block VII: External Validator
======================================

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
from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import heapq

import numpy as np
import networkx as nx

# Import KLTN core definitions
from src.core.definitions import (
    EDGE_TYPE_MAP,
    parse_edge_type_tokens,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


def _is_start_node(data: Dict[str, Any], label_parts: List[str]) -> bool:
    """Accept both legacy `s` labels and current START-typed mission-graph nodes."""
    node_type = str(data.get("type", "")).strip().lower()
    return (
        "s" in label_parts
        or bool(data.get("is_start", False))
        or node_type in {"start", "start_pointer"}
    )


def _is_goal_node(data: Dict[str, Any], label_parts: List[str]) -> bool:
    """Accept both legacy `t` labels and current GOAL-typed mission-graph nodes."""
    node_type = str(data.get("type", "")).strip().lower()
    return (
        "t" in label_parts
        or bool(data.get("has_triforce", False))
        or bool(data.get("is_triforce", False))
        or bool(data.get("is_goal", False))
        or bool(data.get("has_goal", False))
        or node_type in {"goal", "triforce"}
    )

@dataclass
class ValidationState:
    """State for validation pathfinding."""
    position: Any                   # Current graph node
    keys_held: int = 0              # Number of keys in inventory
    keys_collected: Set[Any] = field(default_factory=set)  # Node IDs where keys collected
    key_ids_held: Set[Any] = field(default_factory=set)  # Specific consumable key identities
    boss_key_ids: Set[Any] = field(default_factory=set)  # Permanent boss-key identities
    items_collected: Set[Any] = field(default_factory=set)  # Progression provider nodes consumed
    item_names: Set[str] = field(default_factory=set)  # Named permanent progression items
    switches_activated: Set[Any] = field(default_factory=set)  # Node IDs and switch IDs
    tokens_collected: Set[Any] = field(default_factory=set)
    token_ids: Set[str] = field(default_factory=set)
    token_counts: Dict[str, int] = field(default_factory=dict)
    token_count: int = 0
    resource_names: Set[str] = field(default_factory=set)
    has_boss_key: bool = False
    has_item: bool = False
    doors_opened: Set[Tuple[Any, Any]] = field(default_factory=set)  # Opened door edges
    path: List[Any] = field(default_factory=list)  # Path taken
    
    def copy(self) -> 'ValidationState':
        return ValidationState(
            position=self.position,
            keys_held=self.keys_held,
            keys_collected=self.keys_collected.copy(),
            key_ids_held=self.key_ids_held.copy(),
            boss_key_ids=self.boss_key_ids.copy(),
            items_collected=self.items_collected.copy(),
            item_names=self.item_names.copy(),
            switches_activated=self.switches_activated.copy(),
            tokens_collected=self.tokens_collected.copy(),
            token_ids=self.token_ids.copy(),
            token_counts=self.token_counts.copy(),
            token_count=self.token_count,
            resource_names=self.resource_names.copy(),
            has_boss_key=self.has_boss_key,
            has_item=self.has_item,
            doors_opened=self.doors_opened.copy(),
            path=self.path.copy(),
        )
    
    def __hash__(self):
        return hash((
            self.position,
            self.keys_held,
            frozenset(self.keys_collected),
            frozenset(self.key_ids_held),
            frozenset(self.boss_key_ids),
            frozenset(self.items_collected),
            frozenset(self.item_names),
            frozenset(self.switches_activated),
            frozenset(self.tokens_collected),
            frozenset(self.token_ids),
            frozenset(self.token_counts.items()),
            self.token_count,
            frozenset(self.resource_names),
            self.has_boss_key,
            self.has_item,
            frozenset(self.doors_opened),
        ))
    
    def __eq__(self, other):
        if not isinstance(other, ValidationState):
            return False
        return (self.position == other.position and
                self.keys_held == other.keys_held and
                self.keys_collected == other.keys_collected and
                self.key_ids_held == other.key_ids_held and
                self.boss_key_ids == other.boss_key_ids and
                self.items_collected == other.items_collected and
                self.item_names == other.item_names and
                self.switches_activated == other.switches_activated and
                self.tokens_collected == other.tokens_collected and
                self.token_ids == other.token_ids and
                self.token_counts == other.token_counts and
                self.token_count == other.token_count and
                self.resource_names == other.resource_names and
                self.has_boss_key == other.has_boss_key and
                self.has_item == other.has_item and
                self.doors_opened == other.doors_opened)


@dataclass
class ValidationResult:
    """Result of dungeon validation."""
    is_solvable: bool
    solution_path: Optional[List[int]] = None
    key_collection_order: Optional[List[int]] = None
    doors_opened: Optional[List[Tuple[int, int]]] = None
    path_length: int = 0
    states_explored: int = 0
    failure_reason: Optional[str] = None
    termination_status: str = "unknown"
    proven_unsolvable: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def path(self) -> Optional[List[int]]:
        """Backward-compatible alias for solution_path."""
        return self.solution_path

    def __iter__(self):
        """
        Backward-compatible tuple unpacking:
            is_solvable, path = result
        """
        yield self.is_solvable
        yield self.solution_path


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
        graph: Optional[nx.DiGraph] = None,
        room_data: Optional[Dict[int, np.ndarray]] = None,
        strict_mode: bool = False,
    ):
        self.room_data = room_data
        self.strict_mode = strict_mode
        self.graph: nx.DiGraph = nx.DiGraph()

        # Derived per-graph state
        self.key_nodes: Dict[Any, Tuple[int, Optional[Any]]] = {}
        self.boss_key_nodes: Dict[Any, Optional[Any]] = {}
        self.item_nodes: Dict[Any, Set[str]] = {}
        self.switch_nodes: Dict[Any, Set[Any]] = {}
        self.token_nodes: Dict[Any, Tuple[int, Optional[str]]] = {}
        self.resource_nodes: Dict[Any, Set[str]] = {}
        self.start_node: Optional[Any] = None
        self.goal_node: Optional[Any] = None

        self._bind_graph(graph if graph is not None else nx.DiGraph())

    def _bind_graph(self, graph: nx.DiGraph) -> None:
        """Attach graph and refresh extracted semantic node sets."""
        self.graph = graph
        self.key_nodes.clear()
        self.boss_key_nodes.clear()
        self.item_nodes.clear()
        self.switch_nodes.clear()
        self.token_nodes.clear()
        self.resource_nodes.clear()
        self.start_node = None
        self.goal_node = None
        self._heuristic_goal: Optional[int] = None
        self._heuristic_distances: Dict[int, int] = {}
        try:
            self._undirected_graph = graph.to_undirected(as_view=True)
        except TypeError:
            self._undirected_graph = graph.to_undirected()

        for node_id, data in graph.nodes(data=True):
            label = data.get('label', '')
            label_parts = [p.strip() for p in label.split(',')] if label else []
            lowered_parts = {part.lower() for part in label_parts}
            node_type = str(data.get('type', '')).strip().lower()
            is_boss_key_node = (
                'K' in label_parts
                or bool(data.get('has_boss_key', False))
                or node_type in {'boss_key', 'big_key'}
            )
            is_item_node = (
                'I' in label_parts
                or bool(data.get('has_item', False))
                or node_type in {'item', 'key_item', 'macro_item', 'protection_item'}
            )

            if (
                ('k' in label_parts)
                or bool(data.get('has_key', False))
                or node_type == 'key'
            ) and not is_boss_key_node:
                count = max(
                    1,
                    int(data.get('key_count', data.get('key_count_hint', 1)) or 1),
                )
                self.key_nodes[node_id] = (count, data.get('key_id'))
            if is_boss_key_node:
                self.boss_key_nodes[node_id] = data.get('key_id')
            if is_item_node:
                names: Set[str] = set()
                for raw_name in (
                    data.get('item_type'),
                    data.get('protection_item_id'),
                ):
                    if raw_name is not None and str(raw_name).strip():
                        names.add(str(raw_name).strip().upper())
                # Legacy "I" nodes carry no identity. Preserve compatibility
                # explicitly as a wildcard rather than conflating all typed
                # items. A node that only declares `required_item` is a
                # consumer-side schema and must not become a wildcard provider.
                has_provider_hint = any(
                    data.get(field) is not None and str(data.get(field)).strip()
                    for field in ("item_type", "protection_item_id")
                )
                if not names and not data.get("required_item") and not has_provider_hint:
                    names.add('*')
                if names:
                    self.item_nodes[node_id] = names
            if (
                'switch' in lowered_parts
                or 'puzzle' in lowered_parts
                or node_type in {
                    'switch',
                    'puzzle',
                    'tutorial_puzzle',
                    'combat_puzzle',
                    'complex_puzzle',
                }
                or bool(data.get('has_puzzle', False))
            ):
                switch_ids = {node_id}
                if data.get('switch_id') is not None:
                    switch_ids.add(data.get('switch_id'))
                self.switch_nodes[node_id] = switch_ids
            if node_type == 'token' or 'token' in lowered_parts:
                count = max(
                    1,
                    int(data.get('token_count', data.get('key_count', data.get('key_count_hint', 1))) or 1),
                )
                token_id = data.get('token_id')
                self.token_nodes[node_id] = (
                    count,
                    str(token_id).strip() if token_id is not None and str(token_id).strip() else None,
                )
            resource_names: Set[str] = set()
            for raw_name in (data.get('drops_resource'), data.get('resource_type')):
                if raw_name is not None and str(raw_name).strip():
                    resource_names.add(str(raw_name).strip().upper())
            if resource_names:
                self.resource_nodes[node_id] = resource_names
            if _is_start_node(data, label_parts):
                self.start_node = node_id
            if _is_goal_node(data, label_parts):
                self.goal_node = node_id

    @staticmethod
    def _state_key(state: ValidationState) -> Tuple[Any, ...]:
        """Immutable search key; set/dict equality now handles hash collisions."""
        return (
            state.position,
            state.keys_held,
            frozenset(state.keys_collected),
            frozenset(state.key_ids_held),
            frozenset(state.boss_key_ids),
            frozenset(state.items_collected),
            frozenset(state.item_names),
            frozenset(state.switches_activated),
            frozenset(state.tokens_collected),
            frozenset(state.token_ids),
            frozenset(state.token_counts.items()),
            state.token_count,
            frozenset(state.resource_names),
            state.has_boss_key,
            state.has_item,
            frozenset(state.doors_opened),
        )

    @staticmethod
    def _dominates(
        state_a: ValidationState,
        state_b: ValidationState,
    ) -> bool:
        """Return whether A can replace B without losing any future action."""
        if state_a.position != state_b.position:
            return False
        # Small keys are consumable. Equal collection history prevents pruning
        # a state that intentionally left a key available for later.
        if state_a.keys_collected != state_b.keys_collected:
            return False
        if state_a.keys_held < state_b.keys_held:
            return False
        if not state_a.key_ids_held.issuperset(state_b.key_ids_held):
            return False
        if not state_a.boss_key_ids.issuperset(state_b.boss_key_ids):
            return False
        if state_b.has_boss_key and not state_a.has_boss_key:
            return False
        if state_b.has_item and not state_a.has_item:
            return False
        if not state_a.item_names.issuperset(state_b.item_names):
            return False
        if not state_a.items_collected.issuperset(state_b.items_collected):
            return False
        if not state_a.switches_activated.issuperset(state_b.switches_activated):
            return False
        if not state_a.tokens_collected.issuperset(state_b.tokens_collected):
            return False
        if state_a.token_count < state_b.token_count:
            return False
        if any(
            int(state_a.token_counts.get(token_id, 0)) < int(count)
            for token_id, count in state_b.token_counts.items()
        ):
            return False
        if not state_a.resource_names.issuperset(state_b.resource_names):
            return False
        if not state_a.doors_opened.issuperset(state_b.doors_opened):
            return False
        return True

    @classmethod
    def _update_pareto_frontier(
        cls,
        frontier: List[Tuple[ValidationState, int]],
        candidate: ValidationState,
        candidate_cost: int,
    ) -> Tuple[List[Tuple[ValidationState, int]], bool]:
        """Insert a state unless an equal-or-cheaper state safely dominates it."""
        for existing, existing_cost in frontier:
            if existing_cost <= candidate_cost and cls._dominates(existing, candidate):
                return frontier, False
        retained = [
            (existing, existing_cost)
            for existing, existing_cost in frontier
            if not (
                candidate_cost <= existing_cost
                and cls._dominates(candidate, existing)
            )
        ]
        retained.append((candidate.copy(), int(candidate_cost)))
        return retained, True

    @classmethod
    def _is_frontier_dominated(
        cls,
        frontier: List[Tuple[ValidationState, int]],
        candidate: ValidationState,
        candidate_cost: int,
    ) -> bool:
        return any(
            existing_cost <= candidate_cost and cls._dominates(existing, candidate)
            for existing, existing_cost in frontier
        )
    
    def can_traverse(
        self,
        from_node: Any,
        to_node: Any,
        state: ValidationState,
        *,
        _edge_data_override: Optional[Dict[str, Any]] = None,
        _allow_composite: bool = True,
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
        edge_data = (
            _edge_data_override
            if _edge_data_override is not None
            else self.graph.get_edge_data(from_node, to_node)
        )
        if edge_data is None:
            return False, state, 'none'
        edge_data = self._edge_data_with_endpoint_requirements(to_node, dict(edge_data))
        
        edge_label = edge_data.get('label', '')
        raw_edge_type = edge_data.get(
            'edge_type',
            edge_data.get('type', EDGE_TYPE_MAP.get(edge_label, 'open')),
        )
        constraints = parse_edge_type_tokens(
            label=str(edge_label or ''),
            edge_type=str(getattr(raw_edge_type, 'name', raw_edge_type) or ''),
        )
        if _allow_composite and len(constraints) > 1:
            working_state = state.copy()
            edge_id = (from_node, to_node)
            reverse_edge_id = (to_node, from_node)
            resolved_types: List[str] = []
            for constraint in constraints:
                constraint_data = dict(edge_data)
                constraint_data['label'] = ''
                constraint_data['edge_type'] = constraint
                allowed, candidate_state, resolved_type = self.can_traverse(
                    from_node,
                    to_node,
                    working_state,
                    _edge_data_override=constraint_data,
                    _allow_composite=False,
                )
                if not allowed:
                    return False, state, resolved_type
                working_state = candidate_state
                # Each conjunct must be checked against a closed gate. Opening
                # one requirement must not bypass the remaining requirements.
                working_state.doors_opened.discard(edge_id)
                working_state.doors_opened.discard(reverse_edge_id)
                resolved_types.append(resolved_type)

            persistent_gates = {
                'key_locked',
                'locked',
                'k',
                'bombable',
                'bomb',
                'b',
                'boss_locked',
                'boss',
                'item_locked',
                'item_gate',
                'multi_lock',
                'switch',
                'switch_locked',
                'state_block',
                'on_off_gate',
                'shutter',
            }
            if any(str(constraint).lower() in persistent_gates for constraint in constraints):
                working_state.doors_opened.add(edge_id)
                working_state.doors_opened.add(reverse_edge_id)
            return True, working_state, '+'.join(resolved_types)

        edge_type = str(constraints[0] if constraints else raw_edge_type).strip().lower()
        if edge_type.startswith('edgetype.'):
            edge_type = edge_type.split('.', 1)[1]
        edge_id = (from_node, to_node)
        
        new_state = state.copy()
        
        if self.strict_mode:
            # Only allow open passages
            if edge_type != 'open' and edge_label != '':
                return False, state, edge_type
            return True, new_state, 'open'
        
        # Handle different edge types
        if edge_type in ('open', '', 'path', 'shortcut', 'hidden'):
            return True, new_state, 'open'
        
        if edge_type in ('key_locked', 'locked', 'k'):
            # Check if already opened
            if edge_id in state.doors_opened:
                return True, new_state, 'key_locked'

            key_required = edge_data.get('key_required')
            required_count = max(1, int(edge_data.get('requires_key_count', 0) or 0))
            if key_required is not None and key_required not in state.key_ids_held:
                return False, state, 'key_locked'

            if state.keys_held >= required_count:
                new_state.keys_held -= required_count
                if key_required is not None:
                    new_state.key_ids_held.discard(key_required)
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
                return True, new_state, 'key_locked'
            
            return False, state, 'key_locked'
        
        if edge_type in ('bombable', 'b', 'bomb'):
            if not ({'BOMB', 'BOMBS'} & state.resource_names) and not (
                {'BOMB', 'BOMBS'} & state.item_names
            ) and '*' not in state.item_names:
                return False, state, 'bombable'
            if edge_id not in state.doors_opened:
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
            return True, new_state, 'bombable'

        if edge_type in ('boss_locked', 'boss'):
            if edge_id in state.doors_opened:
                return True, new_state, 'boss_locked'
            key_required = edge_data.get('key_required')
            has_matching_key = (
                state.has_boss_key
                if key_required is None
                else key_required in state.boss_key_ids
            )
            if has_matching_key:
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
                return True, new_state, 'boss_locked'
            return False, state, 'boss_locked'

        if edge_type in ('item_locked', 'item_gate'):
            if edge_id in state.doors_opened:
                return True, new_state, 'item_gate'
            item_required = str(edge_data.get('item_required', '')).strip().upper()
            has_required_item = (
                '*' in state.item_names
                or (item_required in state.item_names if item_required not in {'', 'NONE'} else state.has_item)
                or item_required in state.resource_names
            )
            if has_required_item:
                new_state.doors_opened.add(edge_id)
                new_state.doors_opened.add((to_node, from_node))
                return True, new_state, 'item_gate'
            return False, state, 'item_gate'

        if edge_type in ('multi_lock',):
            if edge_id in state.doors_opened:
                return True, new_state, 'multi_lock'
            required_count = max(1, int(edge_data.get('token_count', 0) or 0))
            required_token_id = edge_data.get('token_id')
            if required_token_id is not None:
                required_token_id = str(required_token_id).strip()
                if (
                    required_token_id
                    and state.token_counts.get(required_token_id, 0) < required_count
                ):
                    return False, state, 'multi_lock'
            elif state.token_count < required_count:
                return False, state, 'multi_lock'
            new_state.doors_opened.add(edge_id)
            new_state.doors_opened.add((to_node, from_node))
            return True, new_state, 'multi_lock'

        if edge_type in ('switch', 'switch_locked', 'state_block', 'on_off_gate', 'shutter'):
            required_switches = {
                node_id
                for node_id in edge_data.get('switches_required', []) or []
                if node_id is not None
            }
            if not required_switches and edge_data.get('switch_id') is not None:
                required_switches.add(edge_data.get('switch_id'))
            if required_switches:
                if required_switches.issubset(state.switches_activated):
                    return True, new_state, edge_type
                return False, state, edge_type
            if state.switches_activated:
                return True, new_state, edge_type
            return False, state, edge_type
        
        if edge_type in ('soft_locked', 'l'):
            # One-way passage, always traversable forward
            return True, new_state, 'soft_locked'

        if edge_type in ('one_way',):
            return True, new_state, 'one_way'
        
        if edge_type in ('stair', 'stairs', 'warp', 's'):
            # Teleport/stair, always traversable
            return True, new_state, 'stair'

        if edge_type in ('hazard',):
            protection_item = edge_data.get('protection_item_id')
            if protection_item is None or not str(protection_item).strip():
                return True, new_state, 'hazard'
            required = str(protection_item).strip().upper()
            if required in state.item_names or '*' in state.item_names:
                return True, new_state, 'hazard'
            return False, state, 'hazard'

        if edge_type in ('visual_link', 'window'):
            return False, state, 'visual_link'
        
        # Unknown edge type - do not silently flatten progression semantics.
        logger.warning(f"Unknown edge type: {edge_type}")
        return False, state, edge_type

    def _edge_data_with_endpoint_requirements(self, to_node: Any, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer missing gate requirements from the target consumer node."""
        if self.graph is None or to_node not in self.graph:
            return edge_data
        target_data = dict(self.graph.nodes[to_node])
        target_type = str(target_data.get("type", target_data.get("label", "")) or "").strip().lower()
        raw_edge_type = edge_data.get(
            "edge_type",
            edge_data.get("type", EDGE_TYPE_MAP.get(edge_data.get("label", ""), "open")),
        )
        constraints = parse_edge_type_tokens(
            label=str(edge_data.get("label", "") or ""),
            edge_type=str(getattr(raw_edge_type, "name", raw_edge_type) or ""),
        )
        lowered = {str(token).strip().lower() for token in constraints}

        is_key_gate = bool({"key_locked", "locked", "k"} & lowered)
        is_boss_gate = bool({"boss_locked", "boss"} & lowered)
        if (is_key_gate or is_boss_gate) and edge_data.get("key_required") is None:
            target_key = target_data.get("key_id")
            if target_key is not None and ("lock" in target_type or "door" in target_type):
                edge_data["key_required"] = target_key

        if bool({"item_locked", "item_gate"} & lowered) and edge_data.get("item_required") is None:
            required_item = target_data.get("required_item")
            if required_item is not None and str(required_item).strip():
                edge_data["item_required"] = str(required_item).strip()

        return edge_data
    
    def collect_items(self, node_id: Any, state: ValidationState) -> ValidationState:
        """Collect any items at the current node."""
        new_state = state.copy()
        
        # Collect key if present and not already collected
        if node_id in self.key_nodes and node_id not in state.keys_collected:
            count, key_id = self.key_nodes[node_id]
            new_state.keys_held += count
            new_state.keys_collected.add(node_id)
            if key_id is not None:
                new_state.key_ids_held.add(key_id)

        if node_id in self.boss_key_nodes and node_id not in state.items_collected:
            new_state.has_boss_key = True
            boss_key_id = self.boss_key_nodes[node_id]
            if boss_key_id is not None:
                new_state.boss_key_ids.add(boss_key_id)
            new_state.items_collected.add(node_id)

        if node_id in self.item_nodes and node_id not in state.items_collected:
            new_state.has_item = True
            new_state.item_names.update(self.item_nodes[node_id])
            new_state.items_collected.add(node_id)

        if node_id in self.switch_nodes:
            new_state.switches_activated.update(self.switch_nodes[node_id])

        if node_id in self.token_nodes and node_id not in state.tokens_collected:
            count, token_id = self.token_nodes[node_id]
            new_state.token_count += count
            new_state.tokens_collected.add(node_id)
            if token_id:
                new_state.token_ids.add(token_id)
                new_state.token_counts[token_id] = (
                    int(new_state.token_counts.get(token_id, 0)) + int(count)
                )

        if node_id in self.resource_nodes:
            new_state.resource_names.update(self.resource_nodes[node_id])
        
        return new_state
    
    def heuristic(self, node: int, goal: int) -> float:
        """A* heuristic (node distance estimate)."""
        if self._heuristic_goal != goal:
            self._heuristic_goal = goal
            try:
                self._heuristic_distances = dict(
                    nx.single_source_shortest_path_length(self._undirected_graph, goal)
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._heuristic_distances = {}
        if node not in self._heuristic_distances:
            return float('inf')
        return float(self._heuristic_distances[node])
    
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
        start = self.start_node if start_node is None else start_node
        goal = self.goal_node if goal_node is None else goal_node
        
        if start is None:
            return ValidationResult(
                is_solvable=False,
                failure_reason="No start node defined",
                termination_status="invalid",
            )
        
        if goal is None:
            return ValidationResult(
                is_solvable=False,
                failure_reason="No goal node defined",
                termination_status="invalid",
            )
        
        # Initialize
        initial_state = ValidationState(position=start)
        initial_state = self.collect_items(start, initial_state)
        initial_key = self._state_key(initial_state)
        
        # Priority queue: (f_cost, g_cost, counter, state)
        counter = 0
        open_set = [(self.heuristic(start, goal), 0, counter, initial_state)]
        visited = set()
        best_g: Dict[Tuple[Any, ...], int] = {initial_key: 0}
        parents: Dict[Tuple[Any, ...], Optional[Tuple[Any, ...]]] = {initial_key: None}
        positions: Dict[Tuple[Any, ...], Any] = {initial_key: start}
        states_explored = 0
        dominated_states_pruned = 0
        pareto_frontiers: Dict[Any, List[Tuple[ValidationState, int]]] = {}
        
        while open_set and states_explored < max_states:
            _, g_cost, _, current = heapq.heappop(open_set)
            
            state_key = self._state_key(current)
            if g_cost != best_g.get(state_key):
                continue
            if state_key in visited:
                continue
            frontier, accepted = self._update_pareto_frontier(
                pareto_frontiers.get(current.position, []),
                current,
                int(g_cost),
            )
            if not accepted:
                dominated_states_pruned += 1
                continue
            pareto_frontiers[current.position] = frontier
            visited.add(state_key)
            states_explored += 1
            
            # Check if goal reached
            if current.position == goal:
                solution_path: List[Any] = []
                cursor: Optional[Tuple[Any, ...]] = state_key
                while cursor is not None:
                    solution_path.append(positions[cursor])
                    cursor = parents[cursor]
                solution_path.reverse()
                return ValidationResult(
                    is_solvable=True,
                    solution_path=solution_path,
                    key_collection_order=list(current.keys_collected),
                    doors_opened=list(current.doors_opened),
                    path_length=max(0, len(solution_path) - 1),
                    states_explored=states_explored,
                    termination_status="solved",
                    metrics={
                        'keys_used': (
                            sum(self.key_nodes[node_id][0] for node_id in current.keys_collected)
                            - current.keys_held
                        ),
                        'doors_opened': len(current.doors_opened) // 2,
                        'states_pruned_dominated': dominated_states_pruned,
                    }
                )
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current.position):
                can_go, new_state, _edge_type = self.can_traverse(
                    current.position, neighbor, current
                )
                
                if not can_go:
                    continue
                
                # Update state
                new_state.position = neighbor
                new_state = self.collect_items(neighbor, new_state)
                
                new_key = self._state_key(new_state)
                if new_key in visited:
                    continue
                
                # Add to open set
                new_g = g_cost + 1
                if new_g >= best_g.get(new_key, float('inf')):
                    continue
                best_g[new_key] = new_g
                parents[new_key] = state_key
                positions[new_key] = neighbor
                new_f = new_g + self.heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (new_f, new_g, counter, new_state))
        
        live_frontier = False
        if states_explored >= max_states:
            for _priority, pending_g, _counter, pending_state in open_set:
                pending_key = self._state_key(pending_state)
                if pending_key in visited or pending_g != best_g.get(pending_key):
                    continue
                if self._is_frontier_dominated(
                    pareto_frontiers.get(pending_state.position, []),
                    pending_state,
                    int(pending_g),
                ):
                    continue
                live_frontier = True
                break
        budget_exhausted = bool(states_explored >= max_states and live_frontier)
        return ValidationResult(
            is_solvable=False,
            states_explored=states_explored,
            failure_reason=(
                f"Search budget exhausted after {states_explored} states"
                if budget_exhausted
                else "No path exists in the exhausted resource-state space"
            ),
            termination_status="budget_exhausted" if budget_exhausted else "exhausted",
            proven_unsolvable=not budget_exhausted,
            metrics={'states_pruned_dominated': dominated_states_pruned},
        )

    def simulate(
        self,
        graph: Optional[nx.DiGraph] = None,
        start_node: Optional[int] = None,
        goal_node: Optional[int] = None,
        max_states: int = 100000,
    ) -> ValidationResult:
        """
        Convenience wrapper for simulation-style usage.

        If ``graph`` is provided, simulator state is rebound to that graph.
        """
        if graph is not None:
            self._bind_graph(graph)
        return self.find_path(start_node=start_node, goal_node=goal_node, max_states=max_states)

    def explore_reachable_nodes(
        self,
        start_node: Optional[Any] = None,
        *,
        max_states: int = 100000,
    ) -> Dict[str, Any]:
        """Exhaust the resource-state space once and report reachable graph nodes."""
        start = self.start_node if start_node is None else start_node
        if start is None:
            return {
                "reachable_nodes": set(),
                "states_explored": 0,
                "termination_status": "invalid",
                "complete": False,
                "failure_reason": "No start node defined",
            }

        initial = self.collect_items(start, ValidationState(position=start))
        queue = deque([initial])
        visited: Set[Tuple[Any, ...]] = set()
        reachable_nodes: Set[Any] = set()
        states_explored = 0
        dominated_states_pruned = 0
        pareto_frontiers: Dict[Any, List[Tuple[ValidationState, int]]] = {}
        state_budget = int(max(1, max_states))

        while queue and states_explored < state_budget:
            current = queue.popleft()
            state_key = self._state_key(current)
            if state_key in visited:
                continue
            frontier, accepted = self._update_pareto_frontier(
                pareto_frontiers.get(current.position, []),
                current,
                0,
            )
            if not accepted:
                dominated_states_pruned += 1
                continue
            pareto_frontiers[current.position] = frontier
            visited.add(state_key)
            states_explored += 1
            reachable_nodes.add(current.position)

            for neighbor in self.graph.neighbors(current.position):
                can_go, new_state, _edge_type = self.can_traverse(
                    current.position,
                    neighbor,
                    current,
                )
                if not can_go:
                    continue
                new_state.position = neighbor
                new_state = self.collect_items(neighbor, new_state)
                if self._state_key(new_state) not in visited:
                    queue.append(new_state)

        live_frontier = False
        if states_explored >= state_budget:
            for pending_state in queue:
                pending_key = self._state_key(pending_state)
                if pending_key in visited:
                    continue
                if self._is_frontier_dominated(
                    pareto_frontiers.get(pending_state.position, []),
                    pending_state,
                    0,
                ):
                    continue
                live_frontier = True
                break
        budget_exhausted = bool(states_explored >= state_budget and live_frontier)
        return {
            "reachable_nodes": reachable_nodes,
            "states_explored": states_explored,
            "termination_status": "budget_exhausted" if budget_exhausted else "exhausted",
            "complete": not budget_exhausted,
            "states_pruned_dominated": dominated_states_pruned,
            "failure_reason": (
                f"Reachability state budget exhausted after {states_explored} states"
                if budget_exhausted
                else ""
            ),
        }


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
        max_states: int = 100000,
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
        
        return simulator.find_path(start_node, goal_node, max_states=max_states)

    def check_tuple(
        self,
        graph: nx.DiGraph,
        room_data: Optional[Dict[int, np.ndarray]] = None,
        start_node: Optional[int] = None,
        goal_node: Optional[int] = None,
    ) -> Tuple[bool, Optional[List[int]]]:
        """Compatibility wrapper returning (is_solvable, solution_path)."""
        result = self.check(graph, room_data, start_node, goal_node)
        return result.is_solvable, result.solution_path
    
    def check_all_rooms_reachable(
        self,
        graph: nx.DiGraph,
        start_node: Optional[Any] = None,
        max_states: int = 100000,
    ) -> Tuple[bool, Set[Any]]:
        """
        Check whether every room has a resource-valid route from start.
        
        Returns:
            (all_reachable, unreachable_nodes)
        """
        details = self.check_all_rooms_reachable_detailed(
            graph,
            start_node=start_node,
            max_states=max_states,
        )
        return bool(details["complete"] and not details["unreachable_nodes"]), set(
            details["unreachable_nodes"]
        )

    def check_all_rooms_reachable_detailed(
        self,
        graph: nx.DiGraph,
        start_node: Optional[Any] = None,
        max_states: int = 100000,
    ) -> Dict[str, Any]:
        """Return a tri-state-safe all-room reachability proof."""
        simulator = AgentSimulator(graph=graph)
        exploration = simulator.explore_reachable_nodes(
            start_node=start_node,
            max_states=max_states,
        )
        reachable = set(exploration["reachable_nodes"])
        return {
            **exploration,
            "unreachable_nodes": set(graph.nodes()) - reachable,
        }


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
            return False, "Empty path", ValidationState(position=-1)
        
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
        max_states: int = 100000,
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
        return self.checker.check(
            graph,
            room_data,
            start_node,
            goal_node,
            max_states=max_states,
        )
    
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
        analysis['termination_status'] = result.termination_status
        analysis['proven_unsolvable'] = result.proven_unsolvable
        
        # Room reachability
        reachability = self.checker.check_all_rooms_reachable_detailed(graph)
        analysis['all_reachable'] = bool(
            reachability["complete"] and not reachability["unreachable_nodes"]
        )
        analysis['unreachable_rooms'] = list(reachability["unreachable_nodes"])
        analysis['all_rooms_reachability_status'] = str(
            reachability["termination_status"]
        )
        analysis['all_rooms_reachability_states_explored'] = int(
            reachability["states_explored"]
        )
        
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
