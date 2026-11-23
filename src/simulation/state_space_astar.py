"""State-Space A* Pathfinding Solver with Inventory and Logic Simulation."""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.core.definitions import (
    ID_TO_NAME,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    parse_edge_type_tokens,
)
from src.simulation.edge_logic import (
    can_traverse_edge_type,
    edge_constraints_from_data,
    edge_type_from_data,
    graph_edge_state_key,
    graph_predecessors,
    graph_successors,
)
from src.simulation.environment import ZeldaLogicEnv
from src.simulation.graph_progression import (
    PhysicalGraphBridge,
    apply_graph_gate,
    required_switch_ids,
    starting_bomb_count,
    switch_provider_ids,
)
from src.simulation.state import (
    ACTION_DELTAS,
    BLOCKING_IDS,
    BRIDGE_FILL_IDS,
    CARDINAL_COST,
    CONDITIONAL_IDS,
    DIAGONAL_COST,
    EDGE_TYPE_MAP,
    PICKUP_IDS,
    PUSHABLE_IDS,
    TRANSITION_IDS,
    WALKABLE_IDS,
    WATER_IDS,
    Action,
    GameState,
    dominates,
    dynamic_geometry_key,
    game_state_key,
    has_pushed_block_at,
    is_push_destination_available,
    was_block_vacated,
)
from src.simulation.validation_types import (
    BatchValidationResult,
    SolverDiagnostics,
    SolverOptions,
    ValidationResult,
)

logger = logging.getLogger(__name__)

__all__ = ["StateSpaceAStar"]

class StateSpaceAStar:
    """
    A* pathfinder that operates on game state space, not just positions.
    
    This allows finding solutions that require:
    - Picking up keys before opening doors
    - Getting bombs before bombing walls
    - Proper sequencing of item collection
    
    Supports multiple search modes:
    - 'astar': f = g + h (default, optimal)
    - 'bfs': f = depth (breadth-first, explores more)
    - 'dijkstra': f = g (uniform cost, no heuristic)
    - 'greedy': f = h (heuristic only, fast but suboptimal)
    """

    def __init__(self, env: ZeldaLogicEnv, timeout: int = 10000000, heuristic_mode: str = "balanced", priority_options: dict = None, search_mode: str = "astar"):
        """
        Initialize the solver.
        
        Args:
            env: ZeldaLogicEnv instance to solve
            timeout: Maximum states to explore (default 10M for complex dungeons)
                    Large Zelda dungeons (96x66) solve in ~7K states with diagonals
            priority_options: dict with keys 'tie_break', 'key_boost',
                'enable_weighted_astar', 'heuristic_weight', and
                'allow_diagonals'. Legacy 'enable_ara'/'ara_weight' aliases are
                accepted for compatibility; this is fixed-weight weighted A*,
                not Anytime Repairing A*.
            search_mode: Search strategy - 'astar', 'bfs', 'dijkstra', or 'greedy'
        """
        self.env = env
        self.timeout = timeout
        self.heuristic_mode = heuristic_mode
        self.search_mode = search_mode.lower() if search_mode else 'astar'
        self.last_termination_status = "not_run"
        self.last_failure_reason = ""
        self.pickup_positions = self._cache_pickups()

        # PERFORMANCE: Cache stair destinations to avoid repeated graph traversals
        self._stair_dest_cache = {}

        # VIRTUAL NODE TRAVERSAL: Cache for graph-based room-to-room transitions
        # This enables traversal through "virtual nodes" (graph nodes without physical rooms)
        self._node_to_room = None  # Lazy-initialized reverse mapping
        self._best_at_pos = {}
        self._best_g_at_pos = {}

        # Priority options
        self.priority_options = priority_options or {}
        if bool(getattr(self.env, 'strict_original_mode', False)):
            env_profile = 'strict_original'
        elif bool(getattr(self.env, 'vglc_strict_mode', False)):
            env_profile = 'vglc_strict'
        else:
            env_profile = 'extended'
        from .search_base import apply_rules_profile
        self.rules_profile = apply_rules_profile(self.env, self.priority_options.get('rules_profile', env_profile) or env_profile)
        self.strict_original_mode = self.env.strict_original_mode
        self.vglc_strict_mode = self.env.vglc_strict_mode

        rep_raw = str(self.priority_options.get('representation', 'hybrid')).strip().lower()
        if rep_raw not in {'tile', 'graph', 'hybrid'}:
            rep_raw = 'hybrid'
        self.representation = rep_raw
        self.tie_break = bool(self.priority_options.get('tie_break', False))
        self.key_boost = bool(self.priority_options.get('key_boost', False))
        secondary_heuristic = self.priority_options.get('secondary_heuristic')
        if secondary_heuristic is not None and not callable(secondary_heuristic):
            raise TypeError("priority_options['secondary_heuristic'] must be callable.")
        self.secondary_heuristic = secondary_heuristic
        self.secondary_heuristic_name = str(
            self.priority_options.get('secondary_heuristic_name', 'secondary')
        )
        self._secondary_heuristic_cache: Dict[Tuple[Any, ...], float] = {}
        self.enable_ara = bool(
            self.priority_options.get(
                'enable_weighted_astar',
                self.priority_options.get('enable_ara', False),
            )
        )
        # Hierarchical front-end can be toggled explicitly, otherwise inferred
        # from representation mode.
        if 'enable_hierarchical' in self.priority_options:
            self.enable_hierarchical = bool(self.priority_options.get('enable_hierarchical', True))
        else:
            self.enable_hierarchical = self.representation in {'graph', 'hybrid'}
        self.graph_only = self.representation == 'graph'
        if self.graph_only and not self.enable_hierarchical:
            # Graph-only representation requires graph/hierarchical front-end.
            self.enable_hierarchical = True
        # Diagonal movement is a rules change, not a generic search optimization.
        self.allow_diagonals = bool(self.priority_options.get('allow_diagonals', False))
        if self.strict_original_mode:
            self.allow_diagonals = False
            # Strict-original mode prioritizes tile-accurate semantics.
            self.enable_hierarchical = False
            self.graph_only = False
        try:
            self.ara_weight = float(
                self.priority_options.get(
                    'heuristic_weight',
                    self.priority_options.get('ara_weight', 1.0),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Heuristic weight must be a finite positive number.") from exc
        if not math.isfinite(self.ara_weight) or self.ara_weight <= 0:
            raise ValueError("Heuristic weight must be a finite positive number.")

        # Precompute minimal locked-door counts from each graph node to goal
        self.min_locked_needed_node = {}
        try:
            G = getattr(self.env, 'graph', None)
            room_to_node = getattr(self.env, 'room_to_node', None)
            room_positions = getattr(self.env, 'room_positions', None)
            goal_pos = getattr(self.env, 'goal_pos', None)
            if G and room_to_node and goal_pos:
                goal_node = None
                # Preferred mapping: locate goal inside a physical room then map room->node.
                if room_positions:
                    for room_pos, (r_off, c_off) in room_positions.items():
                        r_end = r_off + ROOM_HEIGHT
                        c_end = c_off + ROOM_WIDTH
                        if r_off <= goal_pos[0] < r_end and c_off <= goal_pos[1] < c_end:
                            goal_node = room_to_node.get(room_pos)
                            if goal_node is not None:
                                break
                # Legacy fallback for maps where room_to_node is keyed directly by positions.
                if goal_node is None and goal_pos in room_to_node:
                    goal_node = room_to_node[goal_pos]
            else:
                goal_node = None
            if goal_node is not None:
                # Dijkstra-like on locked edge counts
                dist = {goal_node: 0}
                pq = [(0, goal_node)]
                while pq:
                    d, u = heapq.heappop(pq)
                    if d != dist.get(u, 1e9):
                        continue
                    for v in graph_predecessors(G, u):
                        edata = G.get_edge_data(v, u, {}) or {}
                        etype = self._edge_type_from_data(edata)
                        cost = 1 if etype in ('locked', 'key_locked') else 0
                        nd = d + cost
                        if nd < dist.get(v, 1e9):
                            dist[v] = nd
                            heapq.heappush(pq, (nd, v))
                self.min_locked_needed_node = dist
        except (AttributeError, TypeError, ValueError, KeyError):
            self.min_locked_needed_node = {}

        # PERFORMANCE FIX: Cache door and element positions at initialization
        # Avoids O(width x height) scan on every heuristic call
        self._locked_doors_cache = self.env.find_all_positions(SEMANTIC_PALETTE['DOOR_LOCKED'])
        self._boss_doors_cache = self.env.find_all_positions(SEMANTIC_PALETTE['DOOR_BOSS'])
        self._bomb_doors_cache = self.env.find_all_positions(SEMANTIC_PALETTE['DOOR_BOMB'])
        self._element_tiles_cache = self.env.find_all_positions(SEMANTIC_PALETTE['ELEMENT'])

        # -- HIERARCHICAL SOLVER PRECOMPUTATION --
        # Precompute graph BFS distance (hops) from every node to the goal node.
        # Used by the hierarchical solver and as a tighter heuristic.
        self._graph_bfs_dist: Dict[Any, int] = {}
        self._room_node_to_pos: Dict[Any, Tuple[int, int]] = {}  # node -> representative position
        self._node_items: Dict[Any, List[Tuple[str, Tuple[int, int]]]] = {}  # items in each node's room
        self._node_walkable_count: Dict[Any, int] = {}
        try:
            G = getattr(self.env, 'graph', None)
            r2n = getattr(self.env, 'room_to_node', None)
            rpos = getattr(self.env, 'room_positions', None)
            goal = getattr(self.env, 'goal_pos', None)
            if G and r2n and rpos and goal:
                # Find goal node
                goal_node = None
                for rp, (ro, co) in rpos.items():
                    if rp in r2n:
                        nd = r2n[rp]
                        re = min(ro + ROOM_HEIGHT, self.env.height)
                        ce = min(co + ROOM_WIDTH, self.env.width)
                        if ro <= goal[0] < re and co <= goal[1] < ce:
                            goal_node = nd
                            break
                if goal_node is not None:
                    # Reverse BFS from the goal preserves directed reachability.
                    bfs_q = deque([(goal_node, 0)])
                    self._graph_bfs_dist[goal_node] = 0
                    while bfs_q:
                        u, d = bfs_q.popleft()
                        for v in graph_predecessors(G, u):
                            if v not in self._graph_bfs_dist:
                                self._graph_bfs_dist[v] = d + 1
                                bfs_q.append((v, d + 1))

                # Build node->representative-position and node->items maps
                grid = self.env.original_grid
                for rp, (ro, co) in rpos.items():
                    nd = r2n.get(rp)
                    if nd is None:
                        continue
                    re = min(ro + ROOM_HEIGHT, self.env.height)
                    ce = min(co + ROOM_WIDTH, self.env.width)
                    # Find center walkable tile
                    center_r, center_c = ro + ROOM_HEIGHT // 2, co + ROOM_WIDTH // 2
                    best_pos = None
                    best_d = 9999
                    wcount = 0
                    items_in_room: List[Tuple[str, Tuple[int, int]]] = []
                    for r in range(ro, re):
                        for c in range(co, ce):
                            t = grid[r, c]
                            if t in WALKABLE_IDS or t in CONDITIONAL_IDS or t in PUSHABLE_IDS or t in WATER_IDS:
                                wcount += 1
                            if t in WALKABLE_IDS:
                                dd = abs(r - center_r) + abs(c - center_c)
                                if dd < best_d:
                                    best_d = dd
                                    best_pos = (r, c)
                            # Catalog items
                            if t == SEMANTIC_PALETTE['KEY_SMALL']:
                                items_in_room.append(('key', (r, c)))
                            elif t == SEMANTIC_PALETTE['KEY_BOSS']:
                                items_in_room.append(('boss_key', (r, c)))
                            elif t == SEMANTIC_PALETTE['KEY_ITEM']:
                                items_in_room.append(('key_item', (r, c)))
                            elif t == SEMANTIC_PALETTE['ITEM_MINOR']:
                                items_in_room.append(('bomb', (r, c)))
                    # If semantic tiles don't carry pickups, fall back to graph node metadata.
                    # This avoids false negatives on VGLC maps where keys are encoded in graph labels.
                    try:
                        node_attrs = G.nodes.get(nd, {}) if G is not None else {}
                    except (AttributeError, TypeError, KeyError):
                        node_attrs = {}
                    raw_label = str(node_attrs.get('label', ''))
                    raw_tokens = [tok.strip() for tok in raw_label.replace('\n', ',').split(',') if tok.strip()]
                    lower_tokens = {tok.lower() for tok in raw_tokens}
                    existing_kinds = {kind for kind, _ in items_in_room}
                    synthetic_pos = best_pos if best_pos is not None else (center_r, center_c)

                    has_small_key = (
                        bool(node_attrs.get('is_key') or node_attrs.get('has_key'))
                        or ('k' in raw_tokens)
                        or ('key' in lower_tokens)
                    )
                    has_boss_key = (
                        bool(node_attrs.get('is_boss_key') or node_attrs.get('has_boss_key'))
                        or ('K' in raw_tokens)
                        or ('boss_key' in lower_tokens)
                    )
                    has_key_item = (
                        bool(node_attrs.get('is_item') or node_attrs.get('has_item'))
                        or ('I' in raw_tokens)
                        or ('key_item' in lower_tokens)
                        or ('item' in lower_tokens)
                    )

                    if has_small_key and 'key' not in existing_kinds:
                        items_in_room.append(('key', synthetic_pos))
                    if has_boss_key and 'boss_key' not in existing_kinds:
                        items_in_room.append(('boss_key', synthetic_pos))
                    if has_key_item and 'key_item' not in existing_kinds:
                        items_in_room.append(('key_item', synthetic_pos))
                    # Always keep a representative position per node.
                    # Some transition rooms have no floor tile; fall back to room center
                    # so hierarchical paths can still include the room deterministically.
                    rep_pos = best_pos if best_pos is not None else synthetic_pos
                    self._room_node_to_pos[nd] = rep_pos
                    self._node_items[nd] = items_in_room
                    self._node_walkable_count[nd] = wcount
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            logger.debug("Failed to precompute node item hints; continuing without hints: %s", exc)

        # -- PLAN-GUIDED HEURISTIC STATE (Upgrade 3) --
        # Populated lazily in solve() after room-level A* runs.
        # _abstract_plan: ordered list of graph node IDs from start->goal
        # _abstract_plan_rooms: dict node->index-in-plan for O(1) lookup
        # _abstract_plan_avg_cost: average room cost for remaining-rooms estimate
        self._abstract_plan: Optional[List] = None
        self._abstract_plan_rooms: Optional[Dict] = None
        self._abstract_plan_avg_cost: float = 15.0
        self._room_entry_point_cache: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        self._has_nonlocal_transitions: bool = self._detect_nonlocal_transitions()

    def _detect_nonlocal_transitions(self) -> bool:
        """Return True iff iter_successors can make any multi-tile warp/stair jump."""
        if self.env.find_all_positions(SEMANTIC_PALETTE['STAIR']):
            return True
        bridge = getattr(self.env, 'graph_bridge', None)
        if bridge is not None:
            return bool(getattr(bridge, 'portals', None))
        if self.strict_original_mode:
            return False
        graph = getattr(self.env, 'graph', None)
        room_to_node = getattr(self.env, 'room_to_node', None)
        room_positions = getattr(self.env, 'room_positions', None)
        if graph and room_to_node and room_positions:
            node_to_room = getattr(self.env, 'node_to_room', None) or {
                v: k for k, v in room_to_node.items()
            }
            for node, attrs in graph.nodes(data=True):
                if attrs.get('is_virtual', False):
                    return True
                if node not in node_to_room or node_to_room[node] not in room_positions:
                    return True
            for u, v, data in graph.edges(data=True):
                ru = node_to_room.get(u)
                rv = node_to_room.get(v)
                if ru is None or rv is None:
                    return True
                edge_type = str(self._edge_type_from_data(data) or "open").strip().lower()
                if abs(ru[0] - rv[0]) + abs(ru[1] - rv[1]) > 1:
                    return True
                if edge_type not in {"", "open", "path"}:
                    return True
        return False

    @staticmethod
    def _state_key(state: GameState) -> Tuple[Any, ...]:
        """Immutable key for closed/g-score maps; equality handles hash collisions."""
        return game_state_key(state)

    def _secondary_heuristic_score(self, state: GameState) -> float:
        """Evaluate the optional secondary ordering signal once per state."""
        if self.secondary_heuristic is None:
            return 0.0
        state_key = self._state_key(state)
        cached = self._secondary_heuristic_cache.get(state_key)
        if cached is not None:
            return cached
        try:
            score = float(self.secondary_heuristic(state))
        except Exception as exc:
            raise RuntimeError(
                f"Secondary heuristic {self.secondary_heuristic_name!r} failed for state {state_key!r}."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"Secondary heuristic {self.secondary_heuristic_name!r} returned non-finite score {score}."
            )
        score = max(0.0, score)
        self._secondary_heuristic_cache[state_key] = score
        return score

    def _locked_needed_for_state(self, state: GameState) -> int:
        """Return the graph lower-bound hint used only for queue tie-breaking."""
        room_pos = None
        if getattr(self.env, 'room_positions', None):
            for candidate, (row_offset, col_offset) in self.env.room_positions.items():
                if (
                    row_offset <= state.position[0] < row_offset + ROOM_HEIGHT
                    and col_offset <= state.position[1] < col_offset + ROOM_WIDTH
                ):
                    room_pos = candidate
                    break
        if room_pos is None or not self.env.room_to_node:
            return 0
        node = self.env.room_to_node.get(room_pos)
        return int(self.min_locked_needed_node.get(node, 0))

    def _make_open_entry(
        self,
        *,
        f_score: float,
        counter: int,
        g_score: float,
        state: GameState,
        state_key: Tuple[Any, ...],
        parent_state: Optional[GameState] = None,
    ) -> Tuple[Any, ...]:
        """Build one canonical heap entry for every tile-level search path."""
        use_secondary_priority = bool(
            self.tie_break or self.key_boost or self.secondary_heuristic is not None
        )
        if not use_secondary_priority:
            return (float(f_score), int(counter), int(counter), float(g_score), state, state_key)

        priority: List[float] = [float(f_score)]
        if self.secondary_heuristic is not None:
            priority.append(self._secondary_heuristic_score(state))
        if self.tie_break:
            priority.append(float(self._locked_needed_for_state(state)))
        if self.key_boost:
            keys_held = int(getattr(state, 'keys', 0) or 0)
            picked_up_key = bool(
                parent_state is not None
                and keys_held > int(getattr(parent_state, 'keys', 0) or 0)
            )
            priority.extend([-float(keys_held), -0.01 if picked_up_key else 0.0])
        priority.append(float(counter))
        return (tuple(priority), int(counter), float(g_score), state, state_key)

    @staticmethod
    def _state_dominates_with_cost(
        state_a: GameState,
        cost_a: float,
        state_b: GameState,
        cost_b: float,
    ) -> bool:
        """True when state_a is at least as capable as state_b and no costlier."""
        if float(cost_a) > float(cost_b):
            return False
        return dominates(state_a, state_b)

    def _pareto_frontier_dominates(
        self,
        frontier: List[Tuple[GameState, float]],
        candidate: GameState,
        candidate_cost: float,
    ) -> bool:
        return any(
            self._state_dominates_with_cost(existing, existing_cost, candidate, candidate_cost)
            for existing, existing_cost in frontier
        )

    def _add_to_pareto_frontier(
        self,
        frontier: List[Tuple[GameState, float]],
        candidate: GameState,
        candidate_cost: float,
    ) -> List[Tuple[GameState, float]]:
        if self._pareto_frontier_dominates(frontier, candidate, candidate_cost):
            return frontier
        return [
            (existing, existing_cost)
            for existing, existing_cost in frontier
            if not self._state_dominates_with_cost(candidate, candidate_cost, existing, existing_cost)
        ] + [(candidate, float(candidate_cost))]

    def _has_live_open_entry(
        self,
        open_set: List[Any],
        closed_set: Set[Any],
        g_scores: Dict[Any, float],
    ) -> bool:
        """Return whether a queued state remains expandable after lazy pruning."""
        for entry in open_set:
            if len(entry) == 6:
                _, _, _hash_hint, current_g, current_state, _path_key = entry
            elif len(entry) == 5 and isinstance(entry[0], tuple):
                _priority, _hash_hint, current_g, current_state, _path_key = entry
            elif len(entry) == 5:
                _, _, _hash_hint, current_state, _path_key = entry
                current_g = g_scores.get(self._state_key(current_state), float("inf"))
            else:
                continue
            current_key = self._state_key(current_state)
            if current_key in closed_set:
                continue
            if float(current_g) != float(g_scores.get(current_key, float("inf"))):
                continue
            state_bucket = (
                current_state.position,
                dynamic_geometry_key(current_state),
            )
            if self._pareto_frontier_dominates(
                self._best_at_pos.get(state_bucket, []),
                current_state,
                float(current_g),
            ):
                continue
            return True
        return False

    def _edge_constraints_from_data(self, edge_data: Optional[Dict[str, Any]]) -> List[str]:
        """Return canonical edge constraints from edge attributes."""
        return edge_constraints_from_data(edge_data)

    def _edge_type_from_data(self, edge_data: Optional[Dict[str, Any]]) -> str:
        """Return primary canonical edge type from edge attributes."""
        return edge_type_from_data(edge_data)

    # ------------------------------------------------------------------
    # UPGRADE 1: DETERMINISTIC SOFT-LOCK DETECTION (Reverse Reachability)
    # ------------------------------------------------------------------
    # Reference: Holzer & Schwoon (2011) - "Reachability vs. Safety in
    #   Graph-Based Planning", ICAPS Workshop on Heuristics & Search.
    # Uses bidirectional BFS to compute F \ B  (forward-reachable minus
    # backward-reachable).  Any tile/node in that difference is a
    # *proven dead-end trap*: the player can reach it from START but
    # can never reach GOAL from there.
    # ------------------------------------------------------------------

    def find_proven_traps(self) -> Dict[str, Any]:
        """Deterministic soft-lock detection via reverse reachability.

        Algorithm
        ---------
        1. **Graph level** - forward BFS from start-node, backward BFS
           from goal-node (reversing directed / soft-locked edges).
           Traps = forward - backward.
        2. **Grid level** - forward flood-fill from START position on
           the walkable tile grid, backward flood-fill from GOAL
           (reversing one-way DOOR_SOFT tiles).  Traps = forward - backward.

        Returns
        -------
        dict with keys:
            'graph_traps'  : set of trapped graph node IDs
            'grid_traps'   : set of trapped (row, col) positions
            'forward_graph' : set of forward-reachable graph nodes
            'backward_graph': set of backward-reachable graph nodes
            'forward_grid'  : int - count of forward-reachable tiles
            'backward_grid' : int - count of backward-reachable tiles
        """
        trap_report: Dict[str, Any] = {
            'graph_traps': set(),
            'grid_traps': set(),
            'forward_graph': set(),
            'backward_graph': set(),
            'forward_grid': 0,
            'backward_grid': 0,
        }

        # -- GRAPH-LEVEL --
        G = getattr(self.env, 'graph', None)
        r2n = getattr(self.env, 'room_to_node', None)
        rpos = getattr(self.env, 'room_positions', None)
        start = getattr(self.env, 'start_pos', None)
        goal = getattr(self.env, 'goal_pos', None)

        if G and r2n and rpos and start and goal:
            # Determine start / goal nodes
            start_node = goal_node = None
            for rp, (ro, co) in rpos.items():
                nd = r2n.get(rp)
                if nd is None:
                    continue
                re = ro + ROOM_HEIGHT
                ce = co + ROOM_WIDTH
                if ro <= start[0] < re and co <= start[1] < ce:
                    start_node = nd
                if ro <= goal[0] < re and co <= goal[1] < ce:
                    goal_node = nd

            if start_node is not None and goal_node is not None:
                # Forward BFS (respecting edge direction for soft-locked)
                fwd = set()
                q = deque([start_node])
                fwd.add(start_node)
                while q:
                    u = q.popleft()
                    for v in graph_successors(G, u):
                        if v in fwd:
                            continue
                        ed = G.get_edge_data(u, v, {}) or {}
                        et = self._edge_type_from_data(ed)
                        if et == 'soft_locked':
                            # Only traverse if directed edge u->v exists
                            if not G.has_edge(u, v):
                                continue
                            dd = G.get_edge_data(u, v, {}) or {}
                            dt = self._edge_type_from_data(dd)
                            if dt != 'soft_locked':
                                continue
                        fwd.add(v)
                        q.append(v)

                # Backward BFS from goal (reverse all edge directions)
                bwd = set()
                q = deque([goal_node])
                bwd.add(goal_node)
                while q:
                    u = q.popleft()
                    for v in graph_predecessors(G, u):
                        if v in bwd:
                            continue
                        # Reversed: to reach u from v, original edge must go v->u
                        ed = G.get_edge_data(v, u, {}) or {}
                        et = self._edge_type_from_data(ed)
                        if et == 'soft_locked':
                            # In reverse, we need the original directed edge v->u
                            if not G.has_edge(v, u):
                                continue
                            dd = G.get_edge_data(v, u, {}) or {}
                            dt = self._edge_type_from_data(dd)
                            if dt != 'soft_locked':
                                continue
                        bwd.add(v)
                        q.append(v)

                trap_report['forward_graph'] = fwd
                trap_report['backward_graph'] = bwd
                trap_report['graph_traps'] = fwd - bwd
                logger.debug('Graph reachability: fwd=%d bwd=%d traps=%d',
                             len(fwd), len(bwd), len(fwd - bwd))

        # -- GRID-LEVEL --
        grid = self.env.original_grid
        h, w = grid.shape

        passable = WALKABLE_IDS | CONDITIONAL_IDS | PUSHABLE_IDS | WATER_IDS

        if start and goal:
            # Forward flood-fill from START
            fwd_grid: Set[Tuple[int, int]] = set()
            q2: deque = deque()
            if grid[start[0], start[1]] in passable:
                fwd_grid.add(start)
                q2.append(start)
            while q2:
                r, c = q2.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in fwd_grid:
                        t = grid[nr, nc]
                        if t in passable:
                            fwd_grid.add((nr, nc))
                            q2.append((nr, nc))

            # Backward flood-fill from GOAL (reverse one-way tiles)
            bwd_grid: Set[Tuple[int, int]] = set()
            q3: deque = deque()
            if grid[goal[0], goal[1]] in passable:
                bwd_grid.add(goal)
                q3.append(goal)
            while q3:
                r, c = q3.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in bwd_grid:
                        t = grid[nr, nc]
                        if t in passable:
                            # For reverse traversal, DOOR_SOFT at (nr,nc)
                            # means (nr,nc)->(r,c) was allowed forward, so
                            # reverse (r,c)->(nr,nc) is allowed.
                            # DOOR_SOFT at (r,c) means forward was into (r,c),
                            # reverse should allow leaving (r,c).
                            # Simplified: allow both directions for grid-level
                            # (graph handles directional constraints)
                            bwd_grid.add((nr, nc))
                            q3.append((nr, nc))

            trap_report['forward_grid'] = len(fwd_grid)
            trap_report['backward_grid'] = len(bwd_grid)
            trap_report['grid_traps'] = fwd_grid - bwd_grid
            logger.debug('Grid reachability: fwd=%d bwd=%d traps=%d',
                         len(fwd_grid), len(bwd_grid), len(fwd_grid - bwd_grid))

        return trap_report

    # ------------------------------------------------------------------
    # UPGRADE 2: MACRO-ACTION A* (Jump Optimization)
    # ------------------------------------------------------------------
    # Reference: Botea et al. (2004) - "Near Optimal Hierarchical
    #   Pathfinding", JAIR 30.  Pre-computes intra-room BFS between
    #   Points of Interest (doors, items, stairs) and uses POI-to-POI
    #   transitions as macro-actions, collapsing ~20 tile steps into
    #   a single edge with pre-computed cost.
    # ------------------------------------------------------------------

    def _extract_pois(self) -> Dict[Any, List[Tuple[str, Tuple[int, int]]]]:
        """Extract Points of Interest per room node.

        POIs include: doors (all types), items, stairs, start, goal.

        Returns
        -------
        dict  node -> list of (poi_type, (row, col))
        """
        rpos = getattr(self.env, 'room_positions', None)
        r2n = getattr(self.env, 'room_to_node', None)
        if not rpos or not r2n:
            return {}

        grid = self.env.original_grid
        h, w = grid.shape
        poi_ids = {
            SEMANTIC_PALETTE['DOOR_OPEN']: 'door',
            SEMANTIC_PALETTE['DOOR_SOFT']: 'door',
            SEMANTIC_PALETTE['DOOR_LOCKED']: 'door_locked',
            SEMANTIC_PALETTE['DOOR_BOMB']: 'door_bomb',
            SEMANTIC_PALETTE['DOOR_BOSS']: 'door_boss',
            SEMANTIC_PALETTE['KEY_SMALL']: 'key',
            SEMANTIC_PALETTE['KEY_BOSS']: 'boss_key',
            SEMANTIC_PALETTE['KEY_ITEM']: 'key_item',
            SEMANTIC_PALETTE['ITEM_MINOR']: 'bomb',
            SEMANTIC_PALETTE['STAIR']: 'stair',
            SEMANTIC_PALETTE['START']: 'start',
            SEMANTIC_PALETTE['TRIFORCE']: 'goal',
        }

        poi_by_node: Dict[Any, List[Tuple[str, Tuple[int, int]]]] = {}
        for rp, (ro, co) in rpos.items():
            nd = r2n.get(rp)
            if nd is None:
                continue
            pois: List[Tuple[str, Tuple[int, int]]] = []
            re = min(ro + ROOM_HEIGHT, h)
            ce = min(co + ROOM_WIDTH, w)
            for r in range(ro, re):
                for c in range(co, ce):
                    t = grid[r, c]
                    if t in poi_ids:
                        pois.append((poi_ids[t], (r, c)))
            poi_by_node[nd] = pois
        return poi_by_node

    def _intra_room_bfs(self, room_node: Any,
                        pois: List[Tuple[str, Tuple[int, int]]]
                        ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
        """BFS between all POI pairs within a single room.

        Parameters
        ----------
        room_node : graph node id
        pois : list of (type, position) in this room

        Returns
        -------
        dict  (pos_a, pos_b) -> shortest tile distance
        """
        rpos = self.env.room_positions
        r2n = self.env.room_to_node
        if not rpos:
            return {}

        # Find room bounds
        room_key = None
        for rp, nd in r2n.items():
            if nd == room_node:
                room_key = rp
                break
        if room_key is None or room_key not in rpos:
            return {}

        ro, co = rpos[room_key]
        re = min(ro + ROOM_HEIGHT, self.env.height)
        ce = min(co + ROOM_WIDTH, self.env.width)
        grid = self.env.original_grid

        passable = WALKABLE_IDS | CONDITIONAL_IDS | PUSHABLE_IDS | WATER_IDS
        poi_positions = [p for _, p in pois]

        distances: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}

        for src_pos in poi_positions:
            # BFS from src_pos within room bounds
            dist_map: Dict[Tuple[int, int], int] = {src_pos: 0}
            q = deque([src_pos])
            while q:
                r, c = q.popleft()
                d = dist_map[(r, c)]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if ro <= nr < re and co <= nc < ce and (nr, nc) not in dist_map:
                        t = grid[nr, nc]
                        if t in passable:
                            dist_map[(nr, nc)] = d + 1
                            q.append((nr, nc))
            for dst_pos in poi_positions:
                if dst_pos != src_pos and dst_pos in dist_map:
                    distances[(src_pos, dst_pos)] = dist_map[dst_pos]

        return distances

    def _solve_with_macro_actions(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """Macro-Action A*: POI-to-POI search with pre-computed intra-room BFS.

        Strategy 1.5 in the solver cascade, sitting between the purely
        graph-level room solver (Strategy 1) and the full tile-level
        solver (Strategy 2).  Provides tile-level cost accuracy at the
        POI granularity while avoiding per-tile state expansion.

        State: (position, keys, bombs, has_boss_key, has_item,
                frozenset(collected), frozenset(opened))

        Returns
        -------
        (success, path_of_positions, states_explored)
        """
        G = getattr(self.env, 'graph', None)
        r2n = getattr(self.env, 'room_to_node', None)
        rpos = getattr(self.env, 'room_positions', None)
        goal = getattr(self.env, 'goal_pos', None)
        start_pos = getattr(self.env, 'start_pos', None)
        if not (G and r2n and rpos and goal and start_pos):
            return False, [], 0

        # Build POIs and intra-room distances
        all_pois = self._extract_pois()
        if not all_pois:
            return False, [], 0

        intra_dist: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        for nd, pois in all_pois.items():
            intra_dist.update(self._intra_room_bfs(nd, pois))

        # Build position -> room-node lookup
        pos_to_node: Dict[Tuple[int, int], Any] = {}
        for rp, (_ro, _co) in rpos.items():
            nd = r2n.get(rp)
            if nd is None:
                continue
            for _, ppos in all_pois.get(nd, []):
                pos_to_node[ppos] = nd

        # Collect all POI positions as a set for quick membership
        all_poi_positions: Set[Tuple[int, int]] = set()
        for pois in all_pois.values():
            for _, p in pois:
                all_poi_positions.add(p)

        # Make sure start and goal are in POI set
        if start_pos not in all_poi_positions or goal not in all_poi_positions:
            return False, [], 0

        # Build cross-room edges between door POIs of adjacent rooms
        # Two POIs in different rooms are connected if their rooms share a graph edge
        cross_edges: List[Tuple[Tuple[int, int], Tuple[int, int], int, str]] = []
        for nd in G.nodes():
            for nb in graph_successors(G, nd):
                ed = G.get_edge_data(nd, nb, {}) or {}
                et = self._edge_type_from_data(ed)
                # Find door POIs in both rooms that are closest
                nd_pois = [p for t, p in all_pois.get(nd, []) if t.startswith('door') or t == 'stair']
                nb_pois = [p for t, p in all_pois.get(nb, []) if t.startswith('door') or t == 'stair']
                for np1 in nd_pois:
                    for np2 in nb_pois:
                        cross_edges.append((np1, np2, 2, et))  # cost=2 for room transition

        # Initial state
        opts = self.env.solver_options or SolverOptions()
        init_keys = opts.start_keys
        init_bombs = opts.start_bombs
        init_bk = opts.start_boss_key
        init_item = opts.start_item
        init_opened: FrozenSet[Tuple[int, int]] = frozenset()

        # Auto-collect items at start position
        grid = self.env.original_grid
        t = grid[start_pos[0], start_pos[1]]
        coll_set: Set[Tuple[int, int]] = set()
        if t == SEMANTIC_PALETTE['KEY_SMALL']:
            init_keys += 1
            coll_set.add(start_pos)
        elif t == SEMANTIC_PALETTE['KEY_BOSS']:
            init_bk = True
            coll_set.add(start_pos)
        elif t == SEMANTIC_PALETTE['KEY_ITEM']:
            init_item = True
            coll_set.add(start_pos)
        elif t == SEMANTIC_PALETTE['ITEM_MINOR']:
            init_bombs += 4
            coll_set.add(start_pos)
        init_collected = frozenset(coll_set)

        init_state = (start_pos, init_keys, init_bombs, init_bk, init_item,
                      init_collected, init_opened)

        # Average room tiles for heuristic
        avg_rt = max(1, int(np.mean(list(self._node_walkable_count.values())))) if self._node_walkable_count else 15

        def h_macro(st):
            pos = st[0]
            md = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            nd = pos_to_node.get(pos)
            if nd is not None:
                bd = self._graph_bfs_dist.get(nd, 999)
                gh = bd * avg_rt * 0.4
                if gh > md:
                    md = gh
            return md

        open_set: list = []
        counter = 0
        h0 = h_macro(init_state)
        heapq.heappush(open_set, (h0, counter, 0.0, init_state, init_state))
        counter += 1

        visited: Dict[Tuple, float] = {}  # state -> best g
        parents: Dict[Any, Optional[Any]] = {init_state: None}
        positions: Dict[Any, Tuple[int, int]] = {init_state: start_pos}
        parent_costs: Dict[Any, float] = {init_state: 0.0}
        states_explored = 0
        macro_timeout = min(self.timeout, 500000)

        while open_set and states_explored < macro_timeout:
            _, _, g, state, state_parent_key = heapq.heappop(open_set)
            pos, keys, bombs, bk, has_item, collected, opened = state

            # Simple visited check (exact state)
            state_key = (pos, keys, bombs, bk, has_item, collected, opened)
            if state_key in visited and visited[state_key] <= g:
                continue
            visited[state_key] = g
            states_explored += 1

            # Goal check
            if pos == goal:
                logger.debug('Macro-action solver succeeded: %d states', states_explored)
                path = self._reconstruct_parent_path(parents, positions, state_key)
                return True, path, states_explored

            # Expand: intra-room POI transitions
            for (src, dst), dist in intra_dist.items():
                if src != pos:
                    continue
                # Handle item collection at destination
                nk, nb, nbk, ni = keys, bombs, bk, has_item
                nc = set(collected)
                no = set(opened)
                dt = grid[dst[0], dst[1]]

                # Check if door needs resources
                if dst not in opened:
                    if dt == SEMANTIC_PALETTE['DOOR_LOCKED']:
                        if nk <= 0:
                            continue
                        nk -= 1
                        no.add(dst)
                    elif dt == SEMANTIC_PALETTE['DOOR_BOMB']:
                        if nb <= 0:
                            continue
                        nb -= 1
                        no.add(dst)
                    elif dt == SEMANTIC_PALETTE['DOOR_BOSS']:
                        if not nbk:
                            continue
                        nbk = False
                        no.add(dst)

                # Collect items at destination
                if dst not in collected:
                    if dt == SEMANTIC_PALETTE['KEY_SMALL']:
                        nk += 1
                        nc.add(dst)
                    elif dt == SEMANTIC_PALETTE['KEY_BOSS']:
                        nbk = True
                        nc.add(dst)
                    elif dt == SEMANTIC_PALETTE['KEY_ITEM']:
                        ni = True
                        nc.add(dst)
                    elif dt == SEMANTIC_PALETTE['ITEM_MINOR']:
                        nb += 4
                        nc.add(dst)

                new_st = (dst, nk, nb, nbk, ni, frozenset(nc), frozenset(no))
                ng = g + dist
                nh = h_macro(new_st)
                if ng < parent_costs.get(new_st, float('inf')):
                    parent_costs[new_st] = ng
                    parents[new_st] = state_key
                    positions[new_st] = dst
                    heapq.heappush(open_set, (ng + nh, counter, ng, new_st, new_st))
                    counter += 1

            # Expand: cross-room transitions
            for src, dst, cost, et in cross_edges:
                if src != pos:
                    continue
                nk, nb, nbk, ni = keys, bombs, bk, has_item
                nc = set(collected)
                no = set(opened)

                # Check edge traversability
                if et in ('key_locked', 'locked'):
                    if nk <= 0:
                        continue
                    nk -= 1
                elif et == 'bombable':
                    if nb <= 0:
                        continue
                    nb -= 1
                elif et == 'boss_locked':
                    if not nbk:
                        continue
                elif et == 'item_locked':
                    if not ni:
                        continue
                elif et == 'switch':
                    # This abstraction does not carry activated switch IDs.
                    continue
                elif et == 'soft_locked':
                    # Must check directed edge
                    src_nd = pos_to_node.get(src)
                    dst_nd = pos_to_node.get(dst)
                    if src_nd is not None and dst_nd is not None and not G.has_edge(src_nd, dst_nd):
                        continue
                elif et not in ('open', 'stair'):
                    continue

                # Collect items at destination
                dt = grid[dst[0], dst[1]]
                if dst not in collected:
                    if dt == SEMANTIC_PALETTE['KEY_SMALL']:
                        nk += 1
                        nc.add(dst)
                    elif dt == SEMANTIC_PALETTE['KEY_BOSS']:
                        nbk = True
                        nc.add(dst)
                    elif dt == SEMANTIC_PALETTE['KEY_ITEM']:
                        ni = True
                        nc.add(dst)
                    elif dt == SEMANTIC_PALETTE['ITEM_MINOR']:
                        nb += 4
                        nc.add(dst)

                new_st = (dst, nk, nb, nbk, ni, frozenset(nc), frozenset(no))
                ng = g + cost
                nh = h_macro(new_st)
                if ng < parent_costs.get(new_st, float('inf')):
                    parent_costs[new_st] = ng
                    parents[new_st] = state_key
                    positions[new_st] = dst
                    heapq.heappush(open_set, (ng + nh, counter, ng, new_st, new_st))
                    counter += 1

        logger.debug('Macro-action solver exhausted: %d states', states_explored)
        return False, [], states_explored

    # ------------------------------------------------------------------
    # HIERARCHICAL ROOM-LEVEL A* SOLVER
    # ------------------------------------------------------------------
    # Operates on the dungeon graph at room granularity.
    # State = (node, keys, bombs, has_boss_key, has_item,
    #          frozenset(collected_item_positions), frozenset(opened_door_positions))
    # Successor = graph neighbors, filtered by edge-type item requirements.
    # Heuristic = graph BFS distance x average room diameter.
    # This reduces the search space from thousands of tiles to tens of nodes.
    # ------------------------------------------------------------------

    def _solve_room_level(self, search_mode: Optional[str] = None) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Room-level A* on the dungeon graph.

        Returns (success, path_of_representative_positions, states_explored).
        """
        G = getattr(self.env, 'graph', None)
        r2n = getattr(self.env, 'room_to_node', None)
        rpos = getattr(self.env, 'room_positions', None)
        goal = getattr(self.env, 'goal_pos', None)
        start = getattr(self.env, 'start_pos', None)
        if not (G and r2n and rpos and goal and start):
            return False, [], 0

        if self.strict_original_mode:
            soft_door_id = int(SEMANTIC_PALETTE['DOOR_SOFT'])
            if bool(np.any(self.env.grid == soft_door_id)):
                return False, [], 0

        # Determine start/goal nodes
        start_node = None
        goal_node = None
        for rp, (ro, co) in rpos.items():
            nd = r2n.get(rp)
            if nd is None:
                continue
            re = ro + ROOM_HEIGHT
            ce = co + ROOM_WIDTH
            if ro <= start[0] < re and co <= start[1] < ce:
                start_node = nd
            if ro <= goal[0] < re and co <= goal[1] < ce:
                goal_node = nd
        if start_node is None or goal_node is None:
            return False, [], 0

        # Collect all item positions across all rooms for quick lookup
        all_item_positions: Dict[Tuple[int, int], str] = {}
        for nd, items in self._node_items.items():
            for kind, pos in items:
                all_item_positions[pos] = kind

        # Node->room mapping
        n2r: Dict[Any, Tuple[int, int]] = {}
        if hasattr(self.env, 'node_to_room') and self.env.node_to_room:
            n2r = self.env.node_to_room
        else:
            n2r = {v: k for k, v in r2n.items()}

        # State tuple: (node, keys, bombs, has_boss_key, has_item, collected_fs, opened_fs)
        opts = self.env.solver_options or SolverOptions()
        init_opened: FrozenSet[Tuple[int, int]] = frozenset()

        # Collect items from the starting room automatically
        start_keys = opts.start_keys
        start_bombs = opts.start_bombs
        start_boss_key = opts.start_boss_key
        start_has_item = opts.start_item
        start_collected = set()
        for kind, ipos in self._node_items.get(start_node, []):
            start_collected.add(ipos)
            if kind == 'key':
                start_keys += 1
            elif kind == 'boss_key':
                start_boss_key = True
            elif kind == 'key_item':
                start_has_item = True
            elif kind == 'bomb':
                start_bombs += 4

        init_state = (start_node, start_keys, start_bombs, start_boss_key, start_has_item,
                      frozenset(start_collected), init_opened)

        def h_func(st):
            nd = st[0]
            bd = self._graph_bfs_dist.get(nd, 999)
            # Each remaining graph transition costs at least one. Scaling by
            # average room area can overestimate and break A* optimality.
            return float(bd)

        mode = (search_mode or self.search_mode or 'astar').lower()
        open_set = []
        counter = 0
        g0 = 0
        h0 = h_func(init_state)
        if mode == 'bfs':
            f0 = 0
        elif mode == 'dijkstra':
            f0 = g0
        elif mode == 'greedy':
            f0 = h0
        else:
            f0 = g0 + h0
        start_rep = self._room_node_to_pos.get(start_node, start)
        heapq.heappush(open_set, (f0, counter, g0, init_state, init_state))
        counter += 1
        parents: Dict[Any, Optional[Any]] = {init_state: None}
        positions: Dict[Any, Tuple[int, int]] = {init_state: start_rep}
        parent_costs: Dict[Any, float] = {init_state: 0.0}
        depths: Dict[Any, int] = {init_state: 0}

        # Pareto-frontier domination per node
        # For each node, keep a LIST of non-dominated inventory tuples
        pareto: Dict[Any, List[Tuple[int, int, bool, bool, FrozenSet, FrozenSet, float]]] = defaultdict(list)

        def _inv(st, g):
            return (st[1], st[2], st[3], st[4], st[5], st[6], g)

        def _is_dominated(node, inv_tuple):
            """Check if inv_tuple is dominated by anything in pareto[node]."""
            keys, bombs, bk, item, coll, opn, g = inv_tuple
            for pk, pb, pbk, pi, pc, po, pg in pareto[node]:
                if (pk >= keys and pb >= bombs and
                    (pbk or not bk) and (pi or not item) and
                    pc.issuperset(coll) and po.issuperset(opn) and pg <= g):
                    # Check strict domination (at least one dimension strictly better)
                    if (pk > keys or pb > bombs or
                        (pbk and not bk) or (pi and not item) or
                        len(pc) > len(coll) or len(po) > len(opn) or pg < g):
                        return True
                    # Equal in all dims -> also dominated (duplicate)
                    if (pk == keys and pb == bombs and pbk == bk and pi == item and
                        pc == coll and po == opn and pg == g):
                        return True
            return False

        def _add_pareto(node, inv_tuple):
            """Add to pareto frontier, removing dominated entries."""
            keys, bombs, bk, item, coll, opn, g = inv_tuple
            new_front = []
            for existing in pareto[node]:
                pk, pb, pbk, pi, pc, po, pg = existing
                # Is existing dominated by newx
                if (keys >= pk and bombs >= pb and
                    (bk or not pbk) and (item or not pi) and
                    coll.issuperset(pc) and opn.issuperset(po) and g <= pg):
                    if (keys > pk or bombs > pb or
                        (bk and not pbk) or (item and not pi) or
                        len(coll) > len(pc) or len(opn) > len(po) or g < pg):
                        continue  # Drop dominated existing
                    if (keys == pk and bombs == pb and bk == pbk and item == pi and
                        coll == pc and opn == po and g == pg):
                        continue  # Drop exact duplicate
                new_front.append(existing)
            new_front.append(inv_tuple)
            pareto[node] = new_front

        states_explored = 0
        room_timeout = min(self.timeout, 2000000)

        while open_set and states_explored < room_timeout:
            _, _, g, state, state_key = heapq.heappop(open_set)
            node, keys, bombs, bk, has_item_flag, collected, opened = state

            inv = _inv(state, g)
            if _is_dominated(node, inv):
                continue
            _add_pareto(node, inv)

            states_explored += 1

            # Goal check
            if node == goal_node:
                path = self._reconstruct_parent_path(parents, positions, state)
                return True, path, states_explored

            # Expand: iterate over all graph neighbors
            for neighbor in graph_successors(G, node):
                edata = G.get_edge_data(node, neighbor, {}) or {}
                etype = self._edge_type_from_data(edata)

                # Check traversability and compute new inventory
                new_keys = keys
                new_bombs = bombs
                new_bk = bk
                new_item = has_item_flag
                if etype in ('open', 'stair'):
                    pass  # Free
                elif etype == 'switch':
                    # Room-level state does not model activated switch IDs.
                    continue
                elif etype == 'soft_locked':
                    # One-way: only allowed from node->neighbor (directed)
                    # Check if the directed edge exists
                    if not G.has_edge(node, neighbor):
                        continue
                    ed = G.get_edge_data(node, neighbor, {}) or {}
                    et = self._edge_type_from_data(ed)
                    if et == 'soft_locked':
                        pass  # Can traverse in this direction
                    else:
                        continue
                elif etype in ('key_locked', 'locked'):
                    if new_keys <= 0:
                        continue
                    new_keys -= 1
                elif etype == 'bombable':
                    if new_bombs <= 0:
                        continue
                    new_bombs -= 1
                elif etype == 'boss_locked':
                    if not new_bk:
                        continue
                elif etype == 'item_locked':
                    if not new_item:
                        continue
                else:
                    continue

                # Get neighbor room and check if it's physical
                neighbor_room = n2r.get(neighbor)
                n_data = G.nodes.get(neighbor, {})
                is_virtual = n_data.get('is_virtual', False)

                # If neighbor is virtual, BFS through virtual nodes
                if is_virtual or (neighbor_room and neighbor_room not in rpos):
                    # Traverse through virtual nodes to find physical destinations
                    v_visited = {node, neighbor}
                    v_queue = deque([(neighbor, etype, new_keys, new_bombs, new_bk, new_item)])
                    while v_queue:
                        vn, _, vk, vb, vbk, vi = v_queue.popleft()
                        for vn2 in graph_successors(G, vn):
                            if vn2 in v_visited:
                                continue
                            ved = G.get_edge_data(vn, vn2, {}) or {}
                            vet = self._edge_type_from_data(ved)
                            # Check traversability
                            tk, tb, tbk, ti = vk, vb, vbk, vi
                            can = True
                            if vet in ('key_locked', 'locked'):
                                if tk <= 0:
                                    can = False
                                else:
                                    tk -= 1
                            elif vet == 'bombable':
                                if tb <= 0:
                                    can = False
                                else:
                                    tb -= 1
                            elif vet == 'boss_locked':
                                if not tbk:
                                    can = False
                            elif vet == 'item_locked':
                                if not ti:
                                    can = False
                            elif vet == 'switch':
                                can = False
                            elif vet == 'soft_locked':
                                if not G.has_edge(vn, vn2):
                                    can = False
                                else:
                                    edd = G.get_edge_data(vn, vn2, {}) or {}
                                    ett = self._edge_type_from_data(edd)
                                    if ett != 'soft_locked':
                                        can = False
                            elif vet not in ('open', 'stair'):
                                can = False
                            if not can:
                                continue
                            v_visited.add(vn2)
                            vn2_data = G.nodes.get(vn2, {})
                            vn2_room = n2r.get(vn2)
                            if vn2_data.get('is_virtual', False) or (vn2_room and vn2_room not in rpos):
                                v_queue.append((vn2, vet, tk, tb, tbk, ti))
                            else:
                                # Physical destination found
                                self._enqueue_room_neighbor(
                                    vn2, tk, tb, tbk, ti, collected, opened,
                                    g, h_func, open_set, counter, state,
                                    parents, positions, parent_costs, depths,
                                    goal, start, mode)
                                counter += 1
                    continue

                # Physical neighbor room
                self._enqueue_room_neighbor(
                    neighbor, new_keys, new_bombs, new_bk, new_item,
                    collected, opened, g, h_func, open_set, counter, state,
                    parents, positions, parent_costs, depths,
                    goal, start, mode)
                counter += 1

        return False, [], states_explored

    def _enqueue_room_neighbor(self, neighbor, keys, bombs, bk, has_item_flag,
                                collected, opened, g, h_func, open_set, counter, parent_key,
                                parents, positions, parent_costs, depths,
                                goal, start, mode: str = 'astar'):
        """Helper: collect items in the room and push successor onto open_set."""
        new_keys = keys
        new_bombs = bombs
        new_bk = bk
        new_item = has_item_flag
        new_collected = set(collected)

        # Auto-collect items in this room (room-level abstraction:
        # assume player explores the whole room and picks up everything)
        for kind, ipos in self._node_items.get(neighbor, []):
            if ipos not in new_collected:
                new_collected.add(ipos)
                if kind == 'key':
                    new_keys += 1
                elif kind == 'boss_key':
                    new_bk = True
                elif kind == 'key_item':
                    new_item = True
                elif kind == 'bomb':
                    new_bombs += 4

        new_collected_fs = frozenset(new_collected)

        new_state = (neighbor, new_keys, new_bombs, new_bk, new_item,
                     new_collected_fs, opened)

        # Cost: graph BFS hop = approximate room traversal cost
        avg_cost = max(1, self._node_walkable_count.get(neighbor, 15))
        new_g = g + avg_cost
        new_h = h_func(new_state)
        mode_l = str(mode or 'astar').lower()
        current_depth = int(depths.get(parent_key, 0))
        if mode_l == 'bfs':
            new_f = float(current_depth + 1)
        elif mode_l == 'dijkstra':
            new_f = new_g
        elif mode_l == 'greedy':
            new_f = new_h
        else:
            new_f = new_g + new_h

        rep_pos = self._room_node_to_pos.get(neighbor, goal if goal else start)
        if new_g >= parent_costs.get(new_state, float('inf')):
            return
        parent_costs[new_state] = new_g
        parents[new_state] = parent_key
        if rep_pos:
            positions[new_state] = rep_pos
        depths[new_state] = current_depth + 1

        heapq.heappush(open_set, (new_f, counter, new_g, new_state, new_state))

    def _populate_abstract_plan(self, h_path: List[Tuple[int, int]]) -> None:
        """Convert room-level path into abstract plan for heuristic guidance.

        Populates ``_abstract_plan`` (ordered node list), ``_abstract_plan_rooms``
        (node -> index mapping), and ``_abstract_plan_avg_cost`` (average
        intra-room tile cost for remaining-rooms estimation).

        Called by ``solve()`` after ``_solve_room_level()``.
        """
        if not h_path:
            return
        rpos = getattr(self.env, 'room_positions', None)
        r2n = getattr(self.env, 'room_to_node', None)
        if not rpos or not r2n:
            return

        plan_nodes: List = []
        seen: Set = set()
        for pos in h_path:
            for rp, (ro, co) in rpos.items():
                if (ro <= pos[0] < ro + ROOM_HEIGHT and
                    co <= pos[1] < co + ROOM_WIDTH):
                    nd = r2n.get(rp)
                    if nd is not None and nd not in seen:
                        plan_nodes.append(nd)
                        seen.add(nd)
                    break

        if plan_nodes:
            self._abstract_plan = plan_nodes
            self._abstract_plan_rooms = {nd: i for i, nd in enumerate(plan_nodes)}
            costs = [self._node_walkable_count.get(nd, 15) for nd in plan_nodes]
            self._abstract_plan_avg_cost = float(np.mean(costs)) if costs else 15.0
            logger.debug('Abstract plan populated: %d nodes, avg_cost=%.1f',
                         len(plan_nodes), self._abstract_plan_avg_cost)

    def solve(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Find a solution path using A* on state space.
        
        OPTIMIZED VERSION with HIERARCHICAL FALLBACK:
        1. First tries room-level A* on the dungeon graph (fast, handles D2/D9)
        1.5. Tries Macro-Action A* on POI graph (medium speed, tile-accurate)
        2. Falls back to tile-level A* if room-level fails or graph unavailable
        
        Returns:
            success: Whether a solution was found
            path: List of positions visited
            states_explored: Number of states explored
        """
        self.env.reset()
        self.last_solution_cost: Optional[float] = None
        self.last_solution_inventory: Optional[Dict[str, Any]] = None
        self.last_termination_status = "not_run"
        self.last_failure_reason = ""

        if self.env.goal_pos is None:
            self.last_termination_status = "invalid"
            self.last_failure_reason = "No goal (TRIFORCE) found in map"
            return False, [], 0

        if self.env.start_pos is None:
            self.last_termination_status = "invalid"
            self.last_failure_reason = "No start position found in map"
            return False, [], 0

        # -- STRATEGY 1: Room-level hierarchical solver --
        # Much faster for large dungeons (D2, D9) with many rooms.
        # Operates on the graph, not the tile grid.
        graph_available = bool(
            self.env.graph
            and self.env.room_to_node
            and self.env.room_positions
            and self._graph_bfs_dist
        )
        graph_states_explored = 0
        use_graph_frontend = bool(
            self.enable_hierarchical
            and self.representation in {'graph', 'hybrid'}
            and graph_available
        )
        if use_graph_frontend:
            try:
                h_success, h_path, h_states = self._solve_room_level(search_mode=self.search_mode)
                graph_states_explored += h_states
                if h_success:
                    logger.debug('Hierarchical solver succeeded: %d states', h_states)
                    # -- UPGRADE 3: Store abstract plan for heuristic guidance --
                    self._populate_abstract_plan(h_path)
                    if self.representation == 'graph':
                        self.last_termination_status = "solved"
                        return True, h_path, h_states
                    logger.debug(
                        'Hierarchical solver produced abstract path in %s mode; '
                        'continuing to tile-level refinement',
                        self.representation,
                    )
                else:
                    logger.debug('Hierarchical solver failed with %d states, '
                                'falling back to macro-action', h_states)
                    # Even on failure, try to extract partial plan for heuristic
                    self._populate_abstract_plan(h_path)
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                logger.debug('Hierarchical solver error: %s, falling back', e)

        # -- STRATEGY 1.5: Macro-Action A* (POI-to-POI) --
        # Intermediate resolution: jumps between Points of Interest
        # (doors, items, stairs) with pre-computed intra-room BFS costs.
        # Much faster than tile-level but more precise than room-level.
        if (
            use_graph_frontend
            and self.search_mode == 'astar'
        ):
            try:
                m_success, m_path, m_states = self._solve_with_macro_actions()
                graph_states_explored += m_states
                if m_success:
                    logger.debug('Macro-action solver succeeded: %d states', m_states)
                    if self.representation == 'graph':
                        self.last_termination_status = "solved"
                        return True, m_path, m_states
                    logger.debug(
                        'Macro-action solver produced abstract path in %s mode; '
                        'continuing to tile-level refinement',
                        self.representation,
                    )
                else:
                    logger.debug('Macro-action solver failed with %d states, '
                                'falling back to tile-level', m_states)
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                logger.debug('Macro-action solver error: %s, falling back', e)

        # -- STRATEGY 2: Tile-level A* (original, with improvements) --

        if self.graph_only:
            if not graph_available:
                logger.debug('Graph-only mode requested but topology graph is unavailable')
            self.last_termination_status = "incomplete"
            self.last_failure_reason = "Graph-only front end did not produce a certified route"
            return False, [], graph_states_explored

        # Use read-only grid reference (no copies!)
        grid = self.env.original_grid
        height, width = grid.shape

        # PERFORMANCE: Pre-allocate dominance tracking dictionary
        self._best_at_pos = {}
        self._best_g_at_pos = {}  # Track best g-score at each position

        # Priority queue: (f_score, counter, hash_hint, g_score, state, state_key).
        # hash_hint is only heap metadata; closed/g-score maps use full keys.
        start_state = self.env.state.copy()
        start_key = self._state_key(start_state)
        start_h = self._heuristic(start_state)
        start_g = 0

        # Compute initial f-score based on search mode
        if self.search_mode == 'bfs':
            start_f = 0  # BFS: f = depth (starts at 0)
        elif self.search_mode == 'dijkstra':
            start_f = start_g  # Dijkstra: f = g only
        elif self.search_mode == 'greedy':
            start_f = start_h  # Greedy: f = h only
        else:
            start_f = start_g + start_h  # A*: f = g + h

        open_set = [
            self._make_open_entry(
                f_score=start_f,
                counter=0,
                g_score=start_g,
                state=start_state,
                state_key=start_key,
            )
        ]
        heapq.heapify(open_set)

        closed_set = set()
        g_scores = {start_key: 0}
        parents: Dict[Any, Optional[Any]] = {start_key: None}
        positions: Dict[Any, Tuple[int, int]] = {start_key: start_state.position}
        depths: Dict[Any, int] = {start_key: 0}

        states_explored = 0
        counter = 1  # Tie-breaker for heap
        dominated_states_pruned = 0  # Track pruning statistics

        while open_set and states_explored < self.timeout:
            entry: Any = heapq.heappop(open_set)
            # Support both simple and priority tuple formats
            # Simple: (f, counter, hash_hint, g, state, state_key) - 6 elements
            # Priority: (priority_tuple, hash_hint, g, state, state_key) - 5 elements, first is tuple
            if len(entry) == 6:
                _, _, _hash_hint, current_g, current_state, path_key = entry
            elif len(entry) == 5 and isinstance(entry[0], tuple):
                _priority, _hash_hint, current_g, current_state, path_key = entry
            elif len(entry) == 5:
                _, _, _hash_hint, current_state, path_key = entry
                current_g = g_scores.get(self._state_key(current_state), 0)
            else:
                # Unknown format - skip
                continue
            current_key = self._state_key(current_state)
            if current_key in closed_set:
                continue

            # STATE DOMINATION PRUNING: keep a Pareto frontier per
            # position/dynamic-block bucket. A single "best" state is invalid
            # when incomparable inventories meet at the same tile.
            state_bucket = (current_state.position, dynamic_geometry_key(current_state))
            frontier = self._best_at_pos.get(state_bucket, [])
            if self._pareto_frontier_dominates(frontier, current_state, float(current_g)):
                dominated_states_pruned += 1
                continue
            self._best_at_pos[state_bucket] = self._add_to_pareto_frontier(
                frontier,
                current_state,
                float(current_g),
            )
            self._best_g_at_pos[state_bucket] = min(cost for _state, cost in self._best_at_pos[state_bucket])

            closed_set.add(current_key)
            states_explored += 1

            # Check win condition
            if current_state.position == self.env.goal_pos:
                path = self._reconstruct_parent_path(parents, positions, current_key)
                self.last_solution_cost = float(current_g)
                self.last_solution_inventory = {
                    'keys': current_state.keys,
                    'bomb_count': current_state.bomb_count,
                    'has_bomb': current_state.has_bomb,
                    'has_boss_key': current_state.has_boss_key,
                    'has_item': current_state.has_item,
                    'item_names': sorted(str(name) for name in current_state.item_names),
                    'current_floor': current_state.current_floor,
                    'doors_opened': len(current_state.opened_doors),
                    'items_collected': len(current_state.collected_items),
                }
                self.last_termination_status = "solved"
                return True, path, states_explored

            for new_state, step_cost in self.iter_successors(current_state):
                new_key = self._state_key(new_state)

                if new_key in closed_set:
                    continue

                # COMBAT-AWARE COST CALCULATION
                # Use variable cost based on tile type and current_g from the heap.
                current_depth = int(depths.get(current_key, 0))
                if self.search_mode == 'bfs':
                    # True BFS over full game state: each transition has unit depth cost.
                    # (Inventory/doors/items are still modeled in state transitions.)
                    g_score = float(current_depth + 1)
                else:
                    g_score = current_g + step_cost

                if new_key in g_scores and g_score >= g_scores[new_key]:
                    continue

                g_scores[new_key] = g_score
                h_score = self._heuristic(new_state)
                # Compute f based on search mode
                if self.search_mode == 'bfs':
                    f_score = float(current_depth + 1)  # BFS: f = depth
                elif self.search_mode == 'dijkstra':
                    f_score = g_score  # Dijkstra: f = g only (no heuristic)
                elif self.search_mode == 'greedy':
                    f_score = h_score  # Greedy: f = h only (no cost)
                elif self.enable_ara:
                    f_score = g_score + self.ara_weight * h_score
                else:
                    f_score = g_score + h_score  # A*: f = g + h

                parents[new_key] = current_key
                positions[new_key] = new_state.position
                depths[new_key] = current_depth + 1

                heapq.heappush(
                    open_set,
                    self._make_open_entry(
                        f_score=f_score,
                        counter=counter,
                        g_score=g_score,
                        state=new_state,
                        state_key=new_key,
                        parent_state=current_state,
                    ),
                )
                counter += 1

        # PERFORMANCE LOGGING: Report pruning statistics
        if dominated_states_pruned > 0:
            logger.debug('Solver: %d states explored, %d dominated states pruned (%.1f%% reduction)',
                        states_explored, dominated_states_pruned,
                        100.0 * dominated_states_pruned / (states_explored + dominated_states_pruned))

        live_frontier = bool(
            states_explored >= self.timeout
            and self._has_live_open_entry(open_set, closed_set, g_scores)
        )
        self.last_termination_status = "budget_exhausted" if live_frontier else "exhausted"
        self.last_failure_reason = (
            f"State budget reached with a live frontier after {states_explored:,} expansions"
            if live_frontier
            else "Complete frontier exhaustion: no reachable goal state exists"
        )
        return False, [], states_explored

    @staticmethod
    def _reconstruct_parent_path(
        parents: Mapping[Any, Optional[Any]],
        positions: Mapping[Any, Tuple[int, int]],
        end_key: Any,
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        key: Optional[Any] = end_key
        while key is not None:
            if key not in positions:
                break
            path.append(positions[key])
            key = parents.get(key)
        path.reverse()
        return path

    def verify_position_path(
        self,
        path: Sequence[Tuple[int, int]],
        *,
        start_state: Optional[GameState] = None,
    ) -> Tuple[bool, str, Optional[GameState], Optional[float]]:
        """Replay a reconstructed route from a fresh state.

        Position-only paths can be ambiguous when graph transitions connect the
        same pair of tiles under different inventory states. The replay keeps a
        small deduplicated frontier of every legal state compatible with each
        reported position instead of choosing an arbitrary transition.
        """
        positions = [tuple(map(int, position)) for position in path]
        if not positions:
            return False, "empty route", None, None
        expected_start = start_state.position if start_state is not None else self.env.start_pos
        if expected_start is None or positions[0] != tuple(expected_start):
            return (
                False,
                f"route starts at {positions[0]}, expected {expected_start}",
                None,
                None,
            )

        initial_state = self.env.reset() if start_state is None else start_state.copy()
        candidates: Dict[Tuple[Any, ...], Tuple[GameState, float]] = {
            self._state_key(initial_state): (initial_state, 0.0)
        }
        grid = self.env.original_grid
        height, width = grid.shape

        for step_index, target_pos in enumerate(positions[1:], start=1):
            row, col = target_pos
            if not (0 <= row < height and 0 <= col < width):
                return (
                    False,
                    f"step {step_index}: target {target_pos} is out of bounds",
                    None,
                    None,
                )

            next_candidates: Dict[Tuple[Any, ...], Tuple[GameState, float]] = {}
            target_tile = int(grid[row, col])
            for state, accumulated_cost in candidates.values():
                current_pos = tuple(state.position)
                current_row, current_col = current_pos
                delta_row = abs(row - current_row)
                delta_col = abs(col - current_col)
                transition_specs: list[Tuple[Optional[GraphTransitionSpec], float]] = []
                if (self.env.graph_bridge is not None
                        and target_pos in self.env.graph_bridge.destinations(current_pos)):
                    transition_specs.append((None, 1.0))

                if delta_row + delta_col == 1:
                    transition_specs.append((None, float(CARDINAL_COST)))
                elif (
                    self.allow_diagonals
                    and delta_row == 1
                    and delta_col == 1
                ):
                    vertical_tile = int(grid[row, current_col])
                    horizontal_tile = int(grid[current_row, col])
                    corner_tiles = (vertical_tile, horizontal_tile)
                    if not any(
                        tile in BLOCKING_IDS
                        or tile in CONDITIONAL_IDS
                        or tile in PUSHABLE_IDS
                        or tile in WATER_IDS
                        for tile in corner_tiles
                    ):
                        transition_specs.append((None, float(DIAGONAL_COST)))

                current_tile = int(grid[current_row, current_col])
                is_stair = current_tile == int(SEMANTIC_PALETTE['STAIR'])
                is_door = current_tile in {
                    int(SEMANTIC_PALETTE['DOOR_OPEN']),
                    int(SEMANTIC_PALETTE['DOOR_SOFT']),
                    int(SEMANTIC_PALETTE['DOOR_LOCKED']),
                    int(SEMANTIC_PALETTE['DOOR_BOMB']),
                    int(SEMANTIC_PALETTE['DOOR_BOSS']),
                }
                is_at_boundary = False
                if self.env.room_positions:
                    for row_offset, col_offset in self.env.room_positions.values():
                        if (
                            row_offset <= current_row < row_offset + ROOM_HEIGHT
                            and col_offset <= current_col < col_offset + ROOM_WIDTH
                        ):
                            local_row = current_row - row_offset
                            local_col = current_col - col_offset
                            is_at_boundary = bool(
                                local_row <= 1
                                or local_row >= ROOM_HEIGHT - 2
                                or local_col <= 1
                                or local_col >= ROOM_WIDTH - 2
                            )
                            break

                if self.vglc_strict_mode:
                    can_teleport = is_stair or is_door
                else:
                    can_teleport = is_stair or is_door or is_at_boundary
                if self.strict_original_mode:
                    can_teleport = is_stair

                if is_stair and target_pos in self._get_stair_destinations(current_pos):
                    transition_specs.append(("stair", 1.0))
                if can_teleport and not self.strict_original_mode:
                    graph_destinations = (
                        self._get_controlled_virtual_destinations(current_pos, state)
                        + self._get_graph_warp_destinations(current_pos, state)
                    )
                    transition_specs.extend(
                        (edge_type, float(cost))
                        for destination, cost, edge_type in graph_destinations
                        if tuple(destination) == target_pos
                    )

                for edge_type, base_cost in dict.fromkeys(transition_specs):
                    transition_state = state
                    if edge_type is not None:
                        allowed, transition_state = self.apply_graph_edge_transition(
                            state,
                            current_pos,
                            target_pos,
                            edge_type,
                        )
                        if not allowed:
                            continue
                    allowed, next_state = self._try_move_pure(
                        transition_state,
                        target_pos,
                        target_tile,
                    )
                    if not allowed:
                        continue
                    next_state.current_floor = self.env.floor_for_position(
                        target_pos,
                        default=state.current_floor,
                    )
                    if self.search_mode == "bfs":
                        next_cost = accumulated_cost + 1.0
                    else:
                        move_cost = self._get_movement_cost(
                            target_tile,
                            target_pos,
                            state,
                        )
                        next_cost = accumulated_cost + float(move_cost) * float(base_cost)
                    next_key = self._state_key(next_state)
                    previous = next_candidates.get(next_key)
                    if previous is None or next_cost < previous[1]:
                        next_candidates[next_key] = (next_state, next_cost)

            if not next_candidates:
                return (
                    False,
                    f"step {step_index}: no legal transition reaches {target_pos}",
                    None,
                    None,
                )
            candidates = next_candidates

        if self.env.goal_pos is None or positions[-1] != tuple(self.env.goal_pos):
            return (
                False,
                f"route ends at {positions[-1]}, expected goal {self.env.goal_pos}",
                None,
                None,
            )
        final_key = min(
            candidates,
            key=lambda key: (candidates[key][1], repr(key)),
        )
        final_state, final_cost = candidates[final_key]
        return True, "", final_state, float(final_cost)

    def solve_with_diagnostics(
        self, *, start_state: Optional[GameState] = None,
    ) -> Tuple[bool, List[Tuple[int, int]], SolverDiagnostics]:
        """
        Find a solution path with detailed diagnostics.
        
        Enhanced version of solve() that returns comprehensive statistics
        for debugging, performance analysis, and failure diagnosis.
        
        Returns:
            success: Whether a solution was found
            path: List of positions visited
            diagnostics: SolverDiagnostics with detailed statistics
        """
        import time
        start_time = time.perf_counter()

        if start_state is None:
            self.env.reset()
        else:
            self.env.state = start_state.copy()

        # Early exit conditions
        if self.env.goal_pos is None:
            return False, [], SolverDiagnostics(
                success=False, states_explored=0,
                failure_reason="No goal (TRIFORCE) found in map",
                termination_status="invalid",
            )

        if self.env.start_pos is None:
            return False, [], SolverDiagnostics(
                success=False, states_explored=0,
                failure_reason="No start position found in map",
                termination_status="invalid",
            )

        # Use read-only grid reference
        grid = self.env.original_grid
        height, width = grid.shape

        # Tracking for diagnostics
        self._best_at_pos = {}
        self._best_g_at_pos = {}

        # Priority queue
        start_state = self.env.state.copy()
        start_key = self._state_key(start_state)
        start_h = self._heuristic(start_state)
        start_g = 0.0
        if self.search_mode == 'bfs':
            start_f = 0.0
        elif self.search_mode == 'dijkstra':
            start_f = start_g
        elif self.search_mode == 'greedy':
            start_f = start_h
        elif self.enable_ara:
            start_f = start_g + self.ara_weight * start_h
        else:
            start_f = start_g + start_h

        open_set = [
            self._make_open_entry(
                f_score=start_f,
                counter=0,
                g_score=start_g,
                state=start_state,
                state_key=start_key,
            )
        ]
        heapq.heapify(open_set)

        closed_set = set()
        g_scores = {start_key: 0.0}
        parents: Dict[Any, Optional[Any]] = {start_key: None}
        positions: Dict[Any, Tuple[int, int]] = {start_key: start_state.position}
        depths: Dict[Any, int] = {start_key: 0}

        states_explored = 0
        counter = 1
        dominated_states_pruned = 0
        max_queue_size = 1
        final_state = None

        while open_set and states_explored < self.timeout:
            max_queue_size = max(max_queue_size, len(open_set))

            entry: Any = heapq.heappop(open_set)

            # Parse entry format
            if len(entry) == 6:
                _, _, _hash_hint, current_g, current_state, path_key = entry
            elif len(entry) == 5 and isinstance(entry[0], tuple):
                _priority, _hash_hint, current_g, current_state, path_key = entry
            elif len(entry) == 5:
                _, _, _hash_hint, current_state, path_key = entry
                current_g = g_scores.get(self._state_key(current_state), 0.0)
            else:
                continue

            current_key = self._state_key(current_state)
            if current_key in closed_set:
                continue

            state_bucket = (current_state.position, dynamic_geometry_key(current_state))
            frontier = self._best_at_pos.get(state_bucket, [])
            if self._pareto_frontier_dominates(frontier, current_state, float(current_g)):
                dominated_states_pruned += 1
                continue
            self._best_at_pos[state_bucket] = self._add_to_pareto_frontier(
                frontier,
                current_state,
                float(current_g),
            )
            self._best_g_at_pos[state_bucket] = min(cost for _state, cost in self._best_at_pos[state_bucket])

            closed_set.add(current_key)
            states_explored += 1
            final_state = current_state

            # Check win condition
            if current_state.position == self.env.goal_pos:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                path = self._reconstruct_parent_path(parents, positions, current_key)
                return True, path, SolverDiagnostics(
                    success=True,
                    states_explored=states_explored,
                    states_pruned_dominated=dominated_states_pruned,
                    max_queue_size=max_queue_size,
                    time_taken_ms=elapsed_ms,
                    failure_reason="",
                    path_length=max(0, len(path) - 1),
                    path_cost=float(current_g),
                    final_inventory={
                        'keys': current_state.keys,
                        'bomb_count': current_state.bomb_count,
                        'has_bomb': current_state.has_bomb,
                        'has_boss_key': current_state.has_boss_key,
                        'has_item': current_state.has_item,
                        'item_names': sorted(str(name) for name in current_state.item_names),
                        'current_floor': current_state.current_floor,
                        'doors_opened': len(current_state.opened_doors),
                        'items_collected': len(current_state.collected_items),
                    },
                    termination_status="solved",
                )

            for new_state, step_cost in self.iter_successors(current_state):
                new_key = self._state_key(new_state)
                if new_key in closed_set:
                    continue

                if self.search_mode == 'bfs':
                    current_depth = int(depths.get(current_key, 0))
                    g_score = float(current_depth + 1)
                else:
                    current_depth = int(depths.get(current_key, 0))
                    g_score = current_g + step_cost

                if new_key in g_scores and g_score >= g_scores[new_key]:
                    continue

                g_scores[new_key] = g_score
                h_score = self._heuristic(new_state)
                if self.search_mode == 'bfs':
                    f_score = float(current_depth + 1)
                elif self.search_mode == 'dijkstra':
                    f_score = g_score
                elif self.search_mode == 'greedy':
                    f_score = h_score
                elif self.enable_ara:
                    f_score = g_score + self.ara_weight * h_score
                else:
                    f_score = g_score + h_score
                parents[new_key] = current_key
                positions[new_key] = new_state.position
                depths[new_key] = current_depth + 1

                heapq.heappush(
                    open_set,
                    self._make_open_entry(
                        f_score=f_score,
                        counter=counter,
                        g_score=g_score,
                        state=new_state,
                        state_key=new_key,
                        parent_state=current_state,
                    ),
                )
                counter += 1

        # Search failed - determine reason
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        live_frontier = (
            states_explored >= self.timeout
            and self._has_live_open_entry(open_set, closed_set, g_scores)
        )
        if live_frontier:
            failure_reason = f"Timeout: explored {states_explored:,} states (limit: {self.timeout:,})"
        elif not open_set:
            failure_reason = "No path: all reachable states explored without finding goal"
        else:
            failure_reason = "Unknown failure"

        return False, [], SolverDiagnostics(
            success=False,
            states_explored=states_explored,
            states_pruned_dominated=dominated_states_pruned,
            max_queue_size=max_queue_size,
            time_taken_ms=elapsed_ms,
            failure_reason=failure_reason,
            path_length=0,
            final_inventory={
                'keys': final_state.keys if final_state else 0,
                'bomb_count': final_state.bomb_count if final_state else 0,
                'has_bomb': final_state.has_bomb if final_state else False,
                'has_boss_key': final_state.has_boss_key if final_state else False,
                'has_item': final_state.has_item if final_state else False,
                'item_names': (
                    sorted(str(name) for name in final_state.item_names)
                    if final_state
                    else []
                ),
                'current_floor': final_state.current_floor if final_state else 0,
                'doors_opened': len(final_state.opened_doors) if final_state else 0,
                'items_collected': len(final_state.collected_items) if final_state else 0,
            } if final_state else None,
            termination_status="budget_exhausted" if live_frontier else "exhausted",
        )

    def solve_with_verified_diagnostics(
        self,
        *,
        start_state: Optional[GameState] = None,
    ) -> Tuple[bool, List[Tuple[int, int]], SolverDiagnostics]:
        """Solve and require a replayable route certificate on success."""
        if start_state is None:
            success, path, diagnostics = self.solve_with_diagnostics()
        else:
            success, path, diagnostics = self.solve_with_diagnostics(start_state=start_state)
        if not success:
            return False, [], diagnostics
        if str(diagnostics.termination_status) != "solved":
            diagnostics.success = False
            diagnostics.failure_reason = (
                "Solver reported a route with contradictory termination status "
                f"{diagnostics.termination_status!r}."
            )
            diagnostics.termination_status = "validator_error"
            return False, [], diagnostics

        if start_state is None:
            replay_ok, replay_error, replay_state, replay_cost = self.verify_position_path(path)
        else:
            replay_ok, replay_error, replay_state, replay_cost = self.verify_position_path(
                path, start_state=start_state,
            )
        reported_cost = diagnostics.path_cost
        if (
            replay_ok
            and replay_cost is not None
            and reported_cost is not None
            and not math.isclose(
                float(replay_cost),
                float(reported_cost),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            replay_ok = False
            replay_error = (
                "replayed route cost does not match the solver cost: "
                f"replay={float(replay_cost):.12g}, "
                f"solver={float(reported_cost):.12g}"
            )

        diagnostics.route_replay_status = "verified" if replay_ok else "failed"
        diagnostics.route_replay_error = str(replay_error or "")
        diagnostics.route_replay_path_cost = replay_cost
        if not replay_ok:
            diagnostics.success = False
            diagnostics.failure_reason = (
                "Post-route replay rejected the reconstructed solution: "
                f"{diagnostics.route_replay_error}"
            )
            diagnostics.termination_status = "route_replay_failed"
            return False, [], diagnostics

        if replay_state is not None:
            diagnostics.final_inventory = {
                'keys': replay_state.keys,
                'bomb_count': replay_state.bomb_count,
                'has_bomb': replay_state.has_bomb,
                'has_boss_key': replay_state.has_boss_key,
                'has_item': replay_state.has_item,
                'item_names': sorted(str(name) for name in replay_state.item_names),
                'current_floor': replay_state.current_floor,
                'doors_opened': len(replay_state.opened_doors),
                'items_collected': len(replay_state.collected_items),
            }
        return True, list(path), diagnostics

    def iter_successors(self, current_state: GameState):
        """Yield legal full-state successors and costs under canonical rules.

        Search order is supplied by each algorithm; movement, graph gates,
        inventory changes, and corner restrictions are shared. Route replay
        deliberately retains a separate implementation for verification.
        """
        grid = self.env.original_grid
        height, width = grid.shape
        cardinal_deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
        diagonal_deltas = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        curr_r, curr_c = current_state.position
        neighbors = []

        # Determine teleportation eligibility (same as solve())
        curr_tile = grid[curr_r, curr_c]
        is_stair = (curr_tile == SEMANTIC_PALETTE['STAIR'])
        is_door = (curr_tile in {
            SEMANTIC_PALETTE['DOOR_OPEN'],
            SEMANTIC_PALETTE['DOOR_SOFT'],
            SEMANTIC_PALETTE['DOOR_LOCKED'],
            SEMANTIC_PALETTE['DOOR_BOMB'],
            SEMANTIC_PALETTE['DOOR_BOSS'],
        })
        is_at_boundary = False
        if self.env.room_positions:
            curr_room = self.env.get_room_for_position((curr_r, curr_c))
            if curr_room is not None:
                r_off, c_off = self.env.room_positions[curr_room]
                local_r = curr_r - r_off
                local_c = curr_c - c_off
                if (local_r <= 1 or local_r >= ROOM_HEIGHT - 2 or
                    local_c <= 1 or local_c >= ROOM_WIDTH - 2):
                    is_at_boundary = True
        if self.vglc_strict_mode:
            can_teleport = is_stair or is_door
        else:
            can_teleport = is_stair or is_door or is_at_boundary
        if self.strict_original_mode:
            can_teleport = is_stair

        for dr, dc in cardinal_deltas:
            new_r, new_c = curr_r + dr, curr_c + dc
            if 0 <= new_r < height and 0 <= new_c < width:
                next_tile = grid[new_r, new_c]
                if next_tile not in BLOCKING_IDS:
                    neighbors.append(((new_r, new_c), next_tile, CARDINAL_COST, None))

        if self.allow_diagonals:
            for dr, dc in diagonal_deltas:
                new_r, new_c = curr_r + dr, curr_c + dc
                if not (0 <= new_r < height and 0 <= new_c < width):
                    continue
                next_tile = grid[new_r, new_c]
                if next_tile in BLOCKING_IDS:
                    continue
                adj_r_tile = grid[curr_r + dr, curr_c]
                adj_c_tile = grid[curr_r, curr_c + dc]
                if adj_r_tile in BLOCKING_IDS or adj_c_tile in BLOCKING_IDS:
                    continue
                if adj_r_tile in CONDITIONAL_IDS or adj_c_tile in CONDITIONAL_IDS:
                    continue
                # Also block diagonal through pushable blocks or water/lava tiles.
                if adj_r_tile in PUSHABLE_IDS or adj_c_tile in PUSHABLE_IDS:
                    continue
                if adj_r_tile in WATER_IDS or adj_c_tile in WATER_IDS:
                    continue
                neighbors.append(((new_r, new_c), next_tile, DIAGONAL_COST, None))

        # Stair handling
        if grid[curr_r, curr_c] == SEMANTIC_PALETTE['STAIR']:
            for dest_pos in self._get_stair_destinations(current_state.position):
                if 0 <= dest_pos[0] < height and 0 <= dest_pos[1] < width:
                    neighbors.append((dest_pos, grid[dest_pos[0], dest_pos[1]], 1, "stair"))

        # VIRTUAL NODE TRAVERSAL: CONTROLLED VERSION (same as solve())
        if can_teleport and not self.strict_original_mode:
            virtual_destinations = self._get_controlled_virtual_destinations(
                current_state.position, current_state
            )
            for dest_pos, cost, edge_type in virtual_destinations:
                if 0 <= dest_pos[0] < height and 0 <= dest_pos[1] < width:
                    dest_tile = grid[dest_pos[0], dest_pos[1]]
                    neighbors.append((dest_pos, dest_tile, cost, edge_type))

        # GRAPH-BASED ROOM WARPING (same as solve())
        if can_teleport and not self.strict_original_mode:
            warp_destinations = self._get_graph_warp_destinations(
                current_state.position, current_state
            )
            for dest_pos, cost, edge_type in warp_destinations:
                if 0 <= dest_pos[0] < height and 0 <= dest_pos[1] < width:
                    dest_tile = grid[dest_pos[0], dest_pos[1]]
                    neighbors.append((dest_pos, dest_tile, cost, edge_type))

        if self.env.graph_bridge is not None:
            for target_pos in self.env.graph_bridge.destinations(current_state.position):
                neighbors.append((target_pos, grid[target_pos], 1.0, None))

        for target_pos, target_tile, base_cost, graph_edge_type in neighbors:
            transition_state = current_state
            if graph_edge_type is not None:
                can_transition, transition_state = self.apply_graph_edge_transition(
                    current_state,
                    current_state.position,
                    target_pos,
                    graph_edge_type,
                )
                if not can_transition:
                    continue
            can_move, new_state = self._try_move_pure(transition_state, target_pos, target_tile)
            if not can_move:
                continue

            new_state.current_floor = self.env.floor_for_position(
                target_pos,
                default=current_state.current_floor,
            )

            yield new_state, self._get_movement_cost(
                target_tile, target_pos, current_state
            ) * base_cost

    def _cache_pickups(self) -> List[Tuple[int, int]]:
        """Pre-compute pickup locations to support persona heuristics."""
        pickups: List[Tuple[int, int]] = []
        for tile_id in [SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE['KEY_BOSS'],
                        SEMANTIC_PALETTE['KEY_ITEM'], SEMANTIC_PALETTE['ITEM_MINOR']]:
            pickups.extend(self.env.find_all_positions(tile_id))
        return pickups

    def get_stair_destinations(self, current_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Public wrapper for stair-transition destination lookup."""
        return self._get_stair_destinations(current_pos)

    def get_controlled_virtual_destinations(
        self,
        current_pos: Tuple[int, int],
        state: GameState,
    ) -> List[Tuple[Tuple[int, int], int, GraphTransitionSpec]]:
        """Public wrapper for controlled virtual-node transitions."""
        return self._get_controlled_virtual_destinations(current_pos, state)

    def get_graph_warp_destinations(
        self,
        current_pos: Tuple[int, int],
        state: GameState,
    ) -> List[Tuple[Tuple[int, int], int, GraphTransitionSpec]]:
        """Public wrapper for non-adjacent graph warp transitions."""
        return self._get_graph_warp_destinations(current_pos, state)

    def apply_graph_edge_transition(
        self,
        state: GameState,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        edge_type: GraphTransitionSpec,
    ) -> Tuple[bool, GameState]:
        """Apply inventory/open-state effects for a graph teleport edge."""
        if not isinstance(edge_type, str):
            transition_state = state
            for source_node, target_node, step_type in edge_type:
                allowed, transition_state = self._apply_graph_edge_transition_nodes(
                    transition_state,
                    source_node,
                    target_node,
                    step_type,
                )
                if not allowed:
                    return False, state
            return True, transition_state

        current_room = self.env.get_room_for_position(current_pos)
        target_room = self.env.get_room_for_position(target_pos)
        current_node = (
            self.env.room_to_node.get(current_room)
            if current_room is not None and self.env.room_to_node
            else current_room
        )
        target_node = (
            self.env.room_to_node.get(target_room)
            if target_room is not None and self.env.room_to_node
            else target_room
        )
        if current_node is None or target_node is None:
            return False, state
        return self._apply_graph_edge_transition_nodes(
            state,
            current_node,
            target_node,
            edge_type,
        )

    def _apply_graph_edge_transition_nodes(
        self,
        state: GameState,
        current_node: Any,
        target_node: Any,
        edge_type: str,
    ) -> Tuple[bool, GameState]:
        """Authoritative graph-edge gate transition using graph node IDs."""
        if self.env.graph_bridge is not None:
            return apply_graph_gate(self.env, state, current_node, target_node)
        edge_key = graph_edge_state_key(self.env.graph, current_node, target_node)
        normalized = str(edge_type or "open").strip().lower()
        raw_edge_data = self.env.graph.get_edge_data(current_node, target_node, {}) or {}
        edge_data: Dict[str, Any]
        if (
            isinstance(raw_edge_data, Mapping)
            and raw_edge_data
            and not any(
                key in raw_edge_data
                for key in (
                    "edge_type",
                    "type",
                    "label",
                    "item_required",
                    "protection_item_id",
                )
            )
            and all(isinstance(value, Mapping) for value in raw_edge_data.values())
        ):
            candidates = [dict(value) for value in raw_edge_data.values()]
            edge_data = next(
                (
                    candidate
                    for candidate in candidates
                    if normalized
                    in parse_edge_type_tokens(
                        label=str(candidate.get("label", "") or ""),
                        edge_type=str(
                            candidate.get("edge_type", candidate.get("type", ""))
                            or ""
                        ),
                    )
                ),
                candidates[0] if candidates else {},
            )
        else:
            edge_data = dict(raw_edge_data) if isinstance(raw_edge_data, Mapping) else {}

        # Edge labels can combine transport and gates (for example stairs
        # behind a key lock). Enforce every constraint, including on stair hops.
        constraints = self._edge_constraints_from_data(edge_data)
        kinds = list(dict.fromkeys(
            edge_type_from_data({"edge_type": kind, "protection_item_id": edge_data.get("protection_item_id")})
            for kind in constraints
        ))
        if normalized not in kinds and constraints == ["open"]:
            kinds.append(normalized)
        transition_state = state.copy()
        was_open = edge_key in state.opened_graph_edges
        opened_edges = set(state.opened_graph_edges)
        for kind in kinds:
            # Opening one part of a composite gate must not waive its other
            # consumable requirements during the same transition.
            if not was_open:
                transition_state.opened_graph_edges = transition_state.opened_graph_edges - {edge_key}
            allowed, transition_state = self._apply_single_graph_constraint(
                transition_state, kind, edge_key, edge_data,
            )
            if not allowed:
                return False, state
            opened_edges.update(transition_state.opened_graph_edges)
        transition_state.opened_graph_edges = frozenset(opened_edges)
        return True, transition_state

    def _apply_single_graph_constraint(
        self, state: GameState, normalized: str, edge_key: Tuple[Any, Any],
        edge_data: Mapping[str, Any],
    ) -> Tuple[bool, GameState]:
        """Apply one gate without discarding the other constraints on its edge."""
        def _has_required_item(raw_required: Any) -> bool:
            required = self.env._normalize_item_name(raw_required)
            if required is None:
                return bool(state.has_item)
            inventory = {str(name).upper() for name in state.item_names}
            return bool(state.has_item and ("*" in inventory or required in inventory))

        if normalized in {"open", "path", "stair", "soft_locked", "one_way"}:
            return True, state.copy()

        if normalized in {"switch", "puzzle"}:
            if edge_key in state.opened_graph_edges:
                return True, state.copy()
            # Tile-level puzzle completion and mission-graph switch activation
            # are different state spaces. A room ID is not a puzzle-door tile,
            # so treating node_to_room[target] as a tile position is unsound.
            # Without an explicit bridge between those states, fail closed and
            # let ordinary tile movement or the mission progression oracle
            # establish reachability.
            return False, state
        if normalized in {"hazard", "hazard_protected"}:
            if not _has_required_item(edge_data.get("protection_item_id")):
                return False, state
            return True, state.copy()
        if edge_key in state.opened_graph_edges:
            return True, state.copy()

        new_state = state.copy()
        if normalized in {"key_locked", "locked"}:
            if new_state.keys <= 0:
                return False, state
            new_state.keys -= 1
        elif normalized in {"bombable", "bomb"}:
            if new_state.bomb_count <= 0:
                return False, state
            new_state.bomb_count -= 1
        elif normalized in {"boss_locked", "boss"}:
            if not new_state.has_boss_key:
                return False, state
        elif normalized in {"item_locked", "item_gate"}:
            if not _has_required_item(edge_data.get("item_required")):
                return False, state
        else:
            return False, state
        new_state.opened_graph_edges.add(edge_key)
        return True, new_state

    def _can_traverse_graph_edge(
        self,
        current_node: Any,
        target_node: Any,
        edge_type: str,
        state: GameState,
    ) -> bool:
        """Preview the authoritative transition without mutating ``state``."""
        allowed, _ = self._apply_graph_edge_transition_nodes(
            state,
            current_node,
            target_node,
            edge_type,
        )
        return bool(allowed)

    def _get_stair_destinations(self, current_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find stair destinations using graph connectivity (CACHED).
        
        When standing on a stair tile, find the DIRECTLY connected room via
        the graph edge. In Zelda, stairs connect exactly two rooms.
        
        FIXED: Previously this did BFS through the entire graph, allowing
        teleportation to any room. Now it only returns the direct neighbor
        connected by an explicit stair edge (VGLC 's' is impassable).
        """
        # PERFORMANCE: Check cache first
        if self.env.graph_bridge is not None:
            return []
        if current_pos in self._stair_dest_cache:
            return self._stair_dest_cache[current_pos]

        if not self.env.graph or not self.env.room_to_node or not self.env.room_positions:
            self._stair_dest_cache[current_pos] = []
            return []

        # Find which room contains current position
        current_room = self.env.get_room_for_position(current_pos)

        if current_room is None:
            self._stair_dest_cache[current_pos] = []
            return []

        current_node = self.env.room_to_node.get(current_room)
        if current_node is None:
            self._stair_dest_cache[current_pos] = []
            return []

        # Build reverse mapping: node -> room
        node_to_room = {v: k for k, v in self.env.room_to_node.items()}

        # FIXED: Only look at DIRECT neighbors connected by stair edges
        # A stair connects to ONE specific room, not all rooms in the dungeon
        destinations = []

        # Check successors for stair edges
        for neighbor_node in graph_successors(self.env.graph, current_node):
            edge_data = self.env.graph.get_edge_data(current_node, neighbor_node, {}) or {}

            # Only follow stair edges based on canonical constraints.
            constraints = self._edge_constraints_from_data(edge_data)
            is_stair_edge = 'stair' in constraints

            if not is_stair_edge:
                continue

            # Check if neighbor has a physical room
            neighbor_room = node_to_room.get(neighbor_node)
            if neighbor_room is None or neighbor_room not in self.env.room_positions:
                continue

            # Find stair tile in neighbor room
            r_off, c_off = self.env.room_positions[neighbor_room]
            r_end = min(r_off + ROOM_HEIGHT, self.env.height)
            c_end = min(c_off + ROOM_WIDTH, self.env.width)

            found_dest = False
            for r in range(r_off, r_end):
                for c in range(c_off, c_end):
                    if self.env.grid[r, c] == SEMANTIC_PALETTE['STAIR']:
                        destinations.append((r, c))
                        found_dest = True
                        break
                if found_dest:
                    break

            # Fallback: any walkable tile if no stair found
            if not found_dest:
                for r in range(r_off, r_end):
                    for c in range(c_off, c_end):
                        if self.env.grid[r, c] in WALKABLE_IDS:
                            destinations.append((r, c))
                            found_dest = True
                            break
                    if found_dest:
                        break

        # PERFORMANCE: Cache result for future lookups
        self._stair_dest_cache[current_pos] = destinations
        return destinations

    def _get_controlled_virtual_destinations(self, current_pos: Tuple[int, int],
                                              state: GameState) -> List[Tuple[Tuple[int, int], int, GraphTransitionSpec]]:
        """
        Find CONTROLLED virtual node destinations from current position.
        
        Allows transitions through virtual nodes that the current room's graph node
        has a DIRECT edge to. Virtual nodes act as "hubs" connecting non-adjacent
        physical rooms (e.g., hidden passages, bombable shortcuts, stairwells).
        
        Entry is allowed from ANY physical node with a direct edge to the virtual
        node -- not just the virtual_parent. This is required because virtual nodes
        in Zelda dungeons (D7, D9) can be reached from multiple rooms, and the 
        graph encodes all valid entry/exit points.
        
        Still prevents "teleportation everywhere" because:
        1. Only DIRECT neighbors that are virtual are considered
        2. Each edge is checked for item requirements (keys, bombs, etc.)
        3. Player must be on a transition tile (door/stair) to trigger
        
        Args:
            current_pos: Current (row, col) position in the grid
            state: Current game state (for checking item requirements)
            
        Returns:
            List of (dest_pos, cost, edge_type) tuples for valid virtual transitions
        """
        # Quick check: do we have graph connectivityx
        if self.env.graph_bridge is not None:
            return []
        if not self.env.graph or not self.env.room_to_node or not self.env.room_positions:
            return []

        # Find which room contains current position
        current_room = self.env.get_room_for_position(current_pos)

        if current_room is None:
            return []

        current_node = self.env.room_to_node.get(current_room)
        if current_node is None:
            return []

        # Get node_to_room mapping
        if self._node_to_room is None:
            if hasattr(self.env, 'node_to_room') and self.env.node_to_room:
                self._node_to_room = self.env.node_to_room
            else:
                self._node_to_room = {v: k for k, v in self.env.room_to_node.items()}

        destinations = []

        # Check all direct neighbors of current node
        for neighbor in graph_successors(self.env.graph, current_node):
            neighbor_data = self.env.graph.nodes.get(neighbor, {})

            # ONLY process if this is a virtual node (hidden passage / hub)
            if not neighbor_data.get('is_virtual', False):
                continue

            # FIXED: Allow entry from ANY physical node with a direct edge to the
            # virtual node, not just the virtual_parent. The graph explicitly
            # encodes which rooms can access each virtual node via edges.
            # The edge type check below provides the necessary access control.

            # Get edge requirements to access the virtual node
            edge_data = self.env.graph.get_edge_data(current_node, neighbor, {}) or {}
            edge_type = self._edge_type_from_data(edge_data)

            can_traverse, entry_state = self._apply_graph_edge_transition_nodes(
                state,
                current_node,
                neighbor,
                edge_type,
            )
            if not can_traverse:
                continue

            # BFS through virtual nodes to find all reachable physical rooms
            # Exclude the current node to avoid trivial loops back to self
            entry_path: GraphTransitionPath = ((current_node, neighbor, edge_type),)
            virtual_queue = deque(
                [(neighbor, entry_path, entry_state, frozenset({current_node, neighbor}))]
            )
            visited_states = {(neighbor, self._state_key(entry_state))}

            while virtual_queue:
                v_node, transition_path, path_state, path_nodes = virtual_queue.popleft()

                for exit_node in graph_successors(self.env.graph, v_node):
                    if exit_node in path_nodes:
                        continue

                    exit_data = self.env.graph.nodes.get(exit_node, {})
                    exit_edge_data = self.env.graph.get_edge_data(v_node, exit_node, {}) or {}
                    exit_type = self._edge_type_from_data(exit_edge_data)
                    can_exit, exit_state = self._apply_graph_edge_transition_nodes(
                        path_state,
                        v_node,
                        exit_node,
                        exit_type,
                    )
                    if not can_exit:
                        continue
                    exit_path = transition_path + ((v_node, exit_node, exit_type),)
                    exit_path_nodes = path_nodes | {exit_node}

                    if exit_data.get('is_virtual', False):
                        visit_key = (exit_node, self._state_key(exit_state))
                        if visit_key not in visited_states:
                            visited_states.add(visit_key)
                            virtual_queue.append(
                                (exit_node, exit_path, exit_state, exit_path_nodes)
                            )
                    else:
                        exit_room = self._node_to_room.get(exit_node)
                        if exit_room is not None and exit_room in self.env.room_positions:
                            dest_pos = self._find_room_entry_point(exit_room)

                            if dest_pos is None:
                                # TRANSITION ROOM: BFS to find next walkable room
                                dest_pos, traversal_cost, full_path = self._find_next_walkable_room_via_graph(
                                    exit_node,
                                    visited=set(exit_path_nodes),
                                    state=exit_state,
                                    transition_path=exit_path,
                                )
                                if dest_pos:
                                    destinations.append((dest_pos, traversal_cost, full_path))
                            else:
                                # Normal room with walkable tiles
                                destinations.append((dest_pos, 10, exit_path))

        return destinations

    def _can_traverse_edge(self, edge_type: str, state: GameState) -> bool:
        """Check graph-edge traversal using the shared canonical rule table."""
        return can_traverse_edge_type(
            edge_type,
            state,
            strict_original_mode=self.strict_original_mode,
            get_room_for_position=self.env.get_room_for_position,
            is_room_cleared=self.env.is_room_cleared,
        )

    def _get_graph_warp_destinations(self, current_pos: Tuple[int, int],
                                      state: GameState) -> List[Tuple[Tuple[int, int], int, GraphTransitionSpec]]:
        """
        Find non-adjacent room destinations via graph edges (staircases/warps).
        
        In Zelda dungeons, the graph encodes connections between rooms that aren't
        physically adjacent - these represent staircases, hidden passages, or warps
        that you access by bombing walls or using stairs.
        
        This method handles edges between PHYSICAL nodes (not virtual) that connect
        non-adjacent rooms. These are typically:
        - Bombable walls that reveal stairs to another room
        - Key-locked passages to distant rooms
        - Open staircases connecting different dungeon levels
        
        Args:
            current_pos: Current (row, col) position in the grid
            state: Current game state (for checking item requirements)
            
        Returns:
            List of (dest_pos, cost, edge_type) tuples for valid warp transitions
        """
        if self.env.graph_bridge is not None:
            return []
        if not self.env.graph or not self.env.room_to_node or not self.env.room_positions:
            return []

        # Find which room contains current position
        current_room = self.env.get_room_for_position(current_pos)

        if current_room is None:
            return []

        current_node = self.env.room_to_node.get(current_room)
        if current_node is None:
            return []

        # Get node_to_room mapping
        if self._node_to_room is None:
            if hasattr(self.env, 'node_to_room') and self.env.node_to_room:
                self._node_to_room = self.env.node_to_room
            else:
                self._node_to_room = {v: k for k, v in self.env.room_to_node.items()}

        destinations = []

        # Check all neighbors of current node
        for neighbor in graph_successors(self.env.graph, current_node):
            neighbor_data = self.env.graph.nodes.get(neighbor, {})

            # Skip virtual nodes - they're handled by _get_controlled_virtual_destinations
            if neighbor_data.get('is_virtual', False):
                continue

            neighbor_room = self._node_to_room.get(neighbor)
            if neighbor_room is None or neighbor_room not in self.env.room_positions:
                continue

            # Check if this is a non-adjacent room connection. Direct adjacent
            # open/path edges are handled by normal grid movement; direct
            # constrained edges still need this graph transition path because
            # the physical room boundary may not encode the abstract gate.
            dr = abs(current_room[0] - neighbor_room[0])
            dc = abs(current_room[1] - neighbor_room[1])
            manhattan_dist = dr + dc
            edge_data = self.env.graph.get_edge_data(current_node, neighbor, {}) or {}
            edge_type = self._edge_type_from_data(edge_data)
            normalized_edge_type = str(edge_type or "open").strip().lower()
            direct_open_edge = normalized_edge_type in {"", "open", "path"}

            if manhattan_dist <= 1 and direct_open_edge:
                continue

            # Dataset-faithful mode: only explicit stair/warp edges may teleport
            # across non-adjacent rooms. Adjacent constrained graph edges are
            # still real gates and must be checked.
            if self.vglc_strict_mode and manhattan_dist > 1 and edge_type != 'stair':
                continue

            # Check if we can traverse this edge
            if not self._can_traverse_graph_edge(current_node, neighbor, edge_type, state):
                continue

            # Find entry point in destination room
            dest_pos = self._find_room_entry_point(neighbor_room)

            if dest_pos is None:
                # TRANSITION ROOM: This room has no walkable tiles (corridor/staircase placeholder)
                # BFS through the graph to find the next reachable walkable room
                allowed, transition_state = self._apply_graph_edge_transition_nodes(
                    state,
                    current_node,
                    neighbor,
                    edge_type,
                )
                if not allowed:
                    continue
                transition_path: GraphTransitionPath = (
                    (current_node, neighbor, edge_type),
                )
                dest_pos, traversal_cost, full_path = self._find_next_walkable_room_via_graph(
                    neighbor,
                    visited={current_node, neighbor},
                    state=transition_state,
                    transition_path=transition_path,
                )
                if dest_pos:
                    destinations.append((dest_pos, traversal_cost, full_path))
            else:
                # Normal room with walkable tiles
                destinations.append((dest_pos, 10, edge_type))

        return destinations

    def _find_next_walkable_room_via_graph(self, start_node: int, visited: set,
                                            state: 'GameState', max_cost: int = 30,
                                            transition_path: GraphTransitionPath = (),
                                            ) -> Tuple[Optional[Tuple[int, int]], int, GraphTransitionPath]:
        """
        BFS through graph from a transition node to find the next walkable room.
        
        When a graph edge points to a room with no walkable tiles (transition room),
        this method continues through the graph to find the actual destination.
        This handles VGLC dungeon patterns where some rooms are corridor/staircase
        placeholders that players traverse through without actually walking in them.
        
        Args:
            start_node: Graph node ID of the transition room
            visited: Set of already-visited node IDs to prevent cycles
            state: Current game state for edge traversal checks
            max_cost: Maximum accumulated cost before giving up
            
        Returns:
            (dest_pos, cost, transition_path), or ``(None, 0, ())``.
        """
        queue = deque(
            [(start_node, 10, state, transition_path, frozenset(visited))]
        )
        visited_states = {(start_node, self._state_key(state))}

        while queue:
            node, cost, path_state, path, path_nodes = queue.popleft()

            for next_node in graph_successors(self.env.graph, node):
                if next_node in path_nodes:
                    continue

                edge_data = self.env.graph.get_edge_data(node, next_node, {}) or {}
                edge_type = self._edge_type_from_data(edge_data)
                allowed, next_state = self._apply_graph_edge_transition_nodes(
                    path_state,
                    node,
                    next_node,
                    edge_type,
                )
                if not allowed:
                    continue

                next_path = path + ((node, next_node, edge_type),)
                next_path_nodes = path_nodes | {next_node}
                next_room = self._node_to_room.get(next_node)

                if next_room is not None and next_room in self.env.room_positions:
                    dest_pos = self._find_room_entry_point(next_room)
                    if dest_pos:
                        return dest_pos, cost + 5, next_path

                # Continue BFS if within cost limit
                if cost + 5 < max_cost:
                    visit_key = (next_node, self._state_key(next_state))
                    if visit_key not in visited_states:
                        visited_states.add(visit_key)
                        queue.append(
                            (
                                next_node,
                                cost + 5,
                                next_state,
                                next_path,
                                next_path_nodes,
                            )
                        )

        return None, 0, ()

    def _get_room_at_position(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get the room that contains the given position.
        
        Args:
            pos: Position (row, col) in grid coordinates
            
        Returns:
            Room position key (room_row, room_col), or None if not in any room
        """
        return self.env.get_room_for_position(pos)

    def _is_at_room_boundary(self, pos: Tuple[int, int]) -> bool:
        """
        Check if player is at the boundary of their current room.
        Room boundaries are valid transition points for warping to connected rooms.
        
        A position is at the room boundary if it's within 1 tile of the room edge.
        
        Args:
            pos: Player position (row, col)
            
        Returns:
            True if at room boundary, False otherwise
        """
        if not self.env.room_positions:
            return False

        current_room = self._get_room_at_position(pos)
        if current_room is None or current_room not in self.env.room_positions:
            return False

        r_off, c_off = self.env.room_positions[current_room]
        r_end = min(r_off + ROOM_HEIGHT, self.env.height)
        c_end = min(c_off + ROOM_WIDTH, self.env.width)

        row, col = pos

        # Check if within 1 tile of any room edge
        at_top = row <= r_off + 1
        at_bottom = row >= r_end - 2
        at_left = col <= c_off + 1
        at_right = col >= c_end - 2

        return at_top or at_bottom or at_left or at_right

    def _find_room_entry_point(self, room_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Find a walkable entry point in a room for virtual node traversal.
        
        Prefers: STAIR > DOOR_OPEN > FLOOR > any walkable tile
        
        Args:
            room_pos: Room position key
            
        Returns:
            (row, col) of entry point, or None if room not accessible
        """
        if room_pos in self._room_entry_point_cache:
            return self._room_entry_point_cache[room_pos]
        if room_pos not in self.env.room_positions:
            self._room_entry_point_cache[room_pos] = None
            return None

        r_off, c_off = self.env.room_positions[room_pos]
        r_end = min(r_off + ROOM_HEIGHT, self.env.height)
        c_end = min(c_off + ROOM_WIDTH, self.env.width)

        # Priority 1: Look for STAIR tiles
        for r in range(r_off, r_end):
            for c in range(c_off, c_end):
                if self.env.grid[r, c] == SEMANTIC_PALETTE['STAIR']:
                    res = (r, c)
                    self._room_entry_point_cache[room_pos] = res
                    return res

        # Priority 2: Look for open doors
        for r in range(r_off, r_end):
            for c in range(c_off, c_end):
                if self.env.grid[r, c] == SEMANTIC_PALETTE['DOOR_OPEN']:
                    res = (r, c)
                    self._room_entry_point_cache[room_pos] = res
                    return res

        # Priority 3: Find any walkable tile near room center
        center_r = r_off + ROOM_HEIGHT // 2
        center_c = c_off + ROOM_WIDTH // 2

        for radius in range(max(ROOM_HEIGHT, ROOM_WIDTH)):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    r, c = center_r + dr, center_c + dc
                    if r_off <= r < r_end and c_off <= c < c_end:
                        if self.env.grid[r, c] in WALKABLE_IDS:
                            res = (r, c)
                            self._room_entry_point_cache[room_pos] = res
                            return res

        self._room_entry_point_cache[room_pos] = None
        return None

    def _can_traverse_edge_type(self, edge_type: str, state: GameState) -> bool:
        """
        Check if the current state allows traversing an edge of the given type.
        
        Args:
            edge_type: The edge type constraint
            state: Current game state with inventory
            
        Returns:
            True if the edge can be traversed, False otherwise
        """
        return can_traverse_edge_type(
            edge_type,
            state,
            strict_original_mode=self.strict_original_mode,
            get_room_for_position=self.env.get_room_for_position,
            is_room_cleared=self.env.is_room_cleared,
        )

    def _get_movement_cost(self, target_tile: int, target_pos: Tuple[int, int], state: GameState) -> float:
        """
        Calculate the cost of moving to a target tile.
        
        COMBAT-AWARE PATHFINDING:
        - FLOOR tiles: cost = 1.0 (baseline)
        - ENEMY tiles: cost = 10.0 (expensive to walk through)
        - DOOR tiles (unlocked): cost = 2.0 (takes time to open)
        - PICKUP tiles: cost = 1.5 (stop to collect item)
        
        This makes A* prefer safer routes that avoid enemies when possible.
        The higher cost doesn't make enemies impassable, but forces the agent
        to find alternate routes if they exist.
        
        Args:
            target_tile: Semantic ID of the target tile
            target_pos: Position (r, c) of the target
            state: Current game state
            
        Returns:
            Movement cost (float)
        """
        # If position already visited, treat as floor (no repeat cost)
        if self.env.graph_bridge is not None and target_pos in self.env.graph_bridge.cells:
            # The canonical passage transition has already enforced its gate.
            # Rechecking inventory here can assign infinite cost after the
            # passage consumed the last key, despite a legal state transition.
            return 1.0
        if target_pos in state.collected_items or target_pos in state.opened_doors:
            return 1.0

        # ENEMY: High traversal cost (simulates health/time loss from combat)
        if target_tile == SEMANTIC_PALETTE['ENEMY']:
            return 10.0

        # PICKUP items: Slight delay for collection
        if target_tile in PICKUP_IDS:
            return 1.5

        # DOORS (locked): Cost depends on whether we have keys
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if state.keys > 0:
                return 2.0  # Can open, but takes time
            return float('inf')  # Cannot pass

        if target_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
            if state.bomb_count > 0:
                return 3.0  # Bombing takes time
            return float('inf')

        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if state.has_boss_key:
                return 2.0
            return float('inf')

        if target_tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
            return 2.5  # Puzzle solving takes time

        # Standard walkable tiles
        if target_tile in WALKABLE_IDS:
            return 1.0

        # Discourage unnecessary block bulldozing when walking around is possible.
        if (
            has_pushed_block_at(state, target_pos)
            or (target_tile in PUSHABLE_IDS and not was_block_vacated(state, target_pos))
        ):
            return 6.0

        # Blocking tiles
        if target_tile in BLOCKING_IDS:
            return float('inf')

        # Default: standard cost
        return 1.0

    def _try_move_pure(self, state: GameState, target_pos: Tuple[int, int],
                       target_tile: int) -> Tuple[bool, GameState]:
        """
        Pure state-based move attempt (no grid modifications).

        This delegates to ``ZeldaLogicEnv._try_move_pure`` so all solvers share
        one canonical transition function.
        """
        return self.env.try_move_pure(state, target_pos, target_tile)

    def _heuristic(self, state: GameState) -> float:
        """
        Heuristic function for A*.
        
        Uses Manhattan distance to goal, with adjustments for:
        - Graph-based BFS distance (room hops x avg room diameter)
        - Missing keys when locked doors are on path
        - Missing bombs when bomb doors are on path
        - Missing boss key when boss doors are on path
        - Missing ladder (KEY_ITEM) when water/element tiles block path
        
        PERFORMANCE: Uses cached door positions (set at initialization)
        instead of scanning grid on every call.
        """
        if self.env.graph_bridge is not None and self.env.graph_bridge.portals:
            # Coordinate distance is not a lower bound with nonlocal moves.
            return 0.0
        if self.env.goal_pos is None:
            return float('inf')

        pos = state.position
        goal = self.env.goal_pos
        if self.allow_diagonals:
            # Chebyshev distance: admissible when diagonal moves cost 1.0.
            dx = abs(pos[0] - goal[0])
            dy = abs(pos[1] - goal[1])
            manhattan_h = max(dx, dy)
        else:
            manhattan_h = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Track current node and graph-derived lower bounds.
        has_graph_topology = False
        graph_hops_lb = None
        locked_edges_lb = None

        try:
            if (self.env.graph and self.env.room_to_node and
                self.env.room_positions):
                has_graph_topology = True
                room_pos = self.env.get_room_for_position(pos)
                if room_pos is not None:
                    node = self.env.room_to_node.get(room_pos)
                    if node is not None:
                        bfs_dist = self._graph_bfs_dist.get(node, None)
                        if bfs_dist is not None:
                            # Lower bound in abstract graph hops.
                            graph_hops_lb = float(bfs_dist)
                        if node in self.min_locked_needed_node:
                            # Lower bound on number of locked transitions that must be crossed.
                            locked_edges_lb = float(self.min_locked_needed_node[node])
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            logger.debug("Graph heuristic metadata lookup failed; using positional fallback: %s", exc)

        # Strict mode for canonical A* (w=1): keep heuristic admissible.
        # On contiguous stitched dungeons without portals/stairs/non-adjacent warps,
        # coordinate Manhattan (or Chebyshev with diagonals) is an exact lower bound;
        # only fall back to hop/lock bounds when nonlocal transitions exist.
        strict_astar = (self.search_mode == 'astar' and not self.enable_ara)
        if strict_astar:
            if has_graph_topology:
                h_lb = 0.0 if self._has_nonlocal_transitions else float(manhattan_h)
                if graph_hops_lb is not None:
                    h_lb = max(h_lb, graph_hops_lb)
                if locked_edges_lb is not None:
                    h_lb = max(h_lb, locked_edges_lb)
                return float(h_lb)
            return float(manhattan_h)

        # Non-strict modes (e.g., greedy / weighted A*) keep stronger guidance.
        h = float(manhattan_h)
        if graph_hops_lb is not None:
            graph_h = graph_hops_lb * (ROOM_HEIGHT + ROOM_WIDTH) * 0.4
            if graph_h > h:
                h = graph_h
        if locked_edges_lb is not None:
            lock_h = locked_edges_lb * 20.0
            if lock_h > h:
                h = lock_h
        return float(h)
# ==========================================
