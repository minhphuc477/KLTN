"""Zelda Logic Environment - Discrete state machine simulator."""

from __future__ import annotations

import logging
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
from src.simulation import validator_rendering as _validator_rendering

logger = logging.getLogger(__name__)

__all__ = ["ZeldaLogicEnv"]

class ZeldaLogicEnv:
    """
    Discrete state simulator for Zelda dungeon logic.
    
    Handles:
    - Movement with collision detection
    - Item pickup and inventory management
    - Door unlocking (key, bomb, boss key)
    - Win/lose conditions
    
    This is a "headless" environment - no graphics, just logic.
    """

    def __init__(self, semantic_grid: np.ndarray, render_mode: bool = False,
                 graph=None, room_to_node=None, room_positions=None,
                 node_to_room=None,
                 room_puzzle_metadata: Optional[Mapping[str, Any]] = None,
                 solver_options: Optional['SolverOptions'] = None,
                 block_underlay_tiles: Optional[Mapping[Tuple[int, int], int]] = None,
                 connection_realizations: Optional[Mapping] = None):
        """
        Initialize the environment.
        
        Args:
            semantic_grid: 2D numpy array of semantic IDs
            render_mode: If True, enables Pygame rendering (optional)
            graph: Optional NetworkX graph for stair connections
            room_to_node: Optional mapping of room positions to graph nodes
            room_positions: Optional mapping of room positions to grid offsets
            node_to_room: Optional mapping of graph nodes to room positions (includes virtual nodes)
            room_puzzle_metadata: Optional stitched puzzle plan payload
            solver_options: Optional SolverOptions for configurable starting inventory
        """
        self.original_grid = np.array(semantic_grid, dtype=np.int64)
        self.grid = self.original_grid.copy()
        self.height, self.width = self.grid.shape
        self.render_mode = render_mode

        # Store solver options (default if not provided)
        self.solver_options = solver_options or SolverOptions()
        self.solver_options = replace(
            self.solver_options, start_bombs=starting_bomb_count(graph, self.solver_options.start_bombs),
        )
        self.rules_profile = str(getattr(self.solver_options, 'rules_profile', 'vglc_strict') or 'vglc_strict').strip().lower()
        self.strict_original_mode = self.rules_profile in {'strict_original', 'original', 'nes'}
        self.vglc_strict_mode = self.rules_profile in {'vglc_strict', 'vglc', 'dataset'}

        # Store graph connectivity for handling stairs
        self.graph = graph
        self._required_switch_ids = required_switch_ids(graph)
        self.room_to_node = room_to_node
        self.room_positions = room_positions
        self.node_to_room = node_to_room  # Includes virtual node mappings
        self.block_underlay_tiles: Dict[Tuple[int, int], int] = {
            (int(pos[0]), int(pos[1])): int(tile)
            for pos, tile in dict(block_underlay_tiles or {}).items()
        }
        # Cache room-level enemy positions for strict-original shutter logic and
        # precompute O(1) cell->room and room->floor mappings for solver hot paths.
        self._pos_room_cache: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        self._room_floor_cache: Dict[Tuple[int, int], int] = {}
        self._room_positions_stamp: Optional[Tuple[int, int]] = None
        self._room_enemy_tiles: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        if self.room_positions:
            self._build_room_enemy_cache()
            self._build_room_lookup_caches()
        self.room_puzzle_metadata = dict(room_puzzle_metadata or {})
        self._puzzle_plans: Dict[str, Dict[str, Any]] = {}
        self._puzzle_stage_lookup: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self._puzzle_door_lookup: Dict[Tuple[int, int], str] = {}
        self._puzzle_stage_counts: Dict[str, int] = {}
        self._build_puzzle_plan_cache()
        self.graph_bridge = (
            PhysicalGraphBridge(self, connection_realizations)
            if connection_realizations is not None and graph is not None
            else None
        )

        # Find start and goal positions
        self.start_pos = self._find_position(SEMANTIC_PALETTE['START'])
        self.goal_pos = self._find_position(SEMANTIC_PALETTE['TRIFORCE'])

        # Initialize game state with configurable starting inventory
        # Uses solver_options for bombs/keys (allows level-specific configuration)
        self.state = GameState(
            position=self.start_pos if self.start_pos else (0, 0),
            keys=self.solver_options.start_keys,
            bomb_count=self.solver_options.start_bombs,
            has_boss_key=self.solver_options.start_boss_key,
            has_item=self.solver_options.start_item,
            item_names={"*"} if self.solver_options.start_item else set(),
            current_floor=self.floor_for_position(
                self.start_pos if self.start_pos else (0, 0),
                default=0,
            ),
        )
        self.done = False
        self.won = False
        self.step_count = 0
        self.max_steps = 10000  # Prevent infinite loops

        # Initialize rendering if needed
        self._screen = None
        self._font = None
        self._link_img = None
        self._images = {}
        if render_mode:
            self._init_render()

    # Public wrappers used by solver helpers (keeps internals encapsulated for linting)
    def find_all_positions(self, target_id: int) -> List[Tuple[int, int]]:
        return self._find_all_positions(target_id)

    def _build_room_lookup_caches(self) -> None:
        """Populate O(1) cell-to-room and room-to-floor caches from room_positions."""
        self._pos_room_cache = {}
        self._room_floor_cache = {}
        if not self.room_positions:
            self._room_positions_stamp = None
            return
        self._room_positions_stamp = (id(self.room_positions), len(self.room_positions))
        for room_pos, (row_offset, col_offset) in self.room_positions.items():
            r_off = int(row_offset)
            c_off = int(col_offset)
            r_end = min(r_off + ROOM_HEIGHT, self.height)
            c_end = min(c_off + ROOM_WIDTH, self.width)
            for rr in range(max(0, r_off), r_end):
                for cc in range(max(0, c_off), c_end):
                    self._pos_room_cache.setdefault((rr, cc), room_pos)
            if self.room_to_node and self.graph is not None:
                node_id = self.room_to_node.get(room_pos)
                if node_id is not None and node_id in self.graph:
                    attrs = self.graph.nodes[node_id]
                    floor_val: Optional[int] = None
                    for key in ("floor", "floor_id", "z", "level"):
                        value = attrs.get(key)
                        if value is not None:
                            try:
                                floor_val = int(value)
                                break
                            except (TypeError, ValueError):
                                continue
                    if floor_val is None:
                        position_value = attrs.get("position", attrs.get("pos"))
                        if isinstance(position_value, (tuple, list)) and len(position_value) >= 3:
                            try:
                                floor_val = int(position_value[2])
                            except (TypeError, ValueError):
                                pass
                    if floor_val is not None:
                        self._room_floor_cache[room_pos] = floor_val

    def _ensure_room_lookup_caches(self) -> None:
        if not self.room_positions:
            return
        stamp = (id(self.room_positions), len(self.room_positions))
        if not self._pos_room_cache or stamp != getattr(self, "_room_positions_stamp", None):
            self._build_room_lookup_caches()

    def floor_for_position(self, position: Tuple[int, int], *, default: int = 0) -> int:
        """Resolve a stitched-grid position to its mission-node floor."""
        if not self.room_positions or not self.room_to_node or self.graph is None:
            return int(default)

        room_key = self._get_room_for_position((int(position[0]), int(position[1])))
        if room_key is None:
            return int(default)
        return self._room_floor_cache.get(room_key, int(default))

    def get_room_for_position(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        return self._get_room_for_position(pos)

    @staticmethod
    def _normalize_item_name(value: Any) -> Optional[str]:
        text = str(value or "").strip().upper()
        return text or None

    def _item_names_for_position(self, pos: Tuple[int, int]) -> Set[str]:
        """Resolve typed graph-item identities represented by a KEY_ITEM tile."""
        room = self._get_room_for_position(tuple(pos))
        if room is None or not self.room_to_node or self.graph is None:
            return {"*"}
        node_id = self.room_to_node.get(room)
        if node_id is None or node_id not in self.graph:
            return {"*"}
        attrs = dict(self.graph.nodes[node_id])
        names = {
            normalized
            for normalized in (
                self._normalize_item_name(attrs.get("item_type")),
                self._normalize_item_name(attrs.get("protection_item_id")),
                self._normalize_item_name(attrs.get("item_id")),
            )
            if normalized is not None
        }
        return names or {"*"}

    def is_room_cleared(self, room_pos: Optional[Tuple[int, int]], state: GameState) -> bool:
        return self._is_room_cleared(room_pos, state)

    def try_move_pure(self, state: GameState, target_pos: Tuple[int, int], target_tile: int) -> Tuple[bool, GameState]:
        return self._try_move_pure(state, target_pos, target_tile)

    def try_move(self, target_pos: Tuple[int, int], target_tile: int) -> Tuple[bool, GameState, float, Dict[str, Any]]:
        """Public mutating transition used by headless environment adapters."""
        return self._try_move(target_pos, target_tile)

    def _underlay_tile_for_block_origin(self, pos: Tuple[int, int]) -> int:
        return int(self.block_underlay_tiles.get(tuple(pos), SEMANTIC_PALETTE['FLOOR']))

    def _apply_pickup_if_present(
        self,
        state: GameState,
        pos: Tuple[int, int],
        tile: int,
        *,
        mutate_grid: bool,
    ) -> Tuple[GameState, float, Dict[str, Any]]:
        if int(tile) not in PICKUP_IDS or tuple(pos) in state.collected_items:
            return state, 0.0, {}
        if mutate_grid:
            return self._pickup_item(state, pos, int(tile))

        new_state = state.copy()
        new_state.collected_items = state.collected_items | {tuple(pos)}
        if int(tile) == SEMANTIC_PALETTE['KEY_SMALL']:
            new_state.keys = state.keys + 1
        elif int(tile) == SEMANTIC_PALETTE['KEY_BOSS']:
            new_state.has_boss_key = True
        elif int(tile) == SEMANTIC_PALETTE['KEY_ITEM']:
            new_state.has_item = True
            new_state.item_names.update(self._item_names_for_position(pos))
        elif int(tile) == SEMANTIC_PALETTE['ITEM_MINOR']:
            new_state.bomb_count = state.bomb_count + 4
        return new_state, 0.0, {'msg': 'Picked up exposed item', 'item': ID_TO_NAME.get(int(tile), str(tile))}

    def _find_position(self, target_id: int) -> Optional[Tuple[int, int]]:
        """Find the first occurrence of a tile ID."""
        positions = np.where(self.grid == target_id)
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None

    def _find_all_positions(self, target_id: int) -> List[Tuple[int, int]]:
        """Find all occurrences of a tile ID."""
        positions = np.where(self.grid == target_id)
        return list(zip(positions[0].tolist(), positions[1].tolist()))

    def _build_room_enemy_cache(self) -> None:
        """Precompute enemy/boss tiles per room for strict-original semantics."""
        self._room_enemy_tiles = {}
        if not self.room_positions:
            return

        enemy_ids = {SEMANTIC_PALETTE['ENEMY'], SEMANTIC_PALETTE['BOSS']}
        for room_pos, (r_off, c_off) in self.room_positions.items():
            enemies: Set[Tuple[int, int]] = set()
            r_end = min(r_off + ROOM_HEIGHT, self.height)
            c_end = min(c_off + ROOM_WIDTH, self.width)
            for rr in range(r_off, r_end):
                for cc in range(c_off, c_end):
                    if int(self.original_grid[rr, cc]) in enemy_ids:
                        enemies.add((rr, cc))
            self._room_enemy_tiles[room_pos] = enemies

    def _get_room_for_position(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Return room identifier for a global grid position."""
        if not self.room_positions:
            return None
        self._ensure_room_lookup_caches()
        return self._pos_room_cache.get(pos)

    def _is_room_cleared(self, room_pos: Optional[Tuple[int, int]], state: GameState) -> bool:
        """
        A room is cleared when all enemy/boss tiles in that room were defeated.
        """
        if room_pos is None:
            return True
        room_enemies = self._room_enemy_tiles.get(room_pos, set())
        if not room_enemies:
            return True
        return room_enemies.issubset(set(state.defeated_enemies))

    def _is_enemy_progression_relevant(self, pos: Tuple[int, int]) -> bool:
        """Return whether defeating an enemy/boss tile at ``pos`` can affect gating."""
        if self.strict_original_mode:
            return True
        coord = (int(pos[0]), int(pos[1]))
        if coord in self._puzzle_stage_lookup:
            return True
        if self.graph is not None and self.room_to_node:
            room = self._get_room_for_position(coord)
            node = self.room_to_node.get(room) if room is not None else None
            if node is not None and node in self.graph:
                edges = (
                    self.graph.out_edges(node, data=True)
                    if bool(self.graph.is_directed())
                    else self.graph.edges(node, data=True)
                )
                for _u, _v, edata in edges:
                    if "soft_locked" in edge_constraints_from_data(edata):
                        return True
        return False

    def _build_puzzle_plan_cache(self) -> None:
        """Normalize stitched puzzle-plan metadata into fast lookup tables."""
        self._puzzle_plans = {}
        self._puzzle_stage_lookup = defaultdict(list)
        self._puzzle_door_lookup = {}
        self._puzzle_stage_counts = {}

        payload = dict(self.room_puzzle_metadata or {})
        raw_plans = payload.get("plans", payload)
        if not isinstance(raw_plans, Mapping):
            return

        for raw_plan_id, raw_plan in raw_plans.items():
            if not isinstance(raw_plan, Mapping):
                continue
            controlled_doors = [
                (int(door[0]), int(door[1]))
                for door in list(raw_plan.get("controlled_doors_global", raw_plan.get("controlled_doors", [])) or [])
                if isinstance(door, (list, tuple)) and len(door) == 2
            ]
            raw_stages = list(raw_plan.get("stage_sequence", []) or [])
            plan_node = None
            if self.graph is not None and self.room_to_node:
                for stage in raw_stages:
                    if not isinstance(stage, Mapping):
                        continue
                    anchor = stage.get("global_anchor", stage.get("anchor", stage.get("local_anchor")))
                    if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
                        room = self._get_room_for_position((int(anchor[0]), int(anchor[1])))
                        if room is not None and room in self.room_to_node:
                            plan_node = self.room_to_node[room]
                            break
            if plan_node is None and "room_id" in raw_plan:
                plan_node = raw_plan.get("room_id")
            gates_switch = False
            if self.graph is not None and plan_node is not None and plan_node in self.graph:
                providers = switch_provider_ids(plan_node, self.graph.nodes[plan_node])
                gates_switch = bool(providers) and (
                    self._required_switch_ids is None or bool(providers & self._required_switch_ids)
                )
            if not controlled_doors and not gates_switch:
                continue

            plan_id = str(raw_plan.get("plan_id", raw_plan_id))
            normalized_stages: List[Dict[str, Any]] = []
            for stage in raw_stages:
                if not isinstance(stage, Mapping):
                    continue
                anchor = stage.get("global_anchor", stage.get("anchor", stage.get("local_anchor")))
                if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
                    continue
                stage_index = int(stage.get("stage_index", len(normalized_stages)))
                normalized = {
                    "plan_id": plan_id,
                    "stage_index": stage_index,
                    "name": str(stage.get("name", "")),
                    "kind": str(stage.get("kind", "step_on_puzzle")),
                    "anchor": (int(anchor[0]), int(anchor[1])),
                    "trigger_tile_id": (
                        int(stage.get("trigger_tile_id"))
                        if stage.get("trigger_tile_id") is not None
                        else None
                    ),
                }
                normalized_stages.append(normalized)
                self._puzzle_stage_lookup[normalized["anchor"]].append(normalized)

            self._puzzle_plans[plan_id] = {
                **dict(raw_plan),
                "plan_id": plan_id,
                "stage_sequence": normalized_stages,
            }
            self._puzzle_stage_counts[plan_id] = len(normalized_stages)

            for door_pos in controlled_doors:
                self._puzzle_door_lookup[door_pos] = plan_id

    def _prior_puzzle_stages_complete(self, state: GameState, plan_id: str, stage_index: int) -> bool:
        for prior_idx in range(int(stage_index)):
            if (str(plan_id), int(prior_idx)) not in state.completed_puzzle_stages:
                return False
        return True

    def _is_puzzle_plan_complete(self, state: GameState, plan_id: str) -> bool:
        total = int(self._puzzle_stage_counts.get(str(plan_id), 0))
        if total <= 0:
            return True
        for stage_index in range(total):
            if (str(plan_id), int(stage_index)) not in state.completed_puzzle_stages:
                return False
        return True

    def _complete_puzzle_stage(
        self,
        state: GameState,
        *,
        plan_id: str,
        stage_index: int,
    ) -> None:
        state.completed_puzzle_stages = set(state.completed_puzzle_stages) | {
            (str(plan_id), int(stage_index))
        }

    def _update_puzzle_stage_progress(
        self,
        state: GameState,
        *,
        target_pos: Tuple[int, int],
        target_tile: int,
        pushed_block_to: Optional[Tuple[int, int]] = None,
    ) -> GameState:
        """Advance any staged puzzle conditions satisfied by the new state."""
        anchors_to_check: List[Tuple[int, int]] = [tuple(target_pos)]
        if pushed_block_to is not None:
            anchors_to_check.append((int(pushed_block_to[0]), int(pushed_block_to[1])))

        for anchor in anchors_to_check:
            for stage in list(self._puzzle_stage_lookup.get(anchor, []) or []):
                plan_id = str(stage.get("plan_id", ""))
                stage_index = int(stage.get("stage_index", 0))
                if (plan_id, stage_index) in state.completed_puzzle_stages:
                    continue
                if not self._prior_puzzle_stages_complete(state, plan_id, stage_index):
                    continue

                kind = str(stage.get("kind", "step_on_puzzle")).strip().lower()
                trigger_tile_id = stage.get("trigger_tile_id")
                matched = False
                if kind == "collect_key":
                    matched = (
                        anchor == tuple(target_pos)
                        and int(target_tile) in {SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE['KEY_BOSS']}
                    )
                elif kind == "collect_item":
                    matched = (
                        anchor == tuple(target_pos)
                        and int(target_tile) in {SEMANTIC_PALETTE['KEY_ITEM'], SEMANTIC_PALETTE['ITEM_MINOR'], SEMANTIC_PALETTE['STAIR']}
                    )
                elif kind == "defeat_enemy":
                    matched = (
                        anchor == tuple(target_pos)
                        and int(target_tile) in {SEMANTIC_PALETTE['ENEMY'], SEMANTIC_PALETTE['BOSS']}
                    )
                elif kind == "push_block_to_switch":
                    matched = pushed_block_to is not None and tuple(anchor) == tuple(pushed_block_to)
                else:
                    matched = anchor == tuple(target_pos)
                    if trigger_tile_id is not None:
                        matched = matched and int(target_tile) == int(trigger_tile_id)

                if matched:
                    self._complete_puzzle_stage(state, plan_id=plan_id, stage_index=stage_index)

        if int(target_tile) == SEMANTIC_PALETTE['PUZZLE'] and self.graph is not None:
            room = self._get_room_for_position(target_pos)
            node = (self.room_to_node or {}).get(room)
            if node in self.graph:
                plans = {
                    str(stage['plan_id'])
                    for anchor, stages in self._puzzle_stage_lookup.items()
                    if self._get_room_for_position(anchor) == room
                    for stage in stages
                }
                if all(self._is_puzzle_plan_complete(state, plan) for plan in plans):
                    providers = switch_provider_ids(node, self.graph.nodes[node])
                    if self._required_switch_ids is None or providers & self._required_switch_ids:
                        state.switches_activated.update(providers)
        return state

    def _can_pass_puzzle_door(self, state: GameState, target_pos: Tuple[int, int]) -> bool:
        """Return True when a puzzle door is open under the stitched puzzle plan."""
        if target_pos in state.opened_doors:
            return True

        plan_id = self._puzzle_door_lookup.get(tuple(target_pos))
        if plan_id is not None:
            return self._is_puzzle_plan_complete(state, plan_id)

        if self.strict_original_mode and not self._can_pass_soft_door(state, target_pos):
            return False
        return True

    def _can_pass_soft_door(self, state: GameState, target_pos: Tuple[int, int]) -> bool:
        """
        Strict-original shutter rule:
        leaving a room via soft door requires clearing current room enemies.
        """
        if not self.strict_original_mode:
            return True

        current_room = self._get_room_for_position(state.position)
        target_room = self._get_room_for_position(target_pos)
        if current_room is None or target_room is None:
            return True
        if current_room == target_room:
            return True
        return self._is_room_cleared(current_room, state)

    def reset(self) -> GameState:
        """Reset the environment to initial state."""
        self.grid = self.original_grid.copy()
        self._room_positions_stamp = None
        self._pos_room_cache.clear()
        # Use solver_options for configurable starting inventory
        self.state = GameState(
            position=self.start_pos if self.start_pos else (0, 0),
            keys=self.solver_options.start_keys,
            bomb_count=self.solver_options.start_bombs,
            has_boss_key=self.solver_options.start_boss_key,
            has_item=self.solver_options.start_item,
            item_names={"*"} if self.solver_options.start_item else set(),
            current_floor=self.floor_for_position(
                self.start_pos if self.start_pos else (0, 0),
                default=0,
            ),
        )
        self.done = False
        self.won = False
        self.step_count = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            state: New game state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            return self.state.copy(), 0.0, True, {'msg': 'Episode already done'}

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            return self.state.copy(), -100.0, True, {'msg': 'Max steps exceeded'}

        # Get movement delta
        dr, dc = ACTION_DELTAS.get(Action(action), (0, 0))
        current_r, current_c = self.state.position
        new_r, new_c = current_r + dr, current_c + dc

        # Check bounds
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            return self.state.copy(), -1.0, False, {'msg': 'Out of bounds'}

        target_tile = self.grid[new_r, new_c]
        info = {'msg': ''}
        reward = -0.1  # Small step penalty

        # Check if movement is possible
        can_move, new_state, step_reward, step_info = self._try_move(
            (new_r, new_c), target_tile
        )

        if can_move:
            self.state = new_state
            reward += step_reward
            info.update(step_info)

            # Check win condition
            if target_tile == SEMANTIC_PALETTE['TRIFORCE']:
                self.done = True
                self.won = True
                reward = 100.0
                info['msg'] = 'Victory!'
        else:
            reward = -1.0
            info['msg'] = step_info.get('msg', 'Blocked')

        return self.state.copy(), reward, self.done, info

    def _try_move(self, target_pos: Tuple[int, int], target_tile: int
                 ) -> Tuple[bool, GameState, float, Dict]:
        """
        Attempt to move to target position.
        
        Returns:
            can_move: Whether movement is possible
            new_state: Updated state if movement succeeds
            reward: Reward for this action
            info: Additional information
        """
        if self.graph_bridge is not None:
            allowed, candidate = self._try_move_pure(
                self.state, target_pos, int(self.original_grid[target_pos]),
            )
            return allowed, candidate, 0.0, {'msg': 'Moved' if allowed else 'Blocked by mission state'}
        new_state = self.state.copy()
        reward = 0.0
        info = {}

        # Blocking tiles - cannot pass
        if target_tile in BLOCKING_IDS:
            return False, self.state, 0.0, {'msg': 'Blocked by wall'}

        # Strict-original shutter door semantics.
        if target_tile == SEMANTIC_PALETTE['DOOR_SOFT']:
            if not self._can_pass_soft_door(self.state, target_pos):
                return False, self.state, 0.0, {'msg': 'Shutter door closed - clear enemies first'}

        # Walkable tiles - free movement
        if target_tile in WALKABLE_IDS:
            new_state.position = target_pos

            if (
                self.strict_original_mode
                and target_tile in {SEMANTIC_PALETTE['ENEMY'], SEMANTIC_PALETTE['BOSS']}
                and self._is_enemy_progression_relevant(target_pos)
            ):
                new_state.defeated_enemies = self.state.defeated_enemies | {target_pos}

            # Handle item pickup
            if target_tile in PICKUP_IDS and target_pos not in new_state.collected_items:
                new_state, pickup_reward, pickup_info = self._pickup_item(
                    new_state, target_pos, target_tile
                )
                reward += pickup_reward
                info.update(pickup_info)

            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )

            return True, new_state, reward, info

        # Conditional tiles - require inventory items
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if target_pos in new_state.opened_doors:
                new_state.position = target_pos
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 0.0, {'msg': 'Door already open'}
            elif new_state.keys > 0:
                new_state.keys -= 1
                new_state.opened_doors.add(target_pos)
                new_state.position = target_pos
                # Update grid to show door is open
                self.grid[target_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 10.0, {'msg': 'Unlocked door with key'}
            else:
                return False, self.state, 0.0, {'msg': 'Door locked - need key'}

        if target_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
            if target_pos in new_state.opened_doors:
                new_state.position = target_pos
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 0.0, {'msg': 'Wall already bombed'}
            elif new_state.bomb_count > 0:
                new_state.bomb_count -= 1  # Consume one bomb
                new_state.opened_doors.add(target_pos)
                new_state.position = target_pos
                self.grid[target_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 10.0, {'msg': 'Bombed wall'}
            else:
                return False, self.state, 0.0, {'msg': 'Need bombs'}

        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if target_pos in new_state.opened_doors:
                new_state.position = target_pos
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 0.0, {'msg': 'Boss door already open'}
            elif new_state.has_boss_key:
                new_state.opened_doors.add(target_pos)
                new_state.position = target_pos
                self.grid[target_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 20.0, {'msg': 'Opened boss door'}
            else:
                return False, self.state, 0.0, {'msg': 'Need boss key'}

        if target_tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
            if not self._can_pass_puzzle_door(new_state, target_pos):
                return False, self.state, 0.0, {'msg': 'Puzzle door closed - staged puzzle incomplete'}
            new_state.opened_doors.add(target_pos)
            new_state.position = target_pos
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state, 0.0, {'msg': 'Passed puzzle door'}

        # ELEMENT (water/lava) - needs KEY_ITEM (Ladder) to cross.
        # Without the Ladder the tile is impassable.
        if target_tile == SEMANTIC_PALETTE['ELEMENT']:
            if new_state.has_item:
                new_state.position = target_pos
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state, 0.0, {'msg': 'Crossed water/lava with Ladder'}
            return False, self.state, 0.0, {'msg': 'Need Ladder to cross water/lava'}

        # BLOCK - pushable tile.  Attempt to push in the direction of movement.
        if target_tile == SEMANTIC_PALETTE['BLOCK']:
            dr = target_pos[0] - new_state.position[0]
            dc = target_pos[1] - new_state.position[1]
            push_dest_r = target_pos[0] + dr
            push_dest_c = target_pos[1] + dc

            # Bounds check
            if not (0 <= push_dest_r < self.height and 0 <= push_dest_c < self.width):
                return False, self.state, 0.0, {'msg': 'Cannot push block off map'}

            push_dest = (push_dest_r, push_dest_c)
            push_dest_tile = int(self.grid[push_dest_r, push_dest_c])

            if not is_push_destination_available(new_state, push_dest, push_dest_tile):
                return False, self.state, 0.0, {'msg': 'Cannot push block - destination blocked'}

            fills_bridge = (
                int(push_dest_tile) in BRIDGE_FILL_IDS
                and push_dest not in new_state.bridged_tiles
            )
            # Move block in the grid (mutable environment). Pushing into an
            # ELEMENT tile fills it rather than leaving a blocking block there.
            self.grid[push_dest_r, push_dest_c] = (
                SEMANTIC_PALETTE['ELEMENT_FLOOR'] if fills_bridge else SEMANTIC_PALETTE['BLOCK']
            )
            exposed_tile = self._underlay_tile_for_block_origin(target_pos)
            self.grid[target_pos[0], target_pos[1]] = exposed_tile

            # Track push in state
            if fills_bridge:
                new_state.filled_block_origins = new_state.filled_block_origins | {target_pos}
                new_state.bridged_tiles = new_state.bridged_tiles | {push_dest}
            else:
                new_state.pushed_blocks = new_state.pushed_blocks | {(target_pos, push_dest)}
            new_state.position = target_pos

            # Bug #3 fix: if a pickup item was at the player's new position *before*
            # the push moved a block off it, we may have just exposed a collectible
            # that was underneath.  More importantly, check whether the tile the block
            # was pushed *from* (target_pos) was previously a pickup - this cannot
            # happen because the grid showed BLOCK there.  However, if the player
            # stepped onto target_pos after the grid was updated to FLOOR, check for
            # items there (e.g. a key that was already on the floor before the block).
            new_state, pickup_reward, pickup_info = self._apply_pickup_if_present(
                new_state,
                target_pos,
                exposed_tile,
                mutate_grid=True,
            )
            reward += pickup_reward
            info.update(pickup_info)

            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
                pushed_block_to=(int(push_dest_r), int(push_dest_c)),
            )
            return True, new_state, reward, info

        # Default: allow movement (handles FLOOR, STAIR, etc. not caught above)
        new_state.position = target_pos
        return True, new_state, 0.0, info

    def step_graph_transition(self, target_pos: Tuple[int, int]):
        """Interact with a recorded stair endpoint using the same pure rules as search."""
        if self.done:
            return self.state.copy(), 0.0, True, {'msg': 'Episode already done'}
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            return self.state.copy(), -100.0, True, {'msg': 'Max steps exceeded'}
        target_pos = tuple(target_pos)
        if self.graph_bridge is None or target_pos not in self.graph_bridge.destinations(self.state.position):
            return self.state.copy(), -1.0, False, {'msg': 'No recorded transition'}
        allowed, candidate = self.try_move_pure(self.state, target_pos, int(self.original_grid[target_pos]))
        if allowed:
            candidate.current_floor = self.floor_for_position(target_pos, default=self.state.current_floor)
            self.state = candidate
        return self.state.copy(), -0.1 if allowed else -1.0, False, {'msg': 'Transitioned' if allowed else 'Gate closed'}

    def _pickup_item(self, state: GameState, pos: Tuple[int, int], tile: int
                    ) -> Tuple[GameState, float, Dict]:
        """Handle item pickup."""
        state.collected_items.add(pos)

        if tile == SEMANTIC_PALETTE['KEY_SMALL']:
            state.keys += 1
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 5.0, {'msg': 'Picked up key', 'item': 'key'}

        if tile == SEMANTIC_PALETTE['KEY_BOSS']:
            state.has_boss_key = True
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 15.0, {'msg': 'Picked up boss key', 'item': 'boss_key'}

        if tile == SEMANTIC_PALETTE['KEY_ITEM']:
            state.has_item = True
            state.item_names.update(self._item_names_for_position(pos))
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 10.0, {'msg': 'Picked up key item', 'item': 'key_item'}

        if tile == SEMANTIC_PALETTE['ITEM_MINOR']:
            # ITEM_MINOR represents bomb pickups in VGLC Zelda dungeons
            # Without this, dungeons where bombs are behind bombable walls
            # become unsolvable (KEY_ITEM often inaccessible initially)
            state.bomb_count += 4  # Consumable: add 4 bombs
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 1.0, {'msg': 'Picked up bomb', 'item': 'bomb'}

        return state, 0.0, {}

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current state."""
        if self.graph_bridge is not None:
            valid = []
            row, col = self.state.position
            for action, (dr, dc) in ACTION_DELTAS.items():
                target = (row + dr, col + dc)
                if 0 <= target[0] < self.height and 0 <= target[1] < self.width:
                    allowed, _ = self._try_move_pure(self.state, target, int(self.original_grid[target]))
                    if allowed:
                        valid.append(int(action))
            return valid
        valid = []
        r, c = self.state.position

        for action in Action:
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.height and 0 <= nc < self.width:
                tile = self.grid[nr, nc]
                if tile not in BLOCKING_IDS:
                    # Check if we can actually pass this tile
                    if tile in WALKABLE_IDS:
                        if tile == SEMANTIC_PALETTE['DOOR_SOFT'] and not self._can_pass_soft_door(self.state, (nr, nc)):
                            continue
                        valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
                        if self.state.keys > 0 or (nr, nc) in self.state.opened_doors:
                            valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_BOMB']:
                        if self.state.bomb_count > 0 or (nr, nc) in self.state.opened_doors:
                            valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_BOSS']:
                        if self.state.has_boss_key or (nr, nc) in self.state.opened_doors:
                            valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
                        if self._can_pass_puzzle_door(self.state, (nr, nc)):
                            valid.append(int(action))
                    else:
                        valid.append(int(action))

        return valid

    def _try_move_pure(self, state: GameState, target_pos: Tuple[int, int],
                       target_tile: int) -> Tuple[bool, GameState]:
        original_state = state
        if self.graph_bridge is not None:
            allowed, state, owns_gate = self.graph_bridge.transition(
                state, target_pos, copy_on_noop=False
            )
            if not allowed:
                return False, original_state
            if owns_gate and target_tile in (
                CONDITIONAL_IDS | {SEMANTIC_PALETTE['DOOR_SOFT'], SEMANTIC_PALETTE['ELEMENT']}
            ):
                # Recorded graph gates own their inventory cost across the
                # entire aperture; individual door cells must not charge again.
                target_tile = SEMANTIC_PALETTE['DOOR_OPEN']
        allowed, candidate = self._try_move_tiles(state, target_pos, target_tile)
        return (True, candidate) if allowed else (False, original_state)

    def _try_move_tiles(self, state: GameState, target_pos: Tuple[int, int],
                        target_tile: int) -> Tuple[bool, GameState]:
        """
        Pure state-based move attempt (no grid modifications).
        
        This method is used by search algorithms (D* Lite, Bidirectional A*)
        that need to explore state transitions without modifying the environment.
        
        Args:
            state: Current game state
            target_pos: Target position (r, c)
            target_tile: Semantic ID at target position
            
        Returns:
            can_move: Whether the move is valid
            new_state: Updated state if move is valid
        """
        # Blocking tiles - cannot pass
        if target_tile in BLOCKING_IDS:
            return False, state

        # Strict-original shutter door semantics.
        if target_tile == SEMANTIC_PALETTE['DOOR_SOFT']:
            if not self._can_pass_soft_door(state, target_pos):
                return False, state

        new_state = state.copy()
        new_state.position = target_pos

        # Handle special tiles based on STATE, not grid modifications
        if target_pos in state.bridged_tiles and int(target_tile) in BRIDGE_FILL_IDS:
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(SEMANTIC_PALETTE['ELEMENT_FLOOR']),
            )
            return True, new_state

        # Check if this door was already opened (in state)
        if target_pos in state.opened_doors:
            # Door is open, can pass freely
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state

        # Check if this item was already collected (in state)
        if target_pos in state.collected_items:
            # Item already collected, treat as floor
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state

        # Dynamic occupancy wins over the immutable grid. A block may currently
        # occupy the original position vacated by another block.
        for (from_pos, to_pos) in state.pushed_blocks:
            if to_pos == target_pos:
                # There's a pushed block here! Need to try pushing it further
                # Calculate direction of push
                dr = target_pos[0] - state.position[0]
                dc = target_pos[1] - state.position[1]
                push_dest_r = target_pos[0] + dr
                push_dest_c = target_pos[1] + dc

                # Check bounds
                if not (0 <= push_dest_r < self.height and 0 <= push_dest_c < self.width):
                    return False, state  # Can't push off map

                # Check destination - but also check if another block is there!
                push_dest_tile = self.grid[push_dest_r, push_dest_c]
                push_dest = (push_dest_r, push_dest_c)

                if is_push_destination_available(state, push_dest, int(push_dest_tile)):
                    # Can push - update pushed_blocks
                    # CRITICAL: Preserve ORIGINAL from_pos to keep track of empty positions!
                    new_pushed = set()
                    filled_origin: Optional[Tuple[int, int]] = None
                    fills_bridge = (
                        int(push_dest_tile) in BRIDGE_FILL_IDS
                        and push_dest not in state.bridged_tiles
                    )
                    for (fp, tp) in state.pushed_blocks:
                        if tp == target_pos:
                            if fills_bridge:
                                filled_origin = fp
                            else:
                                # Keep original from_pos, update destination to new position
                                new_pushed.add((fp, push_dest))
                        else:
                            new_pushed.add((fp, tp))
                    # Use set (not frozenset) to maintain consistency with GameState.copy()
                    new_state.pushed_blocks = new_pushed
                    if fills_bridge:
                        if filled_origin is not None:
                            new_state.filled_block_origins = state.filled_block_origins | {filled_origin}
                        new_state.bridged_tiles = state.bridged_tiles | {push_dest}
                    # Trigger puzzle progression for re-pushed blocks (was missing).
                    new_state = self._update_puzzle_stage_progress(
                        new_state,
                        target_pos=target_pos,
                        target_tile=int(self.grid[target_pos[0], target_pos[1]]),
                        pushed_block_to=(int(push_dest_r), int(push_dest_c)),
                    )
                    return True, new_state
                else:
                    return False, state  # Can't push

        # A static block origin is floor after its block has moved away.
        if was_block_vacated(state, target_pos):
            underlay_tile = self._underlay_tile_for_block_origin(target_pos)
            new_state, _pickup_reward, _pickup_info = self._apply_pickup_if_present(
                new_state,
                target_pos,
                underlay_tile,
                mutate_grid=False,
            )
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(underlay_tile),
            )
            return True, new_state

        # Walkable tiles - free movement
        if target_tile in WALKABLE_IDS:
            if (
                (self.strict_original_mode or self.graph_bridge is not None)
                and target_tile in {SEMANTIC_PALETTE['ENEMY'], SEMANTIC_PALETTE['BOSS']}
                and self._is_enemy_progression_relevant(target_pos)
            ):
                new_state.defeated_enemies = state.defeated_enemies | {target_pos}
            # Handle item pickup (add to collected_items)
            if target_tile in PICKUP_IDS:
                new_state.collected_items = state.collected_items | {target_pos}

                if target_tile == SEMANTIC_PALETTE['KEY_SMALL']:
                    new_state.keys = state.keys + 1
                elif target_tile == SEMANTIC_PALETTE['KEY_BOSS']:
                    new_state.has_boss_key = True
                elif target_tile == SEMANTIC_PALETTE['KEY_ITEM']:
                    new_state.has_item = True
                    new_state.item_names.update(
                        self._item_names_for_position(target_pos)
                    )
                elif target_tile == SEMANTIC_PALETTE['ITEM_MINOR']:
                    # ITEM_MINOR represents bomb pickups in VGLC Zelda dungeons
                    new_state.bomb_count = state.bomb_count + 4  # Consumable: add 4 bombs
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state

        # Conditional tiles - require inventory items
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if state.keys > 0:
                new_state.keys = state.keys - 1
                new_state.opened_doors = state.opened_doors | {target_pos}
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state
            return False, state

        if target_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
            if state.bomb_count > 0:
                new_state.bomb_count = state.bomb_count - 1  # Consume one bomb
                new_state.opened_doors = state.opened_doors | {target_pos}
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state
            return False, state

        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if state.has_boss_key:
                new_state.opened_doors = state.opened_doors | {target_pos}
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state
            return False, state

        if target_tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
            if not self._can_pass_puzzle_door(state, target_pos):
                return False, state
            new_state.opened_doors = state.opened_doors | {target_pos}
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state

        if target_tile == SEMANTIC_PALETTE['DOOR_OPEN']:
            # Already open door
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state

        if target_tile == SEMANTIC_PALETTE['DOOR_SOFT']:
            # One-way door - can pass
            new_state = self._update_puzzle_stage_progress(
                new_state,
                target_pos=target_pos,
                target_tile=int(target_tile),
            )
            return True, new_state

        # ELEMENT (water/lava) - needs KEY_ITEM (Ladder) to cross.
        # The Stepladder acts as a 1-tile bridge: the agent can step onto
        # ONE ELEMENT tile but cannot continue onto a second consecutive
        # ELEMENT tile (no unlimited swimming).
        if target_tile == SEMANTIC_PALETTE['ELEMENT']:
            if state.has_item:
                # Enforce 1-tile bridge: disallow ELEMENT -> ELEMENT movement.
                cur_r, cur_c = state.position
                if (0 <= cur_r < self.height and 0 <= cur_c < self.width
                        and int(self.grid[cur_r, cur_c]) == SEMANTIC_PALETTE['ELEMENT']):
                    return False, state
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(target_tile),
                )
                return True, new_state
            return False, state

        # BLOCK tiles - pushable
        if target_tile == SEMANTIC_PALETTE['BLOCK']:
            # Calculate push direction
            dr = target_pos[0] - state.position[0]
            dc = target_pos[1] - state.position[1]
            push_dest_r = target_pos[0] + dr
            push_dest_c = target_pos[1] + dc

            # Check bounds
            if not (0 <= push_dest_r < self.height and 0 <= push_dest_c < self.width):
                return False, state

            # Check if destination is walkable
            push_dest_tile = self.grid[push_dest_r, push_dest_c]

            push_dest = (push_dest_r, push_dest_c)

            if is_push_destination_available(state, push_dest, int(push_dest_tile)):
                fills_bridge = (
                    int(push_dest_tile) in BRIDGE_FILL_IDS
                    and push_dest not in state.bridged_tiles
                )
                if fills_bridge:
                    new_state.filled_block_origins = state.filled_block_origins | {target_pos}
                    new_state.bridged_tiles = state.bridged_tiles | {push_dest}
                else:
                    # Can push - record dynamic block occupancy.
                    new_state.pushed_blocks = state.pushed_blocks | {(target_pos, push_dest)}
                underlay_tile = self._underlay_tile_for_block_origin(target_pos)
                new_state, _pickup_reward, _pickup_info = self._apply_pickup_if_present(
                    new_state,
                    target_pos,
                    underlay_tile,
                    mutate_grid=False,
                )
                new_state = self._update_puzzle_stage_progress(
                    new_state,
                    target_pos=target_pos,
                    target_tile=int(underlay_tile if underlay_tile in PICKUP_IDS else target_tile),
                    pushed_block_to=(int(push_dest_r), int(push_dest_c)),
                )
                return True, new_state
            else:
                return False, state

        # Default: treat as walkable
        return True, new_state

    # ==========================================
    # RENDERING (OPTIONAL - PYGAME)
    # ==========================================

    def _init_render(self):
        """Initialize Pygame rendering."""
        _validator_rendering.init_render(self)

    def _load_images(self):
        """Load tile images or create colored fallbacks."""
        _validator_rendering.load_images(self)

    def render(self):
        """Render current state to screen."""
        _validator_rendering.render(self)

    def close(self):
        """Clean up resources."""
        _validator_rendering.close(self)


