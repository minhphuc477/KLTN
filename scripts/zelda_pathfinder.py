"""
Zelda A* Pathfinder - Topology-Aware State-Space Search
========================================================

Advanced pathfinding for The Legend of Zelda (NES) dungeons that properly
handles game mechanics:
- Key collection and consumption
- Locked doors that stay open permanently
- Boss keys and special items
- One-way doors and teleportation stairs
- Backtracking and exploration

Algorithm: A* with state-space search
State: (room_position, inventory_state)
Heuristic: Manhattan distance + key deficit penalty

Author: KLTN Thesis Project
Date: January 19, 2026
"""

import heapq
import logging
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

from Data.zelda_core import (
    Dungeon, Room, InventoryState,
    SEMANTIC_PALETTE, ValidationMode
)


# ==========================================
# SEARCH STATE DEFINITION
# ==========================================
@dataclass
class SearchState:
    """
    State in A* search: room position + inventory + path costs.
    
    This represents a complete game state at a point in time:
    - Current room position in dungeon
    - Keys held and collected
    - Which doors have been unlocked
    - Cost to reach this state (g_cost)
    - Estimated cost to goal (h_cost)
    """
    room: Tuple[int, int]           # (row, col) position
    inventory: InventoryState       # Keys, items, opened doors
    g_cost: int                     # Actual cost from start
    h_cost: float                   # Heuristic estimate to goal
    parent: Optional['SearchState'] = None  # Previous state for path reconstruction
    action_taken: str = 'start'     # Action that led to this state
    
    def f_cost(self) -> float:
        """Total estimated cost: f(n) = g(n) + h(n)"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        """Comparison for heap: prefer lower f_cost."""
        if self.f_cost() != other.f_cost():
            return self.f_cost() < other.f_cost()
        # Tie-breaker: prefer higher g_cost (closer to goal)
        return self.g_cost > other.g_cost
    
    def __hash__(self):
        """Hash for visited set: (room, inventory_hash)"""
        return hash((self.room, hash(self.inventory)))
    
    def __eq__(self, other):
        """Equality check for visited set."""
        return self.room == other.room and self.inventory == other.inventory


# ==========================================
# A* PATHFINDER WITH STATE-SPACE SEARCH
# ==========================================
class ZeldaPathfinder:
    """
    A* pathfinding solver for Zelda dungeons with inventory tracking.
    
    This solver models authentic NES Zelda mechanics:
    - Small keys are consumable (one-time use)
    - Locked doors stay open permanently after unlocking
    - Boss keys are required for boss doors
    - Stairs teleport between rooms
    - Bombable walls are permanent
    
    The solver uses A* with state-space search where each state
    tracks both position and inventory, allowing it to properly
    handle backtracking and key collection strategies.
    """
    
    def __init__(self, dungeon: Dungeon, mode: str = ValidationMode.FULL, admissible_heuristic: bool = True):
        """
        Initialize pathfinder.
        
        Args:
            dungeon: Dungeon object with graph and rooms
            mode: ValidationMode (STRICT, REALISTIC, FULL)
            admissible_heuristic: If True, use strictly admissible heuristic (no key deficit penalty).
        """
        self.dungeon = dungeon
        self.graph = dungeon.graph
        self.rooms = dungeon.rooms
        self.mode = mode
        # Heuristic mode: keep admissible by default to guarantee optimality
        self.admissible_heuristic = bool(admissible_heuristic)
        
        # Extract dungeon metadata
        self.start_pos = dungeon.start_pos
        self.goal_pos = dungeon.triforce_pos
        
        # Build room-to-node mapping
        self.room_to_node = {}
        self.node_to_room = {}
        self._build_room_node_mapping()
        
        # Identify key rooms and locked doors
        self.key_rooms: Set[Tuple[int, int]] = set()
        self.boss_key_rooms: Set[Tuple[int, int]] = set()
        self.locked_doors: Dict[Tuple[Tuple[int,int], Tuple[int,int]], str] = {}
        self._identify_special_rooms()
        
        # Statistics
        self.states_explored = 0
        self.states_generated = 0
        self.max_queue_size = 0
        
    def solve(self) -> Dict:
        """
        Find optimal path from start to triforce using A*.
        
        Returns:
            Dict with:
                - solvable: bool
                - path: List[Tuple[int, int]] (room positions)
                - actions: List[str] (move, collect_key, use_key, etc.)
                - path_length: int
                - keys_collected: int
                - keys_used: int
                - final_inventory: InventoryState
                - stats: Dict (states_explored, time, etc.)
        """
        if self.start_pos is None or self.goal_pos is None:
            return {
                'solvable': False,
                'reason': 'No start or goal position defined'
            }
        
        # Run A* search
        return self._a_star_search()
    
    def _a_star_search(self) -> Dict:
        """
        Core A* implementation with state-space search.
        
        Algorithm:
            1. Initialize open set with start state
            2. While open set not empty:
                a. Pop state with lowest f_cost
                b. Check if goal reached
                c. Expand neighbors (traverse edges)
                d. Add valid successors to open set
            3. Return solution or failure
        """
        import time
        start_time = time.time()
        
        # Initialize starting state
        initial_inventory = InventoryState()
        initial_inventory = self._collect_room_items(self.start_pos, initial_inventory)
        
        initial_state = SearchState(
            room=self.start_pos,
            inventory=initial_inventory,
            g_cost=0,
            h_cost=self._heuristic(self.start_pos, initial_inventory),
            parent=None,
            action_taken='start'
        )
        
        # Open set: priority queue (min-heap by f_cost)
        open_set = []
        counter = 0  # Tie-breaker for heap
        heapq.heappush(open_set, (initial_state.f_cost(), counter, initial_state))
        
        # Visited: (room, inventory_hash) -> best g_cost
        visited: Dict[Tuple, int] = {}
        
        # Statistics
        self.states_explored = 0
        self.states_generated = 1
        self.max_queue_size = 1
        
        # Main search loop
        while open_set:
            # Update stats
            self.max_queue_size = max(self.max_queue_size, len(open_set))
            
            # Pop state with lowest f_cost
            _, _, current_state = heapq.heappop(open_set)
            self.states_explored += 1
            
            # Goal test
            if current_state.room == self.goal_pos:
                elapsed = time.time() - start_time
                return self._reconstruct_solution(current_state, elapsed)
            
            # Get state hash for visited check
            state_hash = (current_state.room, hash(current_state.inventory))
            
            # Skip if we've found a better path to this state
            if state_hash in visited and visited[state_hash] <= current_state.g_cost:
                continue
            visited[state_hash] = current_state.g_cost
            
            # Expand neighbors
            successors = self._expand_state(current_state)
            
            for successor in successors:
                counter += 1
                self.states_generated += 1
                heapq.heappush(open_set, (successor.f_cost(), counter, successor))
        
        # No solution found
        elapsed = time.time() - start_time
        return {
            'solvable': False,
            'reason': 'No path exists with current constraints',
            'mode': self.mode,
            'stats': {
                'states_explored': self.states_explored,
                'states_generated': self.states_generated,
                'max_queue_size': self.max_queue_size,
                'time_elapsed': elapsed,
                'keys_found': len(self.key_rooms)
            }
        }
    
    def _expand_state(self, state: SearchState) -> List[SearchState]:
        """
        Generate all valid successor states from current state.
        
        Args:
            state: Current search state
            
        Returns:
            List of valid successor states
        """
        successors = []
        
        # Get neighbors from graph
        current_node = self.room_to_node.get(state.room)
        if current_node is None:
            return successors
        
        # Try each neighbor
        for neighbor_node in self.graph.neighbors(current_node):
            neighbor_room = self.node_to_room.get(neighbor_node)
            if neighbor_room is None:
                continue
            
            # Check if edge can be traversed
            can_traverse, new_inventory, edge_type = self._can_traverse_edge(
                state.room, neighbor_room, state.inventory
            )
            
            if not can_traverse:
                continue
            
            # Collect items in new room
            new_inventory = self._collect_room_items(neighbor_room, new_inventory)
            
            # Determine action taken
            action = self._describe_action(
                state.room, neighbor_room, edge_type,
                state.inventory, new_inventory
            )
            
            # Create successor state
            successor = SearchState(
                room=neighbor_room,
                inventory=new_inventory,
                g_cost=state.g_cost + 1,
                h_cost=self._heuristic(neighbor_room, new_inventory),
                parent=state,
                action_taken=action
            )
            
            successors.append(successor)
        
        return successors
    
    def _can_traverse_edge(self, from_room: Tuple[int, int], 
                          to_room: Tuple[int, int],
                          inventory: InventoryState) -> Tuple[bool, InventoryState, str]:
        """
        Check if an edge (door) can be traversed with current inventory.
        
        Args:
            from_room: Source room position
            to_room: Destination room position
            inventory: Current inventory state
            
        Returns:
            (can_traverse, new_inventory, edge_type)
        """
        # Get edge data from graph
        from_node = self.room_to_node.get(from_room)
        to_node = self.room_to_node.get(to_room)
        
        if from_node is None or to_node is None:
            return False, inventory, 'none'
        
        edge_data = self.graph.get_edge_data(from_node, to_node)
        if edge_data is None:
            return False, inventory, 'none'
        
        edge_type = edge_data.get('edge_type', 'open')
        edge_id = (from_room, to_room)
        
        # Copy inventory for modification
        new_inventory = inventory.copy()
        
        # Handle different door types
        if edge_type == 'open' or edge_type == '':
            # Normal door - always passable
            return True, new_inventory, 'open'
        
        elif edge_type == 'key_locked':
            # Key-locked door
            if edge_id in inventory.doors_opened:
                # Already unlocked
                return True, new_inventory, 'key_locked'
            
            if inventory.keys_held > 0:
                # Use a key to unlock
                new_inventory.keys_held -= 1
                new_inventory.doors_opened.add(edge_id)
                new_inventory.doors_opened.add((to_room, from_room))  # Bidirectional
                return True, new_inventory, 'key_locked'
            
            # Need a key but don't have one
            return False, inventory, 'key_locked'
        
        elif edge_type == 'bombable':
            # Bombable wall - assume infinite bombs
            if edge_id in inventory.doors_opened:
                return True, new_inventory, 'bombable'
            
            # Bomb it open (permanent)
            new_inventory.doors_opened.add(edge_id)
            new_inventory.doors_opened.add((to_room, from_room))
            return True, new_inventory, 'bombable'
        
        elif edge_type == 'soft_locked':
            # One-way door - always passable forward
            return True, new_inventory, 'soft_locked'
        
        elif edge_type == 'boss_locked':
            # Boss door - requires boss key
            if 'boss_key' in inventory.items_collected:
                return True, new_inventory, 'boss_locked'
            return False, inventory, 'boss_locked'
        
        elif edge_type == 'stair':
            # Stairs - always passable
            return True, new_inventory, 'stair'
        
        # Unknown edge type - allow by default
        return True, new_inventory, edge_type
    
    def _collect_room_items(self, room: Tuple[int, int], 
                           inventory: InventoryState) -> InventoryState:
        """
        Automatically collect keys/items when entering a room.
        
        This models the NES behavior where items are automatically
        collected when Link enters a room.
        
        Args:
            room: Room position
            inventory: Current inventory
            
        Returns:
            Updated inventory with collected items
        """
        new_inventory = inventory.copy()
        
        # Collect small key
        if room in self.key_rooms and room not in inventory.keys_collected:
            new_inventory.keys_held += 1
            new_inventory.keys_collected.add(room)
        
        # Collect boss key
        if room in self.boss_key_rooms and 'boss_key' not in inventory.items_collected:
            new_inventory.items_collected.add('boss_key')
        
        return new_inventory
    
    def _heuristic(self, room: Tuple[int, int], 
                   inventory: InventoryState) -> float:
        """
        Admissible heuristic function for A*.
        
        Components:
        1. Manhattan distance to goal (always admissible)
        2. Key deficit penalty (may be inadmissible but helps guide search)
        
        Args:
            room: Current room position
            inventory: Current inventory state
            
        Returns:
            Estimated cost to goal
        """
        # Component 1: Manhattan distance (always admissible)
        dx = abs(room[0] - self.goal_pos[0])
        dy = abs(room[1] - self.goal_pos[1])
        spatial_cost = dx + dy
        
        # Component 2: Key deficit estimate
        # Count locked doors between current room and goal (rough estimate)
        locked_doors_nearby = 0
        for (from_r, to_r), door_type in self.locked_doors.items():
            if door_type == 'key_locked' and (from_r, to_r) not in inventory.doors_opened:
                # Very rough estimate: is door "between" current and goal?
                if self._is_between(from_r, room, self.goal_pos):
                    locked_doors_nearby += 1
        
        keys_needed = max(0, locked_doors_nearby - inventory.keys_held)
        key_penalty = keys_needed * 1.5  # Weight if enabled
        
        # Component 3: Encourage key collection
        uncollected_keys = len(self.key_rooms - inventory.keys_collected)
        if uncollected_keys > 0 and inventory.keys_held == 0:
            # Slight bonus for exploring if we have no keys
            exploration_bonus = -0.5
        else:
            exploration_bonus = 0

        # If strict admissible heuristic requested, do NOT include the key penalty
        if self.admissible_heuristic:
            return spatial_cost + exploration_bonus
        return spatial_cost + key_penalty + exploration_bonus
    
    def _is_between(self, point: Tuple[int, int], 
                   start: Tuple[int, int], 
                   end: Tuple[int, int]) -> bool:
        """
        Rough check if point is "between" start and end.
        
        Used for heuristic estimation only.
        """
        min_r = min(start[0], end[0])
        max_r = max(start[0], end[0])
        min_c = min(start[1], end[1])
        max_c = max(start[1], end[1])
        
        return min_r <= point[0] <= max_r and min_c <= point[1] <= max_c
    
    def _describe_action(self, from_room: Tuple[int, int], 
                        to_room: Tuple[int, int],
                        edge_type: str,
                        old_inventory: InventoryState,
                        new_inventory: InventoryState) -> str:
        """
        Generate human-readable action description.
        
        Args:
            from_room: Source room
            to_room: Destination room
            edge_type: Type of edge traversed
            old_inventory: Inventory before move
            new_inventory: Inventory after move
            
        Returns:
            Action description string
        """
        # Check if key was used
        if new_inventory.keys_held < old_inventory.keys_held:
            return f'use_key to {to_room}'
        
        # Check if key was collected
        if to_room in new_inventory.keys_collected and to_room not in old_inventory.keys_collected:
            return f'collect_key at {to_room}'
        
        # Check if boss key was collected
        if 'boss_key' in new_inventory.items_collected and 'boss_key' not in old_inventory.items_collected:
            return f'collect_boss_key at {to_room}'
        
        # Check edge type
        if edge_type == 'bombable':
            return f'bomb_wall to {to_room}'
        elif edge_type == 'stair':
            return f'take_stairs to {to_room}'
        elif edge_type == 'soft_locked':
            return f'one_way_door to {to_room}'
        else:
            return f'move to {to_room}'
    
    def _reconstruct_solution(self, final_state: SearchState, elapsed: float) -> Dict:
        """
        Backtrack from goal to start to reconstruct solution path.
        
        Args:
            final_state: Goal state
            elapsed: Time taken for search
            
        Returns:
            Solution dictionary
        """
        # Backtrack to build path
        path = []
        actions = []
        state = final_state
        
        while state is not None:
            path.append(state.room)
            actions.append(state.action_taken)
            state = state.parent
        
        # Reverse to get start->goal order
        path.reverse()
        actions.reverse()
        
        # Count keys used
        keys_used = sum(1 for a in actions if 'use_key' in a)
        
        return {
            'solvable': True,
            'path': path,
            'actions': actions,
            'path_length': len(path) - 1,
            'rooms_traversed': len(path),
            'keys_collected': len(final_state.inventory.keys_collected),
            'keys_used': keys_used,
            'final_inventory': final_state.inventory,
            'stats': {
                'states_explored': self.states_explored,
                'states_generated': self.states_generated,
                'max_queue_size': self.max_queue_size,
                'time_elapsed': elapsed,
                'average_branching_factor': self.states_generated / max(1, self.states_explored)
            }
        }
    
    def _build_room_node_mapping(self):
        """Build mapping between room positions and graph node IDs."""
        if not self.graph or len(self.graph.nodes) == 0:
            return
        
        # Find start and goal nodes
        start_node = None
        goal_node = None
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('is_start'):
                start_node = node
            if attrs.get('is_triforce'):
                goal_node = node
        
        if start_node is None or self.start_pos is None:
            return
        
        # Map start and goal
        self.room_to_node[self.start_pos] = start_node
        self.node_to_room[start_node] = self.start_pos
        
        if goal_node is not None and self.goal_pos is not None:
            self.room_to_node[self.goal_pos] = goal_node
            self.node_to_room[goal_node] = self.goal_pos
        
        # Use BFS to match remaining nodes based on adjacency
        from collections import deque
        
        directions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        visited_nodes = {start_node}
        visited_rooms = {self.start_pos}
        
        if goal_node is not None:
            visited_nodes.add(goal_node)
            visited_rooms.add(self.goal_pos)
        
        queue = deque([(start_node, self.start_pos)])
        
        while queue:
            node, room_pos = queue.popleft()
            room = self.rooms.get(room_pos)
            if not room:
                continue
            
            # Get graph neighbors
            graph_neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
            # Deterministic ordering: sort by degree then node id for stable assignment
            unmatched_nodes = sorted([n for n in graph_neighbors if n not in visited_nodes], key=lambda x: (self.graph.in_degree(x) + self.graph.out_degree(x), x))
            
            # Get room neighbors
            room_neighbors = []
            for direction, (dr, dc) in directions.items():
                if room.doors.get(direction):
                    neighbor_pos = (room_pos[0] + dr, room_pos[1] + dc)
                    if neighbor_pos in self.rooms and neighbor_pos not in visited_rooms:
                        room_neighbors.append(neighbor_pos)
            
            # Match nodes to rooms (greedy matching)
            for i, neighbor_node in enumerate(unmatched_nodes):
                if i < len(room_neighbors):
                    neighbor_room = room_neighbors[i]
                    self.room_to_node[neighbor_room] = neighbor_node
                    self.node_to_room[neighbor_node] = neighbor_room
                    visited_nodes.add(neighbor_node)
                    visited_rooms.add(neighbor_room)
                    queue.append((neighbor_node, neighbor_room))
    
    def _identify_special_rooms(self):
        """Identify rooms with keys, boss keys, and locked doors."""
        # Method 1: Use graph node labels
        for node, attrs in self.graph.nodes(data=True):
            room_pos = self.node_to_room.get(node)
            if room_pos is None:
                continue
            
            if attrs.get('has_key'):
                self.key_rooms.add(room_pos)
            
            if attrs.get('has_boss_key'):
                self.boss_key_rooms.add(room_pos)
        
        # Method 2: Check VGLC tiles
        for pos, room in self.rooms.items():
            if np.any(room.semantic_grid == SEMANTIC_PALETTE.get('KEY', 30)):
                self.key_rooms.add(pos)
            
            if np.any(room.semantic_grid == SEMANTIC_PALETTE.get('KEY_BOSS', 31)):
                self.boss_key_rooms.add(pos)
        
        # Identify locked doors from graph edges
        for node_from, node_to, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'open')
            if edge_type in ['key_locked', 'boss_locked', 'bombable']:
                room_from = self.node_to_room.get(node_from)
                room_to = self.node_to_room.get(node_to)
                if room_from and room_to:
                    self.locked_doors[(room_from, room_to)] = edge_type


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def print_solution(result: Dict):
    """Print solution in human-readable format (uses logging)."""
    if not result['solvable']:
        logger.warning("No solution found: %s", result.get('reason', 'Unknown'))
        if 'stats' in result:
            logger.info("States explored: %d", result['stats']['states_explored'])
            logger.info("Time: %.3fs", result['stats']['time_elapsed'])
        return
    
    logger.info("Solution found!")
    logger.info("Path length: %d moves", result['path_length'])
    logger.info("Rooms traversed: %d", result['rooms_traversed'])
    logger.info("Keys collected: %s", result['keys_collected'])
    logger.info("Keys used: %s", result['keys_used'])
    logger.info("Time: %.4fs", result['stats']['time_elapsed'])
    logger.info("States explored: %d", result['stats']['states_explored'])
    logger.info("States generated: %d", result['stats']['states_generated'])
    logger.info("Max queue size: %d", result['stats']['max_queue_size'])
    logger.info("Avg branching factor: %.2f", result['stats']['average_branching_factor'])
    
    logger.info("Path:")
    for i, room in enumerate(result['path']):
        action = result['actions'][i] if i < len(result['actions']) else ''
        logger.info("  %d. %s - %s", i, room, action)


# ==========================================
# TESTING
# ==========================================
if __name__ == '__main__':
    """Test the pathfinder on sample dungeons."""
    import sys
    from Data.zelda_core import ZeldaDungeonAdapter
    
    logger.info("Zelda A* Pathfinder Test")
    
    # Load a test dungeon
    from pathlib import Path
    data_root = Path(__file__).resolve().parent / 'Data' / 'The Legend of Zelda'
    adapter = ZeldaDungeonAdapter(data_root)
    
    # Try dungeon 1 (simplest)
    dungeon_name = 'tloz1_1'
    logger.info("Loading dungeon: %s", dungeon_name)
    
    try:
        dungeon = adapter.load_dungeon(dungeon_name)
        
        logger.info("Loaded successfully")
        logger.info("Rooms: %d", len(dungeon.rooms))
        logger.info("Start: %s", dungeon.start_pos)
        logger.info("Goal: %s", dungeon.triforce_pos)
        
        # Run pathfinder
        logger.info("Running A* pathfinder...")
        pathfinder = ZeldaPathfinder(dungeon, mode=ValidationMode.FULL)
        result = pathfinder.solve()
        
        # Print results
        print_solution(result)
        
    except Exception as e:
        logger.exception("Error during sample run: %s", e)
        import traceback
        traceback.print_exc()
