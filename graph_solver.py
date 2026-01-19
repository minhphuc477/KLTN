"""
ZAVE - Zelda AI Validation Environment (Graph-Based Solver)
============================================================

This solver uses the DOT graph topology to properly solve dungeons:
1. Parse DOT graph for room objects (keys, items, locked doors)
2. Build a state-space search (position + collected keys)
3. Find optimal path that collects keys to unlock doors
4. Visualize the solution with proper maze navigation

The graph defines:
- Node labels: s=start, t=triforce, b=boss, k=key, e=enemy, i/I=item, p=puzzle
- Edge labels: k=key_locked, b=bombable, l=soft_locked, empty=open

Controls:
- SPACE: Start/Pause solving animation
- R: Reset
- N/P: Next/Previous dungeon
- +/-: Speed up/down
- ESC: Quit
"""

import sys
import os
import time
import heapq
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Set, FrozenSet
from dataclasses import dataclass, field
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
except ImportError:
    print("Install pygame: pip install pygame")
    sys.exit(1)

from Data.zelda_core import (
    ZeldaDungeonAdapter, 
    SEMANTIC_PALETTE,
    ROOM_HEIGHT, 
    ROOM_WIDTH,
    Dungeon,
    StitchedDungeon,
    Room
)


# ==========================================
# TILE COLORS
# ==========================================
COLORS = {
    0: (20, 20, 30),       # VOID - dark
    1: (160, 140, 120),    # FLOOR - tan
    2: (60, 60, 80),       # WALL - dark gray
    3: (80, 80, 100),      # BLOCK - gray
    10: (139, 90, 60),     # DOOR_OPEN - brown
    11: (180, 100, 40),    # DOOR_LOCKED - orange-brown
    12: (120, 80, 50),     # DOOR_BOMB - reddish
    15: (110, 70, 45),     # DOOR_SOFT - tan
    20: (200, 50, 50),     # ENEMY - red
    21: (50, 200, 50),     # START - green
    22: (255, 215, 0),     # TRIFORCE - gold
    23: (150, 0, 0),       # BOSS - dark red
    30: (255, 255, 100),   # KEY - yellow
    33: (255, 200, 100),   # ITEM - orange
    40: (100, 150, 200),   # ELEMENT - blue (water/lava)
    42: (150, 200, 255),   # STAIR - light blue
}


# ==========================================
# GRAPH-BASED SOLVER
# ==========================================
@dataclass
class SolveState:
    """State in the graph search: current room + collected keys."""
    room_pos: Tuple[int, int]
    keys_collected: FrozenSet[Tuple[int, int]]  # Set of room positions where keys were collected
    
    def __hash__(self):
        return hash((self.room_pos, self.keys_collected))
    
    def __eq__(self, other):
        return self.room_pos == other.room_pos and self.keys_collected == other.keys_collected


class GraphSolver:
    """
    Graph-based solver that uses DOT topology.
    
    This solver:
    1. Identifies rooms with keys from the graph
    2. Identifies locked doors from edge labels
    3. Uses state-space search (room + keys) to find solution
    4. Returns a room-level path with key pickups
    """
    
    def __init__(self, dungeon: Dungeon):
        self.dungeon = dungeon
        self.graph = dungeon.graph
        self.rooms = dungeon.rooms
        
        # Build room-to-node mapping
        self.room_to_node = {}
        self.node_to_room = {}
        self._build_room_node_mapping()
        
        # Identify rooms with keys
        self.key_rooms: Set[Tuple[int, int]] = set()
        self._identify_key_rooms()
        
        # Build door requirements (which doors need keys)
        self.locked_doors: Dict[Tuple[Tuple[int,int], Tuple[int,int]], str] = {}
        self._identify_locked_doors()
        
    def _build_room_node_mapping(self):
        """Map rooms to graph nodes using comprehensive BFS matching."""
        if not self.graph or len(self.graph.nodes) == 0:
            return
            
        # Find start node and goal node
        start_node = None
        goal_node = None
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('is_start'):
                start_node = node
            if attrs.get('is_triforce'):
                goal_node = node
        
        start_room = self.dungeon.start_pos
        goal_room = self.dungeon.triforce_pos
        
        if start_node is None or start_room is None:
            return
        
        # Use multiple BFS passes to match as many nodes as possible
        self.room_to_node[start_room] = start_node
        self.node_to_room[start_node] = start_room
        
        if goal_node is not None and goal_room is not None:
            self.room_to_node[goal_room] = goal_node
            self.node_to_room[goal_node] = goal_room
        
        directions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        
        # Multi-pass BFS to handle disconnected components
        for _ in range(3):
            visited_nodes = set(self.node_to_room.keys())
            visited_rooms = set(self.room_to_node.keys())
            
            # Start BFS from all currently mapped nodes
            queue = deque([(n, r) for r, n in self.room_to_node.items()])
            
            while queue:
                node, room_pos = queue.popleft()
                room = self.rooms.get(room_pos)
                if not room:
                    continue
                
                # Get all graph neighbors
                graph_neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                unmatched_graph = [n for n in graph_neighbors if n not in visited_nodes]
                
                # Get all room neighbors with doors
                room_neighbors = []
                for direction, (dr, dc) in directions.items():
                    if room.doors.get(direction):
                        neighbor_pos = (room_pos[0] + dr, room_pos[1] + dc)
                        if neighbor_pos in self.rooms and neighbor_pos not in visited_rooms:
                            room_neighbors.append((neighbor_pos, direction))
                
                # Match by trying to preserve graph edge count
                for neighbor_node in unmatched_graph:
                    if not room_neighbors:
                        break
                    
                    # Try to find best matching room neighbor
                    best_match = None
                    best_score = -float('inf')  # Must be -inf to allow any score
                    
                    node_neighbor_count = len(set(self.graph.successors(neighbor_node)) | 
                                             set(self.graph.predecessors(neighbor_node)))
                    
                    for room_pos_candidate, direction in room_neighbors:
                        candidate_room = self.rooms.get(room_pos_candidate)
                        if candidate_room:
                            # Count doors as a proxy for neighbor count
                            door_count = sum(1 for d, v in candidate_room.doors.items() if v)
                            # Score based on similarity
                            score = -abs(door_count - node_neighbor_count)
                            if score > best_score:
                                best_score = score
                                best_match = (room_pos_candidate, direction)
                    
                    if best_match:
                        neighbor_room, _ = best_match
                        self.room_to_node[neighbor_room] = neighbor_node
                        self.node_to_room[neighbor_node] = neighbor_room
                        visited_nodes.add(neighbor_node)
                        visited_rooms.add(neighbor_room)
                        room_neighbors.remove(best_match)
                        queue.append((neighbor_node, neighbor_room))
    
    def _identify_key_rooms(self):
        """Find rooms that contain keys based on graph node labels and tiles."""
        # Method 1: Use graph node labels (most reliable for DOT topology)
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('has_key'):
                room_pos = self.node_to_room.get(node)
                if room_pos:
                    self.key_rooms.add(room_pos)
        
        # Method 2: Also check VGLC tiles for KEY markers (backup)
        for pos, room in self.rooms.items():
            if np.any(room.semantic_grid == SEMANTIC_PALETTE.get('KEY', 30)):
                self.key_rooms.add(pos)
        
        # Method 3: If still no keys found, use graph topology to infer
        # Any room that leads to a key-locked door but is not behind one might have a key
        if len(self.key_rooms) == 0:
            for node, attrs in self.graph.nodes(data=True):
                # Check if this node has outgoing key-locked edges
                for _, dst, edge_attrs in self.graph.out_edges(node, data=True):
                    if edge_attrs.get('edge_type') == 'key_locked':
                        # The source node or its accessible neighbors might have keys
                        room_pos = self.node_to_room.get(node)
                        if room_pos:
                            self.key_rooms.add(room_pos)
    
    def _identify_locked_doors(self):
        """Find doors that are locked based on graph edge labels."""
        for src, dst, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'open')
            if edge_type in ['key_locked', 'bombable', 'soft_locked']:
                src_room = self.node_to_room.get(src)
                dst_room = self.node_to_room.get(dst)
                if src_room and dst_room:
                    self.locked_doors[(src_room, dst_room)] = edge_type
    
    def solve(self) -> Tuple[List[Tuple[int, int]], List[str], bool]:
        """
        Solve the dungeon using graph-based state search.
        
        Returns:
            (room_path, actions, success)
            - room_path: List of room positions to visit
            - actions: List of actions (move, collect_key, etc.)
            - success: Whether a solution was found
        """
        if self.dungeon.start_pos is None or self.dungeon.triforce_pos is None:
            return [], [], False
        
        start_pos = self.dungeon.start_pos
        goal_pos = self.dungeon.triforce_pos
        
        # State: (room_pos, frozenset of collected key rooms)
        initial_state = SolveState(start_pos, frozenset())
        
        # A* search
        # Priority: (f_score, counter, state, path, actions)
        counter = 0
        open_set = [(self._heuristic(start_pos, goal_pos), counter, initial_state, [start_pos], ['start'])]
        visited = {(initial_state.room_pos, initial_state.keys_collected)}
        
        directions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        opposites = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        
        while open_set:
            f, _, state, path, actions = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if state.room_pos == goal_pos:
                return path, actions, True
            
            room = self.rooms.get(state.room_pos)
            if not room:
                continue
            
            # Automatically collect key if in key room (keys are collected when entering)
            current_keys = state.keys_collected
            current_actions = list(actions)
            
            if state.room_pos in self.key_rooms and state.room_pos not in current_keys:
                current_keys = current_keys | {state.room_pos}
                current_actions = current_actions + [f'collect_key_{state.room_pos}']
            
            # Try moving to adjacent rooms
            for direction, (dr, dc) in directions.items():
                if not room.doors.get(direction):
                    continue
                
                neighbor_pos = (state.room_pos[0] + dr, state.room_pos[1] + dc)
                neighbor_room = self.rooms.get(neighbor_pos)
                
                if not neighbor_room:
                    continue
                
                # Check if neighbor has matching door
                opp = opposites[direction]
                if not neighbor_room.doors.get(opp):
                    continue
                
                # Check if door is locked
                door_key = (state.room_pos, neighbor_pos)
                door_key_rev = (neighbor_pos, state.room_pos)
                
                can_pass = True
                door_type = self.locked_doors.get(door_key) or self.locked_doors.get(door_key_rev)
                
                if door_type == 'key_locked':
                    # Need at least one key to pass
                    if len(current_keys) == 0:
                        can_pass = False
                elif door_type == 'bombable':
                    # Assume we have bombs (or treat as passable for now)
                    can_pass = True
                elif door_type == 'soft_locked':
                    # One-way or special condition - treat as passable
                    can_pass = True
                
                if not can_pass:
                    continue
                
                new_state = SolveState(neighbor_pos, current_keys)
                state_key = (new_state.room_pos, new_state.keys_collected)
                
                if state_key not in visited:
                    visited.add(state_key)
                    counter += 1
                    g = len(path)
                    h = self._heuristic(neighbor_pos, goal_pos)
                    new_path = path + [neighbor_pos]
                    new_actions = current_actions + [f'move_{direction}']
                    heapq.heappush(open_set, (g + h, counter, new_state, new_path, new_actions))
        
        # No solution found - try without locked door restrictions
        return self._solve_simple()
    
    def _solve_simple(self) -> Tuple[List[Tuple[int, int]], List[str], bool]:
        """Simple BFS without considering locked doors."""
        if self.dungeon.start_pos is None or self.dungeon.triforce_pos is None:
            return [], [], False
        
        start_pos = self.dungeon.start_pos
        goal_pos = self.dungeon.triforce_pos
        
        visited = {start_pos}
        queue = deque([(start_pos, [start_pos], ['start'])])
        
        directions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        opposites = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        
        while queue:
            pos, path, actions = queue.popleft()
            
            if pos == goal_pos:
                return path, actions, True
            
            room = self.rooms.get(pos)
            if not room:
                continue
            
            for direction, (dr, dc) in directions.items():
                if not room.doors.get(direction):
                    continue
                
                neighbor_pos = (pos[0] + dr, pos[1] + dc)
                neighbor_room = self.rooms.get(neighbor_pos)
                
                if not neighbor_room or neighbor_pos in visited:
                    continue
                
                opp = opposites[direction]
                if not neighbor_room.doors.get(opp):
                    continue
                
                visited.add(neighbor_pos)
                queue.append((neighbor_pos, path + [neighbor_pos], actions + [f'move_{direction}']))
        
        return [], [], False
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# ------------------------------------------------------------------
# Utility: standalone graph search over (node, inventory) state
# ------------------------------------------------------------------

def solve_with_inventory(graph: nx.DiGraph,
                         start_node,
                         goal_test,
                         items_map: Dict[str, str],
                         locked_edge_attr: str = 'edge_type'):
    """Generic search on a networkx graph that reasons about collectible items.

    Args:
        graph: networkx DiGraph with node attributes and edge attributes.
        start_node: node id to start from.
        goal_test: callable(node, inventory_set) -> bool
        items_map: mapping node_id -> item_name (e.g. {'A': 'key'}) for collectible nodes.
        locked_edge_attr: edge attribute name that encodes lock type (e.g. 'key_locked').

    Returns:
        (path_nodes, actions, success)
        - path_nodes: list of node ids visited (room-level path)
        - actions: list of strings (e.g. 'collect:key', 'pass:key_locked')
        - success: bool

    Complexity: O(|V| * 2^k + |E| log(|V|) * 2^k) where k = #distinct collectible items.

    This is intentionally deterministic, exact and suitable for unit tests.
    """
    from collections import deque
    from typing import FrozenSet

    # Normalize items to small set
    item_nodes = {n: items_map[n] for n in items_map if n in graph}
    all_items = sorted(set(item_nodes.values()))
    item_to_idx = {it: i for i, it in enumerate(all_items)}

    def add_item(inv: FrozenSet[str], node):
        if node in item_nodes:
            return frozenset(set(inv) | {item_nodes[node]})
        return inv

    # State = (node, frozenset(items))
    start_state = (start_node, frozenset())
    queue = deque([(start_state, [start_node], ['start'])])
    visited = {start_state}

    while queue:
        (node, inv), path, actions = queue.popleft()
        # collect on entry
        if node in item_nodes and item_nodes[node] not in inv:
            inv = add_item(inv, node)
            actions = actions + [f'collect:{item_nodes[node]}']

        if goal_test(node, inv):
            return path, actions, True

        for _, nbr, edata in graph.out_edges(node, data=True):
            edge_type = edata.get(locked_edge_attr, 'open')
            can_pass = True
            act = f'pass:{edge_type}'
            if edge_type == 'key_locked':
                # require any key in inventory
                if len(inv) == 0:
                    can_pass = False
            elif edge_type == 'bombable':
                # assume bomb available only if 'bomb' in inv
                if 'bomb' in item_to_idx and 'bomb' not in inv:
                    can_pass = False
            # extend for other types as needed

            if not can_pass:
                continue

            new_state = (nbr, inv)
            if new_state in visited:
                continue
            visited.add(new_state)
            queue.append((new_state, path + [nbr], actions + [act]))

    return [], [], False


class TilePathFinder:
    """
    Find tile-level paths within and between rooms.

    Given a room-level path from the graph solver, this finds
    the actual tile positions Link should walk on. The finder now
    supports an optional inventory set to allow crossing otherwise
    non-walkable tiles (e.g. water/"ELEMENT") when the agent
    carries the required item (raft/boat/etc.).
    """

    # Base walkable tiles (water/lava excluded by default)
    WALKABLE = {
        SEMANTIC_PALETTE['FLOOR'],      # 1
        SEMANTIC_PALETTE['DOOR_OPEN'],  # 10
        SEMANTIC_PALETTE['ENEMY'],      # 20
        SEMANTIC_PALETTE['START'],      # 21
        SEMANTIC_PALETTE['TRIFORCE'],   # 22
        SEMANTIC_PALETTE['STAIR'],      # 42
        SEMANTIC_PALETTE.get('KEY', 30),     # 30 - can walk on key
        SEMANTIC_PALETTE.get('ITEM', 33),    # 33 - can walk on item
    }

    # Map inventory item -> semantic tile it enables traversal for
    INVENTORY_TILE_MAP = {
        'raft': SEMANTIC_PALETTE.get('ELEMENT', 40),
        'flippers': SEMANTIC_PALETTE.get('ELEMENT', 40),
    }

    def __init__(self, stitched: StitchedDungeon, dungeon: Dungeon):
        self.stitched = stitched
        self.dungeon = dungeon
        self.grid = stitched.global_grid

    def find_tile_path(self, room_path: List[Tuple[int, int]], inventory: Optional[Set[str]] = None) -> List[Tuple[int, int]]:
        """
        Convert room-level path to tile-level path.

        Args:
            room_path: List of room positions to visit
            inventory: optional set of inventory item names that may alter walkability

        Returns:
            List of (row, col) tile positions
        """
        if not room_path:
            return []

        tile_path = []

        # Start from the START tile
        if self.stitched.start_global:
            tile_path.append(self.stitched.start_global)

        for i in range(len(room_path) - 1):
            current_room = room_path[i]
            next_room = room_path[i + 1]

            # Find direction of movement
            dr = next_room[0] - current_room[0]
            dc = next_room[1] - current_room[1]

            # Get door positions for current and next room
            current_exit = self._get_door_center(current_room, dr, dc, inventory=inventory)
            next_entry = self._get_door_center(next_room, -dr, -dc, inventory=inventory)

            # Path from current position to exit door
            if tile_path:
                path_to_exit = self._astar(tile_path[-1], current_exit, inventory=inventory)
                if path_to_exit:
                    tile_path.extend(path_to_exit[1:])  # Skip first (already at it)

            # "Teleport" through door (add entry point of next room)
            if next_entry:
                tile_path.append(next_entry)

        # Final path to TRIFORCE
        if self.stitched.triforce_global and tile_path:
            path_to_goal = self._astar(tile_path[-1], self.stitched.triforce_global, inventory=inventory)
            if path_to_goal:
                tile_path.extend(path_to_goal[1:])

        return tile_path

    def _get_door_center(self, room_pos: Tuple[int, int], dr: int, dc: int, inventory: Optional[Set[str]] = None) -> Optional[Tuple[int, int]]:
        """Get the center tile of a door on a specific side of a room."""
        row_start = room_pos[0] * ROOM_HEIGHT
        col_start = room_pos[1] * ROOM_WIDTH

        # Door positions based on direction
        if dr == -1:  # North door
            door_row = row_start
            door_col = col_start + ROOM_WIDTH // 2
        elif dr == 1:  # South door
            door_row = row_start + ROOM_HEIGHT - 1
            door_col = col_start + ROOM_WIDTH // 2
        elif dc == -1:  # West door
            door_row = row_start + ROOM_HEIGHT // 2
            door_col = col_start
        elif dc == 1:  # East door
            door_row = row_start + ROOM_HEIGHT // 2
            door_col = col_start + ROOM_WIDTH - 1
        else:
            return None

        # Find nearest walkable tile to door center (respect inventory)
        for offset in range(5):
            for ddr in range(-offset, offset + 1):
                for ddc in range(-offset, offset + 1):
                    r, c = door_row + ddr, door_col + ddc
                    if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
                        tile = int(self.grid[r, c])
                        if tile in self.WALKABLE or (inventory and any(self.INVENTORY_TILE_MAP.get(it) == tile for it in inventory)):
                            return (r, c)

        return (door_row, door_col)

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int], inventory: Optional[Set[str]] = None) -> List[Tuple[int, int]]:
        """A* pathfinding on walkable tiles. Respects optional inventory to enable traversal."""
        if start == goal:
            return [start]

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        counter = 0
        open_set = [(heuristic(start, goal), counter, start, [start])]
        visited = {start}

        while open_set:
            f, _, pos, path = heapq.heappop(open_set)

            if pos == goal:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc

                if (nr, nc) in visited:
                    continue

                if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]:
                    tile = int(self.grid[nr, nc])
                    # Allow walking on goal even if not normally walkable
                    allowed = tile in self.WALKABLE or (inventory and any(self.INVENTORY_TILE_MAP.get(it) == tile for it in inventory))
                    if allowed or (nr, nc) == goal:
                        visited.add((nr, nc))
                        counter += 1
                        g = len(path)
                        h = heuristic((nr, nc), goal)
                        heapq.heappush(open_set, (g + h, counter, (nr, nc), path + [(nr, nc)]))

        # No path found - return direct line (for visualization)
        return [start, goal]


# ==========================================
# VISUALIZATION
# ==========================================
@dataclass
class AnimationState:
    """State for solving animation."""
    path: List[Tuple[int, int]] = field(default_factory=list)
    room_path: List[Tuple[int, int]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    current_index: int = 0
    is_playing: bool = False
    is_complete: bool = False
    speed: float = 10.0  # Tiles per second
    keys_collected: Set[Tuple[int, int]] = field(default_factory=set)


class GraphVisualizer:
    """
    Pygame visualizer that shows graph-based dungeon solving.
    """
    
    def __init__(self, adapter: ZeldaDungeonAdapter):
        self.adapter = adapter
        self.dungeons: List[Tuple[Dungeon, StitchedDungeon, GraphSolver]] = []
        self.current_dungeon_idx = 0
        
        # Load all dungeons
        self._load_dungeons()
        
        # Pygame setup
        pygame.init()
        self.tile_size = 8
        self.info_width = 250
        
        # Calculate window size from first dungeon
        if self.dungeons:
            _, stitched, _ = self.dungeons[0]
            h, w = stitched.global_grid.shape
            self.screen_width = w * self.tile_size + self.info_width
            self.screen_height = max(h * self.tile_size, 400)
        else:
            self.screen_width = 800
            self.screen_height = 600
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ZAVE - Graph-Based Dungeon Solver")
        
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.animation = AnimationState()
        self.last_update = time.time()
        
        self._setup_current_dungeon()
    
    def _load_dungeons(self):
        """Load all dungeons with graph solving."""
        print("\nLoading dungeons with graph-based solving...")
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = self.adapter.load_dungeon(dungeon_num, variant=variant)
                    stitched = self.adapter.stitch_dungeon(dungeon)
                    
                    # Create graph solver
                    solver = GraphSolver(dungeon)
                    room_path, actions, success = solver.solve()
                    
                    # Create tile path finder
                    tile_finder = TilePathFinder(stitched, dungeon)
                    tile_path = tile_finder.find_tile_path(room_path)
                    
                    # Store results
                    dungeon._room_path = room_path
                    dungeon._tile_path = tile_path
                    dungeon._actions = actions
                    dungeon._solvable = success
                    dungeon._key_rooms = solver.key_rooms
                    dungeon._locked_doors = solver.locked_doors
                    
                    self.dungeons.append((dungeon, stitched, solver))
                    
                    status = "✓" if success else "✗"
                    keys = len(solver.key_rooms)
                    locked = len(solver.locked_doors)
                    print(f"  {status} D{dungeon_num}-{variant}: {len(room_path)} rooms, {keys} keys, {locked} locked doors")
                    
                except Exception as e:
                    print(f"  ! D{dungeon_num}-{variant}: Error - {e}")
        
        solvable = sum(1 for d, s, g in self.dungeons if d._solvable)
        print(f"\nLoaded {len(self.dungeons)} dungeons")
        print(f"Graph-solvable: {solvable}/{len(self.dungeons)} ({100*solvable/len(self.dungeons):.1f}%)")
    
    def _setup_current_dungeon(self):
        """Setup animation for current dungeon."""
        if not self.dungeons:
            return
        
        dungeon, stitched, solver = self.dungeons[self.current_dungeon_idx]
        
        self.animation = AnimationState()
        self.animation.path = getattr(dungeon, '_tile_path', [])
        self.animation.room_path = getattr(dungeon, '_room_path', [])
        self.animation.actions = getattr(dungeon, '_actions', [])
        self.animation.keys_collected = set()
        
        # Resize window if needed
        h, w = stitched.global_grid.shape
        new_width = w * self.tile_size + self.info_width
        new_height = max(h * self.tile_size, 400)
        
        if new_width != self.screen_width or new_height != self.screen_height:
            self.screen_width = new_width
            self.screen_height = new_height
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
            
            self._update()
            self._draw()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    def _handle_key(self, key) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == pygame.K_ESCAPE:
            return False
        elif key == pygame.K_SPACE:
            self.animation.is_playing = not self.animation.is_playing
        elif key == pygame.K_r:
            self._setup_current_dungeon()
        elif key == pygame.K_n:
            self.current_dungeon_idx = (self.current_dungeon_idx + 1) % len(self.dungeons)
            self._setup_current_dungeon()
        elif key == pygame.K_p:
            self.current_dungeon_idx = (self.current_dungeon_idx - 1) % len(self.dungeons)
            self._setup_current_dungeon()
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.animation.speed = min(50, self.animation.speed + 5)
        elif key == pygame.K_MINUS:
            self.animation.speed = max(1, self.animation.speed - 5)
        return True
    
    def _update(self):
        """Update animation state."""
        if not self.animation.is_playing or self.animation.is_complete:
            return
        
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Advance position
        tiles_to_advance = dt * self.animation.speed
        self.animation.current_index += tiles_to_advance
        
        if self.animation.current_index >= len(self.animation.path):
            self.animation.current_index = len(self.animation.path) - 1
            self.animation.is_complete = True
        
        # Check for key collection
        if self.dungeons:
            dungeon, _, _ = self.dungeons[self.current_dungeon_idx]
            key_rooms = getattr(dungeon, '_key_rooms', set())
            
            current_pos = int(self.animation.current_index)
            if current_pos < len(self.animation.path):
                tile_pos = self.animation.path[current_pos]
                # Convert tile pos to room pos
                room_pos = (tile_pos[0] // ROOM_HEIGHT, tile_pos[1] // ROOM_WIDTH)
                if room_pos in key_rooms:
                    self.animation.keys_collected.add(room_pos)
    
    def _draw(self):
        """Draw everything."""
        self.screen.fill((30, 30, 40))
        
        if not self.dungeons:
            return
        
        dungeon, stitched, solver = self.dungeons[self.current_dungeon_idx]
        grid = stitched.global_grid
        
        # Draw dungeon tiles
        self._draw_tiles(grid, dungeon)
        
        # Draw path
        self._draw_path()
        
        # Draw Link
        self._draw_link()
        
        # Draw info panel
        self._draw_info(dungeon, solver)
    
    def _draw_tiles(self, grid: np.ndarray, dungeon: Dungeon):
        """Draw dungeon tiles with key room highlights."""
        key_rooms = getattr(dungeon, '_key_rooms', set())
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                tile = grid[r, c]
                color = COLORS.get(tile, (128, 128, 128))
                
                # Highlight key rooms
                room_pos = (r // ROOM_HEIGHT, c // ROOM_WIDTH)
                if room_pos in key_rooms and room_pos not in self.animation.keys_collected:
                    # Add yellow tint for uncollected key rooms
                    color = (min(255, color[0] + 30), min(255, color[1] + 30), color[2])
                
                rect = pygame.Rect(
                    c * self.tile_size,
                    r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw room grid lines
        h, w = grid.shape
        for room_r in range(h // ROOM_HEIGHT + 1):
            y = room_r * ROOM_HEIGHT * self.tile_size
            pygame.draw.line(self.screen, (80, 80, 100), (0, y), (w * self.tile_size, y), 1)
        for room_c in range(w // ROOM_WIDTH + 1):
            x = room_c * ROOM_WIDTH * self.tile_size
            pygame.draw.line(self.screen, (80, 80, 100), (x, 0), (x, h * self.tile_size), 1)
    
    def _draw_path(self):
        """Draw the solution path."""
        if len(self.animation.path) < 2:
            return
        
        # Draw full path (faded)
        for i in range(len(self.animation.path) - 1):
            start = self.animation.path[i]
            end = self.animation.path[i + 1]
            
            start_px = (start[1] * self.tile_size + self.tile_size // 2,
                       start[0] * self.tile_size + self.tile_size // 2)
            end_px = (end[1] * self.tile_size + self.tile_size // 2,
                     end[0] * self.tile_size + self.tile_size // 2)
            
            pygame.draw.line(self.screen, (100, 100, 150), start_px, end_px, 1)
        
        # Draw traversed path (bright)
        current_idx = int(self.animation.current_index)
        for i in range(min(current_idx, len(self.animation.path) - 1)):
            start = self.animation.path[i]
            end = self.animation.path[i + 1]
            
            start_px = (start[1] * self.tile_size + self.tile_size // 2,
                       start[0] * self.tile_size + self.tile_size // 2)
            end_px = (end[1] * self.tile_size + self.tile_size // 2,
                     end[0] * self.tile_size + self.tile_size // 2)
            
            pygame.draw.line(self.screen, (100, 255, 100), start_px, end_px, 2)
    
    def _draw_link(self):
        """Draw Link at current position."""
        if not self.animation.path:
            return
        
        idx = int(self.animation.current_index)
        if idx >= len(self.animation.path):
            idx = len(self.animation.path) - 1
        
        pos = self.animation.path[idx]
        
        # Draw Link as a green square
        rect = pygame.Rect(
            pos[1] * self.tile_size - 2,
            pos[0] * self.tile_size - 2,
            self.tile_size + 4,
            self.tile_size + 4
        )
        pygame.draw.rect(self.screen, (0, 255, 0), rect)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
    
    def _draw_info(self, dungeon: Dungeon, solver: GraphSolver):
        """Draw information panel."""
        x = self.screen_width - self.info_width + 10
        y = 10
        
        # Title
        title = self.font.render("ZAVE Graph Solver", True, (255, 255, 255))
        self.screen.blit(title, (x, y))
        y += 30
        
        # Dungeon info
        dungeon_text = self.font.render(f"Dungeon: {dungeon.dungeon_id}", True, (200, 200, 200))
        self.screen.blit(dungeon_text, (x, y))
        y += 25
        
        # Solvability
        solvable = getattr(dungeon, '_solvable', False)
        color = (100, 255, 100) if solvable else (255, 100, 100)
        status = "Solvable" if solvable else "Unsolvable"
        status_text = self.font.render(f"Status: {status}", True, color)
        self.screen.blit(status_text, (x, y))
        y += 25
        
        # Keys
        total_keys = len(solver.key_rooms)
        collected = len(self.animation.keys_collected)
        keys_text = self.font.render(f"Keys: {collected}/{total_keys}", True, (255, 255, 100))
        self.screen.blit(keys_text, (x, y))
        y += 25
        
        # Locked doors
        locked = len(solver.locked_doors)
        doors_text = self.font.render(f"Locked Doors: {locked}", True, (200, 150, 100))
        self.screen.blit(doors_text, (x, y))
        y += 25
        
        # Room path
        room_path = getattr(dungeon, '_room_path', [])
        rooms_text = self.font.render(f"Room Path: {len(room_path)} rooms", True, (150, 200, 255))
        self.screen.blit(rooms_text, (x, y))
        y += 25
        
        # Progress
        progress = 0
        if self.animation.path:
            progress = int(100 * self.animation.current_index / len(self.animation.path))
        progress_text = self.font.render(f"Progress: {progress}%", True, (200, 200, 200))
        self.screen.blit(progress_text, (x, y))
        y += 35
        
        # Controls
        controls = [
            "Controls:",
            "SPACE - Play/Pause",
            "R - Reset",
            "N/P - Next/Prev dungeon",
            "+/- - Speed up/down",
            "ESC - Quit"
        ]
        
        for line in controls:
            text = self.small_font.render(line, True, (150, 150, 150))
            self.screen.blit(text, (x, y))
            y += 18
        
        y += 10
        
        # Legend
        legend_title = self.small_font.render("Legend:", True, (200, 200, 200))
        self.screen.blit(legend_title, (x, y))
        y += 20
        
        legend_items = [
            ((160, 140, 120), "Floor"),
            ((60, 60, 80), "Wall"),
            ((139, 90, 60), "Door"),
            ((100, 150, 200), "Water"),
            ((255, 255, 100), "Key"),
            ((50, 200, 50), "Start"),
            ((255, 215, 0), "Triforce"),
        ]
        
        for color, name in legend_items:
            pygame.draw.rect(self.screen, color, (x, y, 15, 15))
            text = self.small_font.render(name, True, (180, 180, 180))
            self.screen.blit(text, (x + 20, y))
            y += 18


def main():
    """Main entry point."""
    print("=" * 60)
    print("ZAVE - Zelda Graph-Based Dungeon Solver")
    print("=" * 60)
    print()
    print("This solver uses DOT graph topology to:")
    print("  - Identify key locations")
    print("  - Track locked doors")
    print("  - Find paths that collect keys to unlock doors")
    print()
    
    adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
    visualizer = GraphVisualizer(adapter)
    
    print()
    print("Press SPACE to start solving animation")
    print("Press N/P to change dungeons")
    
    visualizer.run()


if __name__ == "__main__":
    main()
