"""
Graph Constraint Enforcer
Prevents neural hallucinations by forcing generated layouts to respect mission graph topology.

This addresses the critical thesis defense concern: "The diffusion model is unconstrained - 
what stops it from generating layouts that violate your mission graph?"
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoomBoundary:
    """Represents the boundary region of a room in the spatial grid."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    node_id: int
    
    def get_perimeter(self) -> List[Tuple[int, int]]:
        """Get list of boundary coordinates."""
        perimeter = []
        # Top edge
        for x in range(self.x_min, self.x_max + 1):
            perimeter.append((x, self.y_min))
        # Bottom edge
        for x in range(self.x_min, self.x_max + 1):
            perimeter.append((x, self.y_max))
        # Left edge
        for y in range(self.y_min + 1, self.y_max):
            perimeter.append((self.x_min, y))
        # Right edge
        for y in range(self.y_min + 1, self.y_max):
            perimeter.append((self.x_max, y))
        return perimeter


@dataclass
class Door:
    """Represents a door position connecting two rooms."""
    x: int
    y: int
    direction: str  # 'north', 'south', 'east', 'west'
    connects: Tuple[int, int]  # (room1_id, room2_id)


class GraphConstraintEnforcer:
    """
    Enforces topological consistency between mission graph and spatial layout.
    
    Core Innovation:
    - Step 1: Seal all room boundaries with walls
    - Step 2: Open doors ONLY where mission graph specifies edges
    - Step 3: Verify connectivity matches graph
    
    This prevents the diffusion model from generating:
    - "Phantom corridors" (connections not in graph)
    - "Missing links" (graph edges without spatial connections)
    - "Floating rooms" (disconnected graph components)
    """
    
    def __init__(self, tile_config: Dict[str, int]):
        """
        Args:
            tile_config: Tile ID mapping
                Example: {'wall': 1, 'floor': 0, 'door': 2}
        """
        self.tile_config = tile_config
        self.WALL_ID = tile_config['wall']
        self.FLOOR_ID = tile_config['floor']
        self.DOOR_ID = tile_config.get('door', 2)
    
    def enforce_graph_constraints(
        self,
        visual_grid: np.ndarray,
        node_id: int,
        mission_graph: Dict,
        layout_map: Dict[int, Tuple[int, int, int, int]],
        tile_config: Dict[str, int]
    ) -> np.ndarray:
        """
        Enforce constraints for a single room.
        
        Args:
            visual_grid: Full dungeon grid (H, W)
            node_id: ID of the room to constrain
            mission_graph: Complete mission graph with topology
            layout_map: Mapping of node_id -> (x_min, y_min, x_max, y_max)
            tile_config: Tile type mappings
        
        Returns:
            Modified visual_grid with constraints enforced
        """
        if node_id not in layout_map:
            logger.warning(f"Node {node_id} not in layout_map, skipping")
            return visual_grid
        
        # Get room boundaries
        x_min, y_min, x_max, y_max = layout_map[node_id]
        boundary = RoomBoundary(x_min, y_min, x_max, y_max, node_id)
        
        # Step 1: Seal all boundaries with walls
        visual_grid = self._seal_boundaries(visual_grid, boundary)
        
        # Step 2: Open doors for valid connections
        valid_neighbors = self._get_valid_neighbors(node_id, mission_graph)
        visual_grid = self._create_doors(
            visual_grid, 
            boundary, 
            valid_neighbors, 
            layout_map
        )
        
        return visual_grid
    
    def _seal_boundaries(
        self, 
        grid: np.ndarray, 
        boundary: RoomBoundary
    ) -> np.ndarray:
        """Seal all perimeter tiles with walls."""
        # Top and bottom edges
        grid[boundary.y_min, boundary.x_min:boundary.x_max+1] = self.WALL_ID
        grid[boundary.y_max, boundary.x_min:boundary.x_max+1] = self.WALL_ID
        
        # Left and right edges
        grid[boundary.y_min:boundary.y_max+1, boundary.x_min] = self.WALL_ID
        grid[boundary.y_min:boundary.y_max+1, boundary.x_max] = self.WALL_ID
        
        return grid
    
    def _get_valid_neighbors(
        self, 
        node_id: int, 
        mission_graph: Dict
    ) -> Set[int]:
        """Get set of valid neighbor room IDs from mission graph."""
        neighbors = set()
        
        # Check adjacency dict (format: {node_id: {direction: neighbor_id}})
        if 'adjacency' in mission_graph:
            adjacency = mission_graph['adjacency']
            if node_id in adjacency:
                neighbors.update(adjacency[node_id].values())
        
        # Also check edges list (format: [(src, dst), ...])
        if 'edges' in mission_graph:
            for edge in mission_graph['edges']:
                if len(edge) >= 2:
                    src, dst = edge[0], edge[1]
                    if src == node_id:
                        neighbors.add(dst)
                    elif dst == node_id:
                        neighbors.add(src)
        
        return neighbors
    
    def _create_doors(
        self,
        grid: np.ndarray,
        boundary: RoomBoundary,
        valid_neighbors: Set[int],
        layout_map: Dict[int, Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Create doors to valid neighbor rooms only."""
        
        for neighbor_id in valid_neighbors:
            if neighbor_id not in layout_map:
                continue
            
            neighbor_bounds = layout_map[neighbor_id]
            nx_min, ny_min, nx_max, ny_max = neighbor_bounds
            
            # Determine relative position and create door
            door_pos = self._find_door_position(
                boundary, 
                neighbor_bounds
            )
            
            if door_pos is not None:
                x, y = door_pos
                grid[y, x] = self.DOOR_ID
                logger.debug(f"Created door at ({x}, {y}) connecting rooms {boundary.node_id} and {neighbor_id}")
        
        return grid
    
    def _find_door_position(
        self,
        room_boundary: RoomBoundary,
        neighbor_bounds: Tuple[int, int, int, int]
    ) -> Tuple[int, int]:
        """
        Find appropriate door position between two adjacent rooms.
        
        Returns:
            (x, y) coordinates for door placement, or None if rooms not adjacent
        """
        nx_min, ny_min, nx_max, ny_max = neighbor_bounds
        
        # Check if rooms are adjacent (horizontally or vertically)
        # Horizontal adjacency (neighbor to the right)
        if room_boundary.x_max + 1 == nx_min:
            # Find overlapping Y range
            y_overlap_start = max(room_boundary.y_min + 1, ny_min + 1)
            y_overlap_end = min(room_boundary.y_max - 1, ny_max - 1)
            
            if y_overlap_start <= y_overlap_end:
                # Place door in middle of overlap
                door_y = (y_overlap_start + y_overlap_end) // 2
                return (room_boundary.x_max, door_y)
        
        # Horizontal adjacency (neighbor to the left)
        if nx_max + 1 == room_boundary.x_min:
            y_overlap_start = max(room_boundary.y_min + 1, ny_min + 1)
            y_overlap_end = min(room_boundary.y_max - 1, ny_max - 1)
            
            if y_overlap_start <= y_overlap_end:
                door_y = (y_overlap_start + y_overlap_end) // 2
                return (room_boundary.x_min, door_y)
        
        # Vertical adjacency (neighbor below)
        if room_boundary.y_max + 1 == ny_min:
            x_overlap_start = max(room_boundary.x_min + 1, nx_min + 1)
            x_overlap_end = min(room_boundary.x_max - 1, nx_max - 1)
            
            if x_overlap_start <= x_overlap_end:
                door_x = (x_overlap_start + x_overlap_end) // 2
                return (door_x, room_boundary.y_max)
        
        # Vertical adjacency (neighbor above)
        if ny_max + 1 == room_boundary.y_min:
            x_overlap_start = max(room_boundary.x_min + 1, nx_min + 1)
            x_overlap_end = min(room_boundary.x_max - 1, nx_max - 1)
            
            if x_overlap_start <= x_overlap_end:
                door_x = (x_overlap_start + x_overlap_end) // 2
                return (door_x, room_boundary.y_min)
        
        return None


def enforce_all_rooms(
    visual_grid: np.ndarray,
    mission_graph: Dict,
    layout_map: Dict[int, Tuple[int, int, int, int]],
    tile_config: Dict[str, int]
) -> np.ndarray:
    """
    Apply graph constraint enforcement to all rooms in the dungeon.
    
    Args:
        visual_grid: Full dungeon grid (H, W)
        mission_graph: Mission graph with topology
        layout_map: Node ID -> (x_min, y_min, x_max, y_max)
        tile_config: Tile type mappings
    
    Returns:
        Constrained visual_grid where topology matches mission graph
    """
    enforcer = GraphConstraintEnforcer(tile_config)
    
    logger.info(f"Enforcing graph constraints on {len(layout_map)} rooms...")
    
    for node_id in layout_map.keys():
        visual_grid = enforcer.enforce_graph_constraints(
            visual_grid,
            node_id,
            mission_graph,
            layout_map,
            tile_config
        )
    
    logger.info("Graph constraint enforcement complete")
    
    return visual_grid


def verify_topology_match(
    visual_grid: np.ndarray,
    mission_graph: Dict,
    layout_map: Dict[int, Tuple[int, int, int, int]],
    tile_config: Dict[str, int]
) -> bool:
    """
    Verify that spatial layout matches mission graph topology.
    
    Returns:
        True if topology matches, False otherwise
    """
    DOOR_ID = tile_config.get('door', 2)
    
    # Count expected connections from graph
    expected_edges = set()
    if 'edges' in mission_graph:
        for edge in mission_graph['edges']:
            if len(edge) >= 2:
                src, dst = edge[0], edge[1]
                # Store as sorted tuple to avoid direction issues
                expected_edges.add(tuple(sorted([src, dst])))
    
    # Count actual connections (doors) in spatial layout
    actual_edges = set()
    
    for node_id, bounds in layout_map.items():
        x_min, y_min, x_max, y_max = bounds
        
        # Check perimeter for doors
        # Top edge
        for x in range(x_min, x_max + 1):
            if visual_grid[y_min, x] == DOOR_ID:
                # Find which room this connects to
                neighbor_id = _find_room_at_position(x, y_min - 1, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
        
        # Bottom edge
        for x in range(x_min, x_max + 1):
            if visual_grid[y_max, x] == DOOR_ID:
                neighbor_id = _find_room_at_position(x, y_max + 1, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
        
        # Left edge
        for y in range(y_min, y_max + 1):
            if visual_grid[y, x_min] == DOOR_ID:
                neighbor_id = _find_room_at_position(x_min - 1, y, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
        
        # Right edge
        for y in range(y_min, y_max + 1):
            if visual_grid[y, x_max] == DOOR_ID:
                neighbor_id = _find_room_at_position(x_max + 1, y, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
    
    # Compare sets
    missing_edges = expected_edges - actual_edges
    phantom_edges = actual_edges - expected_edges
    
    if missing_edges:
        logger.warning(f"Missing edges in spatial layout: {missing_edges}")
    
    if phantom_edges:
        logger.warning(f"Phantom edges in spatial layout (not in graph): {phantom_edges}")
    
    match = len(missing_edges) == 0 and len(phantom_edges) == 0
    
    logger.info(f"Topology verification: {'MATCH' if match else 'MISMATCH'}")
    logger.info(f"Expected edges: {len(expected_edges)}, Actual edges: {len(actual_edges)}")
    
    return match


def _find_room_at_position(
    x: int, 
    y: int, 
    layout_map: Dict[int, Tuple[int, int, int, int]]
) -> int:
    """Find which room contains the given position."""
    for node_id, (x_min, y_min, x_max, y_max) in layout_map.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return node_id
    return None
