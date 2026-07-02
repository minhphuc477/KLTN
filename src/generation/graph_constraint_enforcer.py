"""
Graph Constraint Enforcer
Prevents neural hallucinations by forcing generated layouts to respect mission graph topology.

This addresses the critical thesis defense concern: "The diffusion model is unconstrained - 
what stops it from generating layouts that violate your mission graph?"
"""

import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import logging

from src.core.definitions import SEMANTIC_PALETTE

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
        hazard_default = int(tile_config.get('element', SEMANTIC_PALETTE.get('ELEMENT', self.DOOR_ID)))
        self.DOOR_TILE_IDS = {
            'open': self.DOOR_ID,
            'locked': tile_config.get('door_locked', self.DOOR_ID),
            'bombable': tile_config.get('door_bomb', self.DOOR_ID),
            'puzzle': tile_config.get('door_puzzle', self.DOOR_ID),
            'boss': tile_config.get('door_boss', self.DOOR_ID),
            'soft': tile_config.get('door_soft', self.DOOR_ID),
            'hazard': tile_config.get('hazard', hazard_default),
        }
        self.START_ID = tile_config.get('start')
        self.GOAL_ID = tile_config.get('goal')
    
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

        visual_grid = self._place_room_anchor(
            visual_grid,
            boundary,
            dict(mission_graph.get('nodes', {})).get(node_id, {}),
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
    ) -> Dict[int, Dict[str, Any]]:
        """Get neighbor IDs and edge semantics from the mission graph."""
        neighbors: Dict[int, Dict[str, Any]] = {}
        
        # Check adjacency dict (format: {node_id: {direction: neighbor_id}})
        if 'adjacency' in mission_graph:
            adjacency = mission_graph['adjacency']
            if node_id in adjacency:
                for neighbor_id in adjacency[node_id].values():
                    neighbors.setdefault(neighbor_id, {})
        
        # Also check edges list (format: [(src, dst), ...])
        if 'edges' in mission_graph:
            for edge in mission_graph['edges']:
                if len(edge) >= 2:
                    src, dst = edge[0], edge[1]
                    edge_data = dict(edge[2]) if len(edge) >= 3 and isinstance(edge[2], dict) else {}
                    raw_type = edge_data.get('edge_type', edge_data.get('type', edge_data.get('label', 'open')))
                    edge_type = str(getattr(raw_type, 'name', raw_type) or 'open').strip().lower()
                    if edge_type.startswith('edgetype.'):
                        edge_type = edge_type.split('.', 1)[1]
                    if edge_type in {'visual_link', 'window'}:
                        continue
                    metadata = edge_data.get('metadata', {})
                    implied_reverse = bool(
                        isinstance(metadata, dict) and metadata.get('implied_reverse', False)
                    )
                    gate_owner = dst if implied_reverse else src
                    edge_data['_spatial_gate_here'] = bool(node_id == gate_owner)
                    if src == node_id:
                        existing = neighbors.get(dst)
                        if existing is None or edge_data['_spatial_gate_here']:
                            neighbors[dst] = edge_data
                    elif dst == node_id:
                        existing = neighbors.get(src)
                        if existing is None or edge_data['_spatial_gate_here']:
                            neighbors[src] = edge_data
        
        return neighbors

    def _door_tile_for_edge(self, edge_data: Dict[str, Any]) -> int:
        """Map graph gate semantics to the corresponding spatial door tile."""
        if not bool(edge_data.get('_spatial_gate_here', True)):
            return int(self.DOOR_TILE_IDS['open'])
        raw = edge_data.get('edge_type', edge_data.get('type', edge_data.get('label', 'open')))
        edge_type = str(getattr(raw, 'name', raw) or 'open').strip().lower()
        if edge_type.startswith('edgetype.'):
            edge_type = edge_type.split('.', 1)[1]

        if edge_type in {'open', 'path', 'shortcut', 'hidden', ''}:
            return int(self.DOOR_TILE_IDS['open'])
        if edge_type in {'locked', 'key_locked', 'k'}:
            return int(self.DOOR_TILE_IDS['locked'])
        if edge_type in {'bombable', 'bomb', 'b'}:
            return int(self.DOOR_TILE_IDS['bombable'])
        if edge_type in {'boss_locked', 'boss'}:
            return int(self.DOOR_TILE_IDS['boss'])
        if edge_type in {'switch', 'switch_locked', 'state_block', 'on_off_gate', 'shutter'}:
            return int(self.DOOR_TILE_IDS['puzzle'])
        if edge_type in {'one_way', 'soft_locked'}:
            return int(self.DOOR_TILE_IDS['soft'])
        if edge_type == 'hazard':
            if edge_data.get('protection_item_id') is not None:
                return int(self.DOOR_TILE_IDS['hazard'])
            return int(self.DOOR_TILE_IDS['open'])
        if edge_type in {'visual_link', 'window'}:
            raise ValueError('VISUAL_LINK is non-traversable and cannot create a spatial door')
        raise ValueError(
            f"Edge type {edge_type!r} has no faithful tile-level representation; "
            "exclude it from the spatial rule profile or add an explicit mechanic."
        )
    
    def _create_doors(
        self,
        grid: np.ndarray,
        boundary: RoomBoundary,
        valid_neighbors: Dict[int, Dict[str, Any]],
        layout_map: Dict[int, Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Create doors to valid neighbor rooms only."""
        
        for neighbor_id, edge_data in valid_neighbors.items():
            if neighbor_id not in layout_map:
                continue
            
            neighbor_bounds = layout_map[neighbor_id]
            
            # Determine relative position and create door
            door_pos = self._find_door_position(
                boundary, 
                neighbor_bounds
            )
            
            if door_pos is not None:
                x, y = door_pos
                grid[y, x] = self._door_tile_for_edge(edge_data)
                logger.debug(f"Created door at ({x}, {y}) connecting rooms {boundary.node_id} and {neighbor_id}")
        
        return grid

    def _place_room_anchor(
        self,
        grid: np.ndarray,
        boundary: RoomBoundary,
        node_data: Dict[str, Any],
    ) -> np.ndarray:
        """Materialize START/GOAL graph roles on walkable interior tiles."""
        if not node_data:
            return grid
        raw_type = str(node_data.get('type', node_data.get('node_type', '')) or '').strip().lower()
        labels = {
            token.strip().lower()
            for token in str(node_data.get('label', '') or '').split(',')
            if token.strip()
        }
        is_start = bool(node_data.get('is_start', False)) or raw_type == 'start' or 's' in labels
        is_goal = (
            bool(node_data.get('is_goal', node_data.get('goal', False)))
            or bool(node_data.get('has_goal', False))
            or bool(node_data.get('is_triforce', False))
            or bool(node_data.get('has_triforce', False))
            or raw_type in {'goal', 'triforce'}
            or bool({'t', 'goal', 'triforce'} & labels)
        )
        anchor_id = self.GOAL_ID if is_goal else (self.START_ID if is_start else None)
        if anchor_id is None:
            return grid

        center_x = (boundary.x_min + boundary.x_max) // 2
        center_y = (boundary.y_min + boundary.y_max) // 2
        candidates: List[Tuple[int, int, int]] = []
        for y in range(boundary.y_min + 1, boundary.y_max):
            for x in range(boundary.x_min + 1, boundary.x_max):
                if int(grid[y, x]) != int(self.FLOOR_ID):
                    continue
                distance = abs(x - center_x) + abs(y - center_y)
                candidates.append((distance, y, x))
        if not candidates:
            raise ValueError(f"Room {boundary.node_id!r} has no floor tile for its START/GOAL anchor")
        _, y, x = min(candidates)
        grid[y, x] = int(anchor_id)
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
    door_ids = {
        int(tile_config.get('door', 2)),
        int(tile_config.get('door_locked', tile_config.get('door', 2))),
        int(tile_config.get('door_bomb', tile_config.get('door', 2))),
        int(tile_config.get('door_puzzle', tile_config.get('door', 2))),
        int(tile_config.get('door_boss', tile_config.get('door', 2))),
        int(tile_config.get('door_soft', tile_config.get('door', 2))),
        int(tile_config.get('hazard', tile_config.get('door', 2))),
    }
    
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
            if int(visual_grid[y_min, x]) in door_ids:
                # Find which room this connects to
                neighbor_id = _find_room_at_position(x, y_min - 1, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
        
        # Bottom edge
        for x in range(x_min, x_max + 1):
            if int(visual_grid[y_max, x]) in door_ids:
                neighbor_id = _find_room_at_position(x, y_max + 1, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
        
        # Left edge
        for y in range(y_min, y_max + 1):
            if int(visual_grid[y, x_min]) in door_ids:
                neighbor_id = _find_room_at_position(x_min - 1, y, layout_map)
                if neighbor_id is not None:
                    actual_edges.add(tuple(sorted([node_id, neighbor_id])))
        
        # Right edge
        for y in range(y_min, y_max + 1):
            if int(visual_grid[y, x_max]) in door_ids:
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
