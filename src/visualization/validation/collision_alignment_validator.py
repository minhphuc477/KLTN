"""
Feature 2: Collision Alignment Validator
=========================================
Pixel-perfect validation that generated visual tiles match logical collision boxes.

Problem:
    Diffusion model may generate decorative tiles that don't match their
    physical properties (visual floor that acts as wall, or vice versa).

Solution:
    - Extract collision masks from tile semantics
    - Validate alignment with A* pathfinding results
    - Detect "phantom walls" (visual floor, logical wall)
    - Detect "ghost floors" (visual wall, logical floor)
    - Generate diagnostic heatmaps

Integration Point: After symbolic refinement, before MAP-Elites validation
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class CollisionType(Enum):
    """Types of collision mismatches."""
    PERFECT = "perfect"  # Visual matches logical
    PHANTOM_WALL = "phantom_wall"  # Looks walkable but is blocked
    GHOST_FLOOR = "ghost_floor"  # Looks blocked but is walkable
    DOOR_MISMATCH = "door_mismatch"  # Door visual doesn't match navigation


@dataclass
class CollisionMismatch:
    """A single collision alignment error."""
    position: Tuple[int, int]  # (row, col)
    collision_type: CollisionType
    visual_tile_id: int
    logical_tile_id: int  # What it should be based on collision
    severity: float  # 0.0 (cosmetic) to 1.0 (critical)


@dataclass
class ValidationResult:
    """Result of collision alignment validation."""
    is_valid: bool
    total_pixels: int
    mismatched_pixels: int
    alignment_score: float  # 0.0 (all wrong) to 1.0 (perfect)
    mismatches: List[CollisionMismatch] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None  # (H, W) severity map


@dataclass
class CollisionConfig:
    """Configuration for collision validator."""
    # Tile semantics
    walkable_ids: Set[int] = field(default_factory=lambda: {1, 10, 11, 12, 13, 14, 15, 21, 22})
    blocking_ids: Set[int] = field(default_factory=lambda: {0, 2, 3, 20, 23, 40})
    door_ids: Set[int] = field(default_factory=lambda: {10, 11, 12, 13, 14, 15})
    
    # Validation thresholds
    alignment_threshold: float = 0.95  # Minimum for "valid"
    phantom_wall_severity: float = 1.0  # Critical (breaks gameplay)
    ghost_floor_severity: float = 0.7  # Serious (unexpected collision)
    door_mismatch_severity: float = 0.8  # High (confusing navigation)


# ============================================================================
# CORE ALGORITHM
# ============================================================================

class CollisionAlignmentValidator:
    """
    Validates pixel-perfect alignment between visual and logical collision.
    
    Algorithm:
    1. Extract visual collision mask from tile IDs
    2. Compute logical collision mask via A* reachability analysis
    3. Detect mismatches pixel-by-pixel
    4. Classify mismatch types and severity
    5. Generate diagnostic heatmap
    6. Compute overall alignment score
    """
    
    def __init__(self, config: Optional[CollisionConfig] = None):
        self.config = config or CollisionConfig()
    
    def validate_room(
        self,
        room_grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> ValidationResult:
        """
        Validate collision alignment for a single room.
        
        Args:
            room_grid: (H, W) tile ID grid
            start_pos: (row, col) starting position
            goal_pos: Optional goal position for path validation
        
        Returns:
            ValidationResult with alignment metrics and mismatches
        """
        H, W = room_grid.shape
        
        # Step 1: Extract visual collision mask
        visual_walkable = self._compute_visual_walkable(room_grid)
        
        # Step 2: Compute logical reachability from start
        logical_walkable = self._compute_logical_reachability(
            room_grid, start_pos, goal_pos
        )
        
        # Step 3: Detect mismatches
        mismatches = self._detect_mismatches(
            room_grid, visual_walkable, logical_walkable
        )
        
        # Step 4: Compute metrics
        total_pixels = H * W
        mismatched_pixels = len(mismatches)
        alignment_score = 1.0 - (mismatched_pixels / total_pixels)
        
        # Step 5: Generate heatmap
        heatmap = self._generate_heatmap(room_grid.shape, mismatches)
        
        # Step 6: Determine overall validity
        is_valid = alignment_score >= self.config.alignment_threshold
        
        if not is_valid:
            logger.warning(
                f"Collision alignment FAILED: {alignment_score:.2%} "
                f"({mismatched_pixels}/{total_pixels} mismatches)"
            )
        
        return ValidationResult(
            is_valid=is_valid,
            total_pixels=total_pixels,
            mismatched_pixels=mismatched_pixels,
            alignment_score=alignment_score,
            mismatches=mismatches,
            heatmap=heatmap
        )
    
    def validate_dungeon(
        self,
        dungeon_grid: np.ndarray,
        mission_graph,
        layout_map: Dict[int, Tuple[int, int, int, int]]
    ) -> Dict[int, ValidationResult]:
        """
        Validate collision alignment for entire dungeon.
        
        Args:
            dungeon_grid: Full (H, W) stitched dungeon
            mission_graph: Mission graph with room semantics
            layout_map: {room_id: (x_min, y_min, x_max, y_max)}
        
        Returns:
            {room_id: ValidationResult} for each room
        """
        results = {}
        
        for room_id, bbox in layout_map.items():
            x_min, y_min, x_max, y_max = bbox
            room_grid = dungeon_grid[y_min:y_max+1, x_min:x_max+1]
            
            # Extract start/goal from graph
            node_data = mission_graph.nodes[room_id]
            start_pos = node_data.get('start_pos', (1, 1))  # Default interior
            goal_pos = node_data.get('goal_pos', None)
            
            results[room_id] = self.validate_room(room_grid, start_pos, goal_pos)
        
        return results
    
    def _compute_visual_walkable(self, room_grid: np.ndarray) -> np.ndarray:
        """
        Compute walkability based on visual tile IDs.
        
        Returns:
            (H, W) boolean array: True = visually walkable
        """
        walkable_mask = np.isin(room_grid, list(self.config.walkable_ids))
        return walkable_mask
    
    def _compute_logical_reachability(
        self,
        room_grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute logical walkability via flood-fill reachability analysis.
        
        This reveals what the pathfinding algorithm *actually* considers walkable,
        which may differ from visual appearance.
        
        Returns:
            (H, W) boolean array: True = logically reachable
        """
        H, W = room_grid.shape
        reachable = np.zeros((H, W), dtype=bool)
        visited = np.zeros((H, W), dtype=bool)
        
        # Flood fill from start using actual collision rules
        from collections import deque
        queue = deque([start_pos])
        visited[start_pos] = True
        reachable[start_pos] = True
        
        while queue:
            r, c = queue.popleft()
            
            # Check 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                # Bounds check
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                
                if visited[nr, nc]:
                    continue
                
                # Check walkability using actual game logic
                tile_id = int(room_grid[nr, nc])
                if tile_id in self.config.walkable_ids:
                    visited[nr, nc] = True
                    reachable[nr, nc] = True
                    queue.append((nr, nc))
        
        return reachable
    
    def _detect_mismatches(
        self,
        room_grid: np.ndarray,
        visual_walkable: np.ndarray,
        logical_walkable: np.ndarray
    ) -> List[CollisionMismatch]:
        """
        Detect and classify collision mismatches.
        
        Mismatch Types:
        - Phantom Wall: Visual says "go", logic says "blocked"
        - Ghost Floor: Visual says "blocked", logic says "go"
        - Door Mismatch: Door tile, but wrong navigation behavior
        """
        mismatches = []
        H, W = room_grid.shape
        
        for r in range(H):
            for c in range(W):
                visual = visual_walkable[r, c]
                logical = logical_walkable[r, c]
                tile_id = int(room_grid[r, c])
                
                # Determine mismatch type
                if visual == logical:
                    continue  # Perfect alignment
                
                if visual and not logical:
                    # Phantom wall: Looks walkable, but blocked
                    mismatch = CollisionMismatch(
                        position=(r, c),
                        collision_type=CollisionType.PHANTOM_WALL,
                        visual_tile_id=tile_id,
                        logical_tile_id=2,  # Should be wall
                        severity=self.config.phantom_wall_severity
                    )
                    mismatches.append(mismatch)
                
                elif not visual and logical:
                    # Ghost floor: Looks blocked, but walkable
                    mismatch = CollisionMismatch(
                        position=(r, c),
                        collision_type=CollisionType.GHOST_FLOOR,
                        visual_tile_id=tile_id,
                        logical_tile_id=1,  # Should be floor
                        severity=self.config.ghost_floor_severity
                    )
                    mismatches.append(mismatch)
                
                # Check door-specific mismatches
                if tile_id in self.config.door_ids:
                    # Doors should always be logically walkable
                    if not logical:
                        mismatch = CollisionMismatch(
                            position=(r, c),
                            collision_type=CollisionType.DOOR_MISMATCH,
                            visual_tile_id=tile_id,
                            logical_tile_id=10,  # Should be open door
                            severity=self.config.door_mismatch_severity
                        )
                        mismatches.append(mismatch)
        
        return mismatches
    
    def _generate_heatmap(
        self,
        shape: Tuple[int, int],
        mismatches: List[CollisionMismatch]
    ) -> np.ndarray:
        """
        Generate severity heatmap for visualization.
        
        Returns:
            (H, W) float array with severity values [0.0, 1.0]
        """
        heatmap = np.zeros(shape, dtype=np.float32)
        
        for mismatch in mismatches:
            r, c = mismatch.position
            heatmap[r, c] = mismatch.severity
        
        return heatmap


# ============================================================================
# REPAIR ALGORITHM
# ============================================================================

class CollisionAlignmentRepairer:
    """
    Automatically repairs collision mismatches.
    
    Repair Strategy:
    - Phantom walls → Change visual to wall
    - Ghost floors → Change visual to floor
    - Door mismatches → Ensure door connectivity
    """
    
    def __init__(self, config: Optional[CollisionConfig] = None):
        self.config = config or CollisionConfig()
    
    def repair_room(
        self,
        room_grid: np.ndarray,
        validation_result: ValidationResult
    ) -> Tuple[np.ndarray, int]:
        """
        Repair collision mismatches in room.
        
        Args:
            room_grid: (H, W) tile ID grid
            validation_result: Result from CollisionAlignmentValidator
        
        Returns:
            (repaired_grid, num_repairs)
        """
        repaired = room_grid.copy()
        num_repairs = 0
        
        for mismatch in validation_result.mismatches:
            r, c = mismatch.position
            
            # Apply repair based on mismatch type
            if mismatch.collision_type == CollisionType.PHANTOM_WALL:
                # Visual walkable but logically blocked → make wall
                repaired[r, c] = 2  # WALL_ID
                num_repairs += 1
            
            elif mismatch.collision_type == CollisionType.GHOST_FLOOR:
                # Visual blocked but logically walkable → make floor
                repaired[r, c] = 1  # FLOOR_ID
                num_repairs += 1
            
            elif mismatch.collision_type == CollisionType.DOOR_MISMATCH:
                # Door should be walkable → ensure open door
                repaired[r, c] = 10  # DOOR_OPEN
                num_repairs += 1
        
        logger.info(f"Applied {num_repairs} collision repairs")
        return repaired, num_repairs


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/pipeline/dungeon_pipeline.py (after WFC repair):

from src.validation.collision_alignment_validator import (
    CollisionAlignmentValidator,
    CollisionAlignmentRepairer,
    CollisionConfig
)

class NeuralSymbolicDungeonPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.collision_validator = CollisionAlignmentValidator(CollisionConfig())
        self.collision_repairer = CollisionAlignmentRepairer(CollisionConfig())
    
    def generate_room(self, ...):
        # ... after WFC repair ...
        
        # Validate collision alignment
        collision_result = self.collision_validator.validate_room(
            room_grid=final_grid,
            start_pos=start_goal_coords[0] if start_goal_coords else (1, 1),
            goal_pos=start_goal_coords[1] if start_goal_coords else None
        )
        
        # Repair if needed
        if not collision_result.is_valid:
            logger.warning(
                f"Room {room_id}: Collision misalignment detected "
                f"(score={collision_result.alignment_score:.2%}), applying repairs"
            )
            final_grid, num_repairs = self.collision_repairer.repair_room(
                room_grid=final_grid,
                validation_result=collision_result
            )
            
            # Re-validate after repair
            collision_result = self.collision_validator.validate_room(
                room_grid=final_grid,
                start_pos=start_goal_coords[0] if start_goal_coords else (1, 1),
                goal_pos=start_goal_coords[1] if start_goal_coords else None
            )
        
        # Add collision metrics to result
        metrics['collision_alignment_score'] = collision_result.alignment_score
        metrics['collision_mismatches'] = collision_result.mismatched_pixels
        
        return RoomGenerationResult(...)


# In gui_runner.py (for visualization):

def _render_collision_heatmap(self, surface):
    '''Render collision alignment heatmap overlay.'''
    if self.collision_result is None:
        return
    
    heatmap = self.collision_result.heatmap
    for r in range(heatmap.shape[0]):
        for c in range(heatmap.shape[1]):
            severity = heatmap[r, c]
            if severity > 0:
                # Red overlay based on severity
                alpha = int(severity * 180)
                color = (255, 0, 0, alpha)
                rect = pygame.Rect(
                    c * self.tile_size,
                    r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                s = pygame.Surface((self.tile_size, self.tile_size))
                s.set_alpha(alpha)
                s.fill((255, 0, 0))
                surface.blit(s, rect)
"""
