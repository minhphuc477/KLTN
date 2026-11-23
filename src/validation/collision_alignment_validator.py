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
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CollisionType(Enum):
    """Types of collision mismatches."""
    PERFECT = "perfect"
    PHANTOM_WALL = "phantom_wall"
    GHOST_FLOOR = "ghost_floor"
    DOOR_MISMATCH = "door_mismatch"


@dataclass
class CollisionMismatch:
    """A single collision alignment error."""
    position: Tuple[int, int]
    collision_type: CollisionType
    visual_tile_id: int
    logical_tile_id: int
    severity: float


@dataclass
class ValidationResult:
    """Result of collision alignment validation."""
    is_valid: bool
    total_pixels: int
    mismatched_pixels: int
    alignment_score: float
    mismatches: List[CollisionMismatch] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None


@dataclass
class CollisionConfig:
    """Configuration for collision validator."""
    walkable_ids: Set[int] = field(default_factory=lambda: {1, 10, 11, 12, 13, 14, 15, 21, 22})
    blocking_ids: Set[int] = field(default_factory=lambda: {0, 2, 3, 20, 23, 40})
    door_ids: Set[int] = field(default_factory=lambda: {10, 11, 12, 13, 14, 15})
    alignment_threshold: float = 0.95
    phantom_wall_severity: float = 1.0
    ghost_floor_severity: float = 0.7
    door_mismatch_severity: float = 0.8


class CollisionAlignmentValidator:
    """Validates pixel-perfect alignment between visual and logical collision."""

    def __init__(self, config: Optional[CollisionConfig] = None):
        self.config = config or CollisionConfig()

    def validate_room(
        self,
        room_grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> ValidationResult:
        """Validate collision alignment for a single room."""
        height, width = room_grid.shape
        visual_walkable = self._compute_visual_walkable(room_grid)
        logical_walkable = self._compute_logical_reachability(room_grid, start_pos, goal_pos)
        mismatches = self._detect_mismatches(room_grid, visual_walkable, logical_walkable)

        total_pixels = height * width
        # Count unique mismatched coordinates even when a tile triggers multiple rules (e.g., unreachable door).
        mismatched_pixels = len({m.position for m in mismatches})
        alignment_score = 1.0 - (mismatched_pixels / max(1, total_pixels))
        heatmap = self._generate_heatmap(room_grid.shape, mismatches)
        is_valid = alignment_score >= self.config.alignment_threshold

        if not is_valid:
            logger.warning(
                "Collision alignment FAILED: %.2f%% (%d/%d mismatches)",
                alignment_score * 100.0,
                mismatched_pixels,
                total_pixels,
            )

        return ValidationResult(
            is_valid=is_valid,
            total_pixels=total_pixels,
            mismatched_pixels=mismatched_pixels,
            alignment_score=alignment_score,
            mismatches=mismatches,
            heatmap=heatmap,
        )

    def validate_dungeon(
        self,
        dungeon_grid: np.ndarray,
        mission_graph,
        layout_map: Dict[int, Tuple[int, int, int, int]],
    ) -> Dict[int, ValidationResult]:
        """Validate collision alignment for every room in a stitched dungeon."""
        results: Dict[int, ValidationResult] = {}
        for room_id, bbox in layout_map.items():
            x_min, y_min, x_max, y_max = bbox
            room_grid = dungeon_grid[y_min:y_max + 1, x_min:x_max + 1]
            node_data = mission_graph.nodes[room_id]
            start_pos = node_data.get("start_pos", (1, 1))
            goal_pos = node_data.get("goal_pos", None)
            results[room_id] = self.validate_room(room_grid, start_pos, goal_pos)
        return results

    def _compute_visual_walkable(self, room_grid: np.ndarray) -> np.ndarray:
        """Return (H, W) boolean mask of visually walkable tiles."""
        return np.isin(room_grid, list(self.config.walkable_ids))

    def _compute_logical_reachability(
        self,
        room_grid: np.ndarray,
        start_pos: Tuple[int, int],
        goal_pos: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """Compute logical walkability via BFS flood fill from start_pos."""
        del goal_pos
        height, width = room_grid.shape
        reachable = np.zeros((height, width), dtype=bool)
        visited = np.zeros((height, width), dtype=bool)

        sr, sc = int(start_pos[0]), int(start_pos[1])
        if not (0 <= sr < height and 0 <= sc < width):
            return reachable

        queue = deque([(sr, sc)])
        visited[sr, sc] = True
        reachable[sr, sc] = True

        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                    if int(room_grid[nr, nc]) in self.config.walkable_ids:
                        visited[nr, nc] = True
                        reachable[nr, nc] = True
                        queue.append((nr, nc))

        return reachable

    def _detect_mismatches(
        self,
        room_grid: np.ndarray,
        visual_walkable: np.ndarray,
        logical_walkable: np.ndarray,
    ) -> List[CollisionMismatch]:
        """Detect and classify per-tile collision mismatches."""
        mismatches: List[CollisionMismatch] = []
        height, width = room_grid.shape

        for r in range(height):
            for c in range(width):
                visual = bool(visual_walkable[r, c])
                logical = bool(logical_walkable[r, c])
                tile_id = int(room_grid[r, c])

                if visual == logical:
                    continue

                if visual and not logical:
                    mismatches.append(
                        CollisionMismatch(
                            position=(r, c),
                            collision_type=CollisionType.PHANTOM_WALL,
                            visual_tile_id=tile_id,
                            logical_tile_id=2,
                            severity=self.config.phantom_wall_severity,
                        )
                    )
                elif not visual and logical:
                    mismatches.append(
                        CollisionMismatch(
                            position=(r, c),
                            collision_type=CollisionType.GHOST_FLOOR,
                            visual_tile_id=tile_id,
                            logical_tile_id=1,
                            severity=self.config.ghost_floor_severity,
                        )
                    )

                if tile_id in self.config.door_ids and not logical:
                    mismatches.append(
                        CollisionMismatch(
                            position=(r, c),
                            collision_type=CollisionType.DOOR_MISMATCH,
                            visual_tile_id=tile_id,
                            logical_tile_id=10,
                            severity=self.config.door_mismatch_severity,
                        )
                    )

        return mismatches

    def _generate_heatmap(
        self,
        shape: Tuple[int, int],
        mismatches: List[CollisionMismatch],
    ) -> np.ndarray:
        """Generate (H, W) float severity heatmap in [0.0, 1.0]."""
        heatmap = np.zeros(shape, dtype=np.float32)
        for mismatch in mismatches:
            r, c = mismatch.position
            heatmap[r, c] = max(heatmap[r, c], float(mismatch.severity))
        return heatmap


class CollisionAlignmentRepairer:
    """Automatically repairs collision mismatches reported by CollisionAlignmentValidator."""

    def __init__(self, config: Optional[CollisionConfig] = None):
        self.config = config or CollisionConfig()

    def repair_room(
        self,
        room_grid: np.ndarray,
        validation_result: ValidationResult,
    ) -> Tuple[np.ndarray, int]:
        """Repair visual tiles in room_grid using detected collision mismatches."""
        repaired = room_grid.copy()
        num_repairs = 0

        for mismatch in validation_result.mismatches:
            r, c = mismatch.position
            if mismatch.collision_type == CollisionType.PHANTOM_WALL:
                repaired[r, c] = 2
                num_repairs += 1
            elif mismatch.collision_type == CollisionType.GHOST_FLOOR:
                repaired[r, c] = 1
                num_repairs += 1
            elif mismatch.collision_type == CollisionType.DOOR_MISMATCH:
                repaired[r, c] = 10
                num_repairs += 1

        logger.info("Applied %d collision repairs", num_repairs)
        return repaired, num_repairs
