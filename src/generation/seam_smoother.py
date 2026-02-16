"""
Feature 1: Seam Smoothing
==========================
Ensures visual continuity at room boundaries by blending shared edges.

Problem:
    Adjacent rooms may have misaligned walls at their shared boundaries,
    creating visual discontinuities (e.g., two walls meeting at a door).

Solution:
    - Detect shared boundaries between rooms
    - Apply morphological smoothing at seams
    - Enforce door alignment constraints
    - Use bilateral filtering to preserve sharp features

Integration Point: After room stitching in DungeonStitcher
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SeamRegion:
    """A boundary region between two adjacent rooms."""
    room1_id: int
    room2_id: int
    orientation: str  # 'horizontal' or 'vertical'
    room1_slice: Tuple[slice, slice]  # (row_slice, col_slice)
    room2_slice: Tuple[slice, slice]
    shared_slice: Tuple[slice, slice]  # Region in stitched grid
    door_positions: List[Tuple[int, int]]  # Door coordinates in shared region


@dataclass
class SmoothingConfig:
    """Configuration for seam smoothing algorithm."""
    kernel_size: int = 5
    blend_width: int = 3  # Pixels to blend on each side
    door_preserve_radius: int = 2  # Don't smooth near doors
    wall_priority: bool = True  # Prefer walls over floors in conflicts
    use_bilateral: bool = True  # Preserve sharp edges
    sigma_spatial: float = 2.0
    sigma_range: float = 10.0


# ============================================================================
# CORE ALGORITHM
# ============================================================================

class SeamSmoother:
    """
    Smooths visual discontinuities at room boundaries.
    
    Algorithm:
    1. Identify all room adjacencies from mission graph
    2. Extract seam regions (overlapping boundaries)
    3. Detect door positions in seams
    4. Apply bilateral filtering to smooth transitions
    5. Enforce symmetry constraints (both rooms see same boundary)
    6. Preserve door tiles exactly
    """
    
    def __init__(self, config: Optional[SmoothingConfig] = None):
        self.config = config or SmoothingConfig()
        self.DOOR_IDS = {10, 11, 12, 13, 14, 15}  # All door types
        self.WALL_ID = 2
        self.FLOOR_ID = 1
    
    def smooth_dungeon_seams(
        self,
        dungeon_grid: np.ndarray,
        mission_graph,
        layout_map: Dict[int, Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Apply seam smoothing to complete dungeon.
        
        Args:
            dungeon_grid: (H, W) stitched dungeon grid
            mission_graph: NetworkX graph with edges = adjacencies
            layout_map: {room_id: (x_min, y_min, x_max, y_max)}
        
        Returns:
            Smoothed dungeon grid (same shape)
        """
        smoothed = dungeon_grid.copy()
        
        # Step 1: Identify all seams from graph edges
        seams = self._extract_seams(mission_graph, layout_map, dungeon_grid.shape)
        logger.info(f"SeamSmoother: Processing {len(seams)} room boundaries")
        
        # Step 2: Smooth each seam
        for seam in seams:
            smoothed = self._smooth_seam(smoothed, seam)
        
        return smoothed
    
    def _extract_seams(
        self,
        mission_graph,
        layout_map: Dict[int, Tuple[int, int, int, int]],
        grid_shape: Tuple[int, int]
    ) -> List[SeamRegion]:
        """Extract all seam regions from graph adjacencies."""
        seams = []
        
        for edge in mission_graph.edges():
            room1, room2 = edge
            if room1 not in layout_map or room2 not in layout_map:
                continue
            
            bbox1 = layout_map[room1]  # (x_min, y_min, x_max, y_max)
            bbox2 = layout_map[room2]
            
            # Detect orientation and compute shared boundary
            seam = self._compute_shared_boundary(room1, room2, bbox1, bbox2, grid_shape)
            if seam is not None:
                seams.append(seam)
        
        return seams
    
    def _compute_shared_boundary(
        self,
        room1_id: int,
        room2_id: int,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        grid_shape: Tuple[int, int]
    ) -> Optional[SeamRegion]:
        """Compute the shared boundary between two rooms."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Check for horizontal adjacency (left-right)
        if x1_max + 1 == x2_min or x2_max + 1 == x1_min:
            # Vertical seam (shared column)
            if x1_max + 1 == x2_min:
                seam_col = x1_max
                left_room, right_room = room1_id, room2_id
            else:
                seam_col = x2_max
                left_room, right_room = room2_id, room1_id
            
            # Find overlapping rows
            row_min = max(y1_min, y2_min)
            row_max = min(y1_max, y2_max)
            
            if row_min <= row_max and 0 <= seam_col < grid_shape[1]:
                return SeamRegion(
                    room1_id=left_room,
                    room2_id=right_room,
                    orientation='vertical',
                    room1_slice=(slice(row_min, row_max+1), slice(seam_col, seam_col+1)),
                    room2_slice=(slice(row_min, row_max+1), slice(seam_col+1, seam_col+2)),
                    shared_slice=(slice(row_min, row_max+1), slice(seam_col, seam_col+2)),
                    door_positions=[]
                )
        
        # Check for vertical adjacency (top-bottom)
        elif y1_max + 1 == y2_min or y2_max + 1 == y1_min:
            # Horizontal seam (shared row)
            if y1_max + 1 == y2_min:
                seam_row = y1_max
                top_room, bottom_room = room1_id, room2_id
            else:
                seam_row = y2_max
                top_room, bottom_room = room2_id, room1_id
            
            # Find overlapping columns
            col_min = max(x1_min, x2_min)
            col_max = min(x1_max, x2_max)
            
            if col_min <= col_max and 0 <= seam_row < grid_shape[0]:
                return SeamRegion(
                    room1_id=top_room,
                    room2_id=bottom_room,
                    orientation='horizontal',
                    room1_slice=(slice(seam_row, seam_row+1), slice(col_min, col_max+1)),
                    room2_slice=(slice(seam_row+1, seam_row+2), slice(col_min, col_max+1)),
                    shared_slice=(slice(seam_row, seam_row+2), slice(col_min, col_max+1)),
                    door_positions=[]
                )
        
        return None
    
    def _smooth_seam(self, dungeon_grid: np.ndarray, seam: SeamRegion) -> np.ndarray:
        """
        Apply smoothing to a single seam region.
        
        Algorithm:
        1. Extract seam region + context
        2. Detect door positions (preserve exactly)
        3. Apply bilateral filter for edge-preserving smoothing
        4. Enforce wall priority (walls win over floors)
        5. Apply symmetry (both sides match)
        """
        result = dungeon_grid.copy()
        
        # Extract seam region with context (for filtering)
        row_slice, col_slice = seam.shared_slice
        context_size = self.config.blend_width
        
        row_start = max(0, row_slice.start - context_size)
        row_end = min(dungeon_grid.shape[0], row_slice.stop + context_size)
        col_start = max(0, col_slice.start - context_size)
        col_end = min(dungeon_grid.shape[1], col_slice.stop + context_size)
        
        region = dungeon_grid[row_start:row_end, col_start:col_end].astype(float)
        
        # Step 1: Detect doors in seam (these are sacred, never touch)
        door_mask = np.isin(region, list(self.DOOR_IDS))
        
        # Step 2: Apply bilateral filtering to smooth non-door tiles
        if self.config.use_bilateral:
            smoothed_region = self._bilateral_filter(region, door_mask)
        else:
            smoothed_region = median_filter(region, size=self.config.kernel_size)
        
        # Step 3: Enforce wall priority and symmetry
        smoothed_region = self._enforce_wall_priority(smoothed_region, region)
        smoothed_region = self._enforce_symmetry(smoothed_region, seam, row_start, col_start)
        
        # Step 4: Restore doors exactly
        smoothed_region[door_mask] = region[door_mask]
        
        # Write back to result
        result[row_start:row_end, col_start:col_end] = np.round(smoothed_region).astype(dungeon_grid.dtype)
        
        return result
    
    def _bilateral_filter(self, region: np.ndarray, door_mask: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving bilateral filter.
        
        Bilateral filter smooths gradual transitions while preserving sharp edges
        (like wall-floor boundaries), making it ideal for seam smoothing.
        """
        from scipy.ndimage import gaussian_filter
        
        # Simple bilateral approximation using domain + range Gaussians
        # For full bilateral, use OpenCV's cv2.bilateralFilter
        
        # Domain filter (spatial smoothing)
        spatial_filtered = gaussian_filter(region, sigma=self.config.sigma_spatial)
        
        # Range filter (value-based smoothing)
        # Only smooth similar values together
        result = region.copy()
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                if door_mask[i, j]:
                    continue  # Skip doors
                
                # Compute weighted average based on value similarity
                window_size = self.config.kernel_size // 2
                i_min = max(0, i - window_size)
                i_max = min(region.shape[0], i + window_size + 1)
                j_min = max(0, j - window_size)
                j_max = min(region.shape[1], j + window_size + 1)
                
                window = region[i_min:i_max, j_min:j_max]
                center_value = region[i, j]
                
                # Gaussian weights based on value difference
                value_diff = np.abs(window - center_value)
                range_weights = np.exp(- (value_diff ** 2) / (2 * self.config.sigma_range ** 2))
                
                # Spatial weights
                y_coords, x_coords = np.ogrid[i_min:i_max, j_min:j_max]
                spatial_diff = (y_coords - i) ** 2 + (x_coords - j) ** 2
                spatial_weights = np.exp(- spatial_diff / (2 * self.config.sigma_spatial ** 2))
                
                # Combined weights
                weights = range_weights * spatial_weights
                result[i, j] = np.sum(window * weights) / (np.sum(weights) + 1e-8)
        
        return result
    
    def _enforce_wall_priority(self, smoothed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Ensure walls are preserved over floors in smoothed result."""
        if not self.config.wall_priority:
            return smoothed
        
        # If original tile was a wall and smoothing softened it, restore wall
        wall_mask = (original == self.WALL_ID)
        smoothed[wall_mask] = self.WALL_ID
        
        return smoothed
    
    def _enforce_symmetry(
        self,
        region: np.ndarray,
        seam: SeamRegion,
        row_offset: int,
        col_offset: int
    ) -> np.ndarray:
        """Ensure both sides of seam see same boundary."""
        # Map global seam coords to local region coords
        global_row_slice, global_col_slice = seam.shared_slice
        local_row_start = global_row_slice.start - row_offset
        local_row_end = global_row_slice.stop - row_offset
        local_col_start = global_col_slice.start - col_offset
        local_col_end = global_col_slice.stop - col_offset
        
        # Clamp to valid region
        local_row_start = max(0, local_row_start)
        local_row_end = min(region.shape[0], local_row_end)
        local_col_start = max(0, local_col_start)
        local_col_end = min(region.shape[1], local_col_end)
        
        if seam.orientation == 'horizontal':
            # Average across the two rows
            if local_row_end > local_row_start + 1:
                seam_region = region[local_row_start:local_row_end, local_col_start:local_col_end]
                averaged = np.mean(seam_region, axis=0, keepdims=True)
                region[local_row_start:local_row_end, local_col_start:local_col_end] = averaged
        
        elif seam.orientation == 'vertical':
            # Average across the two columns
            if local_col_end > local_col_start + 1:
                seam_region = region[local_row_start:local_row_end, local_col_start:local_col_end]
                averaged = np.mean(seam_region, axis=1, keepdims=True)
                region[local_row_start:local_row_end, local_col_start:local_col_end] = averaged
        
        return region


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/data/zelda_core.py (DungeonStitcher class):

from src.generation.seam_smoother import SeamSmoother, SmoothingConfig

class DungeonStitcher:
    def __init__(self, ...):
        # ... existing init ...
        self.seam_smoother = SeamSmoother(SmoothingConfig(
            kernel_size=5,
            blend_width=3,
            use_bilateral=True
        ))
    
    def stitch_rooms(self, rooms_dict, mission_graph):
        # ... existing stitching logic ...
        
        # After stitching, smooth seams
        stitched_grid = self.seam_smoother.smooth_dungeon_seams(
            dungeon_grid=stitched_grid,
            mission_graph=mission_graph,
            layout_map=self.layout_map
        )
        
        return stitched_grid


# In src/pipeline/dungeon_pipeline.py (generate_dungeon):

def generate_dungeon(self, ...):
    # ... after stitching ...
    
    logger.info("Applying seam smoothing for visual continuity")
    dungeon_result.dungeon_grid = self.stitcher.smooth_dungeon_seams(
        dungeon_grid=dungeon_result.dungeon_grid,
        mission_graph=mission_graph,
        layout_map=layout_map
    )
    
    return dungeon_result
"""
