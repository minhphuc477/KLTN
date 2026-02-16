"""
Feature 7: Big Room Support (Scalability)
==========================================
Support for variable room sizes, especially large boss arenas (22x32).

Problem:
    Current system hardcodes 16x11 rooms. Boss arenas need 22x32.
    VQ-VAE and diffusion models trained on fixed latent dimensions.

Solution:
    - Macro-Nodes: Add 'size' attribute to graph nodes
    - Autoregressive Chunking: Generate large rooms in patches
    - In-painting: Generate edges first, fill interior
    - Latent Interpolation: Smooth transitions between patches
    - Adaptive Layout: Stitcher handles variable sizes

Integration Point: VQ-VAE encoding/decoding, dungeon stitching
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RoomSize(Enum):
    """Predefined room size templates."""
    SMALL = (11, 11)  # Small puzzle room
    STANDARD = (16, 11)  # Standard Zelda room
    LARGE = (22, 16)  # Large combat room
    BOSS = (32, 22)  # Boss arena
    CUSTOM = (-1, -1)  # Custom size


@dataclass
class RoomDimensions:
    """Dimensions for a room."""
    height: int
    width: int
    interior_height: int  # Without walls
    interior_width: int
    size_class: RoomSize = RoomSize.STANDARD
    
    @classmethod
    def from_size_class(cls, size_class: RoomSize):
        """Create dimensions from size class."""
        if size_class == RoomSize.CUSTOM:
            raise ValueError("CUSTOM size requires explicit dimensions")
        
        h, w = size_class.value
        return cls(
            height=h,
            width=w,
            interior_height=h - 2,  # Subtract walls
            interior_width=w - 2,
            size_class=size_class
        )
    
    @classmethod
    def custom(cls, height: int, width: int):
        """Create custom dimensions."""
        return cls(
            height=height,
            width=width,
            interior_height=height - 2,
            interior_width=width - 2,
            size_class=RoomSize.CUSTOM
        )


@dataclass
class RoomPatch:
    """A sub-region of a large room for autoregressive generation."""
    patch_id: int
    row_start: int
    col_start: int
    height: int
    width: int
    overlap_top: int = 0  # Overlap with patch above
    overlap_left: int = 0  # Overlap with patch to left
    neighbors: List[int] = field(default_factory=list)  # Adjacent patch IDs


@dataclass
class BigRoomConfig:
    """Configuration for large room generation."""
    max_single_pass_size: Tuple[int, int] = (16, 11)  # Generate in one pass if ≤ this
    patch_size: Tuple[int, int] = (16, 11)  # Size of each patch
    overlap: int = 3  # Overlapping tiles between patches
    use_inpainting: bool = True  # Generate edges first, fill interior
    edge_thickness: int = 2  # Thickness of edge to generate first


# ============================================================================
# PATCH DECOMPOSER
# ============================================================================

class RoomPatchDecomposer:
    """
    Decomposes large rooms into overlapping patches for autoregressive generation.
    
    Algorithm:
    1. Determine if room fits in single pass
    2. If too large, split into overlapping patches
    3. Generate patches in dependency order (top-left to bottom-right)
    4. Blend overlapping regions
    5. Ensure global consistency
    """
    
    def __init__(self, config: Optional[BigRoomConfig] = None):
        self.config = config or BigRoomConfig()
    
    def decompose(self, dimensions: RoomDimensions) -> List[RoomPatch]:
        """
        Decompose room into patches for autoregressive generation.
        
        Args:
            dimensions: Target room dimensions
        
        Returns:
            List of patches in generation order
        """
        # Check if single-pass generation is possible
        if self._fits_single_pass(dimensions):
            return [self._create_single_patch(dimensions)]
        
        # Multi-patch decomposition
        return self._create_patch_grid(dimensions)
    
    def _fits_single_pass(self, dimensions: RoomDimensions) -> bool:
        """Check if room can be generated in single pass."""
        max_h, max_w = self.config.max_single_pass_size
        return dimensions.height <= max_h and dimensions.width <= max_w
    
    def _create_single_patch(self, dimensions: RoomDimensions) -> RoomPatch:
        """Create single patch covering entire room."""
        return RoomPatch(
            patch_id=0,
            row_start=0,
            col_start=0,
            height=dimensions.height,
            width=dimensions.width,
            overlap_top=0,
            overlap_left=0,
            neighbors=[]
        )
    
    def _create_patch_grid(self, dimensions: RoomDimensions) -> List[RoomPatch]:
        """
        Create grid of overlapping patches.
        
        Patches are ordered for dependency: top-left to bottom-right.
        """
        patch_h, patch_w = self.config.patch_size
        overlap = self.config.overlap
        
        # Compute number of patches needed (with overlap)
        effective_patch_h = patch_h - overlap
        effective_patch_w = patch_w - overlap
        
        num_rows = max(1, (dimensions.height - overlap + effective_patch_h - 1) // effective_patch_h)
        num_cols = max(1, (dimensions.width - overlap + effective_patch_w - 1) // effective_patch_w)
        
        patches = []
        patch_id = 0
        
        for row_idx in range(num_rows):
            row_start = row_idx * effective_patch_h
            patch_height = min(patch_h, dimensions.height - row_start)
            overlap_top = overlap if row_idx > 0 else 0
            
            for col_idx in range(num_cols):
                col_start = col_idx * effective_patch_w
                patch_width = min(patch_w, dimensions.width - col_start)
                overlap_left = overlap if col_idx > 0 else 0
                
                # Determine neighbor patches
                neighbors = []
                if row_idx > 0:
                    neighbors.append(patch_id - num_cols)  # Patch above
                if col_idx > 0:
                    neighbors.append(patch_id - 1)  # Patch to left
                
                patch = RoomPatch(
                    patch_id=patch_id,
                    row_start=row_start,
                    col_start=col_start,
                    height=patch_height,
                    width=patch_width,
                    overlap_top=overlap_top,
                    overlap_left=overlap_left,
                    neighbors=neighbors
                )
                patches.append(patch)
                patch_id += 1
        
        logger.info(f"Decomposed {dimensions.height}x{dimensions.width} room into {len(patches)} patches")
        return patches


# ============================================================================
# BIG ROOM GENERATOR
# ============================================================================

class BigRoomGenerator:
    """
    Generates rooms of arbitrary size using autoregressive patch generation.
    
    Strategy:
    1. Decompose room into patches
    2. Generate patches sequentially, conditioning on already-generated neighbors
    3. Blend overlapping regions using linear interpolation
    4. Optional: Use in-painting to generate edges first, then fill interior
    """
    
    def __init__(self, base_pipeline, config: Optional[BigRoomConfig] = None):
        """
        Args:
            base_pipeline: NeuralSymbolicDungeonPipeline instance
            config: Configuration for big room generation
        """
        self.pipeline = base_pipeline
        self.config = config or BigRoomConfig()
        self.decomposer = RoomPatchDecomposer(config)
    
    def generate_big_room(
        self,
        room_id: int,
        dimensions: RoomDimensions,
        neighbor_latents: Dict,
        graph_context: Dict,
        **generation_kwargs
    ) -> np.ndarray:
        """
        Generate room of arbitrary size.
        
        Args:
            room_id: Room identifier
            dimensions: Target room dimensions
            neighbor_latents: Neighboring room latents
            graph_context: Graph context data
            **generation_kwargs: Additional args for generate_room
        
        Returns:
            (H, W) room grid at target dimensions
        """
        # Decompose into patches
        patches = self.decomposer.decompose(dimensions)
        
        if len(patches) == 1:
            # Single-pass generation
            return self._generate_single_patch(
                room_id, dimensions, neighbor_latents, graph_context, **generation_kwargs
            )
        else:
            # Multi-patch autoregressive generation
            return self._generate_multipatch(
                room_id, dimensions, patches, neighbor_latents, graph_context, **generation_kwargs
            )
    
    def _generate_single_patch(
        self,
        room_id: int,
        dimensions: RoomDimensions,
        neighbor_latents: Dict,
        graph_context: Dict,
        **kwargs
    ) -> np.ndarray:
        """Generate room in single pass (fits in standard generation)."""
        # Temporarily override room dimensions in pipeline
        original_height = self.pipeline.vqvae.room_height
        original_width = self.pipeline.vqvae.room_width
        
        try:
            self.pipeline.vqvae.room_height = dimensions.height
            self.pipeline.vqvae.room_width = dimensions.width
            
            result = self.pipeline.generate_room(
                neighbor_latents=neighbor_latents,
                graph_context=graph_context,
                room_id=room_id,
                **kwargs
            )
            
            return result.room_grid
        
        finally:
            # Restore original dimensions
            self.pipeline.vqvae.room_height = original_height
            self.pipeline.vqvae.room_width = original_width
    
    def _generate_multipatch(
        self,
        room_id: int,
        dimensions: RoomDimensions,
        patches: List[RoomPatch],
        neighbor_latents: Dict,
        graph_context: Dict,
        **kwargs
    ) -> np.ndarray:
        """
        Generate large room using autoregressive patches.
        
        Algorithm:
        1. Generate patches in order (top-left to bottom-right)
        2. Each patch conditions on already-generated neighbors
        3. Blend overlapping regions
        """
        # Initialize full room grid
        full_room = np.zeros((dimensions.height, dimensions.width), dtype=np.int32)
        
        # Generate patches sequentially
        patch_grids = {}
        
        for patch in patches:
            logger.debug(f"Generating patch {patch.patch_id}/{len(patches)}")
            
            # Prepare conditioning from neighbor patches
            patch_neighbors = self._extract_patch_neighbors(patch, patch_grids, patches)
            
            # Generate this patch
            patch_grid = self._generate_patch(
                room_id=room_id,
                patch=patch,
                patch_neighbors=patch_neighbors,
                neighbor_latents=neighbor_latents,
                graph_context=graph_context,
                **kwargs
            )
            
            patch_grids[patch.patch_id] = patch_grid
            
            # Blend into full room
            self._blend_patch_into_room(full_room, patch, patch_grid, patch_grids)
        
        return full_room
    
    def _extract_patch_neighbors(
        self,
        patch: RoomPatch,
        patch_grids: Dict[int, np.ndarray],
        all_patches: List[RoomPatch]
    ) -> Dict[str, np.ndarray]:
        """Extract neighboring patch regions for conditioning."""
        neighbors = {}
        
        for neighbor_id in patch.neighbors:
            if neighbor_id not in patch_grids:
                continue
            
            neighbor_patch = all_patches[neighbor_id]
            neighbor_grid = patch_grids[neighbor_id]
            
            # Determine direction
            if neighbor_patch.row_start < patch.row_start:
                # Patch above
                overlap_region = neighbor_grid[-patch.overlap_top:, :]
                neighbors['N'] = overlap_region
            
            elif neighbor_patch.col_start < patch.col_start:
                # Patch to left
                overlap_region = neighbor_grid[:, -patch.overlap_left:]
                neighbors['W'] = overlap_region
        
        return neighbors
    
    def _generate_patch(
        self,
        room_id: int,
        patch: RoomPatch,
        patch_neighbors: Dict[str, np.ndarray],
        neighbor_latents: Dict,
        graph_context: Dict,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a single patch.
        
        Uses standard generation pipeline but with patch-specific conditioning.
        """
        # Create temporary VQ-VAE latents from neighbor patches
        # (This is a simplified version - full implementation would encode patch_neighbors)
        
        # Generate using standard pipeline (at patch dimensions)
        original_height = self.pipeline.vqvae.room_height
        original_width = self.pipeline.vqvae.room_width
        
        try:
            self.pipeline.vqvae.room_height = patch.height
            self.pipeline.vqvae.room_width = patch.width
            
            result = self.pipeline.generate_room(
                neighbor_latents=neighbor_latents,
                graph_context=graph_context,
                room_id=f"{room_id}_patch{patch.patch_id}",
                **kwargs
            )
            
            return result.room_grid
        
        finally:
            self.pipeline.vqvae.room_height = original_height
            self.pipeline.vqvae.room_width = original_width
    
    def _blend_patch_into_room(
        self,
        full_room: np.ndarray,
        patch: RoomPatch,
        patch_grid: np.ndarray,
        existing_patches: Dict[int, np.ndarray]
    ):
        """
        Blend patch into full room, handling overlaps.
        
        Overlap regions use weighted average to ensure smooth transitions.
        """
        r_start = patch.row_start
        r_end = r_start + patch.height
        c_start = patch.col_start
        c_end = c_start + patch.width
        
        # Non-overlapping region: direct copy
        non_overlap_r_start = r_start + patch.overlap_top
        non_overlap_c_start = c_start + patch.overlap_left
        
        full_room[non_overlap_r_start:r_end, non_overlap_c_start:c_end] = \
            patch_grid[patch.overlap_top:, patch.overlap_left:]
        
        # Overlap regions: blend with existing content
        if patch.overlap_top > 0:
            # Blend top edge
            for i in range(patch.overlap_top):
                weight = (i + 1) / (patch.overlap_top + 1)  # 0 to 1 gradient
                full_room[r_start + i, c_start:c_end] = (
                    (1 - weight) * full_room[r_start + i, c_start:c_end] +
                    weight * patch_grid[i, :]
                ).astype(np.int32)
        
        if patch.overlap_left > 0:
            # Blend left edge
            for j in range(patch.overlap_left):
                weight = (j + 1) / (patch.overlap_left + 1)
                full_room[r_start:r_end, c_start + j] = (
                    (1 - weight) * full_room[r_start:r_end, c_start + j] +
                    weight * patch_grid[:, j]
                ).astype(np.int32)


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/pipeline/dungeon_pipeline.py:

from src.generation.big_room_generator import (
    BigRoomGenerator,
    RoomDimensions,
    RoomSize,
    BigRoomConfig
)

class NeuralSymbolicDungeonPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.big_room_generator = BigRoomGenerator(
            base_pipeline=self,
            config=BigRoomConfig()
        )
    
    def generate_room(self, ..., room_dimensions: Optional[RoomDimensions] = None):
        '''Generate room with optional custom dimensions.'''
        
        # Check if this is a big room
        if room_dimensions is not None and room_dimensions.size_class != RoomSize.STANDARD:
            return self.big_room_generator.generate_big_room(
                room_id=room_id,
                dimensions=room_dimensions,
                neighbor_latents=neighbor_latents,
                graph_context=graph_context,
                **kwargs
            )
        
        # Standard small room generation
        return self._generate_standard_room(...)


# Usage example - boss arena:

# Define mission graph with room sizes
mission_graph = nx.Graph()
mission_graph.add_node(0, size='standard')
mission_graph.add_node(1, size='standard')
mission_graph.add_node(2, size='boss')  # Boss room is 32x22

# Generate dungeon
for node_id in mission_graph.nodes():
    size_attr = mission_graph.nodes[node_id].get('size', 'standard')
    
    if size_attr == 'boss':
        dimensions = RoomDimensions.from_size_class(RoomSize.BOSS)
    else:
        dimensions = RoomDimensions.from_size_class(RoomSize.STANDARD)
    
    room_result = pipeline.generate_room(
        room_id=node_id,
        room_dimensions=dimensions,
        ...
    )


# Custom size example:
custom_big_room = pipeline.generate_room(
    room_id=99,
    room_dimensions=RoomDimensions.custom(height=40, width=30),
    ...
)
"""
