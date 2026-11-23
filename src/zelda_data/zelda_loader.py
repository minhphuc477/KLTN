"""Zelda Dungeon Dataset Loader (Facade & DataLoader Factory).

Consolidates and re-exports:
- Symmetry transforms from `src.zelda_data.symmetry`
- Frequency calculation from `src.zelda_data.frequencies`
- Batching & collation from `src.zelda_data.batching`
- Graph extraction from `src.zelda_data.graph_extraction`
- Dataset classes from `src.zelda_data.datasets`
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.definitions import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    semantic_grid_to_vglc_lines,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
)
from src.utils.data_loading import dataloader_runtime_kwargs, seed_dataloader_worker
from src.zelda_data.batching import DungeonBatchSampler, graph_collate_fn
from src.zelda_data.datasets import (
    TILE_MAPPING,
    VGLC_AVAILABLE,
    ZeldaDungeonDataset,
    ZeldaRoomDataset,
)
from src.zelda_data.frequencies import compute_room_tile_frequencies
from src.zelda_data.graph_extraction import (
    _EDGE_TYPE_ENCODING,
    _build_room_graph_sample,
    _extract_explicit_style_id,
    _extract_graph_from_dungeon,
    _extract_graph_spatial_from_dungeon,
    _room_role_flags,
    validate_floor_conditioning_signal,
)
from src.zelda_data.symmetry import (
    RandomD4Symmetry,
    _flip_room_coord_pair,
    _flip_spatial_tensor_or_array,
    _remap_symmetry_anchor_name,
    _remap_symmetry_direction,
    _swap_topology_directional_channels,
    _transform_puzzle_stage_condition,
    apply_room_symmetry_transform,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ZeldaDungeonDataset",
    "ZeldaRoomDataset",
    "RandomD4Symmetry",
    "apply_room_symmetry_transform",
    "compute_room_tile_frequencies",
    "graph_collate_fn",
    "DungeonBatchSampler",
    "validate_floor_conditioning_signal",
    "create_dataloader",
    "tensor_to_ascii",
    "extract_start_goal",
    "TILE_MAPPING",
    "VGLC_AVAILABLE",
    "_extract_graph_from_dungeon",
    "_extract_graph_spatial_from_dungeon",
    "_build_room_graph_sample",
]

# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
    use_vglc: bool = False,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    transform: Optional[Callable] = None,
    room_level: bool = False,
    load_graphs: bool = False,
    node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
    edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
    topology_supervision_mode: str = "runtime_aligned",
    semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    puzzle_stage_topology_enabled: bool = False,
    puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    sampler: Optional[Sampler] = None,
    dungeon_batch_mode: bool = False,
    return_sampler: bool = False,
    dungeon_ids: Optional[Iterable[int]] = None,
    variants: Optional[Iterable[int]] = None,
    augment: bool = False,
    lazy_vglc: bool = True,
    categorical_tokens: bool = False,
    allow_spatial_graph_fallback: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for Zelda dungeon training.
    
    Args:
        data_dir: Directory containing dungeon data
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for Windows compatibility)
        use_vglc: Use VGLC format via ZeldaDungeonAdapter
        normalize: Normalize values to [0, 1]
        target_size: Optional (H, W) to resize all dungeons
        transform: Optional transform to apply
        room_level: If True, use ZeldaRoomDataset for individual rooms
        augment: Apply shape-preserving grid augmentation. For ``room_level=True``
            with graph conditioning, room-local metadata, boundary constraints,
            neighbor maps, and topology channels are transformed in lockstep under
            V4 symmetry. Disallowed for stitched ``room_level=False`` datasets
            with ``load_graphs=True``.
        
    Returns:
        PyTorch DataLoader
        
    Example:
        >>> loader = create_dataloader(
        ...     'Data/The Legend of Zelda',
        ...     batch_size=4,
        ...     use_vglc=True
        ... )
        >>> for batch in loader:
        ...     print(batch.shape)  # (4, 1, H, W)
    """
    if bool(augment) and bool(load_graphs) and not bool(room_level):
        raise ValueError(
            "augment=True cannot be combined with load_graphs=True for stitched "
            "full-dungeon datasets (room_level=False); use room_level=True for "
            "topology-equivariant V4 room augmentation."
        )
    if room_level:
        dataset = ZeldaRoomDataset(
            data_dir=data_dir,
            transform=transform,
            normalize=normalize,
            categorical_tokens=categorical_tokens,
            load_graphs=load_graphs,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            topology_supervision_mode=topology_supervision_mode,
            semantic_role_prior_strength=semantic_role_prior_strength,
            semantic_puzzle_offset=semantic_puzzle_offset,
            puzzle_stage_topology_enabled=puzzle_stage_topology_enabled,
            puzzle_stage_trace_decay=puzzle_stage_trace_decay,
            dungeon_ids=dungeon_ids,
            variants=variants,
            augment=augment,
            allow_spatial_graph_fallback=allow_spatial_graph_fallback,
        )
    else:
        dataset = ZeldaDungeonDataset(
            data_dir=data_dir,
            transform=transform,
            use_vglc=use_vglc,
            normalize=normalize,
            categorical_tokens=categorical_tokens,
            target_size=target_size,
            load_graphs=load_graphs,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            dungeon_ids=dungeon_ids,
            variants=variants,
            augment=augment,
            lazy_vglc=lazy_vglc,
            allow_spatial_graph_fallback=allow_spatial_graph_fallback,
        )
    
    batch_sampler = None
    if bool(dungeon_batch_mode):
        if sampler is not None:
            raise ValueError("dungeon_batch_mode cannot be combined with sampler.")
        if not room_level:
            raise ValueError("dungeon_batch_mode is only valid for room_level datasets.")
        batch_sampler = DungeonBatchSampler.from_dataset(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        "collate_fn": graph_collate_fn if load_graphs else None,
        "worker_init_fn": seed_dataloader_worker if int(num_workers) > 0 else None,
    }
    if batch_sampler is not None:
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
        sampler = batch_sampler
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(bool(shuffle) if sampler is None else False),
            sampler=sampler,
            drop_last=bool(drop_last),
            **dataloader_kwargs,
        )
    if return_sampler:
        return dataloader, sampler
    return dataloader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def tensor_to_ascii(tensor: torch.Tensor, threshold: float = 0.5) -> str:
    """
    Convert tensor back to ASCII representation for visualization.
    
    Args:
        tensor: (1, H, W) or (H, W) tensor
        threshold: Threshold for binarization
        
    Returns:
        ASCII string representation
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    arr = tensor.detach().cpu()
    if arr.dim() == 3:
        if arr.shape[0] > 1:
            arr = arr.argmax(dim=0)
        else:
            arr = arr.squeeze(0)
    if arr.dim() != 2:
        raise ValueError(f"Expected 2D tensor/grid for ASCII conversion, got shape={tuple(arr.shape)}")

    grid = arr.numpy()
    # Backward-compatible path for normalized tensors in [0,1].
    if np.issubdtype(grid.dtype, np.floating):
        max_val = float(np.max(grid)) if grid.size > 0 else 0.0
        min_val = float(np.min(grid)) if grid.size > 0 else 0.0
        if min_val >= 0.0 and max_val <= 1.0:
            grid = np.rint(grid * 43.0)
    grid_int = np.rint(grid).astype(np.int32, copy=False)
    return '\n'.join(semantic_grid_to_vglc_lines(grid_int))


def extract_start_goal(
    grid: Union[torch.Tensor, np.ndarray]
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Extract start and goal positions from a dungeon grid.
    
    Args:
        grid: Dungeon grid tensor or array
        
    Returns:
        (start_coords, goal_coords) as (row, col) tuples, or None if not found
    """
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    
    if grid.ndim == 3:
        grid = grid.squeeze(0)
    
    start_coords = None
    goal_coords = None
    
    if SEMANTIC_PALETTE is not None:
        start_id = SEMANTIC_PALETTE.get('START', 21)
        goal_id = SEMANTIC_PALETTE.get('TRIFORCE', 22)
    else:
        start_id = TILE_MAPPING.get('S', 6)
        goal_id = TILE_MAPPING.get('G', 7)
    
    # Find start
    start_pos = np.where(grid == start_id)
    if len(start_pos[0]) > 0:
        start_coords = (int(start_pos[0][0]), int(start_pos[1][0]))
    
    # Find goal
    goal_pos = np.where(grid == goal_id)
    if len(goal_pos[0]) > 0:
        goal_coords = (int(goal_pos[0][0]), int(goal_pos[1][0]))
    
    return start_coords, goal_coords


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Zelda Dataset Loader')
    parser.add_argument('--data-dir', type=str, default='Data/The Legend of Zelda',
                        help='Path to dungeon data')
    parser.add_argument('--use-vglc', action='store_true',
                        help='Use VGLC format')
    parser.add_argument('--batch-size', type=int, default=4)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        use_vglc=args.use_vglc,
    )
    
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    for batch in loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
        break
