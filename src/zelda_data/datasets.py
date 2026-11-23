"""PyTorch Dataset implementations for Zelda dungeons and extracted rooms."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.definitions import (
    CHAR_TO_SEMANTIC,
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    GRAPH_TPE_DIM,
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNEL_COUNT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
)
from src.pipeline.spatial_utils import parse_room_coord
from src.zelda_data.graph_extraction import (
    _build_room_graph_sample,
    _extract_graph_from_dungeon,
    _extract_graph_spatial_from_dungeon,
)
from src.zelda_data.symmetry import (
    RandomD4Symmetry,
    apply_room_symmetry_transform,
)
from src.zelda_data.splits import (
    DEFAULT_TRAIN_DUNGEONS,
    normalize_dungeon_ids,
    normalize_variants,
)
from src.zelda_data.zelda_core import ZeldaDungeonAdapter

logger = logging.getLogger(__name__)

TILE_MAPPING = dict(CHAR_TO_SEMANTIC)
VGLC_AVAILABLE = True

__all__ = [
    "ZeldaDungeonDataset",
    "ZeldaRoomDataset",
    "TILE_MAPPING",
    "VGLC_AVAILABLE",
]

class ZeldaDungeonDataset(Dataset):
    """
    PyTorch Dataset for Zelda dungeon grids.
    
    Supports loading from:
    1. Directory of .txt files (ASCII format)
    2. VGLC format via ZeldaDungeonAdapter
    3. Pre-loaded numpy arrays
    4. Paired NPZ format with (image, graph) pairs
    
    Args:
        data_dir: Directory containing dungeon files or VGLC data
        transform: Optional transform to apply to each sample
        use_vglc: Whether to use VGLC format via ZeldaDungeonAdapter
        normalize: Whether to normalize values to [0, 1]
        target_size: Target (height, width) for resizing, None for original
        load_graphs: Whether to load graph data for dual-stream training
        
    Returns:
        torch.Tensor of shape (1, H, W) representing the dungeon grid
        OR (image_tensor, graph_dict) if load_graphs=True
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        use_vglc: bool = False,
        normalize: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        pad_to_max: bool = True,  # Pad all samples to max size for batching
        load_graphs: bool = False,  # NEW: Load graph data for dual-stream
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        dungeon_ids: Optional[Iterable[int]] = None,
        variants: Optional[Iterable[int]] = None,
        augment: bool = False,  # Apply D4 dihedral symmetry augmentation
        lazy_vglc: bool = True,
        categorical_tokens: bool = False,
        allow_spatial_graph_fallback: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.categorical_tokens = bool(categorical_tokens)
        self.normalize = bool(normalize) and not self.categorical_tokens
        self.target_size = target_size
        self.use_vglc = use_vglc and VGLC_AVAILABLE
        self.pad_to_max = pad_to_max
        self.augment = bool(augment)
        if self.augment and bool(load_graphs):
            raise ValueError(
                "augment=True cannot be combined with load_graphs=True until graph "
                "metadata and directional room-conditioning tensors are transformed "
                "with the same symmetry as the target grid."
            )
        self._d4_augment = RandomD4Symmetry()
        self.load_graphs = load_graphs
        self.allow_spatial_graph_fallback = bool(allow_spatial_graph_fallback)
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        self.dungeon_ids = normalize_dungeon_ids(dungeon_ids)
        self.variants = normalize_variants(variants)
        self.sample_metadata: List[Dict[str, Any]] = []
        self.lazy_vglc = bool(lazy_vglc)
        self._vglc_index: List[Tuple[int, int]] = []
        
        # Track max dimensions for padding
        self.max_h = 0
        self.max_w = 0
        
        # Graph data storage
        self.graphs = [] if load_graphs else None
        
        if self.use_vglc:
            self._init_vglc()
        else:
            self._init_text_files()
        
        # If target_size not specified but pad_to_max is True, use max dims
        if self.target_size is None and self.pad_to_max and self.max_h > 0:
            self.target_size = (self.max_h, self.max_w)
            logger.info(f"Auto-set target_size to ({self.max_h}, {self.max_w}) for batching")
            
        logger.info(f"Loaded {len(self)} dungeon samples from {data_dir}")
    
    def _init_text_files(self) -> None:
        """Initialize dataset from text files."""
        self.files = [
            self.data_dir / f
            for f in sorted(os.listdir(self.data_dir))
            if f.endswith('.txt')
        ]
        self.samples = None  # Lazy loading
        if self.pad_to_max and self.target_size is None:
            for path in self.files:
                grid = self._load_text_file(path)
                if grid.ndim != 2:
                    raise ValueError(f"Expected a 2D dungeon grid in {path}, got shape {grid.shape}.")
                self.max_h = max(self.max_h, int(grid.shape[0]))
                self.max_w = max(self.max_w, int(grid.shape[1]))
        
    def _init_vglc(self) -> None:
        """Initialize dataset from VGLC format."""
        self.files = []
        self.samples = None if self.lazy_vglc else []
        
        # Load all dungeons via adapter
        adapter = ZeldaDungeonAdapter(str(self.data_dir))
        
        dungeon_iter = self.dungeon_ids if self.dungeon_ids is not None else DEFAULT_TRAIN_DUNGEONS
        for dungeon_num in dungeon_iter:  # Dungeons 1-9
            for variant in self.variants:  # Two quest variants
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    stitched = adapter.stitch_dungeon(dungeon)
                    grid = stitched.global_grid
                    graph = None
                    if self.load_graphs and self.graphs is not None and not self.lazy_vglc:
                        graph = self._extract_graph(dungeon)
                        if graph is None:
                            raise RuntimeError(
                                "Graph extraction returned no graph for a graph-conditioned "
                                f"sample (dungeon={dungeon_num}, variant={variant})."
                            )
                    self._vglc_index.append((int(dungeon_num), int(variant)))
                    if self.samples is not None:
                        self.samples.append(grid.astype(np.float32))
                    self.sample_metadata.append(
                        {
                            "dungeon_num": int(dungeon_num),
                            "variant": int(variant),
                            "dungeon_id": f"tloz{int(dungeon_num)}_{int(variant)}",
                        }
                    )
                    
                    if graph is not None:
                        self.graphs.append(graph)
                    
                    # Track max dimensions
                    h, w = grid.shape
                    self.max_h = max(self.max_h, h)
                    self.max_w = max(self.max_w, w)
                    
                    logger.debug(f"Loaded dungeon {dungeon_num} variant {variant}: {h}x{w}")
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to load dungeon {dungeon_num}v{variant}: {e}")
                    
        logger.info(f"Indexed {len(self._vglc_index)} VGLC dungeons (max size: {self.max_h}x{self.max_w}, lazy={self.lazy_vglc})")

    def _load_vglc_sample(self, idx: int) -> Tuple[np.ndarray, Optional[dict]]:
        dungeon_num, variant = self._vglc_index[int(idx)]
        adapter = ZeldaDungeonAdapter(str(self.data_dir))
        dungeon = adapter.load_dungeon(dungeon_num, variant)
        stitched = adapter.stitch_dungeon(dungeon)
        grid = stitched.global_grid.astype(np.float32)
        graph = self._extract_graph(dungeon) if self.load_graphs else None
        return grid, graph
    
    def _extract_graph(self, dungeon) -> dict:
        """Extract graph structure from dungeon for GNN training.
        
        Uses the DOT graph topology (dungeon.graph) as the authoritative
        source for node features and edge types. The 's' (start pointer)
        node is NOT included as a room node -- instead, its connected room
        is marked with start_node_id.
        """
        return _extract_graph_from_dungeon(
            dungeon,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            allow_spatial_fallback=self.allow_spatial_graph_fallback,
        )
    
    def _extract_graph_spatial(self, dungeon) -> dict:
        """Fallback: extract graph from spatial adjacency when DOT graph unavailable."""
        return _extract_graph_spatial_from_dungeon(
            dungeon,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
        )
    
    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        if self.use_vglc:
            return len(self._vglc_index)
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Get a single dungeon grid as a tensor.
        
        Returns:
            If load_graphs=False: torch.Tensor of shape (1, H, W)
            If load_graphs=True: (image_tensor, graph_dict) tuple
        """
        graph_override = None
        if self.samples is not None:
            grid = self.samples[idx]
        elif self.use_vglc:
            grid, graph_override = self._load_vglc_sample(idx)
        else:
            grid = self._load_text_file(self.files[idx])
        
        # Convert to tensor
        tensor_map = torch.tensor(grid, dtype=torch.float32)
        
        # Add channel dimension if needed
        if tensor_map.dim() == 2:
            tensor_map = tensor_map.unsqueeze(0)
        
        # Normalize to [0, 1] using fixed num_classes divisor
        # IMPORTANT: Use fixed constant (43 = max tile ID) so that
        # grids_to_onehot / encode_to_latent can invert with *43 exactly.
        if self.normalize:
            NUM_TILE_IDS = 43  # TileID.PUZZLE = 43, the highest ID
            tensor_map = tensor_map / NUM_TILE_IDS
        
        # Resize if target size specified
        if self.target_size is not None:
            tensor_map = self._resize(tensor_map, self.target_size)
        
        # Apply D4 dihedral symmetry augmentation during training
        if bool(getattr(self, "augment", False)):
            tensor_map = self._d4_augment(tensor_map)

        # Apply custom transform
        if self.transform:
            tensor_map = self.transform(tensor_map)
        
        # Return with graph if requested
        if self.load_graphs:
            if graph_override is not None:
                graph = graph_override
            elif self.graphs is not None:
                graph = self.graphs[idx]
            else:
                raise RuntimeError("Graph loading requested but no graph data is available.")
            edge_feature_dim = int(getattr(self, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM))
            return tensor_map, {
                'node_features': torch.tensor(graph['node_features'], dtype=torch.float32),
                'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph.get('edge_attr', np.zeros((0,), dtype=np.int64)), dtype=torch.long),
                'edge_features': torch.tensor(graph.get('edge_features', np.zeros((0, edge_feature_dim), dtype=np.float32)), dtype=torch.float32),
                'edge_rrwp': torch.tensor(graph.get('edge_rrwp', np.zeros((int(graph.get('num_edges', 0)), GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'tpe': torch.tensor(graph.get('tpe', np.zeros((0, GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'node_positions': torch.tensor(graph.get('node_positions', np.zeros((0, 2), dtype=np.float32)), dtype=torch.float32),
                'node_passability_parent_idx': torch.tensor(
                    graph.get('node_passability_parent_idx', np.arange(int(graph.get('num_nodes', 0)), dtype=np.int64)),
                    dtype=torch.long,
                ),
                'num_nodes': graph['num_nodes'],
                'num_edges': graph['num_edges'],
                'start_node_id': graph.get('start_node_id', -1),
                'target_idx': graph.get('target_idx', -1),
                'key_lock_pairs': list(graph.get('key_lock_pairs', []) or []),
                'floor_values': list(graph.get('floor_values', []) or []),
                'floor_labels_present': list(graph.get('floor_labels_present', []) or []),
                'node_to_idx': dict(graph.get('node_to_idx', {})),
            }
            
        return tensor_map
    
    def _load_text_file(self, filepath: Path) -> np.ndarray:
        """Load dungeon grid from text file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        dungeon_grid = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = [TILE_MAPPING.get(c, 0) for c in line]
            dungeon_grid.append(row)
        
        return np.array(dungeon_grid, dtype=np.float32)
    
    def _resize(self, tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Resize or pad tensor to target size for batching compatibility."""
        import torch.nn.functional as F
        
        target_h, target_w = size
        current_h, current_w = tensor.shape[-2], tensor.shape[-1]
        
        # If already correct size, return as-is
        if current_h == target_h and current_w == target_w:
            return tensor
        
        # Pad to target size (zero-padding for void areas)
        pad_h = target_h - current_h
        pad_w = target_w - current_w
        
        if pad_h >= 0 and pad_w >= 0:
            # Pad on right and bottom (left, right, top, bottom)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            # Need to crop or interpolate if target is smaller
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            tensor = F.interpolate(tensor, size=size, mode='nearest')
            tensor = tensor.squeeze(0)  # (C, H, W)
        
        return tensor
    
    def get_raw_grid(self, idx: int) -> np.ndarray:
        """Get raw numpy array for a dungeon (before transforms)."""
        if self.samples is not None:
            return self.samples[idx]
        if self.use_vglc:
            grid, _graph = self._load_vglc_sample(idx)
            return grid
        return self._load_text_file(self.files[idx])


# =============================================================================
# ROOM-LEVEL DATASET
# =============================================================================

class ZeldaRoomDataset(Dataset):
    """
    Dataset for individual rooms extracted from dungeons.
    
    Extracts canonical `(ROOM_HEIGHT, ROOM_WIDTH)` rooms from larger dungeon grids for training
    room-level generation models.
    
    Args:
        data_dir: Directory with VGLC data
        transform: Optional transform for each room
        normalize: Normalize to [0, 1]
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        categorical_tokens: bool = False,
        load_graphs: bool = False,
        node_feature_dim: int = GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim: int = GRAPH_EDGE_FEATURE_DIM,
        topology_supervision_mode: str = "runtime_aligned",
        semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        puzzle_stage_topology_enabled: bool = False,
        puzzle_stage_trace_decay: float = DEFAULT_PUZZLE_STAGE_TRACE_DECAY,
        dungeon_ids: Optional[Iterable[int]] = None,
        variants: Optional[Iterable[int]] = None,
        augment: bool = False,  # Apply D4 dihedral symmetry augmentation
        allow_spatial_graph_fallback: bool = False,
    ):
        self.transform = transform
        self.categorical_tokens = bool(categorical_tokens)
        self.normalize = bool(normalize) and not self.categorical_tokens
        self.load_graphs = load_graphs
        self.augment = bool(augment)
        self._d4_augment = RandomD4Symmetry()
        self.node_feature_dim = int(max(1, node_feature_dim))
        self.edge_feature_dim = int(max(1, edge_feature_dim))
        self.topology_supervision_mode = str(topology_supervision_mode).strip().lower()
        self.semantic_role_prior_strength = float(max(0.0, min(1.0, semantic_role_prior_strength)))
        self.semantic_puzzle_offset = int(max(0, semantic_puzzle_offset))
        self.puzzle_stage_topology_enabled = bool(puzzle_stage_topology_enabled)
        self.puzzle_stage_trace_decay = float(max(0.05, min(1.0, puzzle_stage_trace_decay)))
        self.dungeon_ids = normalize_dungeon_ids(dungeon_ids)
        self.variants = normalize_variants(variants)
        self.allow_spatial_graph_fallback = bool(allow_spatial_graph_fallback)
        self.rooms = []
        self.graphs = [] if load_graphs else None
        self.sample_metadata: List[Dict[str, Any]] = []
        if self.topology_supervision_mode not in {"runtime_aligned", "oracle_room_grid"}:
            raise ValueError(
                "topology_supervision_mode must be 'runtime_aligned' or 'oracle_room_grid'."
            )
        
        if not VGLC_AVAILABLE:
            raise ImportError("VGLC adapter required for room dataset")
        
        adapter = ZeldaDungeonAdapter(str(data_dir))
        
        dungeon_iter = self.dungeon_ids if self.dungeon_ids is not None else DEFAULT_TRAIN_DUNGEONS
        for dungeon_num in dungeon_iter:
            for variant in self.variants:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    dungeon_graph = _extract_graph_from_dungeon(
                        dungeon,
                        node_feature_dim=self.node_feature_dim,
                        edge_feature_dim=self.edge_feature_dim,
                        allow_spatial_fallback=self.allow_spatial_graph_fallback,
                    ) if load_graphs else None
                    if load_graphs and dungeon_graph is None:
                        raise RuntimeError(
                            "Graph extraction returned no graph for a graph-conditioned "
                            f"room dataset (dungeon={dungeon_num}, variant={variant})."
                        )
                    for coord, room in dungeon.rooms.items():
                        grid = getattr(room, 'semantic_grid', None)
                        if grid is None:
                            grid = getattr(room, 'grid', None)
                        if grid is not None:
                            room_graph = None
                            if self.graphs is not None and dungeon_graph is not None:
                                room_graph = _build_room_graph_sample(
                                    dungeon,
                                    coord,
                                    room,
                                    dungeon_graph,
                                    topology_supervision_mode=self.topology_supervision_mode,
                                    semantic_role_prior_strength=self.semantic_role_prior_strength,
                                    semantic_puzzle_offset=self.semantic_puzzle_offset,
                                    puzzle_stage_topology_enabled=self.puzzle_stage_topology_enabled,
                                    puzzle_stage_trace_decay=self.puzzle_stage_trace_decay,
                                )
                                if room_graph is None:
                                    raise RuntimeError(
                                        "Room graph extraction returned no graph for a graph-conditioned "
                                        f"sample (dungeon={dungeon_num}, variant={variant}, coord={coord})."
                                    )
                            self.rooms.append(grid.astype(np.float32))
                            self.sample_metadata.append(
                                {
                                    "dungeon_num": int(dungeon_num),
                                    "variant": int(variant),
                                    "dungeon_id": f"tloz{int(dungeon_num)}_{int(variant)}",
                                    "room_coord": tuple(coord) if isinstance(coord, tuple) else coord,
                                }
                            )
                            if self.graphs is not None and room_graph is not None:
                                self.graphs.append(room_graph)
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"Failed to load room dataset entry for dungeon {dungeon_num} variant {variant}"
                    ) from exc
        
        logger.info(f"Loaded {len(self.rooms)} individual rooms")
    
    def __len__(self) -> int:
        return len(self.rooms)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        grid = self.rooms[idx]
        tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        
        # Use fixed constant (43 = max tile ID) so grids_to_onehot / encode_to_latent
        # can invert exactly with *43
        if self.normalize:
            NUM_TILE_IDS = 43  # TileID.PUZZLE = 43
            tensor = tensor / NUM_TILE_IDS
        
        if self.load_graphs and self.graphs is not None:
            graph = self.graphs[idx]
            num_tile_ids = 43
            edge_feature_dim = int(getattr(self, "edge_feature_dim", GRAPH_EDGE_FEATURE_DIM))
            neighbor_maps = {}
            for direction, room_map in dict(graph.get('neighbor_maps', {})).items():
                if room_map is None:
                    neighbor_maps[direction] = None
                    continue
                room_tensor = torch.tensor(room_map, dtype=torch.float32).unsqueeze(0)
                if self.normalize:
                    room_tensor = room_tensor / num_tile_ids
                neighbor_maps[direction] = room_tensor
            graph_dict = {
                'node_features': torch.tensor(graph['node_features'], dtype=torch.float32),
                'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph.get('edge_attr', np.zeros((0,), dtype=np.int64)), dtype=torch.long),
                'edge_features': torch.tensor(graph.get('edge_features', np.zeros((0, edge_feature_dim), dtype=np.float32)), dtype=torch.float32),
                'edge_rrwp': torch.tensor(graph.get('edge_rrwp', np.zeros((int(graph.get('num_edges', 0)), GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'tpe': torch.tensor(graph.get('tpe', np.zeros((0, GRAPH_TPE_DIM), dtype=np.float32)), dtype=torch.float32),
                'node_positions': torch.tensor(graph.get('node_positions', np.zeros((0, 2), dtype=np.float32)), dtype=torch.float32),
                'node_passability_parent_idx': torch.tensor(
                    graph.get('node_passability_parent_idx', np.arange(int(graph.get('num_nodes', 0)), dtype=np.int64)),
                    dtype=torch.long,
                ),
                'room_topology_map': torch.tensor(graph.get('room_topology_map', np.zeros((ROOM_TOPOLOGY_CHANNEL_COUNT, ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)), dtype=torch.float32),
                'logic_source_mask': torch.tensor(graph.get('logic_source_mask', np.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)), dtype=torch.float32),
                'logic_target_mask': torch.tensor(graph.get('logic_target_mask', np.zeros((1, ROOM_HEIGHT, ROOM_WIDTH), dtype=np.float32)), dtype=torch.float32),
                'boundary_constraints': torch.tensor(graph.get('boundary_constraints', np.zeros((8,), dtype=np.float32)), dtype=torch.float32),
                'room_position': torch.tensor(graph.get('room_position', np.zeros((2,), dtype=np.float32)), dtype=torch.float32),
                'neighbor_maps': neighbor_maps,
                'num_nodes': graph['num_nodes'],
                'num_edges': graph['num_edges'],
                'start_node_id': graph.get('start_node_id', -1),
                'target_idx': graph.get('target_idx', -1),
                'key_lock_pairs': list(graph.get('key_lock_pairs', []) or []),
                'floor_values': list(graph.get('floor_values', []) or []),
                'floor_labels_present': list(graph.get('floor_labels_present', []) or []),
                'current_node_idx': int(graph.get('current_node_idx', 0)),
                'node_to_idx': dict(graph.get('node_to_idx', {})),
                'has_puzzle': bool(graph.get('has_puzzle', False)),
                'puzzle_room_structure_enabled': bool(
                    graph.get('puzzle_room_structure_enabled', graph.get('has_puzzle', False))
                ),
                'puzzle_stage_condition': dict(graph.get('puzzle_stage_condition', {})),
                **({'style_id': int(graph.get('style_id'))} if graph.get('style_id', None) is not None else {}),
            }
            if self.augment:
                flip_h = bool(torch.randint(0, 2, ()).item())
                flip_v = bool(torch.randint(0, 2, ()).item())
                tensor, graph_dict = apply_room_symmetry_transform(
                    tensor,
                    graph_dict,
                    flip_h=flip_h,
                    flip_v=flip_v,
                )
            if self.transform:
                tensor = self.transform(tensor)
            return tensor, graph_dict

        # Apply D4 dihedral symmetry augmentation during training
        if self.augment:
            tensor = self._d4_augment(tensor)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor


