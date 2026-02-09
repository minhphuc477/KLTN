"""
Zelda Dungeon Dataset Loader
============================

PyTorch Dataset and DataLoader for loading Zelda dungeon grids from text files
or from the existing VGLC format via zelda_core.

Supports:
1. Raw text files with ASCII dungeon grids
2. VGLC format via ZeldaDungeonAdapter
3. NumPy array conversion with proper semantic IDs

References:
- VGLC: Video Game Level Corpus (https://github.com/TheVGLC/TheVGLC)
- Zelda dungeon format: 16 rows × 11 columns per room
"""

import os
import logging
import numpy as np
from typing import Optional, List, Callable, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# =============================================================================
# TILE MAPPINGS
# =============================================================================

# Simple ASCII mapping for basic text files (legacy support)
TILE_MAPPING = {
    'F': 0,   # Floor
    'W': 1,   # Wall
    'D': 2,   # Door
    'K': 3,   # Key
    'L': 4,   # Locked door
    'E': 5,   # Enemy
    'S': 6,   # Start
    'G': 7,   # Goal/Triforce
    '.': 0,   # Floor (alternate)
    '-': -1,  # Void
}

# Import semantic palette from local zelda_core module
from .zelda_core import (
    ROOM_HEIGHT,
    ROOM_WIDTH,
    ZeldaDungeonAdapter
)
from src.core.definitions import SEMANTIC_PALETTE, CHAR_TO_SEMANTIC
VGLC_AVAILABLE = True
logger.info("VGLC adapter available via zelda_core")


# =============================================================================
# DATASET CLASS
# =============================================================================

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
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.normalize = normalize
        self.target_size = target_size
        self.use_vglc = use_vglc and VGLC_AVAILABLE
        self.pad_to_max = pad_to_max
        self.load_graphs = load_graphs
        
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
            for f in os.listdir(self.data_dir) 
            if f.endswith('.txt')
        ]
        self.samples = None  # Lazy loading
        
    def _init_vglc(self) -> None:
        """Initialize dataset from VGLC format."""
        self.files = []
        self.samples = []
        
        # Load all dungeons via adapter
        adapter = ZeldaDungeonAdapter(str(self.data_dir))
        
        for dungeon_num in range(1, 10):  # Dungeons 1-9
            for variant in [1, 2]:  # Two quest variants
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    stitched = adapter.stitch_dungeon(dungeon)
                    grid = stitched.global_grid
                    self.samples.append(grid.astype(np.float32))
                    
                    # Extract graph if load_graphs is enabled
                    if self.load_graphs:
                        graph = self._extract_graph(dungeon)
                        self.graphs.append(graph)
                    
                    # Track max dimensions
                    h, w = grid.shape
                    self.max_h = max(self.max_h, h)
                    self.max_w = max(self.max_w, w)
                    
                    logger.debug(f"Loaded dungeon {dungeon_num} variant {variant}: {h}x{w}")
                except Exception as e:
                    logger.warning(f"Failed to load dungeon {dungeon_num}v{variant}: {e}")
                    
        logger.info(f"Loaded {len(self.samples)} VGLC dungeons (max size: {self.max_h}x{self.max_w})")
    
    def _extract_graph(self, dungeon) -> dict:
        """Extract graph structure from dungeon for GNN training.
        
        Uses the DOT graph topology (dungeon.graph) as the authoritative
        source for node features and edge types. The 's' (start pointer)
        node is NOT included as a room node — instead, its connected room
        is marked with start_node_id.
        """
        graph = getattr(dungeon, 'graph', None)
        if graph is None or len(graph.nodes()) == 0:
            # Fallback: spatial adjacency only
            return self._extract_graph_spatial(dungeon)
        
        nodes = []
        edges = []
        edge_attrs = []
        node_id_to_idx = {}
        start_node_idx = -1
        
        # Edge type encoding: open=0, key_locked=1, bombable=2, soft_locked=3, other=4
        EDGE_TYPE_ENCODING = {
            'open': 0, '': 0,
            'key_locked': 1, 'k': 1,
            'bombable': 2, 'b': 2,
            'soft_locked': 3, 'l': 3,
            'boss_locked': 4, 'K': 4,
            'item_locked': 5, 'I': 5,
            'stair': 6, 's': 6,
        }
        
        # Create nodes — skip the 's' start pointer node
        idx = 0
        for node_id, data in sorted(graph.nodes(data=True)):
            if data.get('is_start_pointer', False):
                # Record which room this pointer points to
                continue
            
            node_id_to_idx[node_id] = idx
            label = data.get('label', '')
            parts = [p.strip() for p in label.split(',')]
            
            # Node features: [has_enemy, has_key, has_item, has_triforce, has_boss, has_puzzle]
            node_features = [
                float('e' in parts or data.get('has_enemy', False)),
                float('k' in parts or data.get('has_key', False)),
                float('i' in parts or 'I' in parts or data.get('has_item', False)),
                float('t' in parts or data.get('is_triforce', False)),
                float('b' in parts or data.get('is_boss', False)),
                float('p' in parts or data.get('has_puzzle', False)),
            ]
            nodes.append(node_features)
            
            # Track start node (neighbor of 's' pointer, or node with is_start)
            if data.get('is_start') and not data.get('is_start_pointer', False):
                start_node_idx = idx
            
            idx += 1
        
        # If start not found yet, find the neighbor of the 's' pointer
        if start_node_idx == -1:
            for node_id, data in graph.nodes(data=True):
                if data.get('is_start_pointer', False):
                    for neighbor in list(graph.successors(node_id)) + list(graph.predecessors(node_id)):
                        if neighbor in node_id_to_idx:
                            start_node_idx = node_id_to_idx[neighbor]
                            break
        
        # Create edges from graph topology (skip edges involving pointer node)
        for u, v, data in graph.edges(data=True):
            if u not in node_id_to_idx or v not in node_id_to_idx:
                continue
            src_idx = node_id_to_idx[u]
            dst_idx = node_id_to_idx[v]
            edges.append([src_idx, dst_idx])
            
            # Edge type from edge_type or label
            edge_type = data.get('edge_type', '')
            if not edge_type:
                edge_type = data.get('label', '')
            edge_attrs.append(EDGE_TYPE_ENCODING.get(edge_type, 0))
        
        import numpy as np
        return {
            'node_features': np.array(nodes, dtype=np.float32) if nodes else np.zeros((0, 6), dtype=np.float32),
            'edge_index': np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64),
            'edge_attr': np.array(edge_attrs, dtype=np.int64) if edge_attrs else np.zeros((0,), dtype=np.int64),
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'start_node_id': start_node_idx,
        }
    
    def _extract_graph_spatial(self, dungeon) -> dict:
        """Fallback: extract graph from spatial adjacency when DOT graph unavailable."""
        nodes = []
        edges = []
        room_to_idx = {}
        
        for idx, (coord, room) in enumerate(dungeon.rooms.items()):
            room_to_idx[coord] = idx
            node_features = [0.0] * 6  # No label info available
            nodes.append(node_features)
        
        for coord in dungeon.rooms:
            src_idx = room_to_idx[coord]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (coord[0] + dr, coord[1] + dc)
                if neighbor in room_to_idx:
                    dst_idx = room_to_idx[neighbor]
                    edges.append([src_idx, dst_idx])
        
        import numpy as np
        return {
            'node_features': np.array(nodes, dtype=np.float32) if nodes else np.zeros((0, 6), dtype=np.float32),
            'edge_index': np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64),
            'edge_attr': np.zeros((len(edges),), dtype=np.int64),
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'start_node_id': -1,
        }
    
    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Get a single dungeon grid as a tensor.
        
        Returns:
            If load_graphs=False: torch.Tensor of shape (1, H, W)
            If load_graphs=True: (image_tensor, graph_dict) tuple
        """
        if self.samples is not None:
            grid = self.samples[idx]
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
        if self.normalize and tensor_map.max() > 1:
            NUM_TILE_IDS = 43  # TileID.PUZZLE = 43, the highest ID
            tensor_map = tensor_map / NUM_TILE_IDS
        
        # Resize if target size specified
        if self.target_size is not None:
            tensor_map = self._resize(tensor_map, self.target_size)
        
        # Apply custom transform
        if self.transform:
            tensor_map = self.transform(tensor_map)
        
        # Return with graph if requested
        if self.load_graphs and self.graphs is not None:
            graph = self.graphs[idx]
            return tensor_map, {
                'node_features': torch.tensor(graph['node_features'], dtype=torch.float32),
                'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph.get('edge_attr', np.zeros((0,), dtype=np.int64)), dtype=torch.long),
                'num_nodes': graph['num_nodes'],
                'num_edges': graph['num_edges'],
                'start_node_id': graph.get('start_node_id', -1),
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
        return self._load_text_file(self.files[idx])


# =============================================================================
# ROOM-LEVEL DATASET
# =============================================================================

class ZeldaRoomDataset(Dataset):
    """
    Dataset for individual rooms extracted from dungeons.
    
    Extracts 16x11 rooms from larger dungeon grids for training
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
    ):
        self.transform = transform
        self.normalize = normalize
        self.rooms = []
        
        if not VGLC_AVAILABLE:
            raise ImportError("VGLC adapter required for room dataset")
        
        adapter = ZeldaDungeonAdapter(str(data_dir))
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    for coord, room in dungeon.rooms.items():
                        grid = getattr(room, 'semantic_grid', None)
                        if grid is None:
                            grid = getattr(room, 'grid', None)
                        if grid is not None:
                            self.rooms.append(grid.astype(np.float32))
                except Exception as e:
                    logger.debug(f"Skipping dungeon {dungeon_num}v{variant}: {e}")
        
        logger.info(f"Loaded {len(self.rooms)} individual rooms")
    
    def __len__(self) -> int:
        return len(self.rooms)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        grid = self.rooms[idx]
        tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        
        # Use fixed constant (43 = max tile ID) so grids_to_onehot / encode_to_latent
        # can invert exactly with *43
        if self.normalize and tensor.max() > 1:
            NUM_TILE_IDS = 43  # TileID.PUZZLE = 43
            tensor = tensor / NUM_TILE_IDS
        
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor


# =============================================================================
# GRAPH COLLATION FOR VARIABLE-SIZE GRAPHS
# =============================================================================

def graph_collate_fn(batch):
    """
    Custom collation function for batches containing (image, graph_dict) pairs.
    
    Handles variable-size graphs by:
    1. Stacking image tensors normally (they're already padded to same size)
    2. Storing graph dicts as a list (since node counts differ per dungeon)
    
    For per-sample graph processing during training, we iterate over
    individual graphs rather than trying to batch them into a single tensor.
    
    Args:
        batch: List of (image_tensor, graph_dict) tuples from __getitem__
        
    Returns:
        (images_batch, graph_list) where:
            - images_batch: [B, 1, H, W] stacked image tensors
            - graph_list: List of B graph dicts, each containing:
                - node_features: [N_i, 6] (variable N_i per graph)
                - edge_index: [2, E_i] (variable E_i per graph)
                - edge_attr: [E_i] edge type labels
                - num_nodes: int
                - num_edges: int
                - start_node_id: int
    """
    if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
        images = torch.stack([item[0] for item in batch])
        graphs = [item[1] for item in batch]
        return images, graphs
    else:
        # No graph data — plain image batch
        return torch.stack(batch)


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    use_vglc: bool = False,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    transform: Optional[Callable] = None,
    room_level: bool = False,
    load_graphs: bool = False,
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
    if room_level:
        dataset = ZeldaRoomDataset(
            data_dir=data_dir,
            transform=transform,
            normalize=normalize,
        )
    else:
        dataset = ZeldaDungeonDataset(
            data_dir=data_dir,
            transform=transform,
            use_vglc=use_vglc,
            normalize=normalize,
            target_size=target_size,
            load_graphs=load_graphs,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Ensure consistent batch sizes
        collate_fn=graph_collate_fn if load_graphs else None,
    )


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
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    grid = tensor.numpy()
    
    # Reverse mapping
    inv_mapping = {v: k for k, v in TILE_MAPPING.items() if v >= 0}
    
    lines = []
    for row in grid:
        line = ''
        for val in row:
            idx = int(round(val * max(inv_mapping.keys()))) if val <= 1 else int(val)
            line += inv_mapping.get(idx, '?')
        lines.append(line)
    
    return '\n'.join(lines)


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
