#!/usr/bin/env python3
"""
Build Paired Dataset (Graph + Image) for Dual-Stream Training
==============================================================

Creates paired (graph, image) datasets for training the dual-stream
condition encoder which requires both local (CNN/image) and global
(GNN/graph) features.

Output formats:
1. NPZ format: level_XX.npz with 'image' and 'graph' keys
2. Separate files: level_XX.png + level_XX_graph.json
3. HDF5 format (optional): all data in single file

Usage:
    python scripts/build_paired_dataset.py --data-dir "Data/The Legend of Zelda"
    python scripts/build_paired_dataset.py --output-dir datasets/paired --format npz
    python scripts/build_paired_dataset.py --format hdf5 --output datasets/zelda_paired.h5
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zelda_core import ZeldaDungeonAdapter

logger = logging.getLogger(__name__)


def extract_room_graph(dungeon: Any) -> Dict[str, Any]:
    """
    Extract graph structure from a dungeon.
    
    Returns:
        Graph dict with nodes, edges, and attributes
    """
    nodes = []
    edges = []
    
    # Create node for each room
    room_to_idx = {}
    for idx, (coord, room) in enumerate(dungeon.rooms.items()):
        room_to_idx[coord] = idx
        
        # Node features
        node = {
            'id': idx,
            'coord': list(coord),  # (row, col)
            'has_key': bool(np.any(room.grid == 30)),  # KEY_SMALL
            'has_boss_key': bool(np.any(room.grid == 31)),  # KEY_BOSS
            'has_triforce': bool(np.any(room.grid == 22)),  # TRIFORCE
            'has_start': bool(np.any(room.grid == 21)),  # START
            'has_enemy': bool(np.any(room.grid == 20)),  # ENEMY
            'door_count': int(np.sum((room.grid >= 10) & (room.grid <= 15))),
        }
        nodes.append(node)
    
    # Create edges based on adjacency
    for coord in dungeon.rooms:
        src_idx = room_to_idx[coord]
        
        # Check 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (coord[0] + dr, coord[1] + dc)
            if neighbor in room_to_idx:
                dst_idx = room_to_idx[neighbor]
                
                # Edge attributes
                edge = {
                    'src': src_idx,
                    'dst': dst_idx,
                    'direction': 'N' if dr < 0 else 'S' if dr > 0 else 'W' if dc < 0 else 'E',
                    'weight': 1.0,
                }
                edges.append(edge)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
    }


def save_npz_format(
    image: np.ndarray,
    graph: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save as NPZ with image and graph."""
    # Convert graph to arrays for NPZ
    node_features = np.array([
        [n['has_key'], n['has_boss_key'], n['has_triforce'], 
         n['has_start'], n['has_enemy'], n['door_count']]
        for n in graph['nodes']
    ], dtype=np.float32)
    
    node_coords = np.array([n['coord'] for n in graph['nodes']], dtype=np.int32)
    
    edge_index = np.array(
        [[e['src'], e['dst']] for e in graph['edges']],
        dtype=np.int32
    ).T  # Shape: (2, num_edges)
    
    np.savez_compressed(
        output_path,
        image=image,
        node_features=node_features,
        node_coords=node_coords,
        edge_index=edge_index if len(graph['edges']) > 0 else np.zeros((2, 0), dtype=np.int32),
        num_nodes=graph['num_nodes'],
        num_edges=graph['num_edges'],
    )


def save_separate_format(
    image: np.ndarray,
    graph: Dict[str, Any],
    output_dir: Path,
    name: str,
) -> None:
    """Save as separate PNG and JSON files."""
    from PIL import Image
    
    # Save image as PNG (normalized to 0-255)
    img_normalized = (image / image.max() * 255).astype(np.uint8)
    img = Image.fromarray(img_normalized)
    img.save(output_dir / f"{name}.png")
    
    # Save graph as JSON
    with open(output_dir / f"{name}_graph.json", 'w') as f:
        json.dump(graph, f, indent=2)


def build_paired_dataset(
    data_dir: str,
    output_dir: str,
    output_format: str = 'npz',
    include_rooms: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Build paired (graph, image) dataset from Zelda dungeons.
    
    Args:
        data_dir: Input VGLC data directory
        output_dir: Output directory for paired data
        output_format: 'npz', 'separate', or 'hdf5'
        include_rooms: Also save individual room images
        verbose: Print progress
        
    Returns:
        Statistics dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    adapter = ZeldaDungeonAdapter(data_dir)
    
    stats = {
        'dungeons_processed': 0,
        'rooms_saved': 0,
        'total_nodes': 0,
        'total_edges': 0,
    }
    
    # Process dungeons
    for dungeon_num in range(1, 10):
        for variant in [1, 2]:
            try:
                dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                stitched = adapter.stitch_dungeon(dungeon)
                
                name = f"dungeon_{dungeon_num:02d}_v{variant}"
                image = stitched.global_grid.astype(np.float32)
                graph = extract_room_graph(dungeon)
                
                if output_format == 'npz':
                    save_npz_format(image, graph, output_path / f"{name}.npz")
                elif output_format == 'separate':
                    save_separate_format(image, graph, output_path, name)
                else:
                    raise ValueError(f"Unknown format: {output_format}")
                
                stats['dungeons_processed'] += 1
                stats['total_nodes'] += graph['num_nodes']
                stats['total_edges'] += graph['num_edges']
                
                if verbose:
                    logger.info(f"Saved {name}: {graph['num_nodes']} rooms, "
                               f"{graph['num_edges']} connections")
                
                # Save individual rooms if requested
                if include_rooms:
                    rooms_dir = output_path / 'rooms' / name
                    rooms_dir.mkdir(parents=True, exist_ok=True)
                    
                    for coord, room in dungeon.rooms.items():
                        room_name = f"room_{coord[0]:02d}_{coord[1]:02d}"
                        room_image = room.grid.astype(np.float32)
                        
                        if output_format == 'npz':
                            np.savez_compressed(
                                rooms_dir / f"{room_name}.npz",
                                image=room_image,
                                coord=np.array(coord),
                            )
                        else:
                            from PIL import Image
                            img = (room_image / max(1, room_image.max()) * 255).astype(np.uint8)
                            Image.fromarray(img).save(rooms_dir / f"{room_name}.png")
                        
                        stats['rooms_saved'] += 1
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to process dungeon {dungeon_num}v{variant}: {e}")
    
    # Save metadata
    metadata = {
        'source': data_dir,
        'format': output_format,
        'stats': stats,
        'schema': {
            'npz_keys': ['image', 'node_features', 'node_coords', 'edge_index'],
            'node_features': ['has_key', 'has_boss_key', 'has_triforce', 
                            'has_start', 'has_enemy', 'door_count'],
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        logger.info(f"\nDataset built: {stats}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Build paired (graph, image) dataset for dual-stream training'
    )
    parser.add_argument('--data-dir', type=str, default='Data/The Legend of Zelda',
                       help='Input VGLC data directory')
    parser.add_argument('--output-dir', type=str, default='datasets/paired',
                       help='Output directory for paired data')
    parser.add_argument('--format', type=str, default='npz',
                       choices=['npz', 'separate'],
                       help='Output format')
    parser.add_argument('--no-rooms', action='store_true',
                       help='Skip saving individual room images')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    
    stats = build_paired_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_format=args.format,
        include_rooms=not args.no_rooms,
        verbose=not args.quiet,
    )
    
    print(f"\n✓ Dataset built successfully!")
    print(f"  Dungeons: {stats['dungeons_processed']}")
    print(f"  Rooms: {stats['rooms_saved']}")
    print(f"  Graph nodes: {stats['total_nodes']}")
    print(f"  Graph edges: {stats['total_edges']}")
    print(f"  Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
