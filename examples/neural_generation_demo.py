"""
Neural-Symbolic Dungeon Generation Demo
========================================

Complete end-to-end demonstration of the 7-block H-MOLQD pipeline for
Legend of Zelda dungeon generation.

This script demonstrates:
1. Pipeline initialization with checkpoints
2. Single room generation
3. Multi-room dungeon generation
4. Visualization of intermediate outputs
5. MAP-Elites evaluation
6. Saving and loading results

Usage:
    # Basic demo (no checkpoints required)
    python examples/neural_generation_demo.py
    
    # With checkpoints
    python examples/neural_generation_demo.py --checkpoint-dir ./checkpoints
    
    # Custom parameters
    python examples/neural_generation_demo.py --num-rooms 8 --guidance 2.0 --seed 42
    
    # Save outputs
    python examples/neural_generation_demo.py --output-dir ./demo_outputs
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import create_pipeline, NeuralSymbolicDungeonPipeline
from src.core import ROOM_HEIGHT, ROOM_WIDTH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def print_room_ascii(room_grid: np.ndarray, title: str = "Room"):
    """Print ASCII visualization of a room."""
    import src.core.definitions as defs
    
    # Simple tile-to-char mapping
    tile_chars = {
        defs.TileID.VOID: ' ',
        defs.TileID.FLOOR: '.',
        defs.TileID.WALL: '#',
        defs.TileID.BLOCK: 'B',
        defs.TileID.DOOR_OPEN: 'D',
        defs.TileID.DOOR_LOCKED: 'L',
        defs.TileID.ENEMY: 'M',
        defs.TileID.START: 'S',
        defs.TileID.TRIFORCE: 'T',
        defs.TileID.KEY_SMALL: 'k',
    }
    
    print(f"\n{title}")
    print("=" * (ROOM_WIDTH + 2))
    
    for row in room_grid:
        line = "|"
        for tile_id in row:
            char = tile_chars.get(tile_id, '?')
            line += char
        line += "|"
        print(line)
    
    print("=" * (ROOM_WIDTH + 2))


def visualize_dungeon_structure(result):
    """Print dungeon generation metrics and structure."""
    print("\n" + "="*70)
    print("DUNGEON GENERATION RESULTS")
    print("="*70)
    
    print(f"\nGeneral Metrics:")
    print(f"  Total rooms: {result.metrics['num_rooms']}")
    print(f"  Dungeon shape: {result.metrics['dungeon_shape']}")
    print(f"  Generation time: {result.generation_time:.2f}s")
    print(f"  Repair rate: {result.metrics['repair_rate']:.1%}")
    print(f"  Total tiles repaired: {result.metrics['total_tiles_repaired']}")
    
    if result.map_elites_score:
        print(f"\nMAP-Elites Metrics:")
        print(f"  Linearity: {result.map_elites_score['linearity']:.3f}")
        print(f"  Leniency: {result.map_elites_score['leniency']:.3f}")
        print(f"  Path length: {result.map_elites_score['path_length']}")
    
    print(f"\nPer-Room Statistics:")
    for room_id, room_result in sorted(result.rooms.items()):
        status = "REPAIRED" if room_result.was_repaired else "ORIGINAL"
        tiles_changed = room_result.metrics.get('tiles_changed', 0)
        entropy = room_result.metrics.get('neural_grid_entropy', 0)
        
        print(f"  Room {room_id}: {status} "
              f"(entropy={entropy:.3f}, tiles_changed={tiles_changed})")
    
    print("="*70 + "\n")


def save_results(result, output_dir: Path):
    """Save generation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete dungeon
    np.save(output_dir / "dungeon_grid.npy", result.dungeon_grid)
    logger.info(f"Saved dungeon grid to {output_dir / 'dungeon_grid.npy'}")
    
    # Save individual rooms
    rooms_dir = output_dir / "rooms"
    rooms_dir.mkdir(exist_ok=True)
    
    for room_id, room_result in result.rooms.items():
        # Save room grid
        np.save(rooms_dir / f"room_{room_id}.npy", room_result.room_grid)
        
        # Save latent
        torch.save(room_result.latent, rooms_dir / f"room_{room_id}_latent.pt")
        
        # Save neural output (before repair)
        np.save(rooms_dir / f"room_{room_id}_neural.npy", room_result.neural_grid)
        
        # Save repair mask if available
        if room_result.repair_mask is not None:
            np.save(rooms_dir / f"room_{room_id}_repair_mask.npy", room_result.repair_mask)
    
    logger.info(f"Saved {len(result.rooms)} rooms to {rooms_dir}")
    
    # Save metrics as JSON
    import json
    
    metrics_dict = {
        'general': result.metrics,
        'map_elites': result.map_elites_score,
        'rooms': {
            room_id: room_result.metrics
            for room_id, room_result in result.rooms.items()
        }
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")


# =============================================================================
# EXAMPLE MISSION GRAPHS
# =============================================================================

def create_linear_graph(num_rooms: int = 4) -> nx.Graph:
    """Create a simple linear mission graph (1 → 2 → 3 → ...)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_rooms))
    G.add_edges_from([(i, i+1) for i in range(num_rooms - 1)])
    
    # Add node attributes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['room_type'] = 'start' if i == 0 else 'goal' if i == num_rooms - 1 else 'normal'
    
    return G


def create_branching_graph(num_rooms: int = 6) -> nx.Graph:
    """Create a branching mission graph with choices."""
    G = nx.DiGraph()
    
    # Create tree structure:
    #     0
    #    / \
    #   1   2
    #  / \   \
    # 3   4   5
    
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    limited_edges = edges[:num_rooms - 1]
    
    nodes = set()
    for src, dst in limited_edges:
        nodes.add(src)
        nodes.add(dst)
    
    G.add_nodes_from(sorted(nodes))
    G.add_edges_from(limited_edges)
    
    return G


# =============================================================================
# DEMO SCENARIOS
# =============================================================================

def demo_single_room_generation(pipeline: NeuralSymbolicDungeonPipeline, args):
    """Demo 1: Generate a single room with visualization."""
    print("\n" + "="*70)
    print("DEMO 1: SINGLE ROOM GENERATION")
    print("="*70 + "\n")
    
    # Prepare inputs
    neighbor_latents = {
        'N': torch.randn(1, 64, 4, 3, device=pipeline.device),
        'S': None,
        'E': None,
        'W': torch.randn(1, 64, 4, 3, device=pipeline.device),
    }
    
    graph_context = {
        'node_features': torch.randn(1, 6, device=pipeline.device),
        'edge_index': torch.zeros(2, 0, dtype=torch.long, device=pipeline.device),
        'tpe': torch.randn(1, 8, device=pipeline.device),
        'current_node_idx': 0,
    }
    
    # Generate room
    logger.info("Generating room with neural-symbolic pipeline...")
    
    result = pipeline.generate_room(
        neighbor_latents=neighbor_latents,
        graph_context=graph_context,
        room_id=0,
        guidance_scale=args.guidance_scale,
        logic_guidance_scale=args.logic_guidance_scale,
        num_diffusion_steps=args.num_diffusion_steps,
        apply_repair=True,
        start_goal_coords=((ROOM_HEIGHT // 2, 0), (ROOM_HEIGHT // 2, ROOM_WIDTH - 1)),
        seed=args.seed,
    )
    
    # Visualize
    print_room_ascii(result.neural_grid, "Neural Output (Before Repair)")
    print_room_ascii(result.room_grid, "Final Room (After Repair)")
    
    print(f"\nRoom Metrics:")
    print(f"  Entropy: {result.metrics['neural_grid_entropy']:.3f}")
    print(f"  Was repaired: {result.was_repaired}")
    print(f"  Tiles changed: {result.metrics.get('tiles_changed', 0)}")
    
    return result


def demo_dungeon_generation(pipeline: NeuralSymbolicDungeonPipeline, args):
    """Demo 2: Generate complete multi-room dungeon."""
    print("\n" + "="*70)
    print("DEMO 2: MULTI-ROOM DUNGEON GENERATION")
    print("="*70 + "\n")
    
    # Create mission graph
    if args.graph_type == 'linear':
        mission_graph = create_linear_graph(args.num_rooms)
        logger.info(f"Created linear mission graph with {args.num_rooms} rooms")
    elif args.graph_type == 'branching':
        mission_graph = create_branching_graph(args.num_rooms)
        logger.info(f"Created branching mission graph with {len(mission_graph.nodes)} rooms")
    else:
        raise ValueError(f"Unknown graph type: {args.graph_type}")
    
    # Generate dungeon
    logger.info("Generating complete dungeon...")
    
    result = pipeline.generate_dungeon(
        mission_graph=mission_graph,
        guidance_scale=args.guidance_scale,
        logic_guidance_scale=args.logic_guidance_scale,
        num_diffusion_steps=args.num_diffusion_steps,
        apply_repair=True,
        seed=args.seed,
        enable_map_elites=True,
    )
    
    # Visualize results
    visualize_dungeon_structure(result)
    
    # Show first and last room
    first_room_id = min(result.rooms.keys())
    last_room_id = max(result.rooms.keys())
    
    print_room_ascii(result.rooms[first_room_id].room_grid, f"Room {first_room_id} (Start)")
    print_room_ascii(result.rooms[last_room_id].room_grid, f"Room {last_room_id} (End)")
    
    return result


def demo_guidance_comparison(pipeline: NeuralSymbolicDungeonPipeline, args):
    """Demo 3: Compare generation with/without LogicNet guidance."""
    print("\n" + "="*70)
    print("DEMO 3: GUIDANCE COMPARISON")
    print("="*70 + "\n")
    
    mission_graph = create_linear_graph(3)
    
    # Without guidance
    logger.info("Generating dungeon WITHOUT LogicNet guidance...")
    result_no_guidance = pipeline.generate_dungeon(
        mission_graph=mission_graph,
        logic_guidance_scale=0.0,
        num_diffusion_steps=args.num_diffusion_steps,
        apply_repair=False,
        seed=args.seed,
    )
    
    # With guidance
    logger.info("Generating dungeon WITH LogicNet guidance...")
    result_with_guidance = pipeline.generate_dungeon(
        mission_graph=mission_graph,
        logic_guidance_scale=2.0,
        num_diffusion_steps=args.num_diffusion_steps,
        apply_repair=False,
        seed=args.seed,
    )
    
    # Compare
    print("\nComparison Results:")
    print(f"  Without guidance - Repair rate: {result_no_guidance.metrics['repair_rate']:.1%}")
    print(f"  With guidance    - Repair rate: {result_with_guidance.metrics['repair_rate']:.1%}")
    
    print("\n  Room 0 comparison:")
    print_room_ascii(result_no_guidance.rooms[0].room_grid, "Without Guidance")
    print_room_ascii(result_with_guidance.rooms[0].room_grid, "With Guidance")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural-Symbolic Dungeon Generation Demo"
    )
    
    # Pipeline setup
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run on'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num-rooms',
        type=int,
        default=4,
        help='Number of rooms in dungeon'
    )
    parser.add_argument(
        '--graph-type',
        type=str,
        default='linear',
        choices=['linear', 'branching'],
        help='Mission graph topology'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Classifier-free guidance scale'
    )
    parser.add_argument(
        '--logic-guidance-scale',
        type=float,
        default=1.0,
        help='LogicNet gradient guidance scale'
    )
    parser.add_argument(
        '--num-diffusion-steps',
        type=int,
        default=50,
        help='Number of diffusion sampling steps'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Demo control
    parser.add_argument(
        '--demo',
        type=str,
        default='all',
        choices=['all', '1', '2', '3'],
        help='Which demo to run (1=single room, 2=dungeon, 3=comparison, all=all demos)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save outputs (optional)'
    )
    
    return parser.parse_args()


def main():
    """Main demo entry point."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("NEURAL-SYMBOLIC DUNGEON GENERATION - INTERACTIVE DEMO")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    print(f"  Diffusion steps: {args.num_diffusion_steps}")
    print(f"  Guidance scales: CFG={args.guidance_scale}, Logic={args.logic_guidance_scale}")
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = create_pipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    logger.info(f"Pipeline ready on {pipeline.device}")
    
    # Run demos
    results = {}
    
    if args.demo in ['all', '1']:
        results['single_room'] = demo_single_room_generation(pipeline, args)
    
    if args.demo in ['all', '2']:
        results['dungeon'] = demo_dungeon_generation(pipeline, args)
    
    if args.demo in ['all', '3']:
        demo_guidance_comparison(pipeline, args)
    
    # Save results if output directory specified
    if args.output_dir and 'dungeon' in results:
        output_dir = Path(args.output_dir)
        save_results(results['dungeon'], output_dir)
        logger.info(f"All outputs saved to {output_dir}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
