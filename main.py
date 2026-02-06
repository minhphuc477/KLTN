"""
KLTN ZELDA DUNGEON PIPELINE - Main Entry Point
===============================================
The Golden Pipeline: Load -> Stitch -> Validate

This is the SINGLE entry point for the Zelda dungeon validation system.

Usage:
    # Basic validation of a single dungeon
    python main.py --dungeon 1 --variant 1
    
    # Validate all dungeons
    python main.py --all
    
    # Run with visualization
    python main.py --dungeon 1 --gui
    
    # Export processed data
    python main.py --dungeon 1 --export output.npz

"""

import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root path
PROJECT_ROOT = Path(__file__).parent

# Import from the canonical source
from Data.zelda_core import (
    # Main classes
    ZeldaDungeonAdapter,
    DungeonStitcher,
    DungeonSolver,
    
    # ML Features
    MLFeatureExtractor,
    GridBasedRoomExtractor,
    
    # Data classes
    Dungeon,
    StitchedDungeon,
    DungeonData,
    RoomData,
    
    # Constants
    SEMANTIC_PALETTE,
    ID_TO_NAME,
    ValidationMode,
    
    # Utilities
    visualize_semantic_grid,
    convert_dungeon_to_dungeondata,
    test_all_dungeons,
)

import numpy as np


def load_dungeon(dungeon_num: int, variant: int = 1, data_root: str = None) -> Dungeon:
    """
    Load a dungeon from VGLC data.
    
    Args:
        dungeon_num: Dungeon number (1-9)
        variant: Quest variant (1 or 2)
        data_root: Path to data folder (auto-detected if None)
        
    Returns:
        Dungeon object with rooms and graph
    """
    if data_root is None:
        data_root = str(PROJECT_ROOT / "Data" / "The Legend of Zelda")
    
    adapter = ZeldaDungeonAdapter(data_root)
    return adapter.load_dungeon(dungeon_num, variant=variant)


def stitch_dungeon(dungeon: Dungeon, compact: bool = True) -> StitchedDungeon:
    """
    Stitch dungeon rooms into global grid.
    
    Args:
        dungeon: Dungeon to stitch
        compact: Remove empty rows/columns
        
    Returns:
        StitchedDungeon with global grid
    """
    stitcher = DungeonStitcher()
    return stitcher.stitch(dungeon, compact=compact)


def validate_dungeon(stitched: StitchedDungeon, mode: str = ValidationMode.FULL) -> dict:
    """
    Validate dungeon solvability.
    
    Args:
        stitched: Stitched dungeon
        mode: ValidationMode (STRICT, REALISTIC, FULL)
        
    Returns:
        Validation result dict
    """
    solver = DungeonSolver()
    return solver.solve(stitched, mode=mode)


def run_pipeline(dungeon_num: int, variant: int = 1, 
                 mode: str = ValidationMode.FULL,
                 verbose: bool = True) -> dict:
    """
    Run the complete pipeline: Load -> Stitch -> Validate
    
    Args:
        dungeon_num: Dungeon number (1-9)
        variant: Quest variant (1 or 2)
        mode: Validation mode
        verbose: Print progress
        
    Returns:
        Complete result dict
    """
    if verbose:
        # User-facing output - keep print() for CLI summary
        print(f"\n{'='*60}")
        print(f"PIPELINE: Dungeon {dungeon_num} (Quest {variant})")
        print(f"{'='*60}")
    
    # Step 1: Load
    logger.info("[STEP 1] Loading dungeon data...")
    try:
        dungeon = load_dungeon(dungeon_num, variant)
    except FileNotFoundError as e:
        logger.error(f"Dungeon data not found: {e}")
        raise
    except Exception as e:
        logger.exception("Error loading dungeon")
        raise
    
    if verbose:
        print(f"\n[STEP 1] Loading dungeon data...")
        print(f"  ✓ Loaded {len(dungeon.rooms)} rooms")
        print(f"  ✓ Graph: {dungeon.graph.number_of_nodes()} nodes, {dungeon.graph.number_of_edges()} edges")
    
    # Step 2: Stitch
    logger.info("[STEP 2] Stitching rooms...")
    try:
        stitched = stitch_dungeon(dungeon)
    except Exception as e:
        logger.exception("Error stitching dungeon")
        raise
        raise
    
    if verbose:
        print(f"\n[STEP 2] Stitching rooms...")
        print(f"  ✓ Global grid: {stitched.global_grid.shape}")
        print(f"  ✓ Start: {stitched.start_global}")
        print(f"  ✓ Triforce: {stitched.triforce_global}")
    
    # Step 3: Validate
    logger.info(f"[STEP 3] Validating solvability (mode: {mode})...")
    try:
        result = validate_dungeon(stitched, mode=mode)
    except Exception as e:
        logger.exception("Error validating dungeon")
        raise
    
    if verbose:
        if result['solvable']:
            print(f"  ✓ SOLVABLE!")
            print(f"  ✓ Path length: {result.get('path_length', 'N/A')} steps")
            print(f"  ✓ Rooms traversed: {result.get('rooms_traversed', 'N/A')}")
            if 'keys_available' in result:
                print(f"  ✓ Keys available: {result['keys_available']}")
                print(f"  ✓ Keys used: {result['keys_used']}")
        else:
            print(f"  ✗ NOT SOLVABLE")
            print(f"  ✗ Reason: {result.get('reason', 'Unknown')}")
    
    # Return complete result
    return {
        'dungeon_num': dungeon_num,
        'variant': variant,
        'dungeon': dungeon,
        'stitched': stitched,
        'validation': result,
        'solvable': result['solvable']
    }


def export_dungeon_data(dungeon: Dungeon, output_path: str):
    """
    Export dungeon data with ML features to NPZ file.
    
    Args:
        dungeon: Dungeon to export
        output_path: Output file path
    """
    # Convert to DungeonData format with ML features
    dungeon_data = convert_dungeon_to_dungeondata(dungeon)
    
    # Prepare arrays for export
    room_grids = {}
    for room_id, room in dungeon_data.rooms.items():
        room_grids[f"room_{room_id}"] = room.grid
    
    np.savez(
        output_path,
        dungeon_id=dungeon_data.dungeon_id,
        layout=dungeon_data.layout,
        tpe_vectors=dungeon_data.tpe_vectors,
        p_matrix=dungeon_data.p_matrix,
        node_features=dungeon_data.node_features,
        **room_grids
    )
    logger.info(f"Exported dungeon data to: {output_path}")
    print(f"Exported to: {output_path}")  # User-facing output


def main():
    parser = argparse.ArgumentParser(
        description='KLTN Zelda Dungeon Pipeline - Load, Stitch, Validate'
    )
    
    parser.add_argument(
        '--dungeon', '-d', type=int, choices=range(1, 10),
        help='Dungeon number (1-9)'
    )
    parser.add_argument(
        '--variant', '-v', type=int, default=1, choices=[1, 2],
        help='Quest variant (1 or 2, default: 1)'
    )
    parser.add_argument(
        '--all', '-a', action='store_true',
        help='Validate all dungeons'
    )
    parser.add_argument(
        '--mode', '-m', 
        choices=['strict', 'realistic', 'full'],
        default='full',
        help='Validation mode (default: full)'
    )
    parser.add_argument(
        '--gui', '-g', action='store_true',
        help='Launch GUI visualizer'
    )
    parser.add_argument(
        '--export', '-e', type=str,
        help='Export to NPZ file'
    )
    parser.add_argument(
        '--data-root', type=str,
        help='Path to data folder'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output'
    )
    parser.add_argument(
        '--ascii', action='store_true',
        help='Print ASCII visualization of the dungeon'
    )
    
    args = parser.parse_args()
    
    # Map mode string to ValidationMode
    mode_map = {
        'strict': ValidationMode.STRICT,
        'realistic': ValidationMode.REALISTIC,
        'full': ValidationMode.FULL
    }
    mode = mode_map[args.mode]
    
    # Handle --all
    if args.all:
        if args.data_root:
            data_root = args.data_root
        else:
            data_root = str(PROJECT_ROOT / "Data" / "The Legend of Zelda")
        
        test_all_dungeons(data_root, include_variants=True)
        return
    
    # Require dungeon number for single dungeon operations
    if args.dungeon is None:
        parser.error("Either --dungeon or --all is required")
    
    # Run pipeline
    result = run_pipeline(
        args.dungeon, 
        args.variant,
        mode=mode,
        verbose=not args.quiet
    )
    
    # Export if requested
    if args.export:
        export_dungeon_data(result['dungeon'], args.export)
    
    # ASCII visualization
    if args.ascii:
        print("\n" + "="*60)
        print("ASCII VISUALIZATION")
        print("="*60)
        print(visualize_semantic_grid(result['stitched'].global_grid))
    
    # GUI if requested
    if args.gui:
        try:
            # Try new replay engine first
            from src.visualization.replay_engine import DungeonReplayEngine, ReplayConfig
            
            stitched = result['stitched']
            grid = stitched.global_grid
            
            # Get solution path from validation result
            validation = result['validation']
            solution_path = validation.get('path', [])
            
            # If no path from solver, try to solve again
            if not solution_path and validation.get('solvable'):
                from simulation.validator import ZeldaLogicEnv, StateSpaceAStar
                env = ZeldaLogicEnv(grid, render_mode=False)
                solver = StateSpaceAStar(env)
                success, solution_path, _ = solver.solve()
            
            config = ReplayConfig(
                window_title=f"ZAVE - Dungeon {args.dungeon} Variant {args.variant}",
                show_minimap=True,
                show_path_overlay=True,
            )
            
            engine = DungeonReplayEngine(
                dungeon_grid=grid,
                solution_path=solution_path if solution_path else [],
                config=config,
                solver_result=validation
            )
            
            engine.run()
            
        except ImportError as e:
            # Fall back to old GUI
            logger.warning(f"New visualization not available, using legacy GUI: {e}")
            try:
                from gui_runner import ZeldaGUI
                import pygame
                
                pygame.init()
                gui = ZeldaGUI([result['stitched']], [f"Dungeon {args.dungeon} Variant {args.variant}"])
                gui.run()
            except ImportError as e2:
                logger.error(f"GUI not available - missing dependency: {e2}")
                print(f"\nGUI not available: {e2}")
                print("Make sure pygame is installed: pip install pygame")
        except Exception as e:
            logger.exception("GUI error")
            print(f"\nGUI error: {e}")


if __name__ == "__main__":
    main()
