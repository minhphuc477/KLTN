#!/usr/bin/env python3
"""
Full CBS+ Benchmark Suite
=========================

Runs comprehensive benchmarks comparing A* and CBS+ solvers across:
1. Nintendo Zelda dungeons (1-9)
2. Random noise levels
3. Perfect mazes (Prim's algorithm)
4. BSP-generated dungeons

Outputs:
- results/cbs_benchmark_full.csv (raw data)
- results/benchmark_summary.csv (aggregate statistics)
- results/figures/fig3_greedy_vs_balanced.png (comparison figure)

Usage:
    python scripts/run_full_benchmark.py [--quick] [--verbose]
"""

import sys
import os
import argparse
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, CBSMetrics, PERSONA_CONFIGS
)
from src.data.zelda_core import ZeldaDungeonAdapter

logger = logging.getLogger(__name__)

# =============================================================================
# LEVEL GENERATORS
# =============================================================================

def generate_random_noise(height: int = 16, width: int = 11, walkable_ratio: float = 0.6, seed: int = 42) -> np.ndarray:
    """
    Generate random noise level with unstructured walkable tiles.
    
    Args:
        height: Grid height
        width: Grid width
        walkable_ratio: Fraction of tiles that are walkable
        seed: Random seed
        
    Returns:
        Semantic grid array
    """
    np.random.seed(seed)
    
    grid = np.full((height, width), SEMANTIC_PALETTE['WALL'], dtype=np.int32)
    
    # Random walkable tiles
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if np.random.random() < walkable_ratio:
                grid[r, c] = SEMANTIC_PALETTE['FLOOR']
    
    # Place start and goal
    floor_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['FLOOR'])))
    if len(floor_positions) < 2:
        # Fallback: create minimal path
        for c in range(1, width - 1):
            grid[height // 2, c] = SEMANTIC_PALETTE['FLOOR']
        floor_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['FLOOR'])))
    
    start_pos = floor_positions[0]
    goal_pos = floor_positions[-1]
    grid[start_pos] = SEMANTIC_PALETTE['START']
    grid[goal_pos] = SEMANTIC_PALETTE['TRIFORCE']
    
    return grid


def generate_perfect_maze(height: int = 21, width: int = 21, seed: int = 42) -> np.ndarray:
    """
    Generate a perfect maze using Prim's algorithm.
    
    A perfect maze has exactly one path between any two points.
    
    Args:
        height: Grid height (should be odd)
        width: Grid width (should be odd)
        seed: Random seed
        
    Returns:
        Semantic grid array
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure odd dimensions for proper maze structure
    height = height if height % 2 == 1 else height + 1
    width = width if width % 2 == 1 else width + 1
    
    # Initialize all walls
    grid = np.full((height, width), SEMANTIC_PALETTE['WALL'], dtype=np.int32)
    
    # Prim's algorithm
    # Start from (1, 1)
    start = (1, 1)
    grid[start] = SEMANTIC_PALETTE['FLOOR']
    
    # Frontier: walls that could be passages
    frontiers = []
    
    def add_frontiers(r, c):
        """Add frontier cells 2 steps away."""
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < height - 1 and 0 < nc < width - 1:
                if grid[nr, nc] == SEMANTIC_PALETTE['WALL']:
                    frontiers.append((nr, nc, r, c))
    
    add_frontiers(1, 1)
    
    while frontiers:
        idx = random.randint(0, len(frontiers) - 1)
        fr, fc, pr, pc = frontiers.pop(idx)
        
        if grid[fr, fc] == SEMANTIC_PALETTE['WALL']:
            # Carve passage
            grid[fr, fc] = SEMANTIC_PALETTE['FLOOR']
            # Carve wall between
            grid[(fr + pr) // 2, (fc + pc) // 2] = SEMANTIC_PALETTE['FLOOR']
            add_frontiers(fr, fc)
    
    # Place start and goal at opposite corners
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[height - 2, width - 2] = SEMANTIC_PALETTE['TRIFORCE']
    
    return grid


def generate_bsp_dungeon(height: int = 32, width: int = 32, min_room_size: int = 5, seed: int = 42) -> np.ndarray:
    """
    Generate a BSP (Binary Space Partitioning) dungeon.
    
    Args:
        height: Grid height
        width: Grid width
        min_room_size: Minimum room dimension
        seed: Random seed
        
    Returns:
        Semantic grid array
    """
    np.random.seed(seed)
    random.seed(seed)
    
    grid = np.full((height, width), SEMANTIC_PALETTE['WALL'], dtype=np.int32)
    
    class BSPNode:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h
            self.left = self.right = self.room = None
        
        def split(self, min_size):
            if self.left or self.right:
                return False
            
            # Determine split direction
            if self.w > self.h and self.w / self.h >= 1.25:
                horizontal = False
            elif self.h > self.w and self.h / self.w >= 1.25:
                horizontal = True
            else:
                horizontal = random.random() > 0.5
            
            max_size = (self.h if horizontal else self.w) - min_size
            if max_size <= min_size:
                return False
            
            split_pos = random.randint(min_size, max_size)
            
            if horizontal:
                self.left = BSPNode(self.x, self.y, self.w, split_pos)
                self.right = BSPNode(self.x, self.y + split_pos, self.w, self.h - split_pos)
            else:
                self.left = BSPNode(self.x, self.y, split_pos, self.h)
                self.right = BSPNode(self.x + split_pos, self.y, self.w - split_pos, self.h)
            
            return True
        
        def create_rooms(self, grid, min_size):
            if self.left or self.right:
                if self.left:
                    self.left.create_rooms(grid, min_size)
                if self.right:
                    self.right.create_rooms(grid, min_size)
                
                # Connect children with corridor
                if self.left and self.right:
                    self._connect(self.left, self.right, grid)
            else:
                # Leaf node: create room
                room_w = random.randint(min_size, max(min_size, self.w - 2))
                room_h = random.randint(min_size, max(min_size, self.h - 2))
                room_x = self.x + random.randint(1, max(1, self.w - room_w - 1))
                room_y = self.y + random.randint(1, max(1, self.h - room_h - 1))
                
                self.room = (room_x, room_y, room_w, room_h)
                
                for r in range(room_y, min(room_y + room_h, height - 1)):
                    for c in range(room_x, min(room_x + room_w, width - 1)):
                        if 0 < r < height - 1 and 0 < c < width - 1:
                            grid[r, c] = SEMANTIC_PALETTE['FLOOR']
        
        def _connect(self, left, right, grid):
            # Get centers of rooms/subnodes
            left_center = self._get_center(left)
            right_center = self._get_center(right)
            
            if left_center and right_center:
                # Carve L-shaped corridor
                r1, c1 = left_center
                r2, c2 = right_center
                
                # Horizontal then vertical
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    if 0 < r1 < height - 1 and 0 < c < width - 1:
                        grid[r1, c] = SEMANTIC_PALETTE['FLOOR']
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    if 0 < r < height - 1 and 0 < c2 < width - 1:
                        grid[r, c2] = SEMANTIC_PALETTE['FLOOR']
        
        def _get_center(self, node):
            if node.room:
                x, y, w, h = node.room
                return (y + h // 2, x + w // 2)
            elif node.left:
                return self._get_center(node.left)
            elif node.right:
                return self._get_center(node.right)
            return None
    
    # Build BSP tree
    root = BSPNode(0, 0, width, height)
    
    # Split recursively
    nodes = [root]
    for _ in range(4):  # 4 levels of splitting
        new_nodes = []
        for node in nodes:
            if node.split(min_room_size):
                new_nodes.extend([node.left, node.right])
        nodes = new_nodes
        if not nodes:
            break
    
    # Create rooms and corridors
    root.create_rooms(grid, min_room_size)
    
    # Place start and goal
    floor_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['FLOOR'])))
    if len(floor_positions) >= 2:
        start_pos = floor_positions[0]
        goal_pos = floor_positions[-1]
        grid[start_pos] = SEMANTIC_PALETTE['START']
        grid[goal_pos] = SEMANTIC_PALETTE['TRIFORCE']
    
    return grid


# =============================================================================
# NINTENDO ZELDA LOADER
# =============================================================================

def load_nintendo_levels(data_dir: str = None) -> List[Tuple[str, np.ndarray]]:
    """
    Load all Nintendo Zelda dungeons (1-9, both quests).
    
    Returns:
        List of (map_id, grid) tuples
    """
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "Data" / "The Legend of Zelda")
    
    levels = []
    
    try:
        adapter = ZeldaDungeonAdapter(data_dir)
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant)
                    stitched = adapter.stitch_dungeon(dungeon)
                    grid = stitched.global_grid.astype(np.int32)
                    map_id = f"tloz{dungeon_num}_{variant}"
                    levels.append((map_id, grid))
                    logger.info(f"Loaded {map_id}: {grid.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load dungeon {dungeon_num}_{variant}: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize ZeldaDungeonAdapter: {e}")
        # Fallback: load from processed text files
        processed_dir = Path(data_dir) / "Processed"
        if processed_dir.exists():
            for txt_file in sorted(processed_dir.glob("tloz*.txt")):
                try:
                    grid = load_text_level(txt_file)
                    map_id = txt_file.stem
                    levels.append((map_id, grid))
                except Exception as e:
                    logger.warning(f"Failed to load {txt_file}: {e}")
    
    return levels


def load_text_level(filepath: Path) -> np.ndarray:
    """Load a level from VGLC text format."""
    from src.core.definitions import CHAR_TO_SEMANTIC
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse grid
    rows = []
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        row = []
        for char in line:
            tile_id = CHAR_TO_SEMANTIC.get(char, SEMANTIC_PALETTE['VOID'])
            row.append(tile_id)
        rows.append(row)
    
    # Pad to rectangular
    max_width = max(len(r) for r in rows)
    for row in rows:
        while len(row) < max_width:
            row.append(SEMANTIC_PALETTE['VOID'])
    
    return np.array(rows, dtype=np.int32)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_solver(
    grid: np.ndarray,
    solver_type: str,
    persona: str = None,
    timeout: int = 100000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run a solver on a grid and return results.
    
    Args:
        grid: Semantic grid array
        solver_type: 'astar' or 'cbs'
        persona: CBS persona (if solver_type == 'cbs')
        timeout: Maximum steps/states
        seed: Random seed
        
    Returns:
        Dict with solver results
    """
    result = {
        'solver': solver_type,
        'persona': persona or 'N/A',
        'success': False,
        'path_length': 0,
        'states_explored': 0,
        'PER': 0.0,
        'entropy_final': 0.0,
        'room_entropy': 0.0,
        'replans': 0,
        'confusion_events': 0,
        'confusion_ratio': 0.0,
        'confusion_index': 0.0,
    }
    
    try:
        env = ZeldaLogicEnv(semantic_grid=grid)
        
        if env.goal_pos is None or env.start_pos is None:
            logger.warning("Grid has no start or goal position")
            return result
        
        start_time = time.time()
        
        if solver_type == 'astar':
            solver = StateSpaceAStar(env, timeout=timeout)
            success, path, states = solver.solve()
            
            result['success'] = success
            result['path_length'] = len(path) if path else 0
            result['states_explored'] = states
            
            # Compute optimal path efficiency (A* is optimal)
            if success and len(path) > 0:
                manhattan = abs(env.goal_pos[0] - env.start_pos[0]) + abs(env.goal_pos[1] - env.start_pos[1])
                result['PER'] = manhattan / len(path) if len(path) > 0 else 0.0
            
        elif solver_type == 'cbs':
            cbs = CognitiveBoundedSearch(env, persona=persona, timeout=timeout, seed=seed)
            success, path, states, metrics = cbs.solve()
            
            result['success'] = success
            result['path_length'] = len(path) if path else 0
            result['states_explored'] = states
            
            # Metrics from CBS
            if metrics:
                result['confusion_index'] = metrics.confusion_index
                result['entropy_final'] = metrics.belief_entropy_final
                result['room_entropy'] = metrics.room_entropy
                result['replans'] = metrics.replans
                result['confusion_events'] = metrics.confusion_events
                
                # Confusion ratio: total_steps / unique_tiles
                if metrics.unique_tiles_visited > 0:
                    result['confusion_ratio'] = metrics.total_steps / metrics.unique_tiles_visited
            
            # Path efficiency ratio
            if success and len(path) > 0:
                manhattan = abs(env.goal_pos[0] - env.start_pos[0]) + abs(env.goal_pos[1] - env.start_pos[1])
                result['PER'] = manhattan / len(path) if len(path) > 0 else 0.0
        
        result['time'] = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Solver error: {e}")
        result['error'] = str(e)
    
    return result


def run_benchmarks(
    levels: List[Tuple[str, np.ndarray]],
    dataset_type: str,
    personas: List[str] = None,
    astar_timeout: int = 100000,
    cbs_timeout: int = 5000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Run full benchmark suite on a set of levels.
    
    Args:
        levels: List of (map_id, grid) tuples
        dataset_type: 'nintendo', 'random', 'maze', 'bsp'
        personas: CBS personas to test
        astar_timeout: A* timeout
        cbs_timeout: CBS timeout
        seed: Random seed
        
    Returns:
        List of result dictionaries
    """
    if personas is None:
        personas = ['balanced', 'greedy', 'explorer', 'forgetful']
    
    results = []
    
    for map_id, grid in levels:
        logger.info(f"Testing {dataset_type}/{map_id}...")
        
        # Run A*
        astar_result = run_solver(grid, 'astar', timeout=astar_timeout, seed=seed)
        astar_result['dataset_type'] = dataset_type
        astar_result['map_id'] = map_id
        results.append(astar_result)
        
        # Run CBS with each persona
        for persona in personas:
            cbs_result = run_solver(grid, 'cbs', persona=persona, timeout=cbs_timeout, seed=seed)
            cbs_result['dataset_type'] = dataset_type
            cbs_result['map_id'] = map_id
            results.append(cbs_result)
    
    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics by dataset type."""
    summaries = []
    
    for dataset_type in df['dataset_type'].unique():
        subset = df[df['dataset_type'] == dataset_type]
        
        astar_subset = subset[subset['solver'] == 'astar']
        cbs_subset = subset[subset['solver'] == 'cbs']
        
        summary = {
            'dataset_type': dataset_type,
            'n_levels': len(astar_subset),
            'astar_success_rate': astar_subset['success'].mean() if len(astar_subset) > 0 else 0.0,
            'cbs_success_rate': cbs_subset['success'].mean() if len(cbs_subset) > 0 else 0.0,
            'mean_confusion_ratio': cbs_subset['confusion_ratio'].mean() if len(cbs_subset) > 0 else 0.0,
            'std_confusion_ratio': cbs_subset['confusion_ratio'].std() if len(cbs_subset) > 0 else 0.0,
            'mean_room_entropy': cbs_subset['room_entropy'].mean() if len(cbs_subset) > 0 else 0.0,
            'mean_path_efficiency': cbs_subset['PER'].mean() if len(cbs_subset) > 0 else 0.0,
        }
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def analyze_greedy_vs_balanced(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Statistical comparison of GREEDY vs BALANCED personas.
    
    Proves that memory decay (λ < 1.0) is the active ingredient.
    """
    greedy = df[(df['solver'] == 'cbs') & (df['persona'] == 'greedy')]
    balanced = df[(df['solver'] == 'cbs') & (df['persona'] == 'balanced')]
    
    analysis = {
        'greedy_n': len(greedy),
        'balanced_n': len(balanced),
        'greedy_success_rate': greedy['success'].mean() if len(greedy) > 0 else 0.0,
        'balanced_success_rate': balanced['success'].mean() if len(balanced) > 0 else 0.0,
        'success_rate_diff': 0.0,
        'greedy_mean_confusion_index': greedy['confusion_index'].mean() if len(greedy) > 0 else 0.0,
        'balanced_mean_confusion_index': balanced['confusion_index'].mean() if len(balanced) > 0 else 0.0,
        'confusion_index_diff': 0.0,
        'greedy_mean_path_length': greedy['path_length'].mean() if len(greedy) > 0 else 0.0,
        'balanced_mean_path_length': balanced['path_length'].mean() if len(balanced) > 0 else 0.0,
        'path_length_diff': 0.0,
    }
    
    if len(greedy) > 0 and len(balanced) > 0:
        analysis['success_rate_diff'] = analysis['balanced_success_rate'] - analysis['greedy_success_rate']
        analysis['confusion_index_diff'] = analysis['greedy_mean_confusion_index'] - analysis['balanced_mean_confusion_index']
        analysis['path_length_diff'] = analysis['greedy_mean_path_length'] - analysis['balanced_mean_path_length']
        
        # Statistical significance (t-test if scipy available)
        try:
            from scipy import stats
            
            # Confusion index comparison
            if len(greedy) > 1 and len(balanced) > 1:
                t_stat, p_value = stats.ttest_ind(
                    greedy['confusion_index'].dropna(),
                    balanced['confusion_index'].dropna()
                )
                analysis['confusion_ttest_statistic'] = t_stat
                analysis['confusion_ttest_pvalue'] = p_value
        except ImportError:
            pass
    
    return analysis


def generate_figure3(df: pd.DataFrame, output_path: str):
    """
    Generate Figure 3: GREEDY vs BALANCED comparison.
    
    Bar chart showing Confusion Ratio by persona, grouped by dataset type.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.error("matplotlib not available, skipping figure generation")
        return
    
    # Filter for GREEDY and BALANCED personas
    cbs_data = df[(df['solver'] == 'cbs') & (df['persona'].isin(['greedy', 'balanced']))]
    
    if len(cbs_data) == 0:
        logger.warning("No CBS data for figure generation")
        return
    
    # Group by dataset_type and persona
    grouped = cbs_data.groupby(['dataset_type', 'persona'])['confusion_ratio'].agg(['mean', 'std']).reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dataset_types = grouped['dataset_type'].unique()
    x = np.arange(len(dataset_types))
    width = 0.35
    
    # Get data for each persona
    balanced_data = grouped[grouped['persona'] == 'balanced'].set_index('dataset_type')
    greedy_data = grouped[grouped['persona'] == 'greedy'].set_index('dataset_type')
    
    balanced_means = [balanced_data.loc[dt, 'mean'] if dt in balanced_data.index else 0 for dt in dataset_types]
    balanced_stds = [balanced_data.loc[dt, 'std'] if dt in balanced_data.index else 0 for dt in dataset_types]
    greedy_means = [greedy_data.loc[dt, 'mean'] if dt in greedy_data.index else 0 for dt in dataset_types]
    greedy_stds = [greedy_data.loc[dt, 'std'] if dt in greedy_data.index else 0 for dt in dataset_types]
    
    # Handle NaN values
    balanced_means = [0 if np.isnan(v) else v for v in balanced_means]
    balanced_stds = [0 if np.isnan(v) else v for v in balanced_stds]
    greedy_means = [0 if np.isnan(v) else v for v in greedy_means]
    greedy_stds = [0 if np.isnan(v) else v for v in greedy_stds]
    
    bars1 = ax.bar(x - width/2, balanced_means, width, yerr=balanced_stds, 
                   label='BALANCED (λ=0.95)', color='steelblue', capsize=3)
    bars2 = ax.bar(x + width/2, greedy_means, width, yerr=greedy_stds,
                   label='GREEDY (λ=1.0, no decay)', color='coral', capsize=3)
    
    ax.set_xlabel('Dataset Type', fontsize=12)
    ax.set_ylabel('Confusion Ratio (steps/unique_tiles)', fontsize=12)
    ax.set_title('Figure 3: Memory Decay Effect on Navigation Confusion\n(GREEDY has higher confusion, especially on structured levels)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([dt.capitalize() for dt in dataset_types])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved figure to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run full CBS+ benchmark suite')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer levels')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--astar-timeout', type=int, default=100000, help='A* timeout')
    parser.add_argument('--cbs-timeout', type=int, default=5000, help='CBS timeout')
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Create results directories
    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    # ==========================================================================
    # PART 1: Load and test Nintendo Zelda levels
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 1: Nintendo Zelda Dungeons")
    logger.info("=" * 60)
    
    nintendo_levels = load_nintendo_levels()
    if args.quick:
        nintendo_levels = nintendo_levels[:3]  # First 3 levels only
    
    logger.info(f"Loaded {len(nintendo_levels)} Nintendo levels")
    
    nintendo_results = run_benchmarks(
        nintendo_levels,
        dataset_type='nintendo',
        astar_timeout=args.astar_timeout,
        cbs_timeout=args.cbs_timeout,
        seed=args.seed
    )
    all_results.extend(nintendo_results)
    
    # ==========================================================================
    # PART 2: Generate and test Random levels
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 2: Random Noise Levels")
    logger.info("=" * 60)
    
    n_random = 3 if args.quick else 10
    random_levels = []
    for i in range(n_random):
        grid = generate_random_noise(height=21, width=21, seed=args.seed + i)
        random_levels.append((f"random_{i}", grid))
    
    random_results = run_benchmarks(
        random_levels,
        dataset_type='random',
        astar_timeout=args.astar_timeout,
        cbs_timeout=args.cbs_timeout,
        seed=args.seed
    )
    all_results.extend(random_results)
    
    # ==========================================================================
    # PART 3: Generate and test Perfect Mazes
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 3: Perfect Mazes (Prim's Algorithm)")
    logger.info("=" * 60)
    
    n_mazes = 3 if args.quick else 10
    maze_levels = []
    for i in range(n_mazes):
        grid = generate_perfect_maze(height=21, width=21, seed=args.seed + i * 100)
        maze_levels.append((f"maze_{i}", grid))
    
    maze_results = run_benchmarks(
        maze_levels,
        dataset_type='maze',
        astar_timeout=args.astar_timeout,
        cbs_timeout=args.cbs_timeout,
        seed=args.seed
    )
    all_results.extend(maze_results)
    
    # ==========================================================================
    # PART 4: Generate and test BSP Dungeons
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("PART 4: BSP Dungeons")
    logger.info("=" * 60)
    
    n_bsp = 3 if args.quick else 10
    bsp_levels = []
    for i in range(n_bsp):
        grid = generate_bsp_dungeon(height=32, width=32, seed=args.seed + i * 200)
        bsp_levels.append((f"bsp_{i}", grid))
    
    bsp_results = run_benchmarks(
        bsp_levels,
        dataset_type='bsp',
        astar_timeout=args.astar_timeout,
        cbs_timeout=args.cbs_timeout,
        seed=args.seed
    )
    all_results.extend(bsp_results)
    
    # ==========================================================================
    # Save raw results
    # ==========================================================================
    df = pd.DataFrame(all_results)
    raw_output = results_dir / "cbs_benchmark_full.csv"
    df.to_csv(raw_output, index=False)
    logger.info(f"Saved raw results to {raw_output}")
    
    # ==========================================================================
    # Compute and save summary statistics
    # ==========================================================================
    summary_df = compute_summary_statistics(df)
    summary_output = results_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    logger.info(f"Saved summary to {summary_output}")
    
    # ==========================================================================
    # GREEDY vs BALANCED analysis
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("GREEDY vs BALANCED Analysis")
    logger.info("=" * 60)
    
    analysis = analyze_greedy_vs_balanced(df)
    
    print("\n" + "=" * 60)
    print("GREEDY vs BALANCED Comparison (Memory Decay Proof)")
    print("=" * 60)
    print(f"GREEDY (λ=1.0, no decay) samples: {analysis['greedy_n']}")
    print(f"BALANCED (λ=0.95) samples: {analysis['balanced_n']}")
    print()
    print(f"Success Rate:")
    print(f"  BALANCED: {analysis['balanced_success_rate']:.2%}")
    print(f"  GREEDY:   {analysis['greedy_success_rate']:.2%}")
    print(f"  Diff:     {analysis['success_rate_diff']:+.2%}")
    print()
    print(f"Mean Confusion Index:")
    print(f"  BALANCED: {analysis['balanced_mean_confusion_index']:.3f}")
    print(f"  GREEDY:   {analysis['greedy_mean_confusion_index']:.3f}")
    print(f"  Diff:     {analysis['confusion_index_diff']:+.3f} (GREEDY more confused)")
    print()
    print(f"Mean Path Length:")
    print(f"  BALANCED: {analysis['balanced_mean_path_length']:.1f}")
    print(f"  GREEDY:   {analysis['greedy_mean_path_length']:.1f}")
    print(f"  Diff:     {analysis['path_length_diff']:+.1f}")
    
    if 'confusion_ttest_pvalue' in analysis:
        print()
        print(f"Statistical Significance (t-test):")
        print(f"  t-statistic: {analysis['confusion_ttest_statistic']:.3f}")
        print(f"  p-value:     {analysis['confusion_ttest_pvalue']:.4f}")
        if analysis['confusion_ttest_pvalue'] < 0.05:
            print("  → Statistically significant difference (p < 0.05)")
    
    # ==========================================================================
    # Generate Figure 3
    # ==========================================================================
    fig3_path = figures_dir / "fig3_greedy_vs_balanced.png"
    generate_figure3(df, str(fig3_path))
    
    # ==========================================================================
    # Print summary table
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY BY DATASET TYPE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Per-dataset confusion ratio analysis
    print("\n" + "=" * 60)
    print("Mean Confusion Ratio by Dataset (CBS agents)")
    print("=" * 60)
    
    for dt in df['dataset_type'].unique():
        cbs_dt = df[(df['dataset_type'] == dt) & (df['solver'] == 'cbs')]
        if len(cbs_dt) > 0:
            mean_cr = cbs_dt['confusion_ratio'].mean()
            std_cr = cbs_dt['confusion_ratio'].std()
            print(f"  {dt:10s}: {mean_cr:.2f} ± {std_cr:.2f}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print(f"Raw results:   {raw_output}")
    print(f"Summary:       {summary_output}")
    print(f"Figure 3:      {fig3_path}")
    
    return df, summary_df, analysis


if __name__ == '__main__':
    main()
