"""
Visualize CBS+ Results for Paper Figures.

Generates three types of figures:

Figure 1: Mind's Eye Comparison
- Left: True map (full grid)
- Right: Belief map at step N (faded regions = low confidence)

Figure 2: Heatmap of Frustration
- Overlay agent path on map
- Highlight backtracking loops in red
- Show "confusion hotspots" where agent circled

Figure 3: Complexity Spectrum
- Bar chart: A* vs CBS path lengths per dungeon
- Bar chart: A* vs CBS states explored per dungeon
- Show Confusion Ratio as line overlay

Usage:
    python scripts/visualize_cbs.py --level 1
    python scripts/visualize_cbs.py --csv results/cbs_benchmark.csv
    python scripts/visualize_cbs.py --all
"""
import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

from src.core.definitions import SEMANTIC_PALETTE, ID_TO_NAME
from src.data.zelda_core import ZeldaDungeonAdapter
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, BeliefMap, CBSMetrics, AgentPersona
)


# Color palette for tiles
TILE_COLORS = {
    SEMANTIC_PALETTE['VOID']: '#1a1a2e',      # Dark blue-black
    SEMANTIC_PALETTE['FLOOR']: '#16213e',     # Dark blue
    SEMANTIC_PALETTE['WALL']: '#4a4e69',      # Gray
    SEMANTIC_PALETTE['BLOCK']: '#5c4033',     # Brown
    SEMANTIC_PALETTE['DOOR_OPEN']: '#0f3460', # Blue
    SEMANTIC_PALETTE['DOOR_LOCKED']: '#e94560', # Red
    SEMANTIC_PALETTE['DOOR_BOMB']: '#ff6b35', # Orange
    SEMANTIC_PALETTE['DOOR_BOSS']: '#7209b7', # Purple
    SEMANTIC_PALETTE['DOOR_SOFT']: '#3a86ff', # Light blue
    SEMANTIC_PALETTE['ENEMY']: '#d62828',     # Red
    SEMANTIC_PALETTE['BOSS']: '#9d0208',      # Dark red
    SEMANTIC_PALETTE['START']: '#00b4d8',     # Cyan
    SEMANTIC_PALETTE['TRIFORCE']: '#ffd700',  # Gold
    SEMANTIC_PALETTE['KEY_SMALL']: '#f4d03f', # Yellow
    SEMANTIC_PALETTE['KEY_BOSS']: '#ff6b6b',  # Pink
    SEMANTIC_PALETTE['KEY_ITEM']: '#06ffa5',  # Green
    SEMANTIC_PALETTE['ITEM_MINOR']: '#4cc9f0', # Light cyan
    SEMANTIC_PALETTE['ELEMENT']: '#48cae4',   # Water blue
    SEMANTIC_PALETTE['STAIR']: '#023e8a',     # Dark blue
    SEMANTIC_PALETTE['PUZZLE']: '#9d4edd',    # Purple
}


def get_tile_color(tile_id: int, confidence: float = 1.0) -> Tuple[float, float, float, float]:
    """Get RGBA color for a tile with confidence-based alpha."""
    hex_color = TILE_COLORS.get(tile_id, '#333333')
    # Convert hex to RGB
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    # Alpha based on confidence
    alpha = 0.3 + 0.7 * confidence  # Range [0.3, 1.0]
    return (r, g, b, alpha)


def grid_to_rgba(grid: np.ndarray, confidence_grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert semantic grid to RGBA image array."""
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            conf = confidence_grid[i, j] if confidence_grid is not None else 1.0
            rgba[i, j] = get_tile_color(grid[i, j], conf)
    
    return rgba


def run_cbs_for_visualization(
    dungeon_num: int,
    variant: int = 1,
    persona: str = 'balanced',
    max_steps: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, BeliefMap, List[Tuple[int, int]], CBSMetrics, np.ndarray]:
    """
    Run CBS+ and return data needed for visualization.
    
    Returns:
        ground_truth_grid: The actual map
        belief_map: Agent's belief map
        path: Agent's path
        metrics: CBS metrics
        astar_path: A* optimal path for comparison
    """
    adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
    dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
    stitched = adapter.stitch_dungeon(dungeon)
    grid = stitched.global_grid.copy()
    start = stitched.start_global
    goal = stitched.triforce_global
    
    if start is None or goal is None:
        raise ValueError(f"Dungeon {dungeon_num} v{variant} missing start/goal")
    
    # Run A* for comparison
    env_astar = ZeldaLogicEnv(semantic_grid=grid)
    env_astar.start_pos = start
    env_astar.goal_pos = goal
    solver_a = StateSpaceAStar(env_astar, timeout=200000)
    success_a, astar_path, _ = solver_a.solve()
    
    # Run CBS+
    env_cbs = ZeldaLogicEnv(semantic_grid=grid)
    env_cbs.start_pos = start
    env_cbs.goal_pos = goal
    cbs = CognitiveBoundedSearch(env_cbs, persona=persona, timeout=max_steps, seed=seed)
    success_c, cbs_path, states_c, metrics = cbs.solve()
    
    return grid, cbs.belief_map, cbs_path, metrics, astar_path


def figure1_minds_eye(
    ground_truth: np.ndarray,
    belief_map: BeliefMap,
    path: List[Tuple[int, int]],
    step: int = 200,
    output_path: str = 'results/figures/fig1_minds_eye.png'
) -> None:
    """
    Generate Figure 1: Mind's Eye Comparison.
    
    Left: True map (full grid)
    Right: Belief map at step N (faded regions = low confidence)
    """
    if not HAS_MATPLOTLIB:
        print("Skipping figure generation - matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Ground truth
    ax1 = axes[0]
    truth_rgba = grid_to_rgba(ground_truth)
    ax1.imshow(truth_rgba, interpolation='nearest')
    ax1.set_title('Ground Truth Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Overlay path on ground truth
    if path:
        path_arr = np.array(path)
        ax1.plot(path_arr[:, 1], path_arr[:, 0], 'w-', alpha=0.7, linewidth=1.5, label='Agent Path')
        ax1.plot(path_arr[0, 1], path_arr[0, 0], 'go', markersize=10, label='Start')
        ax1.plot(path_arr[-1, 1], path_arr[-1, 0], 'r*', markersize=12, label='End')
    ax1.legend(loc='upper right')
    
    # Right: Belief map with confidence
    ax2 = axes[1]
    belief_grid = belief_map.to_grid(include_confidence=True)
    tile_grid = belief_grid[:, :, 0].astype(int)
    conf_grid = belief_grid[:, :, 1]
    belief_rgba = grid_to_rgba(tile_grid, conf_grid)
    ax2.imshow(belief_rgba, interpolation='nearest')
    ax2.set_title(f"Agent's Belief Map (Step {step})\nFaded = Low Confidence", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add colorbar for confidence
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.6)
    cbar.set_label('Confidence', fontsize=12)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to {output_path}")


def figure2_frustration_heatmap(
    ground_truth: np.ndarray,
    path: List[Tuple[int, int]],
    metrics: CBSMetrics,
    output_path: str = 'results/figures/fig2_frustration_heatmap.png'
) -> None:
    """
    Generate Figure 2: Heatmap of Frustration.
    
    - Overlay agent path on map
    - Highlight backtracking loops in red
    - Show "confusion hotspots" where agent circled
    """
    if not HAS_MATPLOTLIB:
        print("Skipping figure generation - matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw base map
    truth_rgba = grid_to_rgba(ground_truth)
    ax.imshow(truth_rgba, interpolation='nearest', alpha=0.7)
    
    # Compute visit counts for heatmap
    h, w = ground_truth.shape
    visit_heatmap = np.zeros((h, w), dtype=np.float32)
    for pos in path:
        if 0 <= pos[0] < h and 0 <= pos[1] < w:
            visit_heatmap[pos[0], pos[1]] += 1
    
    # Normalize and overlay heatmap (only where visited > 1 = confusion)
    confusion_mask = visit_heatmap > 1
    if confusion_mask.any():
        # Create custom colormap: transparent -> yellow -> red
        colors = [(0, 0, 0, 0), (1, 1, 0, 0.5), (1, 0, 0, 0.8)]
        cmap = LinearSegmentedColormap.from_list('confusion', colors)
        
        # Normalize visits (1 = visited once, higher = more confusion)
        max_visits = visit_heatmap.max()
        if max_visits > 1:
            norm_visits = (visit_heatmap - 1) / (max_visits - 1)
            norm_visits[~confusion_mask] = 0
            ax.imshow(norm_visits, cmap=cmap, interpolation='nearest', alpha=0.6)
    
    # Draw path with segments colored by order
    if len(path) > 1:
        path_arr = np.array(path)
        
        # Create line segments
        points = np.array([path_arr[:, 1], path_arr[:, 0]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Color by time (blue = early, green = late)
        norm = Normalize(0, len(segments))
        lc = LineCollection(segments, cmap='winter', norm=norm, linewidths=2, alpha=0.8)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)
        
        # Mark start and end
        ax.plot(path_arr[0, 1], path_arr[0, 0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(path_arr[-1, 1], path_arr[-1, 0], 'r*', markersize=14, label='End', zorder=5)
    
    # Find and highlight backtrack loops
    loop_positions = detect_loops(path)
    if loop_positions:
        for pos in loop_positions:
            circle = plt.Circle((pos[1], pos[0]), 1.5, color='red', fill=False, 
                               linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle)
    
    ax.set_title(f'Frustration Heatmap\nConfusion Events: {metrics.confusion_events}, '
                f'Backtrack Loops: {metrics.backtrack_loops}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='yellow', alpha=0.5, label='Mild Confusion (2-3 visits)'),
        mpatches.Patch(color='red', alpha=0.5, label='High Confusion (4+ visits)'),
        plt.Line2D([0], [0], color='green', marker='o', linestyle='', markersize=10, label='Start'),
        plt.Line2D([0], [0], color='red', marker='*', linestyle='', markersize=12, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {output_path}")


def detect_loops(path: List[Tuple[int, int]], min_loop_size: int = 3) -> List[Tuple[int, int]]:
    """Detect positions where loops occur in the path."""
    loop_positions = []
    
    for i, pos in enumerate(path):
        # Check if this position appears again within next N steps
        future = path[i+1:i+20]  # Look ahead 20 steps
        if pos in future:
            loop_positions.append(pos)
    
    return list(set(loop_positions))


def figure3_complexity_spectrum(
    csv_path: str = 'results/cbs_benchmark.csv',
    output_path: str = 'results/figures/fig3_complexity_spectrum.png'
) -> None:
    """
    Generate Figure 3: Complexity Spectrum.
    
    - Bar chart: A* vs CBS path lengths per dungeon
    - Bar chart: A* vs CBS states explored per dungeon
    - Show Confusion Ratio as line overlay
    """
    if not HAS_MATPLOTLIB:
        print("Skipping figure generation - matplotlib not available")
        return
    
    # Load CSV data
    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}. Run benchmarks first.")
        return
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Group by map_id
    astar_data = {}
    cbs_data = {}
    
    for row in rows:
        map_id = row['map_id']
        if row['solver'] == 'A*':
            astar_data[map_id] = {
                'path_length': int(row['path_length']),
                'states': int(row['states_explored']),
                'success': int(row['success'])
            }
        elif row['solver'] == 'CBS+':
            if map_id not in cbs_data:
                cbs_data[map_id] = {}
            persona = row.get('persona', 'balanced')
            cbs_data[map_id][persona] = {
                'path_length': int(row['path_length']),
                'states': int(row['states_explored']),
                'success': int(row['success']),
                'confusion_ratio': float(row.get('confusion_ratio', 1.0))
            }
    
    # Get common maps and sort
    common_maps = sorted(set(astar_data.keys()) & set(cbs_data.keys()))
    if not common_maps:
        print("No common maps found in CSV")
        return
    
    # Prepare data arrays
    n_maps = len(common_maps)
    x = np.arange(n_maps)
    width = 0.35
    
    astar_paths = [astar_data[m]['path_length'] for m in common_maps]
    astar_states = [astar_data[m]['states'] for m in common_maps]
    
    # Use first persona for CBS data
    first_persona = next(iter(cbs_data[common_maps[0]].keys()))
    cbs_paths = [cbs_data[m][first_persona]['path_length'] for m in common_maps]
    cbs_states = [cbs_data[m][first_persona]['states'] for m in common_maps]
    confusion_ratios = [cbs_data[m][first_persona]['confusion_ratio'] for m in common_maps]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Path lengths
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, astar_paths, width, label='A* (Optimal)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, cbs_paths, width, label=f'CBS+ ({first_persona})', color='#3498db', alpha=0.8)
    
    # Add confusion ratio line on secondary axis
    ax1_twin = ax1.twinx()
    # Filter out inf values for plotting
    valid_ratios = [r if r != float('inf') and not np.isnan(r) else 0 for r in confusion_ratios]
    line = ax1_twin.plot(x, valid_ratios, 'r-o', linewidth=2, markersize=6, 
                         label='Confusion Ratio', alpha=0.8)
    ax1_twin.set_ylabel('Confusion Ratio (CBS/A*)', color='red', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    max_ratio = max(valid_ratios) if valid_ratios and max(valid_ratios) > 0 else 5.0
    ax1_twin.set_ylim(0, max_ratio * 1.2)
    ax1_twin.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    
    ax1.set_ylabel('Path Length', fontsize=12)
    ax1.set_title('Path Length Comparison: A* vs CBS+', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Add value labels on bars
    for bar, val in zip(bars1, astar_paths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, cbs_paths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', va='bottom', fontsize=8)
    
    # Plot 2: States explored
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, astar_states, width, label='A* States', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width/2, cbs_states, width, label='CBS+ States', color='#3498db', alpha=0.8)
    
    ax2.set_ylabel('States Explored', fontsize=12)
    ax2.set_xlabel('Dungeon', fontsize=12)
    ax2.set_title('States Explored Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_maps, rotation=45, ha='right')
    ax2.legend(loc='upper left')
    
    # Use log scale if values vary widely
    if max(astar_states + cbs_states) > 10 * min(astar_states + cbs_states):
        ax2.set_yscale('log')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {output_path}")


def generate_all_figures(
    dungeon_num: int = 1,
    variant: int = 1,
    persona: str = 'balanced',
    max_steps: int = 2000,
    seed: int = 42,
    output_dir: str = 'results/figures'
) -> None:
    """Generate all figures for a given dungeon."""
    print(f"Generating figures for Dungeon {dungeon_num} v{variant} with {persona} persona...")
    
    # Run CBS+ to get visualization data
    try:
        grid, belief_map, path, metrics, astar_path = run_cbs_for_visualization(
            dungeon_num, variant, persona, max_steps, seed
        )
    except Exception as e:
        print(f"Error running CBS+: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Path length: {len(path)}, Success: {path[-1] == astar_path[-1] if astar_path else 'N/A'}")
    
    # Generate figures
    step = min(200, len(path))
    figure1_minds_eye(
        grid, belief_map, path[:step], step,
        f'{output_dir}/fig1_minds_eye_D{dungeon_num}_v{variant}.png'
    )
    
    figure2_frustration_heatmap(
        grid, path, metrics,
        f'{output_dir}/fig2_frustration_D{dungeon_num}_v{variant}.png'
    )
    
    # Figure 3 uses CSV data
    figure3_complexity_spectrum(
        'results/cbs_benchmark.csv',
        f'{output_dir}/fig3_complexity_spectrum.png'
    )
    
    print(f"\nAll figures saved to {output_dir}/")


def visualize(csv_path='results/cbs_benchmark.csv', out_dir='results/figures'):
    """Legacy compatibility function."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    figure3_complexity_spectrum(csv_path, f'{out_dir}/complexity_spectrum.png')


def main():
    parser = argparse.ArgumentParser(description='Visualize CBS+ results')
    parser.add_argument('--level', type=int, default=1,
                        help='Dungeon number (1-9)')
    parser.add_argument('--variant', type=int, default=1,
                        help='Dungeon variant (1 or 2)')
    parser.add_argument('--persona', type=str, default='balanced',
                        help='Agent persona')
    parser.add_argument('--max-steps', type=int, default=2000,
                        help='Maximum steps for CBS+')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                        help='Output directory')
    parser.add_argument('--csv', type=str, default='results/cbs_benchmark.csv',
                        help='CSV file for Figure 3')
    parser.add_argument('--all', action='store_true',
                        help='Generate all figures for levels 1-3')
    parser.add_argument('--fig3-only', action='store_true',
                        help='Only generate Figure 3 from CSV')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.fig3_only:
        figure3_complexity_spectrum(args.csv, f'{args.output_dir}/fig3_complexity_spectrum.png')
    elif args.all:
        for level in [1, 2, 3]:
            generate_all_figures(
                level, 1, args.persona, args.max_steps, args.seed, args.output_dir
            )
    else:
        generate_all_figures(
            args.level, args.variant, args.persona, 
            args.max_steps, args.seed, args.output_dir
        )


if __name__ == '__main__':
    main()
