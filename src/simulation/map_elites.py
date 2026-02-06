"""MAP-Elites evaluator for Zelda dungeons.

Provides a minimal MAP-Elites evaluator that can operate on a list of
stitched dungeons (or dungeon-like objects exposing a 2D semantic grid).

API:
- MAPElitesEvaluator(resolution=20)
- run_map_elites_on_maps(maps, resolution=20, tie_breaker='path_length')

The evaluator is intentionally lightweight and dependency-tolerant for use
from the GUI (optional plotting via matplotlib when available).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

# Import from canonical source
from src.core.definitions import SEMANTIC_PALETTE

try:
    from src.data.zelda_core import DungeonSolver
except Exception:
    DungeonSolver = None


@dataclass
class BinEntry:
    dungeon: Any
    score: float
    metrics: Dict[str, float]


class MAPElitesEvaluator:
    def __init__(self, resolution: int = 20, tie_breaker: str = 'path_length'):
        self.resolution = int(resolution)
        self.grid: Dict[Tuple[int, int], BinEntry] = {}
        self.tie_breaker = tie_breaker

    def calculate_linearity(self, path_len: int, playable_area: int) -> float:
        # Avoid division by zero
        return float(path_len) / max(1.0, float(playable_area))

    def calculate_leniency(self, grid: np.ndarray) -> float:
        if SEMANTIC_PALETTE is None:
            # Fallback heuristic: treat higher values as "more enemies"
            enemies = int((grid == 7).sum())
            floors = int((grid == 1).sum())
        else:
            enemies = int((grid == SEMANTIC_PALETTE['ENEMY']).sum())
            floors = int((grid == SEMANTIC_PALETTE['FLOOR']).sum())
        return 1.0 - (enemies / max(1, floors))

    def _discretize(self, lin: float, len_score: float) -> Tuple[int, int]:
        x = min(int(math.floor(lin * self.resolution)), self.resolution - 1)
        y = min(int(math.floor(len_score * self.resolution)), self.resolution - 1)
        return (x, y)

    def add_dungeon(self, dungeon: Any, grid: np.ndarray, solver_result: Dict[str, Any]) -> None:
        # solver_result expected to contain 'solvable' and 'path_length' when solvable
        if not solver_result or not solver_result.get('solvable', False):
            return

        path_len = int(solver_result.get('path_length', 0))
        playable_area = int((grid == (SEMANTIC_PALETTE['FLOOR'] if SEMANTIC_PALETTE else 1)).sum())
        lin = self.calculate_linearity(path_len, playable_area)
        len_score = self.calculate_leniency(grid)

        key = self._discretize(lin, len_score)
        score = float(solver_result.get(self.tie_breaker, path_len))

        entry = BinEntry(dungeon=dungeon, score=score, metrics={'linearity': lin, 'leniency': len_score, 'path_length': path_len})
        # Keep the better-scoring entry per tie-breaker
        existing = self.grid.get(key)
        if existing is None or score > existing.score:
            self.grid[key] = entry

    def occupancy_grid(self) -> np.ndarray:
        arr = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        for (x, y) in self.grid.keys():
            arr[y, x] = 1
        return arr

    def occupied_bins(self) -> List[Tuple[int, int, BinEntry]]:
        return [(x, y, e) for (x, y), e in self.grid.items()]


def _get_grid_from_dungeon(dungeon) -> Optional[np.ndarray]:
    # Accept multiple dungeon representations
    if dungeon is None:
        return None
    if hasattr(dungeon, 'global_grid'):
        return getattr(dungeon, 'global_grid')
    if hasattr(dungeon, 'layout'):
        return getattr(dungeon, 'layout')
    # Some adapters store as 'grid'
    if hasattr(dungeon, 'grid'):
        return getattr(dungeon, 'grid')
    return None


def run_map_elites_on_maps(maps: List[Any], resolution: int = 20, tie_breaker: str = 'path_length', solver: Optional[Any] = None) -> Tuple[MAPElitesEvaluator, np.ndarray]:
    """Run MAP-Elites on a provided list of dungeon-like objects.

    Returns a tuple (evaluator, occupancy_grid) where occupancy_grid is a
    numpy array (resolution,resolution) with 1 for occupied bins.
    """
    if solver is None:
        solver = DungeonSolver() if DungeonSolver is not None else None

    evaluator = MAPElitesEvaluator(resolution=resolution, tie_breaker=tie_breaker)

    for d in maps:
        grid = _get_grid_from_dungeon(d)
        if grid is None:
            continue
        # Validate solvability
        solver_result = {}
        try:
            if solver is not None:
                solver_result = solver.solve(d)
            else:
                solver_result = {'solvable': True, 'path_length': int(max(1, grid.size // 10))}
        except Exception:
            solver_result = {'solvable': False}

        evaluator.add_dungeon(d, grid, solver_result)

    occ = evaluator.occupancy_grid()
    return evaluator, occ


# Optional plotting helper (uses matplotlib if available)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def plot_heatmap(occ_grid: np.ndarray, output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Plot a simple heatmap of occupancy grid. Saves to output_path if given and
    returns the image as an RGB numpy array when matplotlib is available.
    """
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(occ_grid, origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Linearity')
    ax.set_ylabel('Leniency')
    ax.set_xticks([])
    ax.set_yticks([])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    # Return numpy rgba buffer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img
