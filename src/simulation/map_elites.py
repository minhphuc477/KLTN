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
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Iterable

import networkx as nx

# Import from canonical source
from src.core.definitions import SEMANTIC_PALETTE, parse_edge_type_tokens

try:
    from src.data.zelda_core import DungeonSolver
except Exception:
    DungeonSolver = None

try:
    from src.simulation.validator import ZeldaValidator
except Exception:
    ZeldaValidator = None

try:
    # Reuse richer QD archive implementation as an auxiliary backend.
    from src.evaluation.map_elites import CVTEliteArchive
except Exception:
    CVTEliteArchive = None

logger = logging.getLogger(__name__)

def _clip01(value: float) -> float:
    """Clamp scalar to [0, 1] as float."""
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass
class BinEntry:
    dungeon: Any
    score: float
    metrics: Dict[str, float]


class MAPElitesEvaluator:
    def __init__(
        self,
        resolution: int = 20,
        tie_breaker: str = 'path_length',
        enable_advanced_archive: bool = True,
        descriptor_mode: str = 'hybrid',
        seed: Optional[int] = None,
    ):
        self.resolution = int(resolution)
        self.grid: Dict[Tuple[int, int], BinEntry] = {}
        self.tie_breaker = tie_breaker
        self.descriptor_mode = str(descriptor_mode).strip().lower()
        self.rng = np.random.default_rng(seed)

        # Optional CVT archive (from src.evaluation.map_elites) to keep
        # grid-based and research-grade QD tracking aligned.
        self._advanced_archive = None
        if enable_advanced_archive and CVTEliteArchive is not None:
            try:
                num_cells = max(64, self.resolution * self.resolution)
                self._advanced_archive = CVTEliteArchive(
                    num_cells=num_cells,
                    feature_dims=4,
                    feature_ranges=[(0.0, 1.0)] * 4,
                    num_cvt_samples=min(4000, num_cells * 40),
                    seed=seed,
                )
            except Exception as e:
                logger.warning("Advanced CVT archive unavailable, using legacy grid archive only: %s", e)
                self._advanced_archive = None

    def calculate_linearity(self, path_len: int, playable_area: int) -> float:
        # Proxy linearity from fraction of traversed space.
        raw = float(path_len) / max(1.0, float(playable_area))
        return float(np.clip(raw, 0.0, 1.0))

    def calculate_leniency(self, grid: np.ndarray) -> float:
        if SEMANTIC_PALETTE is None:
            # Fallback heuristic: treat higher values as "more enemies"
            enemies = int((grid == 7).sum())
            floors = int((grid == 1).sum())
        else:
            enemies = int((grid == SEMANTIC_PALETTE['ENEMY']).sum())
            floors = int((grid == SEMANTIC_PALETTE['FLOOR']).sum())
        return float(np.clip(1.0 - (enemies / max(1, floors)), 0.0, 1.0))

    def _discretize(self, lin: float, len_score: float) -> Tuple[int, int]:
        x = min(int(math.floor(np.clip(lin, 0.0, 1.0) * self.resolution)), self.resolution - 1)
        y = min(int(math.floor(np.clip(len_score, 0.0, 1.0) * self.resolution)), self.resolution - 1)
        return (x, y)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _iter_locked_edge_types() -> Iterable[str]:
        # Canonical progression-gating edge types used across the codebase.
        return (
            'locked',
            'key_locked',
            'boss_locked',
            'item_locked',
            'item_gate',
            'state_block',
            'switch',
            'switch_locked',
        )

    def _extract_graph_topology_metrics(self, mission_graph: Optional[nx.Graph]) -> Dict[str, float]:
        """
        Extract progression-aware topology descriptors from mission graph.
        """
        if mission_graph is None:
            return {
                'branching_factor': 0.0,
                'cycle_density': 0.0,
                'gating_density': 0.0,
                'topology_complexity': 0.0,
            }

        try:
            n_nodes = int(mission_graph.number_of_nodes())
            n_edges = int(mission_graph.number_of_edges())
            if n_nodes <= 1:
                return {
                    'branching_factor': 0.0,
                    'cycle_density': 0.0,
                    'gating_density': 0.0,
                    'topology_complexity': 0.0,
                }

            if isinstance(mission_graph, nx.DiGraph):
                branch_nodes = sum(
                    1 for node in mission_graph.nodes()
                    if mission_graph.out_degree(node) >= 2
                )
            else:
                branch_nodes = sum(
                    1 for node in mission_graph.nodes()
                    if mission_graph.degree(node) >= 3
                )
            branching_factor = self._clip01(branch_nodes / max(1, n_nodes))

            undirected = mission_graph.to_undirected()
            cycle_rank = max(0, int(undirected.number_of_edges()) - int(undirected.number_of_nodes()) + nx.number_connected_components(undirected))
            cycle_norm = max(1, n_nodes // 2)
            cycle_density = self._clip01(cycle_rank / cycle_norm)

            locked_types = set(self._iter_locked_edge_types())
            gated_edges = 0
            for _, _, data in mission_graph.edges(data=True):
                edge_tokens = parse_edge_type_tokens(
                    label=str(data.get('label', '') or ''),
                    edge_type=str(data.get('edge_type', data.get('type', '')) or ''),
                )
                if any(tok in locked_types for tok in edge_tokens):
                    gated_edges += 1
            gating_density = self._clip01(gated_edges / max(1, n_edges))

            topology_complexity = self._clip01(
                (0.45 * branching_factor) + (0.35 * cycle_density) + (0.20 * gating_density)
            )
            return {
                'branching_factor': branching_factor,
                'cycle_density': cycle_density,
                'gating_density': gating_density,
                'topology_complexity': topology_complexity,
            }
        except Exception:
            return {
                'branching_factor': 0.0,
                'cycle_density': 0.0,
                'gating_density': 0.0,
                'topology_complexity': 0.0,
            }

    def _build_behavior_descriptor(
        self,
        grid: np.ndarray,
        solver_result: Dict[str, Any],
        mission_graph: Optional[nx.Graph] = None,
    ) -> Tuple[Tuple[float, float, float, float], Dict[str, float]]:
        """
        Build progression-aware 4D descriptor and aligned metric dict.

        Dimensions:
        1) linearity
        2) leniency
        3) progression_complexity (gating pressure + backtracking + path pressure)
        4) topology_complexity (branching + cycles + gate density)
        """
        floor_id = int(SEMANTIC_PALETTE['FLOOR']) if SEMANTIC_PALETTE else 1
        enemy_id = int(SEMANTIC_PALETTE['ENEMY']) if SEMANTIC_PALETTE else 7
        locked_door_id = int(SEMANTIC_PALETTE['DOOR_LOCKED']) if SEMANTIC_PALETTE else 11
        key_id = int(SEMANTIC_PALETTE.get('KEY_SMALL', SEMANTIC_PALETTE.get('KEY', 8))) if SEMANTIC_PALETTE else 8

        playable_area = int((grid == floor_id).sum())
        path_len = int(solver_result.get('path_length', 0))
        linearity = self._clip01(solver_result.get('linearity', self.calculate_linearity(path_len, playable_area)))
        leniency = self._clip01(solver_result.get('leniency', self.calculate_leniency(grid)))

        total_tiles = max(1, int(grid.size))
        density = self._clip01(playable_area / total_tiles)

        enemy_count = int((grid == enemy_id).sum())
        lock_count = int(solver_result.get('lock_count', int((grid == locked_door_id).sum())))
        key_count = int(solver_result.get('key_count', int((grid == key_id).sum())))
        backtracking = self._clip01(float(solver_result.get('backtracking_score', 0.0)))

        enemy_pressure = min(1.0, enemy_count / max(1.0, playable_area * 0.2))
        lock_pressure = min(1.0, lock_count / max(1.0, float(max(1, key_count))))
        path_pressure = min(1.0, float(path_len) / max(1.0, math.sqrt(total_tiles) * 4.0))
        progression_complexity = self._clip01(
            (0.40 * lock_pressure) + (0.30 * backtracking) + (0.20 * path_pressure) + (0.10 * enemy_pressure)
        )

        topology_metrics = self._extract_graph_topology_metrics(mission_graph)
        topology_complexity = self._clip01(topology_metrics.get('topology_complexity', density))

        if self.descriptor_mode == 'legacy':
            # Preserve old behavior for ablations.
            topology_complexity = density

        features = (linearity, leniency, progression_complexity, topology_complexity)
        metrics = {
            'linearity': linearity,
            'leniency': leniency,
            'path_length': float(path_len),
            'progression_complexity': progression_complexity,
            'topology_complexity': topology_complexity,
            'density': density,
            'backtracking_score': backtracking,
            'key_count': float(key_count),
            'lock_count': float(lock_count),
            'enemy_count': float(enemy_count),
        }
        metrics.update(topology_metrics)
        return features, metrics

    def add_dungeon(
        self,
        dungeon: Any,
        grid: np.ndarray,
        solver_result: Dict[str, Any],
        mission_graph: Optional[nx.Graph] = None,
    ) -> None:
        # solver_result expected to contain 'solvable' and 'path_length' when solvable
        if not solver_result or not solver_result.get('solvable', False):
            return

        path_len = int(solver_result.get('path_length', 0))
        features, descriptor_metrics = self._build_behavior_descriptor(
            grid=grid,
            solver_result=solver_result,
            mission_graph=mission_graph,
        )
        lin = descriptor_metrics['linearity']
        len_score = descriptor_metrics['leniency']

        key = self._discretize(lin, len_score)
        score = float(
            solver_result.get(
                self.tie_breaker,
                solver_result.get('quality_score', path_len)
            )
        )

        entry = BinEntry(
            dungeon=dungeon,
            score=score,
            metrics=descriptor_metrics,
        )
        # Keep the better-scoring entry per tie-breaker
        existing = self.grid.get(key)
        if existing is None or score > existing.score:
            self.grid[key] = entry

        # Mirror into advanced CVT archive when available.
        if self._advanced_archive is not None:
            try:
                self._advanced_archive.add(
                    solution=dungeon,
                    fitness=score,
                    features=features,
                    metadata=descriptor_metrics,
                )
            except Exception as e:
                logger.debug("Advanced archive add failed: %s", e)

    def occupancy_grid(self) -> np.ndarray:
        arr = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        for (x, y) in self.grid.keys():
            arr[y, x] = 1
        return arr

    def occupied_bins(self) -> List[Tuple[int, int, BinEntry]]:
        return [(x, y, e) for (x, y), e in self.grid.items()]

    def clear(self) -> None:
        """Clear both legacy and auxiliary archives."""
        self.grid.clear()
        if self._advanced_archive is not None:
            self._advanced_archive.clear()

    def advanced_archive_stats(self) -> Optional[Dict[str, float]]:
        """Return auxiliary CVT archive stats if enabled."""
        if self._advanced_archive is None:
            return None
        stats = self._advanced_archive.get_stats()
        return {
            'coverage': float(stats.coverage),
            'qd_score': float(stats.total_fitness),
            'mean_fitness': float(stats.mean_fitness),
            'num_elites': float(stats.num_elites),
            'feature_diversity': float(stats.feature_diversity),
        }


def _get_grid_from_dungeon(dungeon) -> Optional[np.ndarray]:
    # Accept multiple dungeon representations
    if dungeon is None:
        return None
    if isinstance(dungeon, np.ndarray):
        return dungeon
    if hasattr(dungeon, 'global_grid'):
        return getattr(dungeon, 'global_grid')
    if hasattr(dungeon, 'layout'):
        return getattr(dungeon, 'layout')
    # Some adapters store as 'grid'
    if hasattr(dungeon, 'grid'):
        return getattr(dungeon, 'grid')
    return None


def _quick_solver_result_from_grid(grid: np.ndarray) -> Dict[str, Any]:
    """Deterministic fallback when no solver is available."""
    floor_id = int(SEMANTIC_PALETTE['FLOOR']) if SEMANTIC_PALETTE else 1
    playable = int((grid == floor_id).sum())
    path_length = int(max(1, np.sqrt(max(1, playable))))
    return {
        'solvable': playable > 0,
        'path_length': path_length,
        'quality_score': _clip01(path_length / max(1.0, playable)),
    }


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

        # Validate solvability with best available backend.
        solver_result: Dict[str, Any] = {}
        try:
            # Case 1: A proper dungeon object + graph-aware solver.
            if solver is not None and hasattr(solver, "solve") and not isinstance(d, np.ndarray):
                solver_result = solver.solve(d)
            # Case 2: Grid-only validation via ZeldaValidator.
            elif ZeldaValidator is not None:
                validator = ZeldaValidator()
                result = validator.validate_single(np.asarray(grid))
                path_length = int(result.path_length) if result.is_solvable else 0
                solver_result = {
                    'solvable': bool(result.is_solvable),
                    'path_length': path_length,
                    'backtracking_score': float(getattr(result, 'backtracking_score', 0.0)),
                    'quality_score': _clip01(
                        (0.6 * float(getattr(result, 'reachability', 0.0)))
                        + (0.4 * float(getattr(result, 'is_valid_syntax', False)))
                    ),
                }
            # Case 3: Coarse deterministic fallback.
            else:
                solver_result = _quick_solver_result_from_grid(np.asarray(grid))
        except Exception as e:
            logger.debug("Solver failed in run_map_elites_on_maps, using fallback: %s", e)
            solver_result = _quick_solver_result_from_grid(np.asarray(grid))

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


# ===== DIVERSITY METRICS FOR THESIS DEFENSE =====
# Addresses the concern: "How do you prove your system doesn't suffer from mode collapse?"

def calculate_diversity_score(evaluator: MAPElitesEvaluator) -> float:
    """
    Compute diversity score across archived solutions.
    
    Method: Average pairwise difference in 4D behavioral descriptors:
    (linearity, leniency, progression_complexity, topology_complexity).
    
    Target: >35% average difference indicates healthy diversity.
    
    Args:
        evaluator: MAPElitesEvaluator instance with populated archive
    
    Returns:
        Diversity score in [0, 1] where higher = more diverse
    """
    if len(evaluator.grid) < 2:
        return 0.0
    
    # Extract behavioral descriptors from archive entries.
    descriptors = []
    for entry in evaluator.grid.values():
        descriptors.append([
            float(entry.metrics.get('linearity', 0.0)),
            float(entry.metrics.get('leniency', 0.0)),
            float(entry.metrics.get('progression_complexity', 0.0)),
            float(entry.metrics.get('topology_complexity', 0.0)),
        ])
    
    descriptors = np.array(descriptors)
    
    # Calculate pairwise Euclidean distances in behavior space
    distances = []
    n = len(descriptors)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(descriptors[i] - descriptors[j])
            # Normalize by max possible distance in unit hypercube.
            normalized_dist = dist / np.sqrt(float(descriptors.shape[1]))
            distances.append(normalized_dist)
    
    # Average pairwise diversity
    diversity_score = float(np.mean(distances)) if distances else 0.0
    return diversity_score


def calculate_feature_coverage(evaluator: MAPElitesEvaluator) -> float:
    """
    Calculate percentage of feature space covered by archive.
    
    Args:
        evaluator: MAPElitesEvaluator instance
    
    Returns:
        Coverage percentage [0, 1]
    """
    resolution = evaluator.resolution
    total_bins = resolution * resolution
    filled_bins = len(evaluator.grid)
    return filled_bins / total_bins


def generate_diversity_report(
    evaluator: MAPElitesEvaluator, 
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive diversity analysis report.
    
    This addresses thesis defense questions about generalization and mode collapse.
    
    Args:
        evaluator: MAPElitesEvaluator with archived solutions
        output_path: Optional path to save JSON report
    
    Returns:
        Dictionary with diversity metrics
    """
    diversity_score = calculate_diversity_score(evaluator)
    feature_coverage = calculate_feature_coverage(evaluator)
    
    # Additional metrics
    num_solutions = len(evaluator.grid)
    
    # Performance spread (variance in scores)
    scores = [entry.score for entry in evaluator.grid.values()]
    score_std = float(np.std(scores)) if scores else 0.0
    score_range = (float(np.min(scores)), float(np.max(scores))) if scores else (0.0, 0.0)
    
    # Behavioral descriptor ranges
    linearities = [e.metrics.get('linearity', 0.0) for e in evaluator.grid.values()]
    leniencies = [e.metrics.get('leniency', 0.0) for e in evaluator.grid.values()]
    
    progression = [e.metrics.get('progression_complexity', 0.0) for e in evaluator.grid.values()]
    topology = [e.metrics.get('topology_complexity', 0.0) for e in evaluator.grid.values()]

    linearity_range = (float(np.min(linearities)), float(np.max(linearities))) if linearities else (0.0, 0.0)
    leniency_range = (float(np.min(leniencies)), float(np.max(leniencies))) if leniencies else (0.0, 0.0)
    progression_range = (float(np.min(progression)), float(np.max(progression))) if progression else (0.0, 0.0)
    topology_range = (float(np.min(topology)), float(np.max(topology))) if topology else (0.0, 0.0)
    
    report = {
        'diversity_score': float(diversity_score),
        'feature_coverage': float(feature_coverage),
        'num_solutions': int(num_solutions),
        'score_std': score_std,
        'score_range': score_range,
        'linearity_range': linearity_range,
        'leniency_range': leniency_range,
        'progression_complexity_range': progression_range,
        'topology_complexity_range': topology_range,
        'diversity_classification': _classify_diversity(diversity_score),
        'coverage_classification': _classify_coverage(feature_coverage)
    }
    
    # Print report
    print("\n" + "="*60)
    print("MAP-ELITES DIVERSITY REPORT")
    print("="*60)
    print(f"Diversity Score:        {diversity_score:.3f} (target: >0.35)")
    print(f"Feature Coverage:       {feature_coverage*100:.1f}%")
    print(f"Solutions in Archive:   {num_solutions}")
    print(f"Score Std Dev:          {score_std:.3f}")
    print(f"Score Range:            [{score_range[0]:.1f}, {score_range[1]:.1f}]")
    print(f"Linearity Range:        [{linearity_range[0]:.3f}, {linearity_range[1]:.3f}]")
    print(f"Leniency Range:         [{leniency_range[0]:.3f}, {leniency_range[1]:.3f}]")
    print(f"Progression Range:      [{progression_range[0]:.3f}, {progression_range[1]:.3f}]")
    print(f"Topology Range:         [{topology_range[0]:.3f}, {topology_range[1]:.3f}]")
    print(f"Diversity Class:        {report['diversity_classification']}")
    print(f"Coverage Class:         {report['coverage_classification']}")
    print("="*60)
    
    # Save to file
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved diversity report to {output_path}")
    
    return report


def _classify_diversity(score: float) -> str:
    """Classify diversity level."""
    if score >= 0.35:
        return "HEALTHY DIVERSITY ✓"
    elif score >= 0.20:
        return "MODERATE DIVERSITY"
    else:
        return "LOW DIVERSITY (Mode Collapse Risk) ✗"


def _classify_coverage(coverage: float) -> str:
    """Classify feature space coverage."""
    if coverage >= 0.15:
        return "GOOD COVERAGE ✓"
    elif coverage >= 0.05:
        return "MODERATE COVERAGE"
    else:
        return "LOW COVERAGE ✗"
