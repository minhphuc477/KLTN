"""
H-MOLQD Block VI: MAP-Elites Quality Diversity
===============================================

Quality Diversity Optimization for Zelda Dungeon Generation.

MAP-Elites maintains an archive of elite solutions across a feature space,
ensuring diversity while optimizing for quality. This enables generation
of dungeons with varied characteristics (linearity, leniency, etc.).

Mathematical Formulation:
-------------------------
Feature Space: F ∈ ℝ² (e.g., linearity × leniency)
Archive: A: F → (solution, fitness)

Update Rule:
    Given new solution x with features f(x) and fitness q(x):
    if f(x) ∈ A and q(x) > A[f(x)].fitness:
        A[f(x)] = (x, q(x))
    elif f(x) ∉ A:
        A[f(x)] = (x, q(x))

Metrics:
- Coverage: |{cells ∈ A : occupied}| / |F|
- QD-Score: Σ_{c ∈ A} fitness(c)
- Diversity: variance of features in archive

"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import random

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Elite:
    """An elite solution in the archive."""
    solution: Any                   # The dungeon (graph/grid data)
    fitness: float                  # Quality score
    features: Tuple[float, ...]     # Behavior features
    cell: Tuple[int, ...]           # Discretized cell coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Elite(fitness={self.fitness:.3f}, features={self.features}, cell={self.cell})"


@dataclass 
class ArchiveStats:
    """Statistics about the archive."""
    coverage: float              # Fraction of cells filled
    total_fitness: float         # Sum of all fitnesses (QD-score)
    mean_fitness: float          # Average fitness
    max_fitness: float           # Best fitness
    min_fitness: float           # Worst fitness
    num_elites: int              # Number of solutions
    feature_diversity: float     # Variance of features


# ============================================================================
# FEATURE EXTRACTORS
# ============================================================================

class FeatureExtractor:
    """
    Base class for extracting behavioral features from dungeons.
    
    Features are used to characterize dungeons in the MAP-Elites
    feature space, enabling diversity along meaningful dimensions.
    """
    
    def extract(self, dungeon: Any) -> Tuple[float, ...]:
        """Extract features from a dungeon."""
        raise NotImplementedError


class LinearityLeniencyExtractor(FeatureExtractor):
    """
    Extract Linearity and Leniency features.
    
    Linearity: How linear is the path from start to goal?
        L = 1 - (branches / max_branches)
        Low linearity = many branching paths
        High linearity = single path
    
    Leniency: How forgiving is the dungeon?
        E = available_keys / required_keys
        Low leniency = tight key management
        High leniency = many extra keys
    """
    
    def extract(self, graph: nx.DiGraph) -> Tuple[float, float]:
        """
        Extract linearity and leniency from dungeon graph.
        
        Args:
            graph: Dungeon connectivity graph
            
        Returns:
            (linearity, leniency) in [0, 1]
        """
        linearity = self._compute_linearity(graph)
        leniency = self._compute_leniency(graph)
        return (linearity, leniency)
    
    def _compute_linearity(self, graph: nx.DiGraph) -> float:
        """Compute linearity score."""
        if graph.number_of_nodes() < 2:
            return 1.0
        
        # Find start and goal
        start = None
        goal = None
        for node, data in graph.nodes(data=True):
            label = data.get('label', '')
            if 's' in label.split(','):
                start = node
            if 't' in label.split(','):
                goal = node
        
        if start is None or goal is None:
            # Fall back to first and last nodes
            nodes = list(graph.nodes())
            start = nodes[0]
            goal = nodes[-1]
        
        # Compute shortest path length
        try:
            undirected = graph.to_undirected()
            shortest_path = nx.shortest_path_length(undirected, start, goal)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0
        
        # Count nodes on shortest path vs total nodes
        # High linearity = most nodes are on the main path
        try:
            path = nx.shortest_path(undirected, start, goal)
            nodes_on_path = len(path)
        except:
            nodes_on_path = 1
        
        total_nodes = graph.number_of_nodes()
        
        # Linearity: fraction of nodes on main path
        linearity = nodes_on_path / total_nodes
        
        return min(1.0, linearity)
    
    def _compute_leniency(self, graph: nx.DiGraph) -> float:
        """Compute leniency score."""
        # Count keys and locked doors
        num_keys = 0
        num_locks = 0
        
        for node, data in graph.nodes(data=True):
            label = data.get('label', '')
            if 'k' in label.split(','):
                num_keys += 1
        
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('edge_type', data.get('label', ''))
            if edge_type in ('key_locked', 'k'):
                num_locks += 1
        
        # Leniency: keys available vs keys needed
        if num_locks == 0:
            return 1.0  # No locks = maximum leniency
        
        # Extra keys ratio
        leniency = num_keys / num_locks
        return min(1.0, leniency)


class DensityDifficultyExtractor(FeatureExtractor):
    """
    Extract Density and Difficulty features.
    
    Density: How densely connected is the dungeon?
        D = edges / max_possible_edges
        
    Difficulty: Estimated challenge level
        Based on path length, key requirements, enemies, etc.
    """
    
    def extract(self, graph: nx.DiGraph) -> Tuple[float, float]:
        """
        Extract density and difficulty.
        
        Args:
            graph: Dungeon connectivity graph
            
        Returns:
            (density, difficulty) in [0, 1]
        """
        density = self._compute_density(graph)
        difficulty = self._compute_difficulty(graph)
        return (density, difficulty)
    
    def _compute_density(self, graph: nx.DiGraph) -> float:
        """Compute graph density."""
        n = graph.number_of_nodes()
        if n < 2:
            return 0.0
        
        max_edges = n * (n - 1)  # Directed graph
        actual_edges = graph.number_of_edges()
        
        return actual_edges / max_edges
    
    def _compute_difficulty(self, graph: nx.DiGraph) -> float:
        """Estimate difficulty score."""
        factors = []
        
        # Lock count factor
        num_locks = sum(
            1 for _, _, d in graph.edges(data=True)
            if d.get('edge_type', d.get('label', '')) in ('key_locked', 'k', 'boss_locked', 'K')
        )
        lock_factor = min(1.0, num_locks / 5.0)
        factors.append(lock_factor)
        
        # Enemy count factor
        num_enemies = sum(
            1 for _, d in graph.nodes(data=True)
            if 'e' in d.get('label', '').split(',')
        )
        enemy_factor = min(1.0, num_enemies / 10.0)
        factors.append(enemy_factor)
        
        # Size factor
        size_factor = min(1.0, graph.number_of_nodes() / 15.0)
        factors.append(size_factor)
        
        return sum(factors) / len(factors) if factors else 0.5


# ============================================================================
# ELITE ARCHIVE
# ============================================================================

class EliteArchive:
    """
    Archive of elite solutions organized by feature cells.
    
    Discretizes the continuous feature space into a grid of cells,
    storing the best solution found for each cell.
    
    Args:
        feature_dims: Number of feature dimensions
        cells_per_dim: Number of cells per dimension
        feature_ranges: (min, max) range for each feature dimension
    """
    
    def __init__(
        self,
        feature_dims: int = 2,
        cells_per_dim: int = 10,
        feature_ranges: Optional[List[Tuple[float, float]]] = None,
    ):
        self.feature_dims = feature_dims
        self.cells_per_dim = cells_per_dim
        
        # Default ranges: [0, 1] for each dimension
        if feature_ranges is None:
            feature_ranges = [(0.0, 1.0) for _ in range(feature_dims)]
        self.feature_ranges = feature_ranges
        
        # Archive storage: cell -> Elite
        self.archive: Dict[Tuple[int, ...], Elite] = {}
        
        # Statistics
        self.total_evaluations = 0
        self.total_additions = 0
        self.total_replacements = 0
    
    def _discretize(self, features: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert continuous features to discrete cell coordinates."""
        cell = []
        for i, f in enumerate(features):
            f_min, f_max = self.feature_ranges[i]
            # Normalize to [0, 1]
            normalized = (f - f_min) / (f_max - f_min + 1e-8)
            # Discretize to [0, cells_per_dim - 1]
            cell_idx = int(np.clip(normalized * self.cells_per_dim, 0, self.cells_per_dim - 1))
            cell.append(cell_idx)
        return tuple(cell)
    
    def add(
        self,
        solution: Any,
        fitness: float,
        features: Tuple[float, ...],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Attempt to add a solution to the archive.
        
        Args:
            solution: The dungeon solution
            fitness: Quality score
            features: Behavioral features
            metadata: Optional additional data
            
        Returns:
            True if solution was added/replaced, False otherwise
        """
        self.total_evaluations += 1
        
        cell = self._discretize(features)
        
        elite = Elite(
            solution=solution,
            fitness=fitness,
            features=features,
            cell=cell,
            metadata=metadata or {},
        )
        
        # Check if cell is empty or new solution is better
        if cell not in self.archive:
            self.archive[cell] = elite
            self.total_additions += 1
            return True
        elif fitness > self.archive[cell].fitness:
            self.archive[cell] = elite
            self.total_replacements += 1
            return True
        
        return False
    
    def get(self, cell: Tuple[int, ...]) -> Optional[Elite]:
        """Get elite at a specific cell."""
        return self.archive.get(cell)
    
    def get_random_elite(self) -> Optional[Elite]:
        """Get a random elite from the archive."""
        if not self.archive:
            return None
        return random.choice(list(self.archive.values()))
    
    def get_all_elites(self) -> List[Elite]:
        """Get all elites in the archive."""
        return list(self.archive.values())
    
    def get_stats(self) -> ArchiveStats:
        """Compute archive statistics."""
        if not self.archive:
            return ArchiveStats(
                coverage=0.0,
                total_fitness=0.0,
                mean_fitness=0.0,
                max_fitness=0.0,
                min_fitness=0.0,
                num_elites=0,
                feature_diversity=0.0,
            )
        
        fitnesses = [e.fitness for e in self.archive.values()]
        features = np.array([e.features for e in self.archive.values()])
        
        total_cells = self.cells_per_dim ** self.feature_dims
        
        return ArchiveStats(
            coverage=len(self.archive) / total_cells,
            total_fitness=sum(fitnesses),
            mean_fitness=np.mean(fitnesses),
            max_fitness=max(fitnesses),
            min_fitness=min(fitnesses),
            num_elites=len(self.archive),
            feature_diversity=np.var(features).mean() if len(features) > 1 else 0.0,
        )
    
    def clear(self):
        """Clear the archive."""
        self.archive.clear()
        self.total_evaluations = 0
        self.total_additions = 0
        self.total_replacements = 0


# ============================================================================
# DIVERSITY METRICS
# ============================================================================

class DiversityMetrics:
    """
    Compute diversity metrics for generated dungeons.
    
    Metrics:
    - Coverage: Fraction of feature space covered
    - QD-Score: Sum of fitness across archive
    - Uniformity: How evenly distributed are solutions
    - Novelty: Average distance from nearest neighbors
    """
    
    def __init__(self, archive: EliteArchive):
        self.archive = archive
    
    def coverage(self) -> float:
        """Compute coverage metric."""
        stats = self.archive.get_stats()
        return stats.coverage
    
    def qd_score(self) -> float:
        """Compute QD-Score (quality-diversity score)."""
        stats = self.archive.get_stats()
        return stats.total_fitness
    
    def uniformity(self) -> float:
        """
        Compute uniformity of distribution in feature space.
        
        Higher = more evenly spread across cells.
        """
        elites = self.archive.get_all_elites()
        if len(elites) < 2:
            return 1.0
        
        # Compute pairwise distances in feature space
        features = np.array([e.features for e in elites])
        
        # Mean pairwise distance
        distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Uniformity: coefficient of variation (lower = more uniform)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        cv = std_dist / (mean_dist + 1e-8)
        
        # Invert so higher = better
        return 1.0 / (1.0 + cv)
    
    def novelty(self, solution_features: Tuple[float, ...], k: int = 5) -> float:
        """
        Compute novelty score for a solution.
        
        Novelty = average distance to k-nearest neighbors in archive.
        
        Args:
            solution_features: Features of solution to evaluate
            k: Number of nearest neighbors
            
        Returns:
            Novelty score
        """
        elites = self.archive.get_all_elites()
        if not elites:
            return 1.0
        
        features = np.array([e.features for e in elites])
        query = np.array(solution_features)
        
        # Compute distances to all elites
        distances = np.linalg.norm(features - query, axis=1)
        
        # Average distance to k nearest
        k = min(k, len(distances))
        nearest_k = np.partition(distances, k-1)[:k]
        
        return np.mean(nearest_k)
    
    def summary(self) -> Dict[str, float]:
        """Get summary of all diversity metrics."""
        return {
            'coverage': self.coverage(),
            'qd_score': self.qd_score(),
            'uniformity': self.uniformity(),
            'num_elites': len(self.archive.archive),
        }


# ============================================================================
# MAP-ELITES ALGORITHM
# ============================================================================

class MAPElites:
    """
    MAP-Elites Quality Diversity Optimization for H-MOLQD Block VI.
    
    Maintains an archive of elite dungeon solutions across a feature space,
    enabling diverse, high-quality generation.
    
    Args:
        feature_extractor: Extracts features from dungeons
        fitness_fn: Computes fitness/quality score
        cells_per_dim: Archive resolution per dimension
        feature_ranges: Range for each feature dimension
    
    Usage:
        map_elites = MAPElites(
            feature_extractor=LinearityLeniencyExtractor(),
            fitness_fn=lambda g: validator.validate(g).is_solvable * 1.0,
        )
        
        # Add generated dungeons
        for dungeon in generated_dungeons:
            map_elites.add(dungeon)
        
        # Get diverse set
        elites = map_elites.get_diverse_set(n=10)
        
        # Analyze diversity
        metrics = map_elites.get_diversity_metrics()
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        fitness_fn: Callable[[Any], float],
        cells_per_dim: int = 10,
        feature_ranges: Optional[List[Tuple[float, float]]] = None,
        feature_dims: int = 2,
    ):
        self.feature_extractor = feature_extractor
        self.fitness_fn = fitness_fn
        
        self.archive = EliteArchive(
            feature_dims=feature_dims,
            cells_per_dim=cells_per_dim,
            feature_ranges=feature_ranges,
        )
        
        self.diversity_metrics = DiversityMetrics(self.archive)
    
    def add(
        self,
        dungeon: Any,
        precomputed_fitness: Optional[float] = None,
        precomputed_features: Optional[Tuple[float, ...]] = None,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, float, Tuple[float, ...]]:
        """
        Add a dungeon to the archive.
        
        Args:
            dungeon: Dungeon solution (graph or other representation)
            precomputed_fitness: Optional pre-computed fitness
            precomputed_features: Optional pre-computed features
            metadata: Optional metadata
            
        Returns:
            (was_added, fitness, features)
        """
        # Extract features
        features = precomputed_features
        if features is None:
            features = self.feature_extractor.extract(dungeon)
        
        # Compute fitness
        fitness = precomputed_fitness
        if fitness is None:
            fitness = self.fitness_fn(dungeon)
        
        # Try to add to archive
        was_added = self.archive.add(
            solution=dungeon,
            fitness=fitness,
            features=features,
            metadata=metadata,
        )
        
        return was_added, fitness, features
    
    def add_batch(
        self,
        dungeons: List[Any],
    ) -> Tuple[int, int]:
        """
        Add multiple dungeons to archive.
        
        Returns:
            (num_added, num_total)
        """
        num_added = 0
        for dungeon in dungeons:
            was_added, _, _ = self.add(dungeon)
            if was_added:
                num_added += 1
        return num_added, len(dungeons)
    
    def get_diverse_set(self, n: int = 10) -> List[Elite]:
        """
        Get a diverse set of elites from the archive.
        
        Selects elites to maximize coverage of feature space.
        
        Args:
            n: Number of elites to return
            
        Returns:
            List of diverse elites
        """
        elites = self.archive.get_all_elites()
        if len(elites) <= n:
            return elites
        
        # Greedy selection maximizing pairwise distance
        selected = []
        remaining = elites.copy()
        
        # Start with random elite
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        while len(selected) < n and remaining:
            # Find elite with maximum minimum distance to selected
            best_elite = None
            best_min_dist = -1
            
            for elite in remaining:
                min_dist = min(
                    np.linalg.norm(
                        np.array(elite.features) - np.array(s.features)
                    )
                    for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_elite = elite
            
            if best_elite:
                selected.append(best_elite)
                remaining.remove(best_elite)
            else:
                break
        
        return selected
    
    def get_best(self, n: int = 1) -> List[Elite]:
        """Get top-n elites by fitness."""
        elites = self.archive.get_all_elites()
        elites.sort(key=lambda e: e.fitness, reverse=True)
        return elites[:n]
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Get diversity metrics summary."""
        return self.diversity_metrics.summary()
    
    def get_archive_stats(self) -> ArchiveStats:
        """Get archive statistics."""
        return self.archive.get_stats()
    
    def save_archive(self, filepath: str):
        """Save archive to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'archive': self.archive.archive,
                'stats': {
                    'total_evaluations': self.archive.total_evaluations,
                    'total_additions': self.archive.total_additions,
                    'total_replacements': self.archive.total_replacements,
                }
            }, f)
        logger.info(f"Saved archive to {filepath}")
    
    def load_archive(self, filepath: str):
        """Load archive from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.archive.archive = data['archive']
        stats = data.get('stats', {})
        self.archive.total_evaluations = stats.get('total_evaluations', 0)
        self.archive.total_additions = stats.get('total_additions', 0)
        self.archive.total_replacements = stats.get('total_replacements', 0)
        logger.info(f"Loaded archive from {filepath}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_map_elites(
    feature_type: str = 'linearity_leniency',
    fitness_fn: Optional[Callable] = None,
    cells_per_dim: int = 10,
) -> MAPElites:
    """
    Create a MAP-Elites instance.
    
    Args:
        feature_type: Type of features ('linearity_leniency' or 'density_difficulty')
        fitness_fn: Fitness function (defaults to solvability)
        cells_per_dim: Archive resolution
        
    Returns:
        MAPElites instance
    """
    # Select feature extractor
    if feature_type == 'linearity_leniency':
        extractor = LinearityLeniencyExtractor()
    elif feature_type == 'density_difficulty':
        extractor = DensityDifficultyExtractor()
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Default fitness: solvability check
    if fitness_fn is None:
        from src.evaluation.validator import ExternalValidator
        validator = ExternalValidator()
        fitness_fn = lambda g: float(validator.validate(g).is_solvable)
    
    return MAPElites(
        feature_extractor=extractor,
        fitness_fn=fitness_fn,
        cells_per_dim=cells_per_dim,
    )
