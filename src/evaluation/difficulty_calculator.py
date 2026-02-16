"""
Weighted Difficulty Calculator for Dungeon Generation
======================================================

Decouples difficulty into cognitive and mechanical components:
1. Combat Score (Mechanical): Time-to-kill based on enemy stats
2. Navigation Complexity (Cognitive): Path tortuosity/maze-likeness
3. Resource Scarcity (Survival): Health availability vs damage pressure

Prevents genetic algorithm from converging to trivial "enemy spam" solutions
by rewarding structural complexity and resource management.

Defense Statement:
    "We utilize a multi-objective difficulty heuristic that decouples mechanical
    skill (combat) from cognitive load (navigation). The weighted formula
    (0.4*Combat + 0.4*Navigation + 0.2*Resource) prevents the genetic algorithm
    from exploiting local minima (enemy spam) by requiring balanced
    structural complexity."
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DifficultyComponents:
    """Breakdown of difficulty into sub-metrics."""
    combat_score: float
    navigation_complexity: float
    resource_scarcity: float
    overall_difficulty: float
    
    def to_dict(self) -> Dict:
        return {
            'combat': self.combat_score,
            'navigation': self.navigation_complexity,
            'resource': self.resource_scarcity,
            'overall': self.overall_difficulty
        }


class DifficultyCalculator:
    """
    Computes weighted difficulty using multi-objective formula.
    
    Formula:
        Difficulty = 0.4*Combat + 0.4*Navigation + 0.2*Resource
    
    Usage:
        calc = DifficultyCalculator(player_dps=10.0)
        difficulty = calc.compute(
            enemy_count=5,
            avg_enemy_hp=30,
            path_length=45,
            room_size=(11, 7),
            health_drops=1
        )
        print(f"Difficulty: {difficulty.overall_difficulty:.2f}")
        print(f"  Combat: {difficulty.combat_score:.2f}")
        print(f"  Navigation: {difficulty.navigation_complexity:.2f}")
    """
    
    def __init__(
        self,
        player_dps: float = 10.0,
        combat_weight: float = 0.4,
        navigation_weight: float = 0.4,
        resource_weight: float = 0.2,
    ):
        """
        Initialize calculator with game parameters.
        
        Args:
            player_dps: Player damage per second (default: 10 for Zelda)
            combat_weight: Weight for combat score (default: 0.4)
            navigation_weight: Weight for navigation complexity (default: 0.4)
            resource_weight: Weight for resource scarcity (default: 0.2)
        """
        self.player_dps = player_dps
        self.combat_weight = combat_weight
        self.navigation_weight = navigation_weight
        self.resource_weight = resource_weight
        
        # Normalization constants (tuned for Zelda-like dungeons)
        self.combat_norm = 30.0  # 6 enemies @ 30hp / 10dps = 18s → 0.6
        self.nav_base_tortuosity = 2.0  # Tortuosity of 2.0 = 50% penalty
        self.damage_per_enemy = 0.5  # Hearts of damage per enemy
    
    def compute(
        self,
        enemy_count: int,
        avg_enemy_hp: float,
        path_length: int,
        room_size: Tuple[int, int],
        health_drops: int,
        optimal_path_length: Optional[int] = None,
    ) -> DifficultyComponents:
        """
        Compute weighted difficulty for a room.
        
        Args:
            enemy_count: Number of enemies in room
            avg_enemy_hp: Average HP per enemy
            path_length: Actual shortest path length (tiles)
            room_size: (width, height) of room
            health_drops: Number of health pickups in room
            optimal_path_length: Pre-computed optimal path (if available)
            
        Returns:
            DifficultyComponents with all sub-metrics
        """
        combat = self._compute_combat_score(enemy_count, avg_enemy_hp)
        navigation = self._compute_navigation_complexity(
            path_length, room_size, optimal_path_length
        )
        resource = self._compute_resource_scarcity(enemy_count, health_drops)
        
        overall = (
            self.combat_weight * combat +
            self.navigation_weight * navigation +
            self.resource_weight * resource
        )
        
        logger.debug(
            f"Difficulty breakdown: Combat={combat:.2f}, Nav={navigation:.2f}, "
            f"Resource={resource:.2f}, Overall={overall:.2f}"
        )
        
        return DifficultyComponents(
            combat_score=combat,
            navigation_complexity=navigation,
            resource_scarcity=resource,
            overall_difficulty=overall
        )
    
    def _compute_combat_score(self, enemy_count: int, avg_enemy_hp: float) -> float:
        """
        Combat difficulty based on time to kill all enemies.
        
        Formula:
            Combat = (enemy_count * avg_hp) / player_dps / normalization
        
        Returns:
            Normalized combat score [0.0, 1.0]
        """
        if enemy_count == 0:
            return 0.0
        
        total_hp = enemy_count * avg_enemy_hp
        time_to_kill = total_hp / self.player_dps
        
        # Normalize: 18 seconds = 0.6 difficulty
        combat_score = time_to_kill / self.combat_norm
        
        return min(combat_score, 1.0)
    
    def _compute_navigation_complexity(
        self,
        path_length: int,
        room_size: Tuple[int, int],
        optimal_path: Optional[int] = None
    ) -> float:
        """
        Navigation complexity based on path tortuosity.
        
        Tortuosity = shortest_path / euclidean_distance
        - Tortuosity = 1.0: Straight line path
        - Tortuosity = 2.0: Path is 2x longer than straight (maze-like)
        - Tortuosity = 3.0: Highly convoluted maze
        
        Formula:
            Complexity = (tortuosity - 1.0) / base_tortuosity
        
        Returns:
            Normalized navigation complexity [0.0, 1.0]
        """
        width, height = room_size
        
        # Euclidean distance from entrance to exit (approximate)
        euclidean_distance = np.sqrt(width**2 + height**2) * 0.5
        euclidean_distance = max(euclidean_distance, 1.0)
        
        # Use provided optimal path or actual path length
        actual_path = optimal_path if optimal_path is not None else path_length
        
        # Tortuosity: how much longer is the path than straight line?
        tortuosity = actual_path / euclidean_distance
        
        # Normalize: tortuosity of 2.0 → 0.5, tortuosity of 3.0 → 1.0
        nav_complexity = (tortuosity - 1.0) / self.nav_base_tortuosity
        
        return max(min(nav_complexity, 1.0), 0.0)
    
    def _compute_resource_scarcity(
        self,
        enemy_count: int,
        health_drops: int
    ) -> float:
        """
        Resource scarcity based on health availability vs expected damage.
        
        Formula:
            Scarcity = 1.0 - min(health_drops / expected_damage, 1.0)
        
        Returns:
            Normalized scarcity [0.0, 1.0]
        """
        if enemy_count == 0:
            return 0.0
        
        expected_damage = enemy_count * self.damage_per_enemy
        
        if expected_damage > 0:
            health_ratio = health_drops / expected_damage
            scarcity = 1.0 - min(health_ratio, 1.0)
        else:
            scarcity = 0.0
        
        return scarcity
    
    def validate_difficulty_balance(
        self,
        target_difficulty: float,
        actual_components: DifficultyComponents,
        tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """
        Validate that difficulty is balanced across all components.
        
        Prevents degenerate solutions where one component dominates
        (e.g., "enemy spam" has high combat but low navigation).
        
        Args:
            target_difficulty: Desired overall difficulty
            actual_components: Computed difficulty components
            tolerance: Acceptable deviation from target
            
        Returns:
            Dict of validation checks
        """
        checks = {
            'overall_in_range': abs(actual_components.overall_difficulty - target_difficulty) <= tolerance,
            'combat_not_dominant': actual_components.combat_score < 0.8,  # No single enemy spam
            'navigation_present': actual_components.navigation_complexity > 0.1,  # Some maze structure
            'balanced': (
                abs(actual_components.combat_score - actual_components.navigation_complexity) < 0.3
            ),  # Components within 30% of each other
        }
        
        checks['valid'] = all(checks.values())
        
        return checks


def compute_dungeon_difficulty_curve(
    room_difficulties: List[DifficultyComponents]
) -> Dict[str, float]:
    """
    Analyze difficulty progression across an entire dungeon.
    
    Measures:
    - Progression smoothness: Is difficulty increasing gradually?
    - Peak placement: Is the hardest room near the end?
    - Variance: Are there sudden spikes?
    
    Args:
        room_difficulties: List of difficulty components per room
        
    Returns:
        Dict with progression metrics
    """
    if len(room_difficulties) < 2:
        return {'progression': 1.0, 'peak_placement': 1.0, 'variance': 0.0}
    
    difficulties = [d.overall_difficulty for d in room_difficulties]
    
    # Progression: How often does difficulty increase?
    increases = sum(
        1 for i in range(len(difficulties) - 1)
        if difficulties[i+1] >= difficulties[i]
    )
    progression = increases / (len(difficulties) - 1)
    
    # Peak placement: Is max difficulty in last third of dungeon?
    max_idx = difficulties.index(max(difficulties))
    peak_placement = max_idx / len(difficulties)
    
    # Variance: Smoothness of difficulty curve
    variance = np.var(difficulties)
    
    return {
        'progression': progression,
        'peak_placement': peak_placement,
        'variance': variance,
        'avg_difficulty': np.mean(difficulties),
    }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def difficulty_from_room_data(
    room_data: Dict,
    calculator: Optional[DifficultyCalculator] = None
) -> DifficultyComponents:
    """
    Compute difficulty from room data dictionary.
    
    Args:
        room_data: Dict with keys: 'enemies', 'avg_enemy_hp', 'path_length',
                   'room_width', 'room_height', 'health_pickups'
        calculator: Optional calculator instance (creates default if None)
        
    Returns:
        DifficultyComponents
    """
    if calculator is None:
        calculator = DifficultyCalculator()
    
    return calculator.compute(
        enemy_count=room_data.get('enemies', 0),
        avg_enemy_hp=room_data.get('avg_enemy_hp', 30),
        path_length=room_data.get('path_length', 20),
        room_size=(
            room_data.get('room_width', 11),
            room_data.get('room_height', 7)
        ),
        health_drops=room_data.get('health_pickups', 1),
    )


def apply_difficulty_constraint_to_genome(
    genome: Dict,
    target_difficulty: float,
    calculator: DifficultyCalculator
) -> Dict:
    """
    Adjust genome parameters to meet target difficulty.
    
    Prevents genetic algorithm from spamming enemies by enforcing
    balanced difficulty across all components.
    
    Args:
        genome: Dict with 'enemy_count', 'enemy_hp_mult', etc.
        target_difficulty: Desired difficulty [0.0, 1.0]
        calculator: Difficulty calculator instance
        
    Returns:
        Adjusted genome dict
    """
    # Limit enemy count based on target difficulty
    max_enemies = int(target_difficulty * 10) + 1
    genome['enemy_count'] = min(genome.get('enemy_count', 0), max_enemies)
    
    # Adjust enemy HP to compensate if count was capped
    if genome['enemy_count'] > 0:
        desired_combat = target_difficulty * 0.4  # Combat component
        current_combat = (genome['enemy_count'] * 30) / 10.0 / 30.0
        
        if current_combat < desired_combat:
            hp_mult = (desired_combat / current_combat) if current_combat > 0 else 1.0
            genome['enemy_hp_mult'] = min(hp_mult, 2.0)  # Cap at 2x HP
    
    return genome
