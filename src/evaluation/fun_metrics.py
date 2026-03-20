"""
Feature 4: Fun Metrics
======================
Quantify player experience: frustration, explorability, flow, pacing.

Problem:
    Current metrics (solvability, difficulty) don't capture "fun".
    Need objective measures of player engagement and experience quality.

Solution:
    - Frustration Score: Backtracking, dead ends, unclear goals
    - Explorability: Discovery potential, secret rooms, rewards
    - Flow Score: Challenge-skill balance (Csikszentmihalyi)
    - Pacing Score: Tension curve alignment
    
Integration Point: MAP-Elites evaluation, after full dungeon generation
"""

import numpy as np
import networkx as nx
from typing import Dict,Tuple, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FrustrationMetrics:
    """Metrics quantifying player frustration."""
    backtracking_ratio: float  # 0.0 (none) to 1.0 (extreme)
    dead_end_count: int
    unclear_goal_score: float  # 0.0 (clear) to 1.0 (confusing)
    empty_room_ratio: float  # Rooms with no content
    total_frustration: float  # Weighted combination


@dataclass
class ExplorabilityMetrics:
    """Metrics quantifying exploration potential."""
    optional_room_ratio: float  # Non-critical-path rooms
    secret_count: int  # Hidden rooms/passages
    reward_density: float  # Rewards per room
    discovery_potential: float  # 0.0 (linear) to 1.0 (highly explorable)


@dataclass
class FlowMetrics:
    """Metrics for flow state (challenge-skill balance)."""
    difficulty_progression: float  # How smoothly difficulty increases
    skill_utilization: float  # Variety of mechanics used
    challenge_balance: float  # Not too hard, not too easy
    flow_score: float  # Overall flow quality


@dataclass
class PacingMetrics:
    """Metrics for tension curve and pacing."""
    tension_variance: float  # How much tension fluctuates
    peak_placement: float  # 0.0 (early) to 1.0 (late)
    rest_areas: int  # Safe rooms for recovery
    pacing_score: float  # Alignment with target curve


@dataclass
class FunMetrics:
    """Comprehensive fun/engagement metrics."""
    frustration: FrustrationMetrics
    explorability: ExplorabilityMetrics
    flow: FlowMetrics
    pacing: PacingMetrics
    overall_fun_score: float  # 0.0 (terrible) to 1.0 (excellent)


# ============================================================================
# FRUSTRATION ANALYZER
# ============================================================================

class FrustrationAnalyzer:
    """
    Quantifies sources of player frustration.
    
    Frustration Sources:
    1. Backtracking: Re-visiting rooms due to keys/items
    2. Dead Ends: Rooms with no reward/progress
    3. Unclear Goals: Opaque next steps
    4. Empty Rooms: Lack of content/challenge
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'backtracking': 0.4,
            'dead_ends': 0.2,
            'unclear_goals': 0.3,
            'empty_rooms': 0.1
        }
    
    def analyze(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict]
    ) -> FrustrationMetrics:
        """
        Compute frustration metrics for dungeon.
        
        Args:
            mission_graph: Mission graph topology
            solution_path: Optimal solution path (room IDs)
            room_contents: {room_id: {'enemies': N, 'keys': N, ...}}
        
        Returns:
            FrustrationMetrics
        """
        # 1. Backtracking analysis
        backtracking_ratio = self._compute_backtracking(mission_graph, solution_path)
        
        # 2. Dead end detection
        dead_end_count = self._count_dead_ends(mission_graph, room_contents)
        
        # 3. Goal clarity
        unclear_goal_score = self._compute_goal_clarity(mission_graph, room_contents)
        
        # 4. Empty rooms
        empty_room_ratio = self._compute_empty_room_ratio(room_contents)
        
        # Total frustration (weighted sum)
        total_frustration = (
            self.weights['backtracking'] * backtracking_ratio +
            self.weights['dead_ends'] * (dead_end_count / max(len(mission_graph.nodes), 1)) +
            self.weights['unclear_goals'] * unclear_goal_score +
            self.weights['empty_rooms'] * empty_room_ratio
        )
        
        return FrustrationMetrics(
            backtracking_ratio=backtracking_ratio,
            dead_end_count=dead_end_count,
            unclear_goal_score=unclear_goal_score,
            empty_room_ratio=empty_room_ratio,
            total_frustration=total_frustration
        )
    
    def _compute_backtracking(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int]
    ) -> float:
        """
        Compute backtracking ratio: revisited rooms / total visits.
        
        High backtracking = frustrating (player must retrace steps often)
        """
        if not solution_path:
            return 0.0
        
        visited_rooms = set()
        revisits = 0
        
        for room in solution_path:
            if room in visited_rooms:
                revisits += 1
            visited_rooms.add(room)
        
        total_visits = len(solution_path)
        unique_rooms = len(visited_rooms)
        
        # Ratio of revisits to total visits
        backtracking_ratio = revisits / max(total_visits, 1)
        
        return backtracking_ratio
    
    def _count_dead_ends(
        self,
        mission_graph: nx.Graph,
        room_contents: Dict[int, Dict]
    ) -> int:
        """
        Count dead ends: leaf nodes with no rewards/progress.
        """
        dead_ends = 0
        
        for node in mission_graph.nodes():
            # Check if leaf node (only 1 neighbor, or 0 if isolated)
            if mission_graph.degree(node) <= 1:
                # Check if room has any content
                content = room_contents.get(node, {})
                has_content = any([
                    content.get('keys', 0) > 0,
                    content.get('treasures', 0) > 0,
                    content.get('boss', False),
                    content.get('goal', False)
                ])
                
                if not has_content:
                    dead_ends += 1
        
        return dead_ends
    
    def _compute_goal_clarity(
        self,
        mission_graph: nx.Graph,
        room_contents: Dict[int, Dict]
    ) -> float:
        """
        Measure goal clarity: how obvious is the next objective?
        
        Clear goals: Linear path, visible rewards
        Unclear goals: Complex branching, hidden objectives
        """
        # Simple heuristic: branching factor and goal visibility
        avg_degree = sum(dict(mission_graph.degree()).values()) / max(len(mission_graph.nodes), 1)
        
        # High branching = more confusing
        branching_confusion = min(avg_degree / 4.0, 1.0)  # Normalize to [0, 1]
        
        # Check if goal room is obvious (has triforce/boss)
        goal_rooms = [
            node for node, content in room_contents.items()
            if content.get('goal', False) or content.get('boss', False)
        ]
        
        goal_visibility = 1.0 if goal_rooms else 0.5  # Penalize if no clear goal
        
        unclear_score = branching_confusion * (1.0 - goal_visibility)
        
        return unclear_score
    
    def _compute_empty_room_ratio(self, room_contents: Dict[int, Dict]) -> float:
        """Ratio of rooms with no entities/items."""
        if not room_contents:
            return 0.0
        
        empty_count = 0
        for content in room_contents.values():
            has_content = any([
                content.get('enemies', 0) > 0,
                content.get('keys', 0) > 0,
                content.get('treasures', 0) > 0,
                content.get('puzzles', 0) > 0
            ])
            if not has_content:
                empty_count += 1
        
        return empty_count / len(room_contents)


# ============================================================================
# EXPLORABILITY ANALYZER
# ============================================================================

class ExplorabilityAnalyzer:
    """
    Quantifies exploration potential and discovery richness.
    
    Explorability Factors:
    1. Optional Content: Non-critical rooms with rewards
    2. Secrets: Hidden rooms, passages
    3. Reward Density: Items/treasures per room
    4. Branching: Multiple paths to explore
    """
    
    def analyze(
        self,
        mission_graph: nx.Graph,
        critical_path: Set[int],
        room_contents: Dict[int, Dict]
    ) -> ExplorabilityMetrics:
        """
        Compute explorability metrics for dungeon.
        
        Args:
            mission_graph: Mission graph topology
            critical_path: Set of room IDs on critical path
            room_contents: Room content data
        
        Returns:
            ExplorabilityMetrics
        """
        total_rooms = len(mission_graph.nodes)
        
        # 1. Optional rooms (not on critical path)
        optional_rooms = total_rooms - len(critical_path)
        optional_ratio = optional_rooms / max(total_rooms, 1)
        
        # 2. Secret count (hidden edges, soft-locked doors)
        secret_count = self._count_secrets(mission_graph)
        
        # 3. Reward density
        reward_density = self._compute_reward_density(room_contents)
        
        # 4. Discovery potential (combination of above)
        discovery_potential = (
            0.4 * optional_ratio +
            0.3 * min(secret_count / max(total_rooms, 1), 1.0) +
            0.3 * reward_density
        )
        
        return ExplorabilityMetrics(
            optional_room_ratio=optional_ratio,
            secret_count=secret_count,
            reward_density=reward_density,
            discovery_potential=discovery_potential
        )
    
    def _count_secrets(self, mission_graph: nx.Graph) -> int:
        """Count hidden elements (soft-locked doors, hidden edges)."""
        secret_count = 0
        
        for u, v, data in mission_graph.edges(data=True):
            edge_type = data.get('type', '')
            if edge_type in ['soft_locked', 'hidden', 'secret']:
                secret_count += 1
        
        return secret_count
    
    def _compute_reward_density(self, room_contents: Dict[int, Dict]) -> float:
        """Average rewards per room."""
        if not room_contents:
            return 0.0
        
        total_rewards = sum(
            content.get('keys', 0) + content.get('treasures', 0) + content.get('items', 0)
            for content in room_contents.values()
        )
        
        reward_density = total_rewards / len(room_contents)
        
        # Normalize to [0, 1] (assume max 3 rewards/room is high)
        return min(reward_density / 3.0, 1.0)


# ============================================================================
# FLOW ANALYZER
# ============================================================================

class FlowAnalyzer:
    """
    Measures flow state quality (Csikszentmihalyi's flow theory).
    
    Flow = optimal challenge-skill balance
    - Too easy → boredom
    - Too hard → anxiety
    - Just right → flow (engagement)
    """
    
    def analyze(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict]
    ) -> FlowMetrics:
        """Compute flow metrics."""
        # 1. Difficulty progression
        difficulty_progression = self._compute_difficulty_progression(solution_path, room_contents)
        
        # 2. Skill utilization (variety of mechanics)
        skill_utilization = self._compute_skill_utilization(room_contents)
        
        # 3. Challenge balance
        challenge_balance = self._compute_challenge_balance(room_contents)
        
        # Overall flow score
        flow_score = (
            0.4 * difficulty_progression +
            0.3 * skill_utilization +
            0.3 * challenge_balance
        )
        
        return FlowMetrics(
            difficulty_progression=difficulty_progression,
            skill_utilization=skill_utilization,
            challenge_balance=challenge_balance,
            flow_score=flow_score
        )
    
    def _compute_difficulty_progression(
        self,
        solution_path: List[int],
        room_contents: Dict[int, Dict]
    ) -> float:
        """Measure smoothness of difficulty increase using weighted heuristic.
        
        NEW: Decouples difficulty into:
        - Combat Score (0.4 weight): Enemy density * HP / player DPS
        - Navigation Complexity (0.4 weight): Path tortuosity (optimal/euclidean)
        - Resource Scarcity (0.2 weight): Health availability vs damage taken
        
        Prevents the 'Enemy Spam' local minimum by considering maze complexity
        and resource balance, not just raw enemy count.
        """
        if len(solution_path) < 2:
            return 1.0
        
        difficulties = []
        for room in solution_path:
            content = room_contents.get(room, {})
            difficulty = self._calculate_weighted_difficulty(content)
            difficulties.append(difficulty)
        
        # Check if difficulty generally increases
        increases = sum(
            1 for i in range(len(difficulties)-1)
            if difficulties[i+1] >= difficulties[i]
        )
        
        progression_score = increases / max(len(difficulties) - 1, 1)
        
        return progression_score
    
    def _calculate_weighted_difficulty(self, room_content: Dict) -> float:
        """Calculate difficulty using weighted multi-objective formula.
        
        Formula:
            Difficulty = 0.4*Combat + 0.4*Navigation + 0.2*Resource
        
        Where:
            Combat = (enemy_count * avg_hp) / player_dps
            Navigation = shortest_path / euclidean_distance (tortuosity)
            Resource = 1.0 - (health_drops / expected_damage)
        
        Returns:
            Normalized difficulty score [0.0, 1.0+]
        """
        # Combat Score: Time-to-kill all enemies
        enemy_count = room_content.get('enemies', 0)
        avg_enemy_hp = room_content.get('avg_enemy_hp', 30)  # Default Zelda enemy HP
        player_dps = 10.0  # Zelda player damage per second (base sword)
        
        if enemy_count > 0:
            combat_score = (enemy_count * avg_enemy_hp) / player_dps
            # Normalize: 6 enemies @ 30hp = 180hp / 10dps = 18s → ~0.6 normalized
            combat_score = min(combat_score / 30.0, 1.0)
        else:
            combat_score = 0.0
        
        # Navigation Complexity: Path tortuosity
        shortest_path_tiles = room_content.get('path_length', 20)  # Tiles in optimal path
        room_width = room_content.get('room_width', 11)
        room_height = room_content.get('room_height', 7)
        
        # Euclidean distance from room entrance to exit (approximate)
        euclidean_distance = np.sqrt(room_width**2 + room_height**2) * 0.5
        euclidean_distance = max(euclidean_distance, 1.0)  # Prevent division by zero
        
        nav_complexity = shortest_path_tiles / euclidean_distance
        # Tortuosity > 1.0: Path is longer than straight line (maze-like)
        # Normalize: tortuosity of 2.0 (double length) = 0.5, 3.0 = 0.75
        nav_complexity = min((nav_complexity - 1.0) / 2.0, 1.0)
        nav_complexity = max(nav_complexity, 0.0)
        
        # Resource Scarcity: Health availability vs damage
        health_drops = room_content.get('health_pickups', 1)  # Hearts in room
        expected_damage = enemy_count * 0.5  # Avg damage per enemy (hearts)
        
        if expected_damage > 0:
            resource_scarcity = 1.0 - min(health_drops / expected_damage, 1.0)
        else:
            resource_scarcity = 0.0
        
        # Weighted combination
        difficulty = (
            0.4 * combat_score +
            0.4 * nav_complexity +
            0.2 * resource_scarcity
        )
        
        return difficulty
    
    def _compute_skill_utilization(self, room_contents: Dict[int, Dict]) -> float:
        """Measure variety of game mechanics used."""
        mechanic_types = set()
        
        for content in room_contents.values():
            if content.get('enemies', 0) > 0:
                mechanic_types.add('combat')
            if content.get('puzzles', 0) > 0:
                mechanic_types.add('puzzle')
            if content.get('keys', 0) > 0:
                mechanic_types.add('key_hunt')
            if content.get('boss', False):
                mechanic_types.add('boss')
        
        # Normalize: 4 mechanic types = 1.0
        return min(len(mechanic_types) / 4.0, 1.0)
    
    def _compute_challenge_balance(self, room_contents: Dict[int, Dict]) -> float:
        """Check if challenge level is balanced (not too easy/hard)."""
        difficulties = []
        
        for content in room_contents.values():
            difficulty = content.get('enemies', 0) * 0.5 + content.get('puzzles', 0) * 0.5
            difficulties.append(difficulty)
        
        if not difficulties:
            return 0.5
        
        avg_difficulty = np.mean(difficulties)
        
        # Ideal difficulty: around 2-3 enemies/puzzles per room
        ideal = 2.5
        balance = 1.0 - min(abs(avg_difficulty - ideal) / ideal, 1.0)
        
        return balance


# ============================================================================
# PACING ANALYZER
# ============================================================================

class PacingAnalyzer:
    """
    Estimates pacing quality from a tension curve over the solution path.

    The analyzer favors:
    - A noticeable build-up toward late-game climax
    - Some low-tension recovery beats (rest areas)
    - Non-flat but not chaotic tension fluctuations
    """

    def __init__(self, target_peak_position: float = 0.75):
        # Late-game peak is a common dungeon pacing target.
        self.target_peak_position = float(np.clip(target_peak_position, 0.0, 1.0))

    def analyze(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
    ) -> PacingMetrics:
        """Compute pacing metrics from the path-level tension signal."""
        del mission_graph  # Reserved for future graph-level pacing features.

        tension_curve = self._compute_tension_curve(solution_path, room_contents)
        if tension_curve.size == 0:
            return PacingMetrics(
                tension_variance=0.0,
                peak_placement=self.target_peak_position,
                rest_areas=0,
                pacing_score=0.5,
            )

        if tension_curve.size == 1:
            return PacingMetrics(
                tension_variance=0.0,
                peak_placement=1.0,
                rest_areas=int(tension_curve[0] < 0.35),
                pacing_score=float(0.6 + 0.4 * tension_curve[0]),
            )

        # Variance of the first derivative captures pacing swings.
        first_diff = np.diff(tension_curve)
        tension_variance = float(np.var(first_diff))

        # Peak should usually appear in later sections of dungeon flow.
        peak_idx = int(np.argmax(tension_curve))
        peak_placement = float(peak_idx / max(1, tension_curve.size - 1))

        rest_areas = int(self._count_rest_areas(tension_curve))

        # Target curve: gradual rise, mild mid-run dip, late climax.
        target_curve = self._target_curve(tension_curve.size)
        rmse = float(np.sqrt(np.mean((tension_curve - target_curve) ** 2)))
        curve_alignment_score = float(np.clip(1.0 - rmse, 0.0, 1.0))

        peak_score = float(np.clip(1.0 - abs(peak_placement - self.target_peak_position), 0.0, 1.0))

        # Encourage moderate variation (not flat, not overly noisy).
        target_variance = 0.02
        variance_score = float(np.exp(-((tension_variance - target_variance) ** 2) / (2.0 * target_variance**2)))

        rest_ratio = rest_areas / max(1, tension_curve.size)
        if rest_ratio < 0.1:
            rest_score = rest_ratio / 0.1
        elif rest_ratio <= 0.35:
            rest_score = 1.0
        else:
            rest_score = max(0.0, 1.0 - (rest_ratio - 0.35) / 0.35)

        pacing_score = (
            0.45 * curve_alignment_score +
            0.25 * peak_score +
            0.15 * variance_score +
            0.15 * rest_score
        )
        pacing_score = float(np.clip(pacing_score, 0.0, 1.0))

        return PacingMetrics(
            tension_variance=tension_variance,
            peak_placement=peak_placement,
            rest_areas=rest_areas,
            pacing_score=pacing_score,
        )

    def _compute_tension_curve(
        self,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
    ) -> np.ndarray:
        """Build and smooth a normalized tension curve from room content."""
        if not solution_path:
            return np.array([], dtype=np.float32)

        tensions: List[float] = []
        for room_id in solution_path:
            content = room_contents.get(room_id, {})

            # Challenge contributors
            challenge = (
                0.45 * float(content.get('enemies', 0)) +
                0.35 * float(content.get('puzzles', 0)) +
                0.40 * float(content.get('locks', 0)) +
                (1.50 if content.get('boss', False) else 0.0)
            )

            # Recovery contributors
            recovery = (
                0.30 * float(content.get('health_pickups', 0)) +
                0.20 * float(content.get('keys', 0)) +
                0.20 * float(content.get('items', 0)) +
                0.15 * float(content.get('treasures', 0)) +
                (0.50 if content.get('safe_room', False) else 0.0)
            )

            tension_value = max(0.0, challenge - 0.6 * recovery)
            tensions.append(tension_value)

        curve = np.asarray(tensions, dtype=np.float32)
        if curve.max() > 0:
            curve = curve / curve.max()

        if curve.size >= 3:
            # Light smoothing to reduce single-room spikes.
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            curve = np.convolve(curve, kernel, mode='same')
            curve = np.clip(curve, 0.0, 1.0)

        return curve

    def _count_rest_areas(self, curve: np.ndarray) -> int:
        """Count low-tension local minima as recovery beats."""
        if curve.size == 0:
            return 0

        rest_threshold = min(0.35, float(np.quantile(curve, 0.35)))
        rest_areas = 0

        for i, value in enumerate(curve):
            prev_v = curve[i - 1] if i > 0 else value
            next_v = curve[i + 1] if i < curve.size - 1 else value
            if value <= rest_threshold and value <= prev_v and value <= next_v:
                rest_areas += 1

        return rest_areas

    def _target_curve(self, length: int) -> np.ndarray:
        """Construct a canonical dungeon pacing template."""
        if length <= 0:
            return np.array([], dtype=np.float32)

        x = np.linspace(0.0, 1.0, length, dtype=np.float32)
        base_rise = 0.15 + 0.70 * x
        mid_dip = 0.18 * np.exp(-((x - 0.55) ** 2) / (2.0 * 0.08**2))
        late_peak = 0.22 * np.exp(-((x - self.target_peak_position) ** 2) / (2.0 * 0.06**2))
        target = np.clip(base_rise - mid_dip + late_peak, 0.0, 1.0)
        return target.astype(np.float32)


# ============================================================================
# MASTER FUN EVALUATOR
# ============================================================================

class FunMetricsEvaluator:
    """
    Master evaluator combining all fun metrics.
    
    Usage:
        evaluator = FunMetricsEvaluator()
        fun_metrics = evaluator.evaluate(
            mission_graph=graph,
            solution_path=path,
            room_contents=contents,
            critical_path=critical_set
        )
        
        print(f"Fun Score: {fun_metrics.overall_fun_score:.2f}")
    """
    
    def __init__(self):
        self.frustration_analyzer = FrustrationAnalyzer()
        self.explorability_analyzer = ExplorabilityAnalyzer()
        self.flow_analyzer = FlowAnalyzer()
        self.pacing_analyzer = PacingAnalyzer()
    
    def evaluate(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
        critical_path: Set[int]
    ) -> FunMetrics:
        """
        Comprehensive fun evaluation.
        
        Returns:
            FunMetrics with all sub-metrics and overall score
        """
        # Compute sub-metrics
        frustration = self.frustration_analyzer.analyze(
            mission_graph, solution_path, room_contents
        )
        
        explorability = self.explorability_analyzer.analyze(
            mission_graph, critical_path, room_contents
        )
        
        flow = self.flow_analyzer.analyze(
            mission_graph, solution_path, room_contents
        )
        
        pacing = self.pacing_analyzer.analyze(
            mission_graph, solution_path, room_contents
        )
        
        # Overall fun score
        # Fun = high explorability + high flow - high frustration
        overall_fun_score = (
            0.3 * explorability.discovery_potential +
            0.3 * flow.flow_score +
            0.2 * pacing.pacing_score +
            0.2 * (1.0 - frustration.total_frustration)
        )
        
        return FunMetrics(
            frustration=frustration,
            explorability=explorability,
            flow=flow,
            pacing=pacing,
            overall_fun_score=overall_fun_score
        )


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

"""
# In src/simulation/map_elites.py (MAP-Elites validator):

from src.evaluation.fun_metrics import FunMetricsEvaluator

class MAPElitesEvaluator:
    def __init__(self, ...):
        # ... existing init ...
        self.fun_evaluator = FunMetricsEvaluator()
    
    def evaluate_dungeon(self, mission_graph, solution_data, room_contents):
        # ... existing metrics ...
        
        # Add fun metrics
        fun_metrics = self.fun_evaluator.evaluate(
            mission_graph=mission_graph,
            solution_path=solution_data['path'],
            room_contents=room_contents,
            critical_path=set(solution_data['critical_path'])
        )
        
        metrics['fun_score'] = fun_metrics.overall_fun_score
        metrics['frustration'] = fun_metrics.frustration.total_frustration
        metrics['explorability'] = fun_metrics.explorability.discovery_potential
        metrics['flow'] = fun_metrics.flow.flow_score
        
        return metrics


# In gui_runner.py (display fun metrics):

def _render_fun_metrics_panel(self, surface):
    '''Render fun metrics in HUD.'''
    if self.fun_metrics is None:
        return
    
    font = pygame.font.Font(None, 24)
    y = 50
    
    # Overall fun score with color coding
    fun_score = self.fun_metrics.overall_fun_score
    color = self._get_score_color(fun_score)
    text = font.render(f'Fun Score: {fun_score:.2f}', True, color)
    surface.blit(text, (10, y))
    y += 30
    
    # Sub-metrics
    metrics = [
        ('Frustration', self.fun_metrics.frustration.total_frustration, True),  # Inverted
        ('Explorability', self.fun_metrics.explorability.discovery_potential, False),
        ('Flow', self.fun_metrics.flow.flow_score, False)
    ]
    
    for label, value, inverted in metrics:
        display_value = (1.0 - value) if inverted else value
        color = self._get_score_color(display_value)
        text = font.render(f'{label}: {value:.2f}', True, color)
        surface.blit(text, (10, y))
        y += 25
    
    def _get_score_color(self, score):
        '''Red (bad) -> Yellow (OK) -> Green (good).'''
        if score < 0.4:
            return (255, 0, 0)
        elif score < 0.7:
            return (255, 255, 0)
        else:
            return (0, 255, 0)
"""
