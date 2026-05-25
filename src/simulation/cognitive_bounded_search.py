"""
COGNITIVE BOUNDED SEARCH (CBS) â€” Human-Like Dungeon Navigation
================================================================

This module implements cognitively-realistic agents for validating dungeon
playability from a human perspective, extending the optimal A* solver with
bounded rationality and memory limitations.

SCIENTIFIC FOUNDATION:
----------------------
- Miller's Law (1956): Working memory capacity ~7Â±2 items
- Kahneman (2011): System 1/System 2 decision-making
- Simon (1955): Bounded rationality and satisficing
- Newell & Simon (1972): Human Problem Solving â€” search space constraints

ARCHITECTURE OVERVIEW:
----------------------

    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚                     COGNITIVE BOUNDED SEARCH (CBS)                   â”‚
    â”‚                                                                      â”‚
    â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
    â”‚  â”‚ VISION SYSTEM  â”‚  â”‚ BELIEF MAP   â”‚  â”‚ WORKING MEMORY         â”‚   â”‚
    â”‚  â”‚ â€¢ FOV cone     â”‚â”€â”€â”‚ â€¢ Known tilesâ”‚â”€â”€â”‚ â€¢ Capacity=7 (Miller)  â”‚   â”‚
    â”‚  â”‚ â€¢ Radius=5     â”‚  â”‚ â€¢ Confidence â”‚  â”‚ â€¢ Decay rate=0.95      â”‚   â”‚
    â”‚  â”‚ â€¢ Occlusion    â”‚  â”‚ â€¢ Last seen  â”‚  â”‚ â€¢ Salience weighting   â”‚   â”‚
    â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
    â”‚          â”‚                  â”‚                      â”‚                 â”‚
    â”‚          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                 â”‚
    â”‚                             â”‚                                        â”‚
    â”‚                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”                               â”‚
    â”‚                    â”‚ DECISION ENGINE â”‚                               â”‚
    â”‚                    â”‚ â€¢ Heuristic mix â”‚                               â”‚
    â”‚                    â”‚ â€¢ Satisficing   â”‚                               â”‚
    â”‚                    â”‚ â€¢ Persona-based â”‚                               â”‚
    â”‚                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜                               â”‚
    â”‚                             â”‚                                        â”‚
    â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”‚
    â”‚  â”‚                    AGENT PERSONAS                            â”‚    â”‚
    â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚    â”‚
    â”‚  â”‚  â”‚SPEEDRUNERâ”‚  â”‚ EXPLORER â”‚  â”‚ CAUTIOUS â”‚  â”‚ FORGETFUL    â”‚ â”‚    â”‚
    â”‚  â”‚  â”‚ A* base  â”‚  â”‚ Curious  â”‚  â”‚ Safe     â”‚  â”‚ High decay   â”‚ â”‚    â”‚
    â”‚  â”‚  â”‚ Min path â”‚  â”‚ All roomsâ”‚  â”‚ Avoids M â”‚  â”‚ Gets lost    â”‚ â”‚    â”‚
    â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚    â”‚
    â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â”‚
    â”‚                                                                      â”‚
    â”‚  OUTPUT: (success, path, states, CBSMetrics)                         â”‚
    â”‚  â€¢ Confusion Index = revisits / unique_visits                        â”‚
    â”‚  â€¢ Navigation Entropy = -Î£ p(dir) log p(dir)                         â”‚
    â”‚  â€¢ Cognitive Load = memory_size Ã— confidence_variance                â”‚
    â”‚  â€¢ Goal-Sighting Latency = time_see_goal - time_reach_goal          â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

INTEGRATION:
-----------
    from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
    from src.simulation.validator import ZeldaLogicEnv

    env = ZeldaLogicEnv(semantic_grid=grid)
    cbs = CognitiveBoundedSearch(env, persona='explorer')
    success, path, states, metrics = cbs.solve()
    
    print(metrics.confusion_index)      # How lost the agent got
    print(metrics.navigation_entropy)   # Decision randomness
    print(metrics.cognitive_load)       # Mental effort estimate
    print(metrics.aha_latency)          # Goal-sighting-to-completion time

VERSION: 1.0.0
"""

from __future__ import annotations

import math
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, Hashable, List, Tuple, Optional, Set,
    TYPE_CHECKING
)
from collections import deque, defaultdict

import numpy as np

if TYPE_CHECKING:
    from src.simulation.validator import ZeldaLogicEnv

# Configure logging
logger = logging.getLogger(__name__)

# Import tile constants from canonical source
from src.core.definitions import (
    ROOM_HEIGHT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE
)

# Re-import blocking/walkable sets for convenience
from src.simulation.validator import (
    WALKABLE_IDS, BLOCKING_IDS, CONDITIONAL_IDS, 
    PICKUP_IDS, Action, ACTION_DELTAS,
    GameState as _GameState,
    StateSpaceAStar,
    game_state_key,
)


# ==============================================================================
# COGNITIVE METRICS DATA STRUCTURES
# ==============================================================================

@dataclass
class CBSMetrics:
    """
    Metrics capturing human-like navigation behavior.
    
    These metrics are inspired by cognitive science research on spatial
    navigation, decision-making under uncertainty, and memory limitations.
    
    Attributes:
        confusion_index: Ratio of tile revisits to unique tile visits.
                        High values indicate the agent got lost or backtracked.
                        Formula: revisits / unique_visits
                        Range: [0, âˆž), optimal â‰ˆ 0, confused > 2.0
                        
        navigation_entropy: Shannon entropy of direction choices.
                           High = random wandering, Low = directed movement.
                           Formula: -Î£ p(dir) logâ‚‚ p(dir)
                           Range: [0, 2] for 4 directions, [0, 3] for 8
                           
        cognitive_load: Estimated mental effort based on memory usage and
                       belief uncertainty. Combines Miller's capacity with
                       confidence variance.
                       Formula: (memory_items / capacity) Ã— (1 + ÏƒÂ²_confidence)
                       Range: [0, âˆž), typical [0.1, 2.0]
                       
        aha_latency: Legacy internal name for goal-sighting latency:
                    steps between first seeing the goal and reaching it.
                    Low = efficient path exploitation.
                    High = poor spatial memory or suboptimal routing.
                    
        unique_tiles_visited: Number of distinct positions explored
        
        total_steps: Total path length including revisits
        
        peak_memory_usage: Maximum items in working memory at once
        
        goal_first_seen_step: Step number when goal was first observed
        
        decisions_made: Total decision points (non-corridor tiles)
        
        suboptimal_decisions: Decisions that moved away from goal
        
        replans: Number of times agent changed planned direction
        
        confusion_events: Number of times agent revisited a tile > 2 times
        
        backtrack_loops: Number of detected looping patterns in path
        
        path_length: Total steps (alias for total_steps for compatibility)
        
        belief_entropy_final: Final entropy of the belief map
    """
    confusion_index: float = 0.0
    navigation_entropy: float = 0.0
    cognitive_load: float = 0.0
    aha_latency: int = 0
    unique_tiles_visited: int = 0
    total_steps: int = 0
    peak_memory_usage: int = 0
    goal_first_seen_step: int = -1  # -1 means never seen
    decisions_made: int = 0
    suboptimal_decisions: int = 0
    exploration_efficiency: float = 0.0  # unique_tiles / total_steps
    
    # Extended metrics for CBS+
    replans: int = 0  # Direction changes after planning
    confusion_events: int = 0  # Tiles revisited > 2 times
    backtrack_loops: int = 0  # Detected loops in path
    path_length: int = 0  # Alias for total_steps
    belief_entropy_final: float = 0.0  # Final belief map entropy
    
    # Paper metrics (CBS+ Paper Section 4)
    room_entropy: float = 0.0  # Navigational entropy H = -Î£ p(room) Ã— logâ‚‚(p(room))
    unique_rooms_visited: int = 0
    total_room_visits: int = 0
    deliberation_events: int = 0
    budget_exhaustion_events: int = 0
    peak_frustration: float = 0.0
    final_deliberation_budget: float = 0.0
    affordance_reactivations: int = 0
    affordance_guided_steps: int = 0
    inventory_change_events: int = 0
    focus_switches: int = 0
    focus_guided_steps: int = 0
    
    # Per-room metrics for detailed analysis
    room_visit_counts: Dict[Hashable, int] = field(default_factory=dict)
    direction_distribution: Dict[str, int] = field(default_factory=dict)
    memory_timeline: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute derived metrics."""
        if self.total_steps > 0:
            self.exploration_efficiency = self.unique_tiles_visited / self.total_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'confusion_index': round(self.confusion_index, 4),
            'navigation_entropy': round(self.navigation_entropy, 4),
            'cognitive_load': round(self.cognitive_load, 4),
            'aha_latency': self.aha_latency,
            'goal_sighting_latency': self.aha_latency,
            'unique_tiles_visited': self.unique_tiles_visited,
            'total_steps': self.total_steps,
            'peak_memory_usage': self.peak_memory_usage,
            'goal_first_seen_step': self.goal_first_seen_step,
            'decisions_made': self.decisions_made,
            'suboptimal_decisions': self.suboptimal_decisions,
            'exploration_efficiency': round(self.exploration_efficiency, 4),
            'room_visit_counts': dict(self.room_visit_counts),
            'direction_distribution': dict(self.direction_distribution),
            'room_entropy': round(self.room_entropy, 4),
            'unique_rooms_visited': int(self.unique_rooms_visited),
            'total_room_visits': int(self.total_room_visits),
            'deliberation_events': self.deliberation_events,
            'budget_exhaustion_events': self.budget_exhaustion_events,
            'peak_frustration': round(self.peak_frustration, 4),
            'final_deliberation_budget': round(self.final_deliberation_budget, 4),
            'affordance_reactivations': self.affordance_reactivations,
            'affordance_guided_steps': self.affordance_guided_steps,
            'inventory_change_events': self.inventory_change_events,
            'focus_switches': self.focus_switches,
            'focus_guided_steps': self.focus_guided_steps,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘                     CBS COGNITIVE METRICS                            â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘ Confusion Index:     {self.confusion_index:>8.3f}  (revisits/unique, low=good)     â•‘
â•‘ Navigation Entropy:  {self.navigation_entropy:>8.3f}  (bits, 0=linear, 2=random)   â•‘
â•‘ Cognitive Load:      {self.cognitive_load:>8.3f}  (memoryÃ—uncertainty)            â•‘
â•‘ Goal-Sighting Lat.:  {self.aha_latency:>8d}  steps (seeâ†’reach goal)             â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘ Unique Tiles:        {self.unique_tiles_visited:>8d}  â”‚ Total Steps: {self.total_steps:>8d}          â•‘
â•‘ Exploration Eff:     {self.exploration_efficiency:>8.3f}  â”‚ Peak Memory: {self.peak_memory_usage:>8d}          â•‘
â•‘ Decisions Made:      {self.decisions_made:>8d}  â”‚ Suboptimal:  {self.suboptimal_decisions:>8d}          â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""


# ==============================================================================
# TILE OBSERVATION TYPE
# ==============================================================================

class TileKnowledge(Enum):
    """Knowledge state for a tile in the belief map."""
    UNKNOWN = auto()      # Never seen, completely uncertain
    GLIMPSED = auto()     # Seen briefly, may have decayed
    OBSERVED = auto()     # Currently in view, high confidence
    EXPLORED = auto()     # Physically visited, full confidence


@dataclass
class TileObservation:
    """
    An observation of a single tile with confidence and temporal data.
    
    Attributes:
        tile_type: The semantic ID of the tile (or best guess)
        confidence: Probability that tile_type is correct [0.0, 1.0]
        last_seen: Timestep when this tile was last observed
        knowledge: Categorical knowledge state
        visited: Whether the agent has physically stepped on this tile
    """
    tile_type: int
    confidence: float = 0.0
    last_seen: int = -1
    knowledge: TileKnowledge = TileKnowledge.UNKNOWN
    visited: bool = False
    
    def decay(self, current_step: int, decay_rate: float = 0.95) -> None:
        """
        Apply memory decay based on time since last observation.
        
        Formula: confidence *= decay_rate^(current_step - last_seen)
        """
        if self.last_seen < 0 or self.knowledge == TileKnowledge.EXPLORED:
            return  # Unknown tiles and explored tiles don't decay
        
        time_delta = current_step - self.last_seen
        if time_delta > 0:
            self.confidence *= (decay_rate ** time_delta)
            
            # Downgrade knowledge level if confidence drops
            if self.confidence < 0.3:
                self.knowledge = TileKnowledge.GLIMPSED
            if self.confidence < 0.1:
                self.knowledge = TileKnowledge.UNKNOWN


# ==============================================================================
# EPISTEMIC STATE: BELIEF MAP
# ==============================================================================

class BeliefMap:
    """
    What the agent THINKS the map looks like (epistemic state).
    
    This is fundamentally different from the ground truth grid:
    - The agent only knows what it has seen
    - Memory decays over time (configurable rate)
    - Confidence varies by recency and exploration
    
    Scientific basis:
    - Tolman (1948): Cognitive maps in rats and men
    - O'Keefe & Nadel (1978): Hippocampus as a cognitive map
    - Gallistel (1990): Spatial representation in animals
    
    Attributes:
        known_tiles: Map from position to TileObservation
        grid_shape: Shape of the underlying grid (height, width)
        default_assumption: What to assume for unknown tiles (WALL by default)
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        default_assumption: int = None,
        decay_rate: float = 0.95
    ):
        """
        Initialize an empty belief map.
        
        Args:
            grid_shape: (height, width) of the map
            default_assumption: Tile ID to assume for unknown areas.
                               Default is WALL (conservative assumption).
            decay_rate: Per-step confidence decay factor [0, 1]
        """
        self.grid_shape = grid_shape
        self.default_assumption = default_assumption or SEMANTIC_PALETTE['WALL']
        self.decay_rate = decay_rate
        
        # Core storage: position -> TileObservation
        self.known_tiles: Dict[Tuple[int, int], TileObservation] = {}
        
        # Statistics for cognitive metrics
        self.total_observations = 0
        self.revisit_count = 0
        self.unique_visits: Set[Tuple[int, int]] = set()

    def reset(self) -> None:
        """Clear all accumulated beliefs and run-local statistics."""
        self.known_tiles.clear()
        self.total_observations = 0
        self.revisit_count = 0
        self.unique_visits.clear()
    
    def observe(
        self,
        position: Tuple[int, int],
        tile_type: int,
        current_step: int,
        is_visit: bool = False
    ) -> None:
        """
        Record an observation of a tile.
        
        Args:
            position: (row, col) position observed
            tile_type: Semantic ID of the tile
            current_step: Current timestep for recency tracking
            is_visit: True if agent is physically at this position
        """
        self.total_observations += 1
        
        if position in self.known_tiles:
            obs = self.known_tiles[position]
            
            # Update observation
            obs.tile_type = tile_type
            obs.confidence = 1.0 if is_visit else max(obs.confidence, 0.8)
            obs.last_seen = current_step
            obs.knowledge = TileKnowledge.EXPLORED if is_visit else TileKnowledge.OBSERVED
            
            if is_visit and obs.visited:
                self.revisit_count += 1
            obs.visited = obs.visited or is_visit
        else:
            # New observation
            self.known_tiles[position] = TileObservation(
                tile_type=tile_type,
                confidence=1.0 if is_visit else 0.8,
                last_seen=current_step,
                knowledge=TileKnowledge.EXPLORED if is_visit else TileKnowledge.OBSERVED,
                visited=is_visit
            )
        
        if is_visit:
            self.unique_visits.add(position)
    
    def get_tile_with_confidence(self, position: Tuple[int, int]) -> Tuple[int, float]:
        """
        Query what the agent believes about a tile with confidence.
        
        Returns:
            (tile_type, confidence) tuple
        """
        if position in self.known_tiles:
            obs = self.known_tiles[position]
            return (obs.tile_type, obs.confidence)
        return (self.default_assumption, 0.0)

    def bayes_update(
        self,
        position: Tuple[int, int],
        observed_tile: int,
        current_step: int,
        obs_accuracy: float = 0.9,
        is_visit: bool = False
    ) -> None:
        """
        Perform a simple Bayesian update for a tile observation.

        This maintains `tile_type` as the MAP estimate and `confidence`
        as P(tile==tile_type). When a new observation arrives we update
        confidence using likelihoods P(obs|true).
        """
        # If we've seen this position before, update posterior
        if position in self.known_tiles:
            obs = self.known_tiles[position]

            # Prior: probability that currently-believed type is correct
            prior = obs.confidence

            # If observed matches current belief, likelihood = obs_accuracy
            if observed_tile == obs.tile_type:
                likelihood = obs_accuracy
                false_likelihood = 1.0 - obs_accuracy
                posterior = (likelihood * prior) / (likelihood * prior + false_likelihood * (1 - prior) + 1e-9)
                obs.confidence = min(1.0, max(0.0, posterior))
                obs.last_seen = current_step
                obs.knowledge = TileKnowledge.EXPLORED if is_visit else TileKnowledge.OBSERVED
                if is_visit:
                    obs.visited = True
                    self.unique_visits.add(position)
            else:
                # If observation disagrees, compute probability that observed is true
                # Treat prior for observed as (1 - prior)
                prior_obs = 1.0 - prior
                likelihood = obs_accuracy
                false_likelihood = 1.0 - obs_accuracy
                posterior_obs = (likelihood * prior_obs) / (likelihood * prior_obs + false_likelihood * (1 - prior_obs) + 1e-9)

                # Update stored belief to observed tile with its posterior
                obs.tile_type = observed_tile
                obs.confidence = min(1.0, max(0.0, posterior_obs))
                obs.last_seen = current_step
                obs.knowledge = TileKnowledge.EXPLORED if is_visit else TileKnowledge.OBSERVED
                obs.visited = obs.visited or is_visit
                if is_visit:
                    self.unique_visits.add(position)
        else:
            # New observation: initialize with observation accuracy
            self.known_tiles[position] = TileObservation(
                tile_type=observed_tile,
                confidence=1.0 if is_visit else obs_accuracy,
                last_seen=current_step,
                knowledge=TileKnowledge.EXPLORED if is_visit else TileKnowledge.OBSERVED,
                visited=is_visit,
            )
            if is_visit:
                self.unique_visits.add(position)

    def expected_info_gain(
        self,
        position: Tuple[int, int],
        vision: 'VisionSystem',
        grid: Optional[np.ndarray] = None,
        direction: Tuple[int, int] = (0, 1),
    ) -> float:
        """
        Approximate expected information gain from moving to a position.

        We model gain as mean uncertainty (binary entropy) over tiles that would
        be visible from `position`. Unknown tiles are treated as prior p=0.5.

        Args:
            position: Candidate location to evaluate.
            vision: Vision model controlling FoV and occlusion.
            grid: Optional occupancy grid for occlusion checks. If omitted,
                uses the current belief grid (conservative fallback).
            direction: Candidate facing direction at `position`.
        """
        if grid is None:
            grid = self.to_grid()

        visible = vision.get_visible_tiles(position, direction, grid)
        if not visible:
            return 0.0

        total_uncertainty = 0.0
        eps = 1e-6

        for pos in visible:
            obs = self.known_tiles.get(pos)
            if obs is None or obs.knowledge == TileKnowledge.UNKNOWN:
                # Unknown tile prior -> maximum binary uncertainty.
                total_uncertainty += 1.0
                continue

            conf = float(min(1.0 - eps, max(eps, obs.confidence)))
            uncertainty = -conf * math.log2(conf) - (1.0 - conf) * math.log2(1.0 - conf)
            total_uncertainty += uncertainty

        return total_uncertainty / len(visible)
    
    def get_knowledge_state(self, position: Tuple[int, int]) -> TileKnowledge:
        """Get the knowledge state of a tile."""
        if position in self.known_tiles:
            return self.known_tiles[position].knowledge
        return TileKnowledge.UNKNOWN
    
    def get_confidence(self, row_or_pos: int | Tuple[int, int], col: Optional[int] = None) -> float:
        """
        Get confidence for a specific tile.
        
        Can be called as:
            get_confidence((row, col))  or  get_confidence(row, col)
        """
        if col is not None:
            position = (row_or_pos, col)
        else:
            position = row_or_pos
            
        if position in self.known_tiles:
            return self.known_tiles[position].confidence
        return 0.0
    
    def get_tile(self, row_or_pos: int | Tuple[int, int], col: Optional[int] = None) -> int:
        """
        Get the believed tile type at a position.
        
        Can be called as:
            get_tile((row, col))  or  get_tile(row, col)
        
        Returns:
            Tile type ID (int)
        """
        if col is not None:
            position = (row_or_pos, col)
        else:
            position = row_or_pos
            
        if position in self.known_tiles:
            return self.known_tiles[position].tile_type
        return self.default_assumption
    
    def apply_decay(self, current_step: int = 0, decay_rate: Optional[float] = None) -> None:
        """
        Apply memory decay to all observations.
        
        Args:
            current_step: Current timestep (optional, used for time-based decay)
            decay_rate: Optional override for decay rate
        """
        rate = decay_rate if decay_rate is not None else self.decay_rate
        for _pos, obs in self.known_tiles.items():
            if obs.confidence > 0:
                obs.confidence *= rate
                # Downgrade knowledge if confidence too low
                if obs.confidence < 0.3:
                    obs.knowledge = TileKnowledge.GLIMPSED
                if obs.confidence < 0.1:
                    obs.knowledge = TileKnowledge.UNKNOWN
    
    def get_unexplored_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get adjacent positions that haven't been visited."""
        unexplored = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (position[0] + dr, position[1] + dc)
            if 0 <= neighbor[0] < self.grid_shape[0] and 0 <= neighbor[1] < self.grid_shape[1]:
                if neighbor not in self.known_tiles or not self.known_tiles[neighbor].visited:
                    unexplored.append(neighbor)
        return unexplored
    
    def compute_confidence_variance(self) -> float:
        """Compute variance in confidence across known tiles."""
        if not self.known_tiles:
            return 1.0  # Maximum uncertainty if nothing known
        confidences = [obs.confidence for obs in self.known_tiles.values()]
        return float(np.var(confidences))
    
    def get_frontier(self) -> Set[Tuple[int, int]]:
        """
        Get the exploration frontier: observed but unvisited tiles.
        
        These are tiles the agent knows about but hasn't stepped on yet.
        """
        frontier = set()
        for pos, obs in self.known_tiles.items():
            if obs.knowledge in (TileKnowledge.OBSERVED, TileKnowledge.GLIMPSED):
                if not obs.visited:
                    frontier.add(pos)
        return frontier
    
    def compute_confusion_index(self) -> float:
        """
        Compute confusion index: revisits / unique_visits.
        
        Higher values indicate the agent got lost or backtracked frequently.
        """
        if len(self.unique_visits) == 0:
            return 0.0
        return self.revisit_count / len(self.unique_visits)
    
    @property
    def height(self) -> int:
        """Height of the belief map grid."""
        return self.grid_shape[0]
    
    @property
    def width(self) -> int:
        """Width of the belief map grid."""
        return self.grid_shape[1]
    
    def update(
        self,
        row: int,
        col: int,
        observed_tile: int,
        current_step: int = 0,
        is_visit: bool = False
    ) -> None:
        """
        Convenience method to update a tile observation (alias for bayes_update).
        
        Args:
            row: Row position
            col: Column position
            observed_tile: The tile type observed
            current_step: Current timestep
            is_visit: True if physically at this position
        """
        self.bayes_update((row, col), observed_tile, current_step, is_visit=is_visit)
    
    def compute_entropy(self, row: int, col: int) -> float:
        """
        Compute Shannon entropy for a single tile based on confidence.
        
        High entropy = high uncertainty (unknown tile)
        Low entropy = high certainty (observed tile)
        
        Formula: H = -p*log2(p) - (1-p)*log2(1-p)
        where p is the confidence.
        """
        pos = (row, col)
        obs = self.known_tiles.get(pos)

        # Unknown tiles should carry high uncertainty, not zero.
        if obs is None or obs.knowledge == TileKnowledge.UNKNOWN:
            conf = 0.5
        else:
            conf = float(obs.confidence)

        eps = 1e-6
        conf = min(1.0 - eps, max(eps, conf))

        # Binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
        entropy = -conf * math.log2(conf) - (1 - conf) * math.log2(1 - conf)
        return float(entropy)
    
    def compute_total_entropy(self) -> float:
        """Compute total entropy across all tiles."""
        total = 0.0
        for r in range(self.height):
            for c in range(self.width):
                total += self.compute_entropy(r, c)
        return total
    
    def to_grid(self, include_confidence: bool = False) -> np.ndarray:
        """
        Convert belief map to grid array.
        
        Args:
            include_confidence: If True, return (H, W, 2) array with
                               [:,:,0] = tile types and [:,:,1] = confidences.
                               If False, return (H, W) array of tile types.
        
        Returns:
            Numpy array representing the belief map.
        """
        if include_confidence:
            grid = np.zeros((self.height, self.width, 2), dtype=np.float32)
            grid[:, :, 0] = self.default_assumption
            grid[:, :, 1] = 0.5  # Prior confidence
            
            for pos, obs in self.known_tiles.items():
                if 0 <= pos[0] < self.height and 0 <= pos[1] < self.width:
                    grid[pos[0], pos[1], 0] = obs.tile_type
                    grid[pos[0], pos[1], 1] = obs.confidence
            
            return grid
        else:
            grid = np.full((self.height, self.width), self.default_assumption, dtype=np.int32)
            
            for pos, obs in self.known_tiles.items():
                if 0 <= pos[0] < self.height and 0 <= pos[1] < self.width:
                    grid[pos[0], pos[1]] = obs.tile_type
            
            return grid


# ==============================================================================
# VISION SYSTEM
# ==============================================================================

class VisionSystem:
    """
    Simulates limited field-of-view for the agent.
    
    Unlike omniscient A*, a human player can only see tiles within
    their view range and field of view angle.
    
    Attributes:
        radius: Maximum visibility distance (default 5 tiles)
        cone_angle: Field of view in degrees (default 120Â°, forward-facing)
        enable_occlusion: Whether walls block line of sight
    """
    
    def __init__(
        self,
        radius: int = 5,
        cone_angle: float = 120.0,
        enable_occlusion: bool = True
    ):
        """
        Initialize vision system.
        
        Args:
            radius: Max tiles visible in any direction
            cone_angle: FOV angle in degrees (360 = full vision)
            enable_occlusion: If True, walls block visibility
        """
        self.radius = radius
        self.cone_angle = cone_angle
        self.enable_occlusion = enable_occlusion
        
        # Precompute visibility offsets for efficiency
        self._visibility_offsets = self._compute_visibility_offsets()
    
    def _compute_visibility_offsets(self) -> List[Tuple[int, int]]:
        """Precompute all (dr, dc) offsets within radius, sorted by distance."""
        offsets = []
        for dr in range(-self.radius, self.radius + 1):
            for dc in range(-self.radius, self.radius + 1):
                dist = math.sqrt(dr*dr + dc*dc)
                if 0 < dist <= self.radius:  # Exclude (0,0) - current position
                    offsets.append((dr, dc, dist))
        # Sort by distance for occlusion processing
        offsets.sort(key=lambda x: x[2])
        return [(dr, dc) for dr, dc, _ in offsets]
    
    def get_visible_tiles(
        self,
        position: Tuple[int, int],
        direction: Tuple[int, int],
        grid: np.ndarray
    ) -> Set[Tuple[int, int]]:
        """
        Get all tiles visible from current position and facing direction.
        
        Args:
            position: Current (row, col) position
            direction: Facing direction as (dr, dc), e.g., (-1, 0) = UP
            grid: Ground truth grid for occlusion checking
        
        Returns:
            Set of visible (row, col) positions
        """
        visible = {position}  # Always see current tile
        height, width = grid.shape
        
        # Handle 360Â° vision (no cone restriction)
        if self.cone_angle >= 360:
            angle_check = lambda dr, dc: True
        else:
            # Compute angle limits
            half_angle = math.radians(self.cone_angle / 2)
            dir_angle = math.atan2(direction[1], direction[0])
            
            def angle_check(dr: int, dc: int) -> bool:
                if dr == 0 and dc == 0:
                    return True
                tile_angle = math.atan2(dc, dr)
                angle_diff = abs(tile_angle - dir_angle)
                # Handle wrap-around
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                return angle_diff <= half_angle
        
        # Track occluded tiles for shadow casting
        occluded: Set[Tuple[int, int]] = set()
        
        for dr, dc in self._visibility_offsets:
            new_r = position[0] + dr
            new_c = position[1] + dc
            
            # Bounds check
            if not (0 <= new_r < height and 0 <= new_c < width):
                continue
            
            # Angle check (field of view cone)
            if not angle_check(dr, dc):
                continue
            
            tile_pos = (new_r, new_c)
            
            # Occlusion check
            if self.enable_occlusion and tile_pos in occluded:
                continue
            
            # Tile is visible
            visible.add(tile_pos)
            
            # If this tile is a wall, add shadow behind it
            if self.enable_occlusion and grid[new_r, new_c] in BLOCKING_IDS:
                # Cast shadow: all tiles further in this direction
                self._cast_shadow(position, (dr, dc), occluded, height, width)
        
        return visible
    
    def _cast_shadow(
        self,
        origin: Tuple[int, int],
        direction: Tuple[int, int],
        occluded: Set[Tuple[int, int]],
        height: int,
        width: int
    ) -> None:
        """Add tiles behind an occluding wall to the shadow set."""
        dr, dc = direction
        # Normalize to unit direction.
        dist = math.sqrt(dr*dr + dc*dc)
        if dist == 0:
            return

        step_r = dr / dist
        step_c = dc / dist

        # Continue the ray from the blocker.  The previous implementation used
        # step * mult * dist, which simplifies back to the original wall offset
        # multiplied by mult and skips valid shadow cells for non-unit diagonals.
        for extra in range(1, self.radius + 1):
            shadow_r = origin[0] + int(round(step_r * (dist + extra)))
            shadow_c = origin[1] + int(round(step_c * (dist + extra)))
            if 0 <= shadow_r < height and 0 <= shadow_c < width:
                occluded.add((shadow_r, shadow_c))
    
    def get_360_visible_tiles(
        self,
        position: Tuple[int, int],
        grid: np.ndarray
    ) -> Set[Tuple[int, int]]:
        """
        Get all tiles visible with 360Â° vision (useful for comparison).
        """
        return self.get_visible_tiles(position, (0, 1), grid)  # Direction ignored with 360Â°


# ==============================================================================
# WORKING MEMORY SYSTEM
# ==============================================================================

class MemoryItemType(Enum):
    """Types of items that can be stored in working memory."""
    POSITION = auto()       # A remembered location
    GOAL = auto()           # Goal location
    ITEM = auto()           # Item location (key, bomb, etc.)
    THREAT = auto()         # Enemy or danger location
    DOOR = auto()           # Door requiring key/bomb
    AFFORDANCE = auto()     # Remembered unresolved progression affordance
    PATH_SEGMENT = auto()   # A remembered path segment
    LANDMARK = auto()       # Distinctive visual landmark


@dataclass
class MemoryItem:
    """
    A single item in working memory.
    
    Scientific basis: Cowan (2001) - "The magical number 4 in short-term memory"
    Items have salience that determines retention priority.
    
    Attributes:
        item_type: Category of the memory
        position: Location associated with this memory
        salience: Importance weight [0, 1], higher = more memorable
        created_step: When this memory was created
        last_accessed: When this memory was last used
        data: Additional associated data (tile type, path, etc.)
    """
    item_type: MemoryItemType
    position: Tuple[int, int]
    salience: float = 0.5
    created_step: int = 0
    last_accessed: int = 0
    data: Any = None
    
    def __hash__(self):
        return hash((self.item_type, self.position))
    
    def __eq__(self, other):
        if not isinstance(other, MemoryItem):
            return False
        return self.item_type == other.item_type and self.position == other.position


class WorkingMemory:
    """
    Capacity-limited working memory based on Miller's Law.
    
    Scientific basis:
    - Miller (1956): "The magical number seven, plus or minus two"
    - Cowan (2001): Modern estimate is 4Â±1 chunks
    - Baddeley (2000): Working memory model with central executive
    
    When at capacity, lowest-salience items are forgotten first.
    Items also decay over time if not refreshed.
    
    Attributes:
        capacity: Maximum items (default 7, Miller's number)
        decay_rate: Per-step salience decay factor
        items: Current memory contents (ordered by salience)
    """
    
    def __init__(
        self,
        capacity: int = 7,
        decay_rate: float = 0.95,
        salience_weights: Optional[Dict[MemoryItemType, float]] = None
    ):
        """
        Initialize working memory.
        
        Args:
            capacity: Max items to retain (7 = Miller's number, 4 = Cowan's)
            decay_rate: Salience decay per step [0, 1]
            salience_weights: Type-specific base salience values
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        
        # Default salience weights by type (goals are most memorable)
        self.salience_weights = salience_weights or {
            MemoryItemType.GOAL: 1.0,
            MemoryItemType.THREAT: 0.9,
            MemoryItemType.ITEM: 0.8,
            MemoryItemType.DOOR: 0.7,
            MemoryItemType.AFFORDANCE: 0.85,
            MemoryItemType.LANDMARK: 0.6,
            MemoryItemType.POSITION: 0.4,
            MemoryItemType.PATH_SEGMENT: 0.3,
        }
        
        # Memory storage: list sorted by salience (high to low)
        self.items: List[MemoryItem] = []
        
        # Statistics
        self.total_remembered = 0
        self.total_forgotten = 0
        self.peak_usage = 0

    def reset(self) -> None:
        """Clear all active memory contents and run-local statistics."""
        self.items.clear()
        self.total_remembered = 0
        self.total_forgotten = 0
        self.peak_usage = 0
    
    def remember(
        self,
        item_type: MemoryItemType,
        position: Tuple[int, int],
        current_step: int,
        data: Any = None,
        salience_boost: float = 0.0
    ) -> bool:
        """
        Add or update an item in memory.
        
        Args:
            item_type: Category of memory
            position: Associated location
            current_step: Current timestep
            data: Additional data to store
            salience_boost: Extra salience for this item
        
        Returns:
            True if item was remembered, False if forgotten immediately
        """
        base_salience = self.salience_weights.get(item_type, 0.5)
        salience = min(1.0, base_salience + salience_boost)
        
        # Check if this item already exists (update it)
        for existing in self.items:
            if existing.item_type == item_type and existing.position == position:
                # Refresh the memory
                existing.salience = max(existing.salience, salience)
                existing.last_accessed = current_step
                existing.data = data or existing.data
                self._sort_by_salience()
                return True
        
        # Create new memory item
        new_item = MemoryItem(
            item_type=item_type,
            position=position,
            salience=salience,
            created_step=current_step,
            last_accessed=current_step,
            data=data
        )
        
        # Add to memory
        self.items.append(new_item)
        self.total_remembered += 1
        self._sort_by_salience()
        
        # Enforce capacity limit
        self._forget_excess()
        
        # Update peak usage
        self.peak_usage = max(self.peak_usage, len(self.items))
        
        return new_item in self.items
    
    def recall(
        self,
        item_type: Optional[MemoryItemType] = None,
        current_step: int = 0
    ) -> List[MemoryItem]:
        """
        Retrieve memories, optionally filtered by type.
        
        Accessing memories refreshes their last_accessed timestamp.
        
        Args:
            item_type: Optional filter by type
            current_step: Current timestep for access tracking
        
        Returns:
            List of matching memory items
        """
        results = []
        for item in self.items:
            if item_type is None or item.item_type == item_type:
                item.last_accessed = current_step
                results.append(item)
        return results
    
    def recall_nearest(
        self,
        position: Tuple[int, int],
        item_type: Optional[MemoryItemType] = None,
        current_step: int = 0
    ) -> Optional[MemoryItem]:
        """
        Recall the nearest remembered item of a given type.
        """
        candidates = self.recall(item_type, current_step)
        if not candidates:
            return None
        
        def distance(item: MemoryItem) -> float:
            dr = item.position[0] - position[0]
            dc = item.position[1] - position[1]
            return math.sqrt(dr*dr + dc*dc)
        
        return min(candidates, key=distance)
    
    def forget(self, item: MemoryItem) -> None:
        """Explicitly forget an item."""
        if item in self.items:
            self.items.remove(item)
            self.total_forgotten += 1
    
    def apply_decay(self, current_step: int) -> None:
        """
        Apply time-based decay to all memories.
        
        Items decay based on time since last access.
        Memories that fall below threshold are forgotten.
        """
        for item in self.items:
            time_since_access = current_step - item.last_accessed
            if time_since_access > 0:
                item.salience *= (self.decay_rate ** time_since_access)
        
        # Remove memories with very low salience
        threshold = 0.05
        to_forget = [item for item in self.items if item.salience < threshold]
        for item in to_forget:
            self.forget(item)
        
        self._sort_by_salience()
    
    def _sort_by_salience(self) -> None:
        """Keep items sorted by salience (high to low)."""
        self.items.sort(key=lambda x: x.salience, reverse=True)
    
    def _forget_excess(self) -> None:
        """Remove lowest-salience items if over capacity."""
        while len(self.items) > self.capacity:
            _ = self.items.pop()  # Remove lowest salience
            self.total_forgotten += 1
    
    def is_remembered(
        self,
        position: Tuple[int, int],
        item_type: Optional[MemoryItemType] = None
    ) -> bool:
        """Check if a position is currently in memory."""
        for item in self.items:
            if item.position == position:
                if item_type is None or item.item_type == item_type:
                    return True
        return False
    
    def get_usage_ratio(self) -> float:
        """Return current usage as fraction of capacity."""
        return len(self.items) / self.capacity


# ==============================================================================
# DECISION HEURISTICS
# ==============================================================================

class DecisionHeuristic(ABC):
    """
    Abstract base class for decision heuristics.
    
    Each heuristic provides a score for a potential move based on
    different criteria (curiosity, safety, goal-seeking, etc.).
    """
    
    @abstractmethod
    def score(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_tile: int,
        belief_map: BeliefMap,
        memory: WorkingMemory,
        goal_pos: Optional[Tuple[int, int]],
        current_step: int
    ) -> float:
        """
        Score a potential move.
        
        Args:
            current_pos: Current agent position
            target_pos: Position being considered
            target_tile: Tile type at target (from belief map or truth)
            belief_map: Agent's epistemic state
            memory: Agent's working memory
            goal_pos: Known or remembered goal position
            current_step: Current timestep
        
        Returns:
            Score where higher = more preferred
        """
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Weight of this heuristic in the decision mix."""
        pass


class CuriosityHeuristic(DecisionHeuristic):
    """
    Prefer unexplored areas.
    
    Scientific basis: Berlyne (1966) - Curiosity and exploration
    
    High scores for:
    - Tiles never visited
    - Tiles with low confidence
    - Directions with more unknown tiles
    """
    
    def __init__(self, weight: float = 1.0):
        self._weight = weight
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def score(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_tile: int,
        belief_map: BeliefMap,
        memory: WorkingMemory,
        goal_pos: Optional[Tuple[int, int]],
        current_step: int
    ) -> float:
        # Base score from knowledge state
        knowledge = belief_map.get_knowledge_state(target_pos)
        
        if knowledge == TileKnowledge.UNKNOWN:
            return 1.0  # Maximum curiosity for unknown tiles
        elif knowledge == TileKnowledge.GLIMPSED:
            return 0.7
        elif knowledge == TileKnowledge.OBSERVED:
            return 0.3
        else:  # EXPLORED
            return 0.0  # No curiosity for visited tiles
    
    def compute_exploration_potential(
        self,
        position: Tuple[int, int],
        belief_map: BeliefMap,
        radius: int = 3
    ) -> float:
        """
        Compute how much unexplored area is accessible from a position.
        """
        unexplored_count = 0
        total_count = 0
        
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                pos = (position[0] + dr, position[1] + dc)
                if 0 <= pos[0] < belief_map.grid_shape[0] and \
                   0 <= pos[1] < belief_map.grid_shape[1]:
                    total_count += 1
                    if belief_map.get_knowledge_state(pos) in \
                       (TileKnowledge.UNKNOWN, TileKnowledge.GLIMPSED):
                        unexplored_count += 1
        
        return unexplored_count / max(1, total_count)


class RecencyHeuristic(DecisionHeuristic):
    """
    Prefer recently seen paths (spatial memory).
    
    Scientific basis: Ebbinghaus forgetting curve
    
    Recently observed tiles are easier to navigate to because
    the agent has fresher spatial memory.
    """
    
    def __init__(self, weight: float = 0.5, decay_steps: int = 20):
        self._weight = weight
        self.decay_steps = decay_steps
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def score(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_tile: int,
        belief_map: BeliefMap,
        memory: WorkingMemory,
        goal_pos: Optional[Tuple[int, int]],
        current_step: int
    ) -> float:
        if target_pos not in belief_map.known_tiles:
            return 0.0  # Unknown = no recency preference
        
        obs = belief_map.known_tiles[target_pos]
        if obs.last_seen < 0:
            return 0.0
        
        # Score based on how recently tile was seen
        time_delta = current_step - obs.last_seen
        if time_delta <= 0:
            return 1.0
        
        # Exponential decay of recency score
        return math.exp(-time_delta / self.decay_steps)


class SafetyHeuristic(DecisionHeuristic):
    """
    Avoid enemies and dangerous tiles.
    
    Applies negative scores to tiles with threats.
    """
    
    def __init__(self, weight: float = 0.8):
        self._weight = weight
        self.threat_tiles = {
            SEMANTIC_PALETTE['ENEMY'],
            SEMANTIC_PALETTE['BOSS'],
            SEMANTIC_PALETTE['ELEMENT'],  # Water/lava hazard
        }
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def score(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_tile: int,
        belief_map: BeliefMap,
        memory: WorkingMemory,
        goal_pos: Optional[Tuple[int, int]],
        current_step: int
    ) -> float:
        # Check if target is a threat
        if target_tile in self.threat_tiles:
            return -1.0  # Strongly avoid
        
        # Check for remembered threats nearby
        for item in memory.recall(MemoryItemType.THREAT, current_step):
            dr = abs(item.position[0] - target_pos[0])
            dc = abs(item.position[1] - target_pos[1])
            if dr + dc <= 2:  # Within 2 tiles of remembered threat
                return -0.5 * item.salience
        
        return 0.1  # Slight preference for safe tiles


class GoalSeekingHeuristic(DecisionHeuristic):
    """
    Move toward the goal (when known).
    
    Classic A* distance heuristic, but only applies when
    the goal has been seen and is remembered.
    """
    
    def __init__(self, weight: float = 1.5):
        self._weight = weight
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def score(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_tile: int,
        belief_map: BeliefMap,
        memory: WorkingMemory,
        goal_pos: Optional[Tuple[int, int]],
        current_step: int
    ) -> float:
        # Check if goal is remembered
        goal_memory = memory.recall_nearest(current_pos, MemoryItemType.GOAL, current_step)
        
        remembered_goal = None
        if goal_memory:
            remembered_goal = goal_memory.position
        elif goal_pos:
            # Goal might be in view but not yet in memory
            remembered_goal = goal_pos
        
        if not remembered_goal:
            return 0.0  # No goal-seeking if goal unknown
        
        # Manhattan distance reduction (higher score = closer to goal)
        current_dist = abs(current_pos[0] - remembered_goal[0]) + \
                       abs(current_pos[1] - remembered_goal[1])
        target_dist = abs(target_pos[0] - remembered_goal[0]) + \
                      abs(target_pos[1] - remembered_goal[1])
        
        # Score positive if moving toward goal
        if current_dist > 0:
            improvement = (current_dist - target_dist) / current_dist
            return improvement
        return 0.0


class ItemSeekingHeuristic(DecisionHeuristic):
    """
    Move toward remembered items (keys, bombs).
    
    Important for dungeons requiring specific items to progress.
    """
    
    def __init__(self, weight: float = 0.6):
        self._weight = weight
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def score(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        target_tile: int,
        belief_map: BeliefMap,
        memory: WorkingMemory,
        goal_pos: Optional[Tuple[int, int]],
        current_step: int
    ) -> float:
        # Find nearest remembered item
        item_memory = memory.recall_nearest(current_pos, MemoryItemType.ITEM, current_step)
        
        if not item_memory:
            return 0.0
        
        item_pos = item_memory.position
        
        # Distance improvement score
        current_dist = abs(current_pos[0] - item_pos[0]) + abs(current_pos[1] - item_pos[1])
        target_dist = abs(target_pos[0] - item_pos[0]) + abs(target_pos[1] - item_pos[1])
        
        if current_dist > 0:
            return (current_dist - target_dist) / current_dist * item_memory.salience
        return 0.0


# ==============================================================================
# AGENT PERSONAS
# ==============================================================================

class AgentPersona(Enum):
    """Pre-defined agent behavior profiles."""
    SPEEDRUNNER = "speedrunner"   # Optimal A* (baseline)
    EXPLORER = "explorer"         # Curiosity-driven, clears all rooms
    CAUTIOUS = "cautious"         # Avoids enemies, prefers safe paths
    FORGETFUL = "forgetful"       # High memory decay, gets lost easily
    BALANCED = "balanced"         # Mix of all heuristics
    COMPLETIONIST = "completionist"  # Collects all items before goal
    NOVICE = "novice"             # Risk-averse, complexity-averse, weak memory
    GREEDY = "greedy"             # Static persona: NO memory decay (Î»=1.0), proves decay is active ingredient


@dataclass
class PersonaConfig:
    """
    Configuration for a bounded-rational player persona.

    The current P-CBS scoring model is richer than a plain goal/risk heuristic.
    At move-scoring time the solver combines:

    U(a) =
        alpha * goal_progress
        + beta * info_gain
        - gamma * risk
        - rho * revisit
        + lambda * loot
        - kappa * combat
        - psi * local_complexity
        - omega * conditional_uncertainty
        + eta * frontier
        + xi * affordance_resumption
        + phi * focus_commitment
        - zeta * affordance_forgetting
        - tau * task_switch
        - frustration_sensitivity * frustration_penalty

    These weights define how each persona trades off progression, exploration,
    uncertainty, memory pressure, and focus persistence.
    """
    name: str
    memory_capacity: int = 7
    memory_decay_rate: float = 0.95
    decay_rate: float = 0.01  # Exponential decay rate Î» for memory
    vision_radius: int = 5
    vision_accuracy: float = 0.9
    vision_cone: float = 360.0
    heuristic_weights: Dict[str, float] = field(default_factory=dict)
    satisficing_threshold: float = 0.8  # Accept "good enough" solutions
    random_tiebreaker: float = 0.1      # Randomness in equal-score decisions
    
    # Utility function weights (Î±, Î², Î³)
    goal_weight: float = 0.6        # Î±: goal progress weight
    curiosity_weight: float = 0.3   # Î²: information gain weight  
    risk_weight: float = 0.1        # Î³: risk avoidance weight
    revisit_penalty_weight: float = 0.15
    loot_weight: float = 0.15
    combat_penalty_weight: float = 0.1
    puzzle_complexity_weight: float = 0.1
    conditional_uncertainty_penalty_weight: float = 0.15
    frontier_bonus_weight: float = 0.1
    affordance_memory_bonus_weight: float = 0.15
    affordance_forgetting_penalty_weight: float = 0.10
    affordance_reactivation_boost: float = 0.30
    deliberation_budget: float = 8.0
    deliberation_recovery: float = 0.35
    deliberation_trigger: float = 0.65
    deliberation_cost_weight: float = 1.0
    frustration_sensitivity: float = 0.2
    focus_commitment_bonus_weight: float = 0.10
    task_switch_penalty_weight: float = 0.05
    
    @classmethod
    def get_persona(cls, persona: AgentPersona) -> 'PersonaConfig':
        """Factory method for predefined personas."""
        
        if persona == AgentPersona.SPEEDRUNNER:
            return cls(
                name="Speedrunner",
                memory_capacity=10,      # Excellent memory
                memory_decay_rate=0.99,  # Very slow decay
                vision_radius=10,        # See far ahead
                vision_cone=360.0,       # Full awareness
                vision_accuracy=0.98,
                heuristic_weights={
                    'goal_seeking': 2.0,
                    'curiosity': 0.1,
                    'safety': 0.0,       # Ignores danger
                    'recency': 0.0,
                    'item_seeking': 0.5,
                },
                satisficing_threshold=1.0,  # Only accepts optimal
                random_tiebreaker=0.0,
                goal_weight=0.8,
                curiosity_weight=0.1,
                risk_weight=0.1,
                revisit_penalty_weight=0.55,
                loot_weight=0.05,
                combat_penalty_weight=0.05,
                puzzle_complexity_weight=0.05,
                conditional_uncertainty_penalty_weight=0.05,
                frontier_bonus_weight=0.0,
                affordance_memory_bonus_weight=0.30,
                affordance_forgetting_penalty_weight=0.05,
                affordance_reactivation_boost=0.40,
                deliberation_budget=10.0,
                deliberation_recovery=0.20,
                deliberation_trigger=0.85,
                deliberation_cost_weight=0.9,
                frustration_sensitivity=0.05,
                focus_commitment_bonus_weight=0.28,
                task_switch_penalty_weight=0.12,
            )
        
        elif persona == AgentPersona.EXPLORER:
            return cls(
                name="Explorer",
                memory_capacity=7,
                memory_decay_rate=0.95,
                decay_rate=0.01,
                vision_radius=5,
                vision_accuracy=0.9,
                vision_cone=360.0,
                heuristic_weights={
                    'curiosity': 2.0,     # Strong exploration drive
                    'goal_seeking': 0.3,  # Weak goal focus
                    'safety': 0.5,
                    'recency': 0.2,
                    'item_seeking': 1.0,  # Likes finding items
                },
                satisficing_threshold=0.7,
                random_tiebreaker=0.2,    # Some randomness
                goal_weight=0.3,      # Î± = 0.3
                curiosity_weight=0.6, # Î² = 0.6
                risk_weight=0.1,      # Î³ = 0.1
                revisit_penalty_weight=0.0,
                loot_weight=0.40,
                combat_penalty_weight=0.05,
                puzzle_complexity_weight=0.05,
                conditional_uncertainty_penalty_weight=0.05,
                frontier_bonus_weight=0.35,
                affordance_memory_bonus_weight=0.25,
                affordance_forgetting_penalty_weight=0.05,
                affordance_reactivation_boost=0.35,
                deliberation_budget=8.5,
                deliberation_recovery=0.45,
                deliberation_trigger=0.55,
                deliberation_cost_weight=0.8,
                frustration_sensitivity=0.10,
                focus_commitment_bonus_weight=0.0,
                task_switch_penalty_weight=0.0,
            )
        
        elif persona == AgentPersona.CAUTIOUS:
            return cls(
                name="Cautious",
                memory_capacity=7,
                memory_decay_rate=0.95,
                decay_rate=0.01,
                vision_radius=5,
                vision_cone=120.0,        # Limited FOV (careful)
                vision_accuracy=0.92,
                heuristic_weights={
                    'safety': 2.0,        # Very safety-conscious
                    'goal_seeking': 0.8,
                    'curiosity': 0.3,
                    'recency': 1.0,       # Prefers known paths
                    'item_seeking': 0.4,
                },
                satisficing_threshold=0.9,
                random_tiebreaker=0.05,
                goal_weight=0.5,      # Î± = 0.5
                curiosity_weight=0.2, # Î² = 0.2
                risk_weight=0.3,      # Î³ = 0.3
                revisit_penalty_weight=0.20,
                loot_weight=0.15,
                combat_penalty_weight=0.55,
                puzzle_complexity_weight=0.35,
                conditional_uncertainty_penalty_weight=0.40,
                frontier_bonus_weight=0.05,
                affordance_memory_bonus_weight=0.20,
                affordance_forgetting_penalty_weight=0.25,
                affordance_reactivation_boost=0.25,
                deliberation_budget=7.5,
                deliberation_recovery=0.30,
                deliberation_trigger=0.50,
                deliberation_cost_weight=1.10,
                frustration_sensitivity=0.30,
                focus_commitment_bonus_weight=0.16,
                task_switch_penalty_weight=0.10,
            )
        
        elif persona == AgentPersona.FORGETFUL:
            return cls(
                name="Forgetful",
                memory_capacity=4,        # Cowan's number
                memory_decay_rate=0.80,   # Fast decay!
                decay_rate=0.03,          # Î» = 0.03 (faster forgetting)
                vision_radius=4,          # Poor awareness
                vision_cone=120.0,
                vision_accuracy=0.85,
                heuristic_weights={
                    'recency': 1.5,       # Relies on recent memory
                    'goal_seeking': 0.5,  # Often forgets goal
                    'curiosity': 0.8,
                    'safety': 0.6,
                    'item_seeking': 0.3,
                },
                satisficing_threshold=0.6,
                random_tiebreaker=0.3,    # More random (confused)
                goal_weight=0.4,      # Î± = 0.4
                curiosity_weight=0.3, # Î² = 0.3
                risk_weight=0.3,      # Î³ = 0.3
                revisit_penalty_weight=0.12,
                loot_weight=0.10,
                combat_penalty_weight=0.25,
                puzzle_complexity_weight=0.30,
                conditional_uncertainty_penalty_weight=0.35,
                frontier_bonus_weight=0.10,
                affordance_memory_bonus_weight=0.12,
                affordance_forgetting_penalty_weight=0.35,
                affordance_reactivation_boost=0.20,
                deliberation_budget=5.5,
                deliberation_recovery=0.25,
                deliberation_trigger=0.45,
                deliberation_cost_weight=1.15,
                frustration_sensitivity=0.35,
                focus_commitment_bonus_weight=0.06,
                task_switch_penalty_weight=0.04,
            )
        
        elif persona == AgentPersona.COMPLETIONIST:
            return cls(
                name="Completionist",
                memory_capacity=10,
                memory_decay_rate=0.98,
                decay_rate=0.01,
                vision_radius=6,
                vision_accuracy=0.92,
                vision_cone=360.0,
                heuristic_weights={
                    'item_seeking': 2.0,  # Must get all items
                    'curiosity': 1.5,     # Must explore everything
                    'goal_seeking': 0.2,  # Goal is last priority
                    'safety': 0.5,
                    'recency': 0.3,
                },
                satisficing_threshold=0.8,
                random_tiebreaker=0.1,
                goal_weight=0.3,      # Low goal priority
                curiosity_weight=0.5, # High exploration
                risk_weight=0.2,
                revisit_penalty_weight=0.05,
                loot_weight=0.80,
                combat_penalty_weight=0.10,
                puzzle_complexity_weight=0.10,
                conditional_uncertainty_penalty_weight=0.10,
                frontier_bonus_weight=0.30,
                affordance_memory_bonus_weight=0.35,
                affordance_forgetting_penalty_weight=0.05,
                affordance_reactivation_boost=0.45,
                deliberation_budget=9.0,
                deliberation_recovery=0.40,
                deliberation_trigger=0.55,
                deliberation_cost_weight=0.85,
                frustration_sensitivity=0.12,
                focus_commitment_bonus_weight=0.18,
                task_switch_penalty_weight=0.05,
            )

        elif persona == AgentPersona.NOVICE:
            return cls(
                name="Novice",
                memory_capacity=5,
                memory_decay_rate=0.86,
                decay_rate=0.025,
                vision_radius=4,
                vision_accuracy=0.86,
                vision_cone=110.0,
                heuristic_weights={
                    'safety': 2.2,
                    'recency': 1.2,
                    'goal_seeking': 0.5,
                    'curiosity': 0.2,
                    'item_seeking': 0.5,
                },
                satisficing_threshold=0.65,
                random_tiebreaker=0.25,
                goal_weight=0.35,
                curiosity_weight=0.15,
                risk_weight=0.50,
                revisit_penalty_weight=0.10,
                loot_weight=0.15,
                combat_penalty_weight=0.75,
                puzzle_complexity_weight=0.65,
                conditional_uncertainty_penalty_weight=0.60,
                frontier_bonus_weight=0.05,
                affordance_memory_bonus_weight=0.10,
                affordance_forgetting_penalty_weight=0.45,
                affordance_reactivation_boost=0.15,
                deliberation_budget=4.5,
                deliberation_recovery=0.20,
                deliberation_trigger=0.35,
                deliberation_cost_weight=1.35,
                frustration_sensitivity=0.45,
                focus_commitment_bonus_weight=0.10,
                task_switch_penalty_weight=0.12,
            )
        
        elif persona == AgentPersona.GREEDY:
            # GREEDY/STATIC BASELINE: No memory decay (Î»=1.0)
            # This persona proves memory decay is the "active ingredient"
            # by comparing performance with decay disabled
            return cls(
                name="Greedy (Static)",
                memory_capacity=7,
                memory_decay_rate=1.0,   # NO DECAY - Î»=1.0 (Ebbinghaus control)
                decay_rate=0.0,          # Zero decay rate
                vision_radius=5,
                vision_accuracy=0.9,
                vision_cone=360.0,
                heuristic_weights={
                    'goal_seeking': 1.5,  # Pure goal-seeking
                    'curiosity': 0.3,
                    'safety': 0.5,
                    'recency': 0.0,       # No recency bias (perfect memory)
                    'item_seeking': 0.5,
                },
                satisficing_threshold=0.9,
                random_tiebreaker=0.05,
                goal_weight=0.7,      # Î± = 0.7 (goal-focused)
                curiosity_weight=0.2, # Î² = 0.2
                risk_weight=0.1,      # Î³ = 0.1
                revisit_penalty_weight=0.35,
                loot_weight=0.10,
                combat_penalty_weight=0.10,
                puzzle_complexity_weight=0.05,
                conditional_uncertainty_penalty_weight=0.05,
                frontier_bonus_weight=0.0,
                affordance_memory_bonus_weight=0.18,
                affordance_forgetting_penalty_weight=0.02,
                affordance_reactivation_boost=0.25,
                deliberation_budget=9.5,
                deliberation_recovery=0.25,
                deliberation_trigger=0.80,
                deliberation_cost_weight=0.75,
                frustration_sensitivity=0.05,
                focus_commitment_bonus_weight=0.22,
                task_switch_penalty_weight=0.08,
            )
        
        else:  # BALANCED
            return cls(
                name="Balanced",
                memory_capacity=7,
                memory_decay_rate=0.95,
                decay_rate=0.01,
                vision_radius=5,
                vision_accuracy=0.9,
                vision_cone=360.0,
                heuristic_weights={
                    'goal_seeking': 1.0,
                    'curiosity': 0.8,
                    'safety': 0.7,
                    'recency': 0.5,
                    'item_seeking': 0.6,
                },
                satisficing_threshold=0.8,
                random_tiebreaker=0.15,
                goal_weight=0.58,      # Î± = 0.58
                curiosity_weight=0.28, # Î² = 0.28
                risk_weight=0.14,      # Î³ = 0.14
                revisit_penalty_weight=0.06,
                loot_weight=0.18,
                combat_penalty_weight=0.12,
                puzzle_complexity_weight=0.04,
                conditional_uncertainty_penalty_weight=0.05,
                frontier_bonus_weight=0.12,
                affordance_memory_bonus_weight=0.10,
                affordance_forgetting_penalty_weight=0.04,
                affordance_reactivation_boost=0.18,
                deliberation_budget=8.5,
                deliberation_recovery=0.40,
                deliberation_trigger=0.48,
                deliberation_cost_weight=0.85,
                frustration_sensitivity=0.10,
                focus_commitment_bonus_weight=0.14,
                task_switch_penalty_weight=0.06,
            )


# Predefined persona configurations dictionary for easy access
PERSONA_CONFIGS: Dict[str, PersonaConfig] = {
    str(persona.value): PersonaConfig.get_persona(persona)
    for persona in AgentPersona
}


# ==============================================================================
# COGNITIVE STATE (COMBINED AGENT STATE)
# ==============================================================================

@dataclass
class CognitiveState:
    """
    Complete cognitive state of the CBS agent.

    Combines game state (position, inventory) with epistemic state
    (beliefs, memory, metacognitive budget, and metrics).
    """
    # Game state (from validator.py)
    game_state: _GameState
    
    # Cognitive components
    belief_map: BeliefMap
    memory: WorkingMemory
    
    # Tracking
    current_step: int = 0
    facing_direction: Tuple[int, int] = (0, 1)  # Default: facing right
    deliberation_budget_remaining: float = 0.0
    frustration: float = 0.0
    active_focus_position: Optional[Tuple[int, int]] = None
    active_focus_kind: str = "none"
    focus_lock_steps: int = 0
    
    # Metrics accumulators
    direction_history: List[Tuple[int, int]] = field(default_factory=list)
    visit_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    def __hash__(self):
        """Hash for state-space search (only game state + step)."""
        return hash((hash(self.game_state), self.current_step))
    
    def copy(self) -> 'CognitiveState':
        """Create a deep copy for branching."""
        return CognitiveState(
            game_state=self.game_state.copy(),
            belief_map=self.belief_map,  # Shared (not copied per-state)
            memory=self.memory,          # Shared
            current_step=self.current_step,
            facing_direction=self.facing_direction,
            deliberation_budget_remaining=self.deliberation_budget_remaining,
            frustration=self.frustration,
            active_focus_position=self.active_focus_position,
            active_focus_kind=self.active_focus_kind,
            focus_lock_steps=self.focus_lock_steps,
            direction_history=list(self.direction_history),
            visit_counts=dict(self.visit_counts),
        )


# ==============================================================================
# COGNITIVE BOUNDED SEARCH SOLVER
# ==============================================================================

class CognitiveBoundedSearch:
    """
    Main P-CBS solver implementing bounded-rational dungeon navigation.

    Unlike StateSpaceAStar which finds optimal paths, P-CBS simulates
    realistic player behavior with:
    - Limited vision (field of view)
    - Decaying memory (forgetting)
    - Bounded working memory
    - Satisficing (accepting "good enough" choices)
    - Persona-conditioned utilities
    - Deliberation budget and frustration
    - Progression-affordance recall and focus persistence
    
    Usage:
        env = ZeldaLogicEnv(semantic_grid=grid)
        cbs = CognitiveBoundedSearch(env, persona='explorer')
        success, path, states, metrics = cbs.solve()
    """
    
    def __init__(
        self,
        env: 'ZeldaLogicEnv',
        persona: AgentPersona | str = AgentPersona.BALANCED,
        timeout: int = 100000,
        seed: Optional[int] = None,
        custom_config: Optional[PersonaConfig] = None,
        representation: str = "hybrid",
    ):
        """
        Initialize CBS solver.
        
        Args:
            env: ZeldaLogicEnv instance to solve
            persona: Agent persona (enum or string name)
            timeout: Maximum steps before giving up
            seed: Random seed for reproducibility
            custom_config: Optional custom PersonaConfig
            representation: Search abstraction hint: tile, graph, or hybrid
        """
        self.env = env
        self.timeout = timeout
        self.seed = seed
        # Instance-local RNG for deterministic tie-breaking without mutating globals.
        self.rng = random.Random(seed)
        
        # Parse persona
        if isinstance(persona, str):
            persona = AgentPersona(persona.lower())
        
        # Get configuration
        if custom_config:
            self.config = custom_config
        else:
            self.config = PersonaConfig.get_persona(persona)
        
        # Alias for compatibility
        self.persona_config = self.config
        rep_raw = str(representation or "hybrid").strip().lower()
        if rep_raw not in {"tile", "graph", "hybrid"}:
            rep_raw = "hybrid"
        self.representation = rep_raw
        
        # Initialize cognitive components
        grid_shape = (env.height, env.width)
        self.belief_map = BeliefMap(
            grid_shape=grid_shape,
            decay_rate=self.config.memory_decay_rate
        )
        self.memory = WorkingMemory(
            capacity=self.config.memory_capacity,
            decay_rate=self.config.memory_decay_rate
        )
        self.vision = VisionSystem(
            radius=self.config.vision_radius,
            cone_angle=self.config.vision_cone,
            enable_occlusion=True
        )
        self._transition_helper = StateSpaceAStar(
            env,
            timeout=max(1, int(timeout)),
            priority_options={
                "representation": self.representation,
                "rules_profile": str(getattr(env, "rules_profile", "vglc_strict") or "vglc_strict"),
                "allow_diagonals": False,
            },
        )
        
        # Initialize heuristics
        self.heuristics: List[DecisionHeuristic] = []
        hw = self.config.heuristic_weights
        
        if hw.get('curiosity', 0) > 0:
            self.heuristics.append(CuriosityHeuristic(hw['curiosity']))
        if hw.get('recency', 0) > 0:
            self.heuristics.append(RecencyHeuristic(hw['recency']))
        if hw.get('safety', 0) > 0:
            self.heuristics.append(SafetyHeuristic(hw['safety']))
        if hw.get('goal_seeking', 0) > 0:
            self.heuristics.append(GoalSeekingHeuristic(hw['goal_seeking']))
        if hw.get('item_seeking', 0) > 0:
            self.heuristics.append(ItemSeekingHeuristic(hw['item_seeking']))
        
        # Metrics tracking
        self._direction_counts: Dict[str, int] = defaultdict(int)
        self._visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._room_visit_counts: Dict[Any, int] = defaultdict(int)
        self._goal_first_seen: int = -1
        self._decisions_made: int = 0
        self._suboptimal_decisions: int = 0
        self._memory_timeline: List[int] = []
        self._deliberation_events: int = 0
        self._budget_exhaustion_events: int = 0
        self._peak_frustration: float = 0.0
        self._affordance_reactivations: int = 0
        self._affordance_guided_steps: int = 0
        self._inventory_change_events: int = 0
        self._focus_switches: int = 0
        self._focus_guided_steps: int = 0
    
    def solve(self) -> Tuple[bool, List[Tuple[int, int]], int, CBSMetrics]:
        """
        Find a path using cognitive bounded search.
        
        Returns:
            success: Whether goal was reached
            path: List of positions visited
            states_explored: Number of decision points evaluated
            metrics: CBSMetrics with cognitive analysis
        """
        self.env.reset()
        self.belief_map.reset()
        self.memory.reset()

        # Reset run-specific metric accumulators to avoid cross-run leakage.
        self._direction_counts.clear()
        self._visit_counts.clear()
        self._room_visit_counts.clear()
        self._goal_first_seen = -1
        self._decisions_made = 0
        self._suboptimal_decisions = 0
        self._memory_timeline = []
        self._deliberation_events = 0
        self._budget_exhaustion_events = 0
        self._peak_frustration = 0.0
        self._affordance_reactivations = 0
        self._affordance_guided_steps = 0
        self._inventory_change_events = 0
        self._focus_switches = 0
        self._focus_guided_steps = 0
        
        # Validation
        if self.env.goal_pos is None:
            return False, [], 0, CBSMetrics()
        if self.env.start_pos is None:
            return False, [], 0, CBSMetrics()
        
        # Initialize cognitive state
        cog_state = CognitiveState(
            game_state=self.env.state.copy(),
            belief_map=self.belief_map,
            memory=self.memory,
            current_step=0,
            facing_direction=(0, 1),
            deliberation_budget_remaining=float(max(0.0, getattr(self.config, 'deliberation_budget', 0.0))),
            frustration=0.0,
            active_focus_position=None,
            active_focus_kind="none",
            focus_lock_steps=0,
        )
        
        path = [cog_state.game_state.position]
        states_explored = 0

        # Include starting position in visit/memory traces for entropy/load metrics.
        self._record_visit(cog_state.game_state.position)
        self._memory_timeline.append(len(self.memory.items))
        
        grid = self.env.original_grid
        
        # Main simulation loop
        while states_explored < self.timeout:
            current_pos = cog_state.game_state.position
            step = cog_state.current_step
            
            # 1. PERCEPTION: Update beliefs from current vision
            self._perceive(cog_state, grid)
            self._refresh_active_focus(cog_state, force=(cog_state.active_focus_position is None))
            
            # 2. CHECK WIN: Did we reach the goal?
            if current_pos == self.env.goal_pos:
                metrics = self._compute_metrics(path, cog_state)
                return True, path, states_explored, metrics
            
            # 3. MEMORY DECAY: Apply forgetting
            self.belief_map.apply_decay(step)
            self.memory.apply_decay(step)
            
            # 4. DECIDE: Choose next move using heuristics
            candidates = self._get_candidate_moves(cog_state, grid)

            if not candidates:
                # Stuck - no valid moves
                logger.debug(f"CBS stuck at {current_pos}, step {step}")
                break

            decision_pressure = self._estimate_decision_pressure(cog_state, candidates)
            should_deliberate = (
                decision_pressure >= float(getattr(self.config, 'deliberation_trigger', 0.0))
                and cog_state.deliberation_budget_remaining > 0.0
            )

            if should_deliberate:
                self._deliberation_events += 1
                budget_before = float(cog_state.deliberation_budget_remaining)
                burn = float(decision_pressure) * float(getattr(self.config, 'deliberation_cost_weight', 1.0))
                cog_state.deliberation_budget_remaining = max(0.0, budget_before - burn)
                if budget_before > 0.0 and cog_state.deliberation_budget_remaining <= 0.0:
                    self._budget_exhaustion_events += 1

                lookahead_depth = max(
                    2,
                    min(8, 2 + int(round(cog_state.deliberation_budget_remaining / 2.0)))
                )
                subgoals = self._generate_subgoals(cog_state, num=4)
                for sg in subgoals:
                    plan = self._plan_short_horizon(cog_state.game_state, sg, max_depth=lookahead_depth)
                    if plan and len(plan) >= 2:
                        first_step = plan[1]
                        candidate = (first_step, grid[first_step[0], first_step[1]])
                        if candidate not in candidates:
                            candidates.append(candidate)
            else:
                budget_cap = float(max(0.0, getattr(self.config, 'deliberation_budget', 0.0)))
                recovery = float(max(0.0, getattr(self.config, 'deliberation_recovery', 0.0)))
                cog_state.deliberation_budget_remaining = min(
                    budget_cap,
                    float(cog_state.deliberation_budget_remaining) + recovery,
                )
            
            # Score all candidates
            scored = []
            for target_pos, target_tile in candidates:
                # Compute approximate expected information gain for moving to target
                facing = (
                    target_pos[0] - current_pos[0],
                    target_pos[1] - current_pos[1],
                )
                info_gain = self.belief_map.expected_info_gain(
                    target_pos,
                    self.vision,
                    grid=grid,
                    direction=facing,
                )
                score = self._score_move(cog_state, target_pos, target_tile, info_gain=info_gain)
                scored.append((score, target_pos, target_tile))
            
            # Sort by score (descending)
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Satisficing: accept if best score meets threshold
            best_score, best_pos, best_tile = scored[0]
            
            # Add randomness for tiebreaking
            if len(scored) > 1:
                threshold = best_score * self.config.satisficing_threshold
                acceptable = [s for s in scored if s[0] >= threshold]
                
                if self.config.random_tiebreaker > 0 and len(acceptable) > 1:
                    # Random selection among acceptable choices
                    if self.rng.random() < self.config.random_tiebreaker:
                        chosen = self.rng.choice(acceptable)
                        best_score, best_pos, best_tile = chosen
            
            # Track decision quality
            self._decisions_made += 1
            if self._estimate_affordance_resumption_bonus(cog_state, best_pos) > 0.0:
                self._affordance_guided_steps += 1
            if self._estimate_focus_commitment_bonus(cog_state, best_pos) > 0.0:
                self._focus_guided_steps += 1
            if self.env.goal_pos:
                current_dist = abs(current_pos[0] - self.env.goal_pos[0]) + \
                              abs(current_pos[1] - self.env.goal_pos[1])
                new_dist = abs(best_pos[0] - self.env.goal_pos[0]) + \
                          abs(best_pos[1] - self.env.goal_pos[1])
                if new_dist > current_dist:
                    self._suboptimal_decisions += 1
            
            # 5. EXECUTE: Resolve the chosen action against the actual environment.
            # Decision-making above is belief-driven; execution is grounded.
            actual_tile = int(grid[best_pos[0], best_pos[1]])
            moved, new_game_state = self._try_move(
                cog_state.game_state, best_pos, actual_tile
            )
            
            if not moved:
                # Move failed (shouldn't happen with valid candidates)
                logger.warning(f"CBS move failed: {current_pos} -> {best_pos}")
                states_explored += 1
                continue

            cog_state.frustration = self._update_frustration(
                cog_state,
                decision_pressure=decision_pressure,
                target_pos=best_pos,
                target_tile=best_tile,
            )
            self._peak_frustration = max(self._peak_frustration, float(cog_state.frustration))
            inventory_changed = self._reactivate_affordances_after_inventory_change(
                previous_state=cog_state.game_state,
                new_state=new_game_state,
                current_step=step,
            )
            
            # Update cognitive state
            direction = (best_pos[0] - current_pos[0], best_pos[1] - current_pos[1])
            dir_name = self._direction_name(direction)
            self._direction_counts[dir_name] += 1
            
            cog_state.game_state = new_game_state
            cog_state.facing_direction = direction
            cog_state.current_step += 1
            self._refresh_active_focus(cog_state, force=inventory_changed)
            
            # Track visits
            self._record_visit(best_pos)
            self._memory_timeline.append(len(self.memory.items))
            
            path.append(best_pos)
            states_explored += 1
        
        # Timeout or stuck
        metrics = self._compute_metrics(path, cog_state)
        return False, path, states_explored, metrics

    def _room_key_for_position(self, position: Tuple[int, int]) -> Any:
        """
        Map a global tile coordinate to the room abstraction used by P-CBS metrics.

        Replay and local movement remain tile-level, but cognitive summary
        metrics should aggregate over rooms. Prefer explicit stitched-dungeon
        room metadata, then fall back to canonical 16x11 Zelda room slots.
        """
        pos = (int(position[0]), int(position[1]))
        try:
            room = self.env.get_room_for_position(pos)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            room = None
        if room is not None:
            if isinstance(room, (tuple, list)) and len(room) >= 2:
                return (int(room[0]), int(room[1]))
            return room
        return (pos[0] // max(1, int(ROOM_HEIGHT)), pos[1] // max(1, int(ROOM_WIDTH)))

    def _record_visit(self, position: Tuple[int, int]) -> None:
        """Track both replay-level tile visits and cognition-level room visits."""
        pos = (int(position[0]), int(position[1]))
        self._visit_counts[pos] += 1
        self._room_visit_counts[self._room_key_for_position(pos)] += 1
    
    def _perceive(self, cog_state: CognitiveState, grid: np.ndarray) -> None:
        """
        Update belief map and memory from current vision.
        """
        pos = cog_state.game_state.position
        step = cog_state.current_step
        direction = cog_state.facing_direction
        
        # Get visible tiles
        visible = self.vision.get_visible_tiles(pos, direction, grid)
        
        for tile_pos in visible:
            tile_type = grid[tile_pos[0], tile_pos[1]]
            is_current = (tile_pos == pos)

            # Bayesian update of belief map using sensor accuracy from persona
            self.belief_map.bayes_update(
                tile_pos, tile_type, step,
                obs_accuracy=getattr(self.config, 'vision_accuracy', 0.9),
                is_visit=is_current
            )
            
            # Update working memory for important discoveries
            if tile_type == SEMANTIC_PALETTE['TRIFORCE']:
                self.memory.remember(
                    MemoryItemType.GOAL, tile_pos, step,
                    salience_boost=0.5
                )
                if self._goal_first_seen < 0:
                    self._goal_first_seen = step
            
            elif tile_type in PICKUP_IDS:
                if tile_pos not in cog_state.game_state.collected_items:
                    self.memory.remember(
                        MemoryItemType.ITEM, tile_pos, step,
                        data={'tile_type': tile_type}
                    )
            
            elif tile_type in {SEMANTIC_PALETTE['ENEMY'], SEMANTIC_PALETTE['BOSS']}:
                self.memory.remember(
                    MemoryItemType.THREAT, tile_pos, step
                )
            
            elif tile_type in CONDITIONAL_IDS:
                if tile_pos not in cog_state.game_state.opened_doors:
                    self.memory.remember(
                        MemoryItemType.DOOR, tile_pos, step,
                        data={'tile_type': tile_type}
                    )
                    self.memory.remember(
                        MemoryItemType.AFFORDANCE,
                        tile_pos,
                        step,
                        data={
                            'tile_type': int(tile_type),
                            'requirement': self._affordance_requirement(int(tile_type)),
                        },
                        salience_boost=0.10,
                    )
            elif tile_type == SEMANTIC_PALETTE['PUZZLE']:
                self.memory.remember(
                    MemoryItemType.AFFORDANCE,
                    tile_pos,
                    step,
                    data={
                        'tile_type': int(tile_type),
                        'requirement': self._affordance_requirement(int(tile_type)),
                    },
                    salience_boost=0.05,
                )
    
    def _get_candidate_moves(
        self,
        cog_state: CognitiveState,
        grid: np.ndarray
    ) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid moves from current position.
        
        Returns list of (target_pos, target_tile) tuples.
        """
        candidates = []
        pos = cog_state.game_state.position
        height, width = grid.shape
        
        for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            dr, dc = ACTION_DELTAS[action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            
            # Bounds check
            if not (0 <= new_pos[0] < height and 0 <= new_pos[1] < width):
                continue
            
            # Use the agent's belief state rather than hidden global knowledge.
            tile_type, _confidence = self.belief_map.get_tile_with_confidence(new_pos)
            tile_type = int(tile_type)
            
            # Check if move is possible
            if self._can_move_to(cog_state.game_state, new_pos, tile_type):
                candidates.append((new_pos, tile_type))

        candidates.extend(self._get_graph_transition_candidates(cog_state, grid, candidates))
        
        return candidates

    def _is_at_room_boundary(self, position: Tuple[int, int]) -> bool:
        """Return True when a position is close enough to a room edge to transition."""
        room_positions = getattr(self.env, "room_positions", None)
        if not room_positions:
            return False
        room = self._room_key_for_position(position)
        if room not in room_positions:
            return False
        r_off, c_off = room_positions[room]
        local_r = int(position[0]) - int(r_off)
        local_c = int(position[1]) - int(c_off)
        return (
            local_r <= 1
            or local_r >= int(ROOM_HEIGHT) - 2
            or local_c <= 1
            or local_c >= int(ROOM_WIDTH) - 2
        )

    def _get_graph_transition_candidates(
        self,
        cog_state: CognitiveState,
        grid: np.ndarray,
        existing_candidates: List[Tuple[Tuple[int, int], int]],
    ) -> List[Tuple[Tuple[int, int], int]]:
        """
        Add room-graph transitions when the current position is a valid doorway/stair.

        P-CBS still outputs replayable tile positions, but in hybrid/graph mode
        it should use the same room connectivity as the hard solvers instead of
        treating the stitched text grid as the whole world.
        """
        if self.representation == "tile":
            return []
        if not getattr(self.env, "graph", None) or not getattr(self.env, "room_to_node", None):
            return []

        state = cog_state.game_state
        current_pos = tuple(int(v) for v in state.position)
        current_tile = int(grid[current_pos[0], current_pos[1]])
        is_stair = current_tile == int(SEMANTIC_PALETTE["STAIR"])
        is_door = current_tile in {
            int(SEMANTIC_PALETTE["DOOR_OPEN"]),
            int(SEMANTIC_PALETTE["DOOR_SOFT"]),
            int(SEMANTIC_PALETTE["DOOR_LOCKED"]),
            int(SEMANTIC_PALETTE["DOOR_BOMB"]),
            int(SEMANTIC_PALETTE["DOOR_BOSS"]),
        }

        strict_original = bool(getattr(self.env, "strict_original_mode", False))
        vglc_strict = bool(getattr(self.env, "vglc_strict_mode", False))
        if strict_original:
            can_transition = is_stair
        elif vglc_strict:
            can_transition = is_stair or is_door
        else:
            can_transition = is_stair or is_door or self._is_at_room_boundary(current_pos)

        transition_positions: List[Tuple[int, int]] = []
        if is_stair:
            try:
                transition_positions.extend(self._transition_helper.get_stair_destinations(current_pos))
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

        if can_transition and not strict_original:
            try:
                transition_positions.extend(
                    pos
                    for pos, _cost, _edge_type in self._transition_helper.get_controlled_virtual_destinations(
                        current_pos,
                        state,
                    )
                )
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            try:
                transition_positions.extend(
                    pos
                    for pos, _cost, _edge_type in self._transition_helper.get_graph_warp_destinations(
                        current_pos,
                        state,
                    )
                )
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

        if not transition_positions:
            return []

        seen = {tuple(pos) for pos, _tile in existing_candidates}
        height, width = grid.shape
        candidates: List[Tuple[Tuple[int, int], int]] = []
        for raw_pos in transition_positions:
            pos = (int(raw_pos[0]), int(raw_pos[1]))
            if pos in seen or pos == current_pos:
                continue
            if not (0 <= pos[0] < height and 0 <= pos[1] < width):
                continue
            tile = int(grid[pos[0], pos[1]])
            try:
                can_move, _new_state = self.env.try_move_pure(state, pos, tile)
            except (AttributeError, RuntimeError, ValueError, TypeError):
                can_move, _new_state = self._try_move(state, pos, tile)
            if not can_move:
                continue
            seen.add(pos)
            candidates.append((pos, tile))

        return candidates

    def _plan_short_horizon(
        self,
        start_state: _GameState,
        goal_pos: Tuple[int, int],
        max_depth: int = 6
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Bounded breadth-first planner over game states (considers inventory)
        up to `max_depth` steps. Returns a path of positions if goal reachable
        within depth, otherwise None.
        """

        if start_state.position == goal_pos:
            return [start_state.position]

        grid = self.env.original_grid
        height, width = grid.shape
        q = deque()
        q.append((start_state.copy(), [start_state.position]))
        visited = set()

        depth = 0
        while q and depth <= max_depth:
            for _ in range(len(q)):
                state, path = q.popleft()
                if state.position == goal_pos:
                    return path

                if len(path) > max_depth:
                    continue

                pos = state.position
                for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                    dr, dc = ACTION_DELTAS[action]
                    new_pos = (pos[0] + dr, pos[1] + dc)
                    if not (0 <= new_pos[0] < height and 0 <= new_pos[1] < width):
                        continue
                    tile, confidence = self.belief_map.get_tile_with_confidence(new_pos)
                    tile = int(tile)
                    if confidence <= 0.0:
                        # Unknown space is not a valid short-horizon plan segment.
                        continue
                    if not self._can_move_to(state, new_pos, tile):
                        continue
                    moved, new_state = self._try_move(state, new_pos, tile)
                    if not moved:
                        continue
                    state_key = game_state_key(new_state)
                    if state_key in visited:
                        continue
                    visited.add(state_key)
                    q.append((new_state, path + [new_pos]))
            depth += 1

        return None

    def _generate_subgoals(self, cog_state: CognitiveState, num: int = 5) -> List[Tuple[int, int]]:
        """
        Generate candidate subgoals using mission grammar and belief map.

        Strategy (compact):
        - Use MissionGrammar to get desired node types
        - Map node types to nearest remembered/observed positions
        - Fallback to frontier (observed-but-unvisited) and unknown tile centers
        """
        try:
            from src.generation.grammar import MissionGrammar, NodeType
        except Exception:
            MissionGrammar = None

        candidates: List[Tuple[int, int]] = []

        # 1) Goal if remembered
        goal_mem = self.memory.recall_nearest(cog_state.game_state.position, MemoryItemType.GOAL, cog_state.current_step)
        if goal_mem:
            candidates.append(goal_mem.position)

        # 2) Items remembered
        items = self.memory.recall(MemoryItemType.ITEM, cog_state.current_step)
        items_sorted = sorted(items, key=lambda m: abs(m.position[0] - cog_state.game_state.position[0]) + abs(m.position[1] - cog_state.game_state.position[1]))
        for it in items_sorted:
            if len(candidates) >= num:
                break
            candidates.append(it.position)

        # 3) Use grammar to propose types and map to observed positions
        if MissionGrammar is not None:
            try:
                g = MissionGrammar(seed=42)
                graph = g.generate(num_rooms=6)
                # Map KEY nodes to observed key tiles
                for _node in graph.get_nodes_by_type(NodeType.KEY):
                    # find nearest observed key in belief_map
                    best = None
                    bestd = 1e9
                    for pos, obs in self.belief_map.known_tiles.items():
                        if obs.tile_type in PICKUP_IDS:
                            d = abs(pos[0] - cog_state.game_state.position[0]) + abs(pos[1] - cog_state.game_state.position[1])
                            if d < bestd:
                                bestd = d
                                best = pos
                    if best and best not in candidates:
                        candidates.append(best)
                        if len(candidates) >= num:
                            break
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.debug("MissionGrammar-assisted subgoal generation unavailable; using belief-map fallback: %s", exc)

        # 4) Frontier (observed but unvisited)
        current = tuple(int(v) for v in cog_state.game_state.position)
        for pos in sorted(
            self.belief_map.get_frontier(),
            key=lambda p: (abs(int(p[0]) - current[0]) + abs(int(p[1]) - current[1]), int(p[0]), int(p[1])),
        ):
            if len(candidates) >= num:
                break
            if pos not in candidates:
                candidates.append(pos)

        # 5) Fill with nearest unknown tiles if still short
        if len(candidates) < num:
            # search nearby tiles ranking by low confidence
            low_conf = sorted(
                self.belief_map.known_tiles.items(),
                key=lambda kv: (
                    float(kv[1].confidence),
                    abs(int(kv[0][0]) - current[0]) + abs(int(kv[0][1]) - current[1]),
                    int(kv[0][0]),
                    int(kv[0][1]),
                ),
            )
            for pos, obs in low_conf:
                if pos not in candidates:
                    candidates.append(pos)
                if len(candidates) >= num:
                    break

        return candidates
    
    def _can_move_to(
        self,
        game_state: _GameState,
        target_pos: Tuple[int, int],
        tile_type: int
    ) -> bool:
        """Check if a move is valid."""
        if tile_type in BLOCKING_IDS:
            return False
        
        if tile_type in WALKABLE_IDS:
            return True
        
        # Conditional tiles
        if tile_type == SEMANTIC_PALETTE['DOOR_LOCKED']:
            return game_state.keys > 0 or target_pos in game_state.opened_doors
        if tile_type == SEMANTIC_PALETTE['DOOR_BOMB']:
            return game_state.bomb_count > 0 or target_pos in game_state.opened_doors
        if tile_type == SEMANTIC_PALETTE['DOOR_BOSS']:
            return game_state.has_boss_key or target_pos in game_state.opened_doors
        
        return True

    def _inventory_signature(self, game_state: _GameState) -> Tuple[int, int, int, int]:
        """Compact signature for progression-relevant inventory state."""
        return (
            int(game_state.keys),
            int(game_state.bomb_count),
            int(bool(game_state.has_boss_key)),
            int(bool(game_state.has_item)),
        )

    def _affordance_requirement(self, tile_id: int) -> str:
        """Map a conditional or puzzle tile to its remembered requirement family."""
        tile_id = int(tile_id)
        if tile_id == int(SEMANTIC_PALETTE['DOOR_LOCKED']):
            return "small_key"
        if tile_id == int(SEMANTIC_PALETTE['DOOR_BOMB']):
            return "bomb"
        if tile_id == int(SEMANTIC_PALETTE['DOOR_BOSS']):
            return "boss_key"
        if tile_id == int(SEMANTIC_PALETTE['PUZZLE']):
            return "item_or_observation"
        return "none"

    def _is_affordance_satisfied(self, game_state: _GameState, affordance: MemoryItem) -> bool:
        """Check whether a remembered progression affordance is currently actionable."""
        requirement = str((affordance.data or {}).get('requirement', 'none'))
        pos = tuple(int(v) for v in affordance.position)
        if requirement == "small_key":
            return int(game_state.keys) > 0 or pos in game_state.opened_doors
        if requirement == "bomb":
            return int(game_state.bomb_count) > 0 or pos in game_state.opened_doors
        if requirement == "boss_key":
            return bool(game_state.has_boss_key) or pos in game_state.opened_doors
        if requirement == "item_or_observation":
            return bool(game_state.has_item) or bool(game_state.has_boss_key) or int(game_state.keys) > 0 or int(game_state.bomb_count) > 0
        return False

    def _reactivate_affordances_after_inventory_change(
        self,
        *,
        previous_state: _GameState,
        new_state: _GameState,
        current_step: int,
    ) -> bool:
        """
        Boost remembered unresolved affordances when inventory changes.

        This models the human tendency to recall a previously seen blocked gate
        after finding the relevant key, bomb, or item.
        """
        if self._inventory_signature(previous_state) == self._inventory_signature(new_state):
            return False

        self._inventory_change_events += 1
        boost = float(max(0.0, getattr(self.config, 'affordance_reactivation_boost', 0.0)))
        if boost <= 0.0:
            return True

        reactivated = 0
        for item in self.memory.items:
            if item.item_type != MemoryItemType.AFFORDANCE:
                continue
            if self._is_affordance_satisfied(new_state, item) and not self._is_affordance_satisfied(previous_state, item):
                item.salience = min(1.0, float(item.salience) + boost)
                item.last_accessed = current_step
                reactivated += 1
        if reactivated > 0:
            self._affordance_reactivations += int(reactivated)
            self.memory._sort_by_salience()
        return True

    def _try_move(
        self,
        game_state: _GameState,
        target_pos: Tuple[int, int],
        tile_type: int
    ) -> Tuple[bool, _GameState]:
        """Execute a move and return new game state."""
        new_state = game_state.copy()
        
        # Handle conditional tiles
        if tile_type == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if target_pos not in new_state.opened_doors:
                if new_state.keys > 0:
                    new_state.keys -= 1
                    new_state.opened_doors.add(target_pos)
                else:
                    return False, game_state
        
        elif tile_type == SEMANTIC_PALETTE['DOOR_BOMB']:
            if target_pos not in new_state.opened_doors:
                if new_state.bomb_count > 0:
                    new_state.bomb_count -= 1  # Consume one bomb
                    new_state.opened_doors.add(target_pos)
                else:
                    return False, game_state
        
        elif tile_type == SEMANTIC_PALETTE['DOOR_BOSS']:
            if target_pos not in new_state.opened_doors:
                if new_state.has_boss_key:
                    new_state.opened_doors.add(target_pos)
                else:
                    return False, game_state
        
        # Handle pickups
        if tile_type in PICKUP_IDS and target_pos not in new_state.collected_items:
            new_state.collected_items.add(target_pos)
            
            if tile_type == SEMANTIC_PALETTE['KEY_SMALL']:
                new_state.keys += 1
            elif tile_type == SEMANTIC_PALETTE['KEY_BOSS']:
                new_state.has_boss_key = True
            elif tile_type == SEMANTIC_PALETTE['KEY_ITEM']:
                new_state.has_item = True
                new_state.bomb_count += 4  # Consumable bombs
            elif tile_type == SEMANTIC_PALETTE['ITEM_MINOR']:
                new_state.bomb_count += 4  # Consumable: add 4 bombs
        
        new_state.position = target_pos
        return True, new_state
    
    def _score_move(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
        target_tile: int
        , info_gain: float = 0.0
    ) -> float:
        """
        Compute aggregate score for a potential move.
        
        Combines all heuristics according to persona weights.
        """
        total_score = 0.0
        total_weight = 0.0
        
        for heuristic in self.heuristics:
            score = heuristic.score(
                current_pos=cog_state.game_state.position,
                target_pos=target_pos,
                target_tile=target_tile,
                belief_map=self.belief_map,
                memory=self.memory,
                goal_pos=self.env.goal_pos,
                current_step=cog_state.current_step
            )
            total_score += score * heuristic.weight
            total_weight += heuristic.weight
        
        base = (total_score / total_weight) if total_weight > 0 else 0.0

        # Explicit bounded-rational utility:
        # U = Î±Â·goal_progress + Î²Â·info_gain - Î³Â·risk
        goal_progress = 0.0
        if self.env.goal_pos is not None:
            curr = cog_state.game_state.position
            current_dist = abs(curr[0] - self.env.goal_pos[0]) + abs(curr[1] - self.env.goal_pos[1])
            target_dist = abs(target_pos[0] - self.env.goal_pos[0]) + abs(target_pos[1] - self.env.goal_pos[1])
            if current_dist > 0:
                goal_progress = (current_dist - target_dist) / current_dist
            elif target_dist == 0:
                goal_progress = 1.0

        risk = self._estimate_risk(cog_state, target_pos, target_tile)

        alpha = float(getattr(self.config, 'goal_weight', 0.6))
        beta = float(getattr(self.config, 'curiosity_weight', 0.3))
        gamma = float(getattr(self.config, 'risk_weight', 0.1))
        revisit_weight = float(getattr(self.config, 'revisit_penalty_weight', 0.0))
        uncertainty_weight = float(getattr(self.config, 'conditional_uncertainty_penalty_weight', 0.0))
        puzzle_complexity_weight = float(getattr(self.config, 'puzzle_complexity_weight', 0.0))
        affordance_bonus_weight = float(getattr(self.config, 'affordance_memory_bonus_weight', 0.0))
        affordance_forgetting_weight = float(getattr(self.config, 'affordance_forgetting_penalty_weight', 0.0))
        focus_commitment_weight = float(getattr(self.config, 'focus_commitment_bonus_weight', 0.0))
        task_switch_weight = float(getattr(self.config, 'task_switch_penalty_weight', 0.0))

        revisit_penalty = self._estimate_revisit_penalty(target_pos) if revisit_weight > 0.0 else 0.0
        loot_value = self._estimate_loot_value(cog_state, target_pos, target_tile)
        combat_penalty = self._estimate_combat_penalty(target_tile)
        puzzle_complexity = self._estimate_local_complexity(target_pos) if puzzle_complexity_weight > 0.0 else 0.0
        conditional_uncertainty = (
            self._estimate_conditional_uncertainty(cog_state, target_pos, target_tile)
            if uncertainty_weight > 0.0 else 0.0
        )
        frontier_bonus = self._estimate_frontier_bonus(target_pos)
        affordance_bonus = (
            self._estimate_affordance_resumption_bonus(cog_state, target_pos)
            if affordance_bonus_weight > 0.0 else 0.0
        )
        affordance_forgetting_penalty = (
            self._estimate_affordance_forgetting_penalty(cog_state, target_pos, target_tile)
            if affordance_forgetting_weight > 0.0 else 0.0
        )
        focus_bonus = (
            self._estimate_focus_commitment_bonus(cog_state, target_pos)
            if focus_commitment_weight > 0.0 else 0.0
        )
        task_switch_penalty = (
            self._estimate_task_switch_penalty(cog_state, target_pos)
            if task_switch_weight > 0.0 else 0.0
        )
        frustration_penalty = float(cog_state.frustration) * (
            revisit_penalty + conditional_uncertainty + affordance_forgetting_penalty + task_switch_penalty + 0.5 * combat_penalty
        )

        utility = (
            alpha * goal_progress
            + beta * info_gain
            - gamma * risk
            - revisit_weight * revisit_penalty
            + float(getattr(self.config, 'loot_weight', 0.0)) * loot_value
            - float(getattr(self.config, 'combat_penalty_weight', 0.0)) * combat_penalty
            - puzzle_complexity_weight * puzzle_complexity
            - uncertainty_weight * conditional_uncertainty
            + float(getattr(self.config, 'frontier_bonus_weight', 0.0)) * frontier_bonus
            + affordance_bonus_weight * affordance_bonus
            + focus_commitment_weight * focus_bonus
            - affordance_forgetting_weight * affordance_forgetting_penalty
            - task_switch_weight * task_switch_penalty
            - float(getattr(self.config, 'frustration_sensitivity', 0.0)) * frustration_penalty
        )

        return base + utility

    def _estimate_risk(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
        target_tile: int,
    ) -> float:
        """Estimate movement risk from tile hazards and remembered nearby threats."""
        risk = 0.0

        # Immediate tile risk
        if target_tile in {
            SEMANTIC_PALETTE['ENEMY'],
            SEMANTIC_PALETTE['BOSS'],
            SEMANTIC_PALETTE['ELEMENT'],
        }:
            risk = max(risk, 1.0)

        # Memory-based nearby threat risk
        for threat in self.memory.recall(MemoryItemType.THREAT, cog_state.current_step):
            d = abs(threat.position[0] - target_pos[0]) + abs(threat.position[1] - target_pos[1])
            if d == 0:
                risk = max(risk, 1.0)
            elif d == 1:
                risk = max(risk, 0.7 * float(threat.salience))
            elif d == 2:
                risk = max(risk, 0.4 * float(threat.salience))

        return float(min(1.5, risk))

    def _estimate_revisit_penalty(self, target_pos: Tuple[int, int]) -> float:
        """Penalty for revisiting already explored tiles."""
        visits = int(self._visit_counts.get(tuple(target_pos), 0))
        if visits <= 0:
            return 0.0
        return float(min(1.5, visits / 2.0))

    def _estimate_loot_value(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
        target_tile: int,
    ) -> float:
        """Reward immediate collectible progress for loot-seeking personas."""
        if int(target_tile) not in PICKUP_IDS:
            return 0.0
        if tuple(target_pos) in cog_state.game_state.collected_items:
            return 0.0
        if int(target_tile) == int(SEMANTIC_PALETTE['KEY_BOSS']):
            return 1.0
        if int(target_tile) in {int(SEMANTIC_PALETTE['KEY_SMALL']), int(SEMANTIC_PALETTE['KEY_ITEM'])}:
            return 0.8
        return 0.5

    def _estimate_combat_penalty(self, target_tile: int) -> float:
        """Persona-specific aversion to entering direct combat tiles."""
        if int(target_tile) == int(SEMANTIC_PALETTE['BOSS']):
            return 1.0
        if int(target_tile) == int(SEMANTIC_PALETTE['ENEMY']):
            return 0.7
        return 0.0

    def _estimate_frontier_bonus(self, target_pos: Tuple[int, int]) -> float:
        """Reward moving onto the current frontier of partially known space."""
        frontier = self.belief_map.get_frontier()
        return 1.0 if tuple(target_pos) in frontier else 0.0

    def _progress_ratio(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        focus_pos: Tuple[int, int],
    ) -> float:
        """Normalized Manhattan progress from current_pos -> target_pos toward focus_pos."""
        current_dist = abs(int(current_pos[0]) - int(focus_pos[0])) + abs(int(current_pos[1]) - int(focus_pos[1]))
        target_dist = abs(int(target_pos[0]) - int(focus_pos[0])) + abs(int(target_pos[1]) - int(focus_pos[1]))
        if current_dist <= 0:
            return 1.0 if target_dist <= 0 else 0.0
        return float(max(0.0, float(current_dist - target_dist) / float(current_dist)))

    def _collect_focus_targets(
        self,
        cog_state: CognitiveState,
    ) -> List[Tuple[str, Tuple[int, int], float]]:
        """Collect meaningful cognitive targets that can anchor a short-term objective."""
        targets: List[Tuple[str, Tuple[int, int], float]] = []
        seen: Set[Tuple[str, Tuple[int, int]]] = set()

        def add_target(kind: str, pos: Tuple[int, int], salience: float) -> None:
            key = (str(kind), tuple(int(v) for v in pos))
            if key in seen:
                return
            seen.add(key)
            targets.append((str(kind), key[1], float(max(0.0, salience))))

        if self.env.goal_pos is not None and tuple(cog_state.game_state.position) != tuple(self.env.goal_pos):
            goal_seen = self._goal_first_seen >= 0 or any(
                item.item_type == MemoryItemType.GOAL for item in self.memory.items
            )
            add_target("goal", tuple(int(v) for v in self.env.goal_pos), 1.0 if goal_seen else 0.65)

        for item in self.memory.recall(MemoryItemType.ITEM, cog_state.current_step):
            pos = tuple(int(v) for v in item.position)
            if pos in cog_state.game_state.collected_items:
                continue
            add_target("item", pos, max(0.25, float(item.salience)))

        for item in self.memory.recall(MemoryItemType.AFFORDANCE, cog_state.current_step):
            pos = tuple(int(v) for v in item.position)
            if not self._is_affordance_satisfied(cog_state.game_state, item):
                continue
            add_target("affordance", pos, max(0.25, float(item.salience)))

        return targets

    def _focus_target_priority(
        self,
        cog_state: CognitiveState,
        kind: str,
        position: Tuple[int, int],
        salience: float,
    ) -> float:
        """Rank candidate short-term objectives by persona bias, salience, and distance."""
        current_pos = cog_state.game_state.position
        distance = abs(int(current_pos[0]) - int(position[0])) + abs(int(current_pos[1]) - int(position[1]))
        if kind == "goal":
            base = 0.55 + float(getattr(self.config, 'goal_weight', 0.0))
        elif kind == "affordance":
            base = 0.20 + 0.85 * float(getattr(self.config, 'affordance_memory_bonus_weight', 0.0)) + 0.25 * float(getattr(self.config, 'goal_weight', 0.0))
        elif kind == "item":
            base = 0.10 + 0.75 * float(getattr(self.config, 'loot_weight', 0.0)) + 0.35 * float(getattr(self.config, 'curiosity_weight', 0.0))
        else:
            base = 0.10
        distance_discount = 1.0 / (1.0 + 0.08 * float(max(0, distance)))
        return float(base * float(max(0.0, salience)) * distance_discount)

    def _recommend_focus_target(
        self,
        cog_state: CognitiveState,
    ) -> Optional[Tuple[str, Tuple[int, int], float]]:
        """Choose the best current short-term objective under bounded cognition."""
        candidates = self._collect_focus_targets(cog_state)
        if not candidates:
            return None

        best: Optional[Tuple[str, Tuple[int, int], float]] = None
        best_priority = -float("inf")
        for kind, position, salience in candidates:
            priority = self._focus_target_priority(cog_state, kind, position, salience)
            if priority > best_priority:
                best_priority = priority
                best = (kind, position, salience)
        return best

    def _is_focus_target_valid(
        self,
        cog_state: CognitiveState,
        *,
        kind: str,
        position: Optional[Tuple[int, int]],
    ) -> bool:
        """Check whether the currently active focus still makes sense."""
        if position is None:
            return False
        position = tuple(int(v) for v in position)
        if tuple(cog_state.game_state.position) == position:
            return False

        if kind == "goal":
            return self.env.goal_pos is not None and tuple(self.env.goal_pos) == position
        if kind == "item":
            if position in cog_state.game_state.collected_items:
                return False
            return any(
                item.item_type == MemoryItemType.ITEM and tuple(int(v) for v in item.position) == position
                for item in self.memory.items
            )
        if kind == "affordance":
            return any(
                item.item_type == MemoryItemType.AFFORDANCE
                and tuple(int(v) for v in item.position) == position
                and self._is_affordance_satisfied(cog_state.game_state, item)
                for item in self.memory.items
            )
        return False

    def _refresh_active_focus(
        self,
        cog_state: CognitiveState,
        *,
        force: bool = False,
    ) -> None:
        """Maintain a bounded short-term objective with optional forced retargeting."""
        current_valid = self._is_focus_target_valid(
            cog_state,
            kind=str(cog_state.active_focus_kind or "none"),
            position=cog_state.active_focus_position,
        )
        recommended = self._recommend_focus_target(cog_state)

        if recommended is None:
            if not current_valid:
                cog_state.active_focus_position = None
                cog_state.active_focus_kind = "none"
                cog_state.focus_lock_steps = 0
            else:
                cog_state.focus_lock_steps += 1
            return

        recommended_kind, recommended_position, _recommended_salience = recommended
        if not current_valid:
            if cog_state.active_focus_position is not None and (
                tuple(int(v) for v in cog_state.active_focus_position) != tuple(int(v) for v in recommended_position)
                or str(cog_state.active_focus_kind) != str(recommended_kind)
            ):
                self._focus_switches += 1
            cog_state.active_focus_position = tuple(int(v) for v in recommended_position)
            cog_state.active_focus_kind = str(recommended_kind)
            cog_state.focus_lock_steps = 0
            return

        active_position = tuple(int(v) for v in (cog_state.active_focus_position or ()))
        same_focus = (
            str(cog_state.active_focus_kind) == str(recommended_kind)
            and active_position == tuple(int(v) for v in recommended_position)
        )
        if same_focus:
            cog_state.focus_lock_steps += 1
            return

        if force or float(cog_state.frustration) >= 0.9:
            self._focus_switches += 1
            cog_state.active_focus_position = tuple(int(v) for v in recommended_position)
            cog_state.active_focus_kind = str(recommended_kind)
            cog_state.focus_lock_steps = 0
        else:
            cog_state.focus_lock_steps += 1

    def _estimate_focus_commitment_bonus(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
    ) -> float:
        """Reward moves that continue progress toward the currently committed objective."""
        if not self._is_focus_target_valid(
            cog_state,
            kind=str(cog_state.active_focus_kind or "none"),
            position=cog_state.active_focus_position,
        ):
            return 0.0
        focus_pos = tuple(int(v) for v in (cog_state.active_focus_position or ()))
        progress = self._progress_ratio(cog_state.game_state.position, target_pos, focus_pos)
        persistence = min(1.5, 1.0 + 0.03 * float(max(0, cog_state.focus_lock_steps)))
        return float(min(1.5, progress * persistence))

    def _estimate_task_switch_penalty(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
    ) -> float:
        """
        Penalize abandoning a viable active objective for a different local target.

        This models human planning rigidity and switching costs rather than
        treating every step as an independent scalarized choice.
        """
        if not self._is_focus_target_valid(
            cog_state,
            kind=str(cog_state.active_focus_kind or "none"),
            position=cog_state.active_focus_position,
        ):
            return 0.0

        active_position = tuple(int(v) for v in (cog_state.active_focus_position or ()))
        active_progress = self._progress_ratio(cog_state.game_state.position, target_pos, active_position)
        best_other_progress = 0.0
        for kind, position, _salience in self._collect_focus_targets(cog_state):
            candidate_position = tuple(int(v) for v in position)
            if candidate_position == active_position and str(kind) == str(cog_state.active_focus_kind):
                continue
            best_other_progress = max(
                best_other_progress,
                self._progress_ratio(cog_state.game_state.position, target_pos, candidate_position),
            )

        if best_other_progress <= active_progress:
            return 0.0

        frustration_relief = min(1.0, float(max(0.0, cog_state.frustration)) / 1.5)
        rigidity = min(1.5, 0.45 + 0.05 * float(max(0, cog_state.focus_lock_steps)))
        penalty = (best_other_progress - active_progress) * rigidity * (1.0 - frustration_relief)
        return float(min(1.5, max(0.0, penalty)))

    def _estimate_affordance_resumption_bonus(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
    ) -> float:
        """
        Reward moves that resume toward remembered, now-actionable progression affordances.

        This is the Zelda-specific bounded-rational extension beyond generic
        revisit and uncertainty penalties.
        """
        current_pos = cog_state.game_state.position
        memories = self.memory.recall(MemoryItemType.AFFORDANCE, cog_state.current_step)
        actionable = [item for item in memories if self._is_affordance_satisfied(cog_state.game_state, item)]
        if not actionable:
            return 0.0

        best_bonus = 0.0
        for item in actionable:
            aff_pos = tuple(int(v) for v in item.position)
            current_dist = abs(current_pos[0] - aff_pos[0]) + abs(current_pos[1] - aff_pos[1])
            target_dist = abs(int(target_pos[0]) - aff_pos[0]) + abs(int(target_pos[1]) - aff_pos[1])
            if current_dist <= 0:
                best_bonus = max(best_bonus, float(item.salience))
                continue
            progress = max(0.0, float(current_dist - target_dist) / float(current_dist))
            best_bonus = max(best_bonus, progress * float(item.salience))
        return float(min(1.5, best_bonus))

    def _estimate_affordance_forgetting_penalty(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
        target_tile: int,
    ) -> float:
        """
        Penalize stepping into unresolved conditional structure without strong
        episodic support in memory.
        """
        tile_id = int(target_tile)
        if tile_id not in CONDITIONAL_IDS and tile_id != int(SEMANTIC_PALETTE['PUZZLE']):
            return 0.0

        matching = [
            item for item in self.memory.recall(MemoryItemType.AFFORDANCE, cog_state.current_step)
            if tuple(int(v) for v in item.position) == tuple(int(v) for v in target_pos)
        ]
        if not matching:
            return 1.0

        salience = max(float(item.salience) for item in matching)
        penalty = 1.0 - float(max(0.0, min(1.0, salience)))
        if not any(self._is_affordance_satisfied(cog_state.game_state, item) for item in matching):
            penalty += 0.35
        return float(min(1.5, penalty))

    def _estimate_local_complexity(self, target_pos: Tuple[int, int]) -> float:
        """Approximate room-local puzzle/branching complexity around a move."""
        grid = self.env.original_grid
        row, col = int(target_pos[0]), int(target_pos[1])
        walkable_neighbors = 0
        local_hazards = 0
        local_conditionals = 0

        for d_r, d_c in ACTION_DELTAS.values():
            next_r = row + int(d_r)
            next_c = col + int(d_c)
            if not (0 <= next_r < grid.shape[0] and 0 <= next_c < grid.shape[1]):
                continue
            tile = int(grid[next_r, next_c])
            if tile in WALKABLE_IDS or tile in CONDITIONAL_IDS or tile in PICKUP_IDS:
                walkable_neighbors += 1
            if tile in {int(SEMANTIC_PALETTE['ENEMY']), int(SEMANTIC_PALETTE['BOSS'])}:
                local_hazards += 1
            if tile in CONDITIONAL_IDS or tile == int(SEMANTIC_PALETTE['PUZZLE']):
                local_conditionals += 1

        complexity = 0.0
        if walkable_neighbors >= 3:
            complexity += 0.35
        complexity += 0.20 * float(local_hazards)
        complexity += 0.15 * float(local_conditionals)
        return float(min(1.5, complexity))

    def _estimate_conditional_uncertainty(
        self,
        cog_state: CognitiveState,
        target_pos: Tuple[int, int],
        target_tile: int,
    ) -> float:
        """
        Penalize interacting with conditional content when the agent's local
        memory/belief support is weak.
        """
        tile_id = int(target_tile)
        if tile_id not in CONDITIONAL_IDS and tile_id != int(SEMANTIC_PALETTE['PUZZLE']):
            return 0.0

        _belief_tile, confidence = self.belief_map.get_tile_with_confidence(target_pos)
        uncertainty = 1.0 - float(max(0.0, min(1.0, confidence)))
        if tile_id == int(SEMANTIC_PALETTE['DOOR_LOCKED']) and cog_state.game_state.keys <= 0:
            uncertainty += 0.75
        elif tile_id == int(SEMANTIC_PALETTE['DOOR_BOMB']) and cog_state.game_state.bomb_count <= 0:
            uncertainty += 0.75
        elif tile_id == int(SEMANTIC_PALETTE['DOOR_BOSS']) and not cog_state.game_state.has_boss_key:
            uncertainty += 1.0
        elif tile_id == int(SEMANTIC_PALETTE['PUZZLE']):
            door_memories = self.memory.recall(MemoryItemType.DOOR, cog_state.current_step)
            if not door_memories:
                uncertainty += 0.35
        return float(min(1.5, uncertainty))

    def _estimate_decision_pressure(
        self,
        cog_state: CognitiveState,
        candidates: List[Tuple[Tuple[int, int], int]],
    ) -> float:
        """
        Estimate how cognitively demanding the current decision point is.

        This drives a bounded deliberation controller rather than simply
        reweighting the local utility formula.
        """
        if not candidates:
            return 0.0

        branch_factor = max(0.0, (len(candidates) - 1) / 3.0)
        uncertainty_terms: List[float] = []
        conditional_terms = 0.0
        local_risk = 0.0

        for target_pos, target_tile in candidates:
            _belief_tile, confidence = self.belief_map.get_tile_with_confidence(target_pos)
            uncertainty_terms.append(1.0 - float(max(0.0, min(1.0, confidence))))
            if int(target_tile) in CONDITIONAL_IDS or int(target_tile) == int(SEMANTIC_PALETTE['PUZZLE']):
                conditional_terms += 1.0
            local_risk += self._estimate_risk(cog_state, target_pos, int(target_tile))

        mean_uncertainty = float(sum(uncertainty_terms) / max(1, len(uncertainty_terms)))
        conditional_density = float(conditional_terms / max(1, len(candidates)))
        mean_risk = float(local_risk / max(1, len(candidates)))

        pressure = (
            0.40 * branch_factor
            + 0.35 * mean_uncertainty
            + 0.30 * conditional_density
            + 0.20 * mean_risk
        )
        return float(min(2.0, pressure))

    def _update_frustration(
        self,
        cog_state: CognitiveState,
        *,
        decision_pressure: float,
        target_pos: Tuple[int, int],
        target_tile: int,
    ) -> float:
        """
        Update a bounded frustration state that modulates future decisions.

        This turns the agent into a search/controller over augmented cognitive
        state, not only a static scalarized move scorer.
        """
        budget_cap = float(max(1e-6, getattr(self.config, 'deliberation_budget', 1.0)))
        budget_ratio = float(max(0.0, cog_state.deliberation_budget_remaining) / budget_cap)
        revisit = self._estimate_revisit_penalty(target_pos)
        uncertainty = self._estimate_conditional_uncertainty(cog_state, target_pos, target_tile)
        complexity = self._estimate_local_complexity(target_pos)
        hazard = self._estimate_combat_penalty(target_tile)

        frustration = (
            0.55 * float(cog_state.frustration)
            + 0.30 * float(decision_pressure)
            + 0.15 * float(revisit + uncertainty + complexity + hazard)
            + 0.20 * float(max(0.0, 1.0 - budget_ratio))
        )
        return float(min(2.5, frustration))
    
    def _direction_name(self, direction: Tuple[int, int]) -> str:
        """Convert direction tuple to name."""
        names = {
            (-1, 0): 'UP',
            (1, 0): 'DOWN',
            (0, -1): 'LEFT',
            (0, 1): 'RIGHT',
        }
        return names.get(direction, 'OTHER')
    
    def _compute_metrics(
        self,
        path: List[Tuple[int, int]],
        cog_state: CognitiveState
    ) -> CBSMetrics:
        """
        Compute final cognitive metrics from the path.
        """
        unique_tiles = len(set(path))
        total_steps = len(path)
        
        # Confusion index
        revisits = total_steps - unique_tiles
        confusion_index = revisits / max(1, unique_tiles)
        
        # Navigation entropy (direction-based)
        nav_entropy = self._compute_entropy(dict(self._direction_counts))
        
        # Room-level navigational entropy (CBS Paper Formula D)
        # H = -Î£ p(room) Ã— logâ‚‚(p(room)) where p(room) = visits(room) / total_visits
        room_visit_counts = dict(self._room_visit_counts)
        room_entropy = self._compute_room_entropy(room_visit_counts)
        unique_rooms = len(room_visit_counts)
        total_room_visits = int(sum(room_visit_counts.values()))
        
        # Cognitive load
        conf_var = self.belief_map.compute_confidence_variance()
        memory_ratio = self.memory.get_usage_ratio()
        cognitive_load = memory_ratio * (1 + conf_var)
        
        # Aha latency
        aha_latency = 0
        if self._goal_first_seen >= 0:
            aha_latency = cog_state.current_step - self._goal_first_seen
        
        # Extended metrics: replans, confusion_events, backtrack_loops
        replans = self._count_replans(path)
        confusion_events = self._count_confusion_events(path)
        backtrack_loops = self._count_backtrack_loops(path)
        belief_entropy = self.belief_map.compute_total_entropy()
        
        return CBSMetrics(
            confusion_index=confusion_index,
            navigation_entropy=nav_entropy,
            cognitive_load=cognitive_load,
            aha_latency=aha_latency,
            unique_tiles_visited=unique_tiles,
            total_steps=total_steps,
            peak_memory_usage=self.memory.peak_usage,
            goal_first_seen_step=self._goal_first_seen,
            decisions_made=self._decisions_made,
            suboptimal_decisions=self._suboptimal_decisions,
            replans=replans,
            confusion_events=confusion_events,
            backtrack_loops=backtrack_loops,
            path_length=total_steps,
            belief_entropy_final=belief_entropy,
            room_entropy=room_entropy,
            unique_rooms_visited=unique_rooms,
            total_room_visits=total_room_visits,
            deliberation_events=self._deliberation_events,
            budget_exhaustion_events=self._budget_exhaustion_events,
            peak_frustration=float(self._peak_frustration),
            final_deliberation_budget=float(cog_state.deliberation_budget_remaining),
            affordance_reactivations=self._affordance_reactivations,
            affordance_guided_steps=self._affordance_guided_steps,
            inventory_change_events=self._inventory_change_events,
            focus_switches=self._focus_switches,
            focus_guided_steps=self._focus_guided_steps,
            room_visit_counts=room_visit_counts,
            direction_distribution=dict(self._direction_counts),
            memory_timeline=list(self._memory_timeline),
        )
    
    def _count_replans(self, path: List[Tuple[int, int]]) -> int:
        """Count direction changes (replans) in the path."""
        if len(path) < 3:
            return 0
        
        replans = 0
        prev_dir = None
        
        for i in range(1, len(path)):
            direction = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            if prev_dir is not None and direction != prev_dir:
                replans += 1
            prev_dir = direction
        
        return replans
    
    def _count_confusion_events(self, path: List[Tuple[int, int]]) -> int:
        """Count tiles visited more than 2 times (confusion)."""
        visit_counts = defaultdict(int)
        for pos in path:
            visit_counts[pos] += 1
        
        return sum(1 for count in visit_counts.values() if count > 2)
    
    def _count_backtrack_loops(self, path: List[Tuple[int, int]]) -> int:
        """Count backtracking loops in the path."""
        if len(path) < 4:
            return 0
        
        loops = 0
        for i in range(len(path) - 3):
            # Check if position repeats within next 10 steps
            for j in range(i + 2, min(i + 10, len(path))):
                if path[i] == path[j]:
                    loops += 1
                    break
        
        return loops
    
    def _compute_entropy(self, distribution: Dict[str, int]) -> float:
        """Compute Shannon entropy of a distribution."""
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy

    def _compute_room_entropy(self, visit_counts: Dict[Hashable, int]) -> float:
        """
        Compute navigational entropy from the room-visit distribution.

        This stays in the base class because both the legacy `CBS` name and the
        thesis-facing `P-CBS` alias use the same bounded-rational metrics.
        """
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return 0.0

        entropy = 0.0
        for count in visit_counts.values():
            if count > 0:
                p = count / total_visits
                entropy -= p * math.log2(p)
        return entropy


class PersonaDrivenCognitiveBoundedSearch(CognitiveBoundedSearch):
    """
    Explicit thesis-facing alias for the bounded-rational persona validator.

    The implementation remains the same engine as `CognitiveBoundedSearch`,
    but this name makes the algorithmic intent unambiguous in reports and
    benchmark tables: this is a persona-driven bounded validator, not MAPF
    Conflict-Based Search.
    """

    algorithm_name = "P-CBS"
    algorithm_long_name = "Persona-Driven Cognitive Bounded Search"


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def solve_with_cbs(
    grid: np.ndarray,
    persona: str = 'balanced',
    timeout: int = 100000,
    seed: Optional[int] = None
) -> Tuple[bool, List[Tuple[int, int]], int, CBSMetrics]:
    """
    Convenience function to solve a grid with CBS.
    
    Args:
        grid: Semantic grid array
        persona: Agent persona name
        timeout: Max steps
        seed: Random seed
    
    Returns:
        (success, path, states, metrics)
    """
    from src.simulation.validator import ZeldaLogicEnv
    
    env = ZeldaLogicEnv(semantic_grid=grid)
    cbs = CognitiveBoundedSearch(env, persona=persona, timeout=timeout, seed=seed)
    return cbs.solve()


def solve_with_pcbs(
    grid: np.ndarray,
    persona: str = 'balanced',
    timeout: int = 100000,
    seed: Optional[int] = None,
    representation: str = 'hybrid',
) -> Tuple[bool, List[Tuple[int, int]], int, CBSMetrics]:
    """Convenience wrapper exposing the thesis-facing P-CBS name."""
    from src.simulation.validator import ZeldaLogicEnv

    env = ZeldaLogicEnv(semantic_grid=grid)
    pcbs = PersonaDrivenCognitiveBoundedSearch(
        env,
        persona=persona,
        timeout=timeout,
        seed=seed,
        representation=representation,
    )
    return pcbs.solve()


def compare_personas(
    grid: np.ndarray,
    personas: Optional[List[str]] = None,
    timeout: int = 100000,
    seed: int = 42
) -> Dict[str, Tuple[bool, int, CBSMetrics]]:
    """
    Run multiple personas on the same grid and compare results.
    
    Args:
        grid: Semantic grid array
        personas: List of persona names (default: all)
        timeout: Max steps per persona
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping persona name to (success, path_length, metrics)
    """
    if personas is None:
        personas = [p.value for p in AgentPersona]
    
    results = {}
    
    for persona_name in personas:
        success, path, _states, metrics = solve_with_cbs(
            grid, persona=persona_name, timeout=timeout, seed=seed
        )
        results[persona_name] = (success, len(path), metrics)
    
    return results


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Main classes
    'CognitiveBoundedSearch',
    'PersonaDrivenCognitiveBoundedSearch',
    'CBSMetrics',
    
    # Cognitive components
    'BeliefMap',
    'VisionSystem',
    'WorkingMemory',
    'MemoryItem',
    'MemoryItemType',
    'TileObservation',
    'TileKnowledge',
    'CognitiveState',
    
    # Heuristics
    'DecisionHeuristic',
    'CuriosityHeuristic',
    'RecencyHeuristic',
    'SafetyHeuristic',
    'GoalSeekingHeuristic',
    'ItemSeekingHeuristic',
    
    # Personas
    'AgentPersona',
    'PersonaConfig',
    'PERSONA_CONFIGS',
    
    # Convenience functions
    'solve_with_cbs',
    'solve_with_pcbs',
    'compare_personas',
]
