# Cognitive Bounded Search (CBS) — Architecture Design Document

**Author**: KLTN Team  
**Version**: 1.0.0  
**Date**: 2026-02-06  
**Status**: Implementation Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scientific Foundation](#scientific-foundation)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [Algorithm Pseudocode](#algorithm-pseudocode)
6. [Agent Personas](#agent-personas)
7. [Cognitive Metrics](#cognitive-metrics)
8. [Integration with Validator](#integration-with-validator)
9. [Usage Examples](#usage-examples)
10. [Performance Considerations](#performance-considerations)
11. [Future Extensions](#future-extensions)

---

## Executive Summary

The Cognitive Bounded Search (CBS) algorithm extends the optimal A* solver in `validator.py` to simulate **human-like navigation behavior** with realistic cognitive limitations. Unlike perfect-information A*, CBS models:

- **Limited vision** (field of view, occlusion)
- **Decaying memory** (forgetting over time)
- **Bounded working memory** (Miller's 7±2 items)
- **Satisficing behavior** (accepting "good enough" solutions)
- **Multiple decision heuristics** (curiosity, safety, goal-seeking)

This enables assessment of dungeon **playability from a human perspective**, not just theoretical solvability.

---

## Scientific Foundation

CBS is grounded in established cognitive science research:

| Principle | Source | Application in CBS |
|-----------|--------|-------------------|
| Working Memory Capacity | Miller (1956) | 7±2 item memory limit |
| Bounded Rationality | Simon (1955) | Satisficing threshold |
| System 1/System 2 | Kahneman (2011) | Heuristic vs. deliberate decisions |
| Cognitive Maps | Tolman (1948) | Belief map representation |
| Forgetting Curve | Ebbinghaus (1885) | Memory decay rate |
| Curiosity | Berlyne (1966) | Exploration heuristic |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     COGNITIVE BOUNDED SEARCH (CBS)                       │
│                                                                          │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────────────┐   │
│  │ VISION SYSTEM  │  │ BELIEF MAP   │  │ WORKING MEMORY             │   │
│  │ • FOV cone     │──│ • Known tiles│──│ • Capacity=7 (Miller)      │   │
│  │ • Radius=5     │  │ • Confidence │  │ • Decay rate=0.95          │   │
│  │ • Occlusion    │  │ • Last seen  │  │ • Salience weighting       │   │
│  └───────┬────────┘  └──────┬───────┘  └───────────┬────────────────┘   │
│          │                  │                      │                     │
│          └──────────────────┼──────────────────────┘                     │
│                             │                                            │
│                    ┌────────▼────────┐                                   │
│                    │ DECISION ENGINE │                                   │
│                    │ • Heuristic mix │                                   │
│                    │ • Satisficing   │                                   │
│                    │ • Persona-based │                                   │
│                    └────────┬────────┘                                   │
│                             │                                            │
│  ┌──────────────────────────▼──────────────────────────────────────┐    │
│  │                    AGENT PERSONAS                                │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│  │  │SPEEDRUNER│  │ EXPLORER │  │ CAUTIOUS │  │ FORGETFUL        │ │    │
│  │  │ A* base  │  │ Curious  │  │ Safe     │  │ High decay       │ │    │
│  │  │ Min path │  │ All rooms│  │ Avoids M │  │ Gets lost        │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  OUTPUT: (success, path, states, CBSMetrics)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### A) Epistemic State (Belief Map)

The belief map represents what the agent **thinks** the map looks like, distinct from ground truth.

```python
class BeliefMap:
    """
    Epistemic state: agent's beliefs about the world.
    
    Attributes:
        known_tiles: Dict[Tuple[int,int], TileObservation]
            - tile_type: int (semantic ID)
            - confidence: float [0.0=unknown, 1.0=certain]
            - last_seen: int (timestep)
            - knowledge: TileKnowledge enum (UNKNOWN/GLIMPSED/OBSERVED/EXPLORED)
            - visited: bool
        
        decay_rate: float [0.95 default]
            - Per-step confidence decay
        
        default_assumption: int
            - What to assume for unknown tiles (WALL = conservative)
    
    Key Methods:
        observe(position, tile_type, step, is_visit)
            - Record an observation, update confidence
        
        get_tile(position) -> (tile_type, confidence)
            - Query belief about a tile
        
        apply_decay(current_step)
            - Apply memory decay to all observations
        
        get_frontier() -> Set[position]
            - Get observed-but-unvisited tiles
        
        compute_confusion_index() -> float
            - revisits / unique_visits
    """
```

### B) Vision System

Simulates limited field-of-view with occlusion.

```python
class VisionSystem:
    """
    Limited field-of-view perception.
    
    Attributes:
        radius: int = 5
            - Maximum visibility distance
        
        cone_angle: float = 120.0
            - Field of view in degrees (360 = full vision)
        
        enable_occlusion: bool = True
            - Whether walls block line of sight
    
    Key Methods:
        get_visible_tiles(position, direction, grid) -> Set[position]
            - Returns all tiles visible from current position
            - Respects FOV cone and occlusion
        
        _cast_shadow(origin, direction, occluded, height, width)
            - Add tiles behind walls to shadow set
    """
```

### C) Working Memory

Capacity-limited memory based on Miller's Law.

```python
class WorkingMemory:
    """
    Bounded working memory (7±2 items).
    
    Attributes:
        capacity: int = 7 (Miller's number)
        decay_rate: float = 0.95
        salience_weights: Dict[MemoryItemType, float]
    
    Memory Item Types:
        - GOAL: Goal location (highest salience)
        - THREAT: Enemy/danger positions
        - ITEM: Key/bomb locations
        - DOOR: Locked door positions
        - LANDMARK: Distinctive features
        - POSITION: General locations
        - PATH_SEGMENT: Remembered routes
    
    Key Methods:
        remember(item_type, position, step, data, salience_boost) -> bool
            - Add item to memory (may displace low-salience items)
        
        recall(item_type=None, step) -> List[MemoryItem]
            - Retrieve memories, optionally filtered
        
        recall_nearest(position, item_type, step) -> MemoryItem
            - Get closest remembered item
        
        apply_decay(current_step)
            - Decay salience over time, forget low-salience items
        
        get_usage_ratio() -> float
            - Current memory usage as fraction of capacity
    """
```

### D) Decision Heuristics

Multiple heuristics combined for decision-making:

```python
class DecisionHeuristic(ABC):
    @abstractmethod
    def score(current_pos, target_pos, target_tile, 
              belief_map, memory, goal_pos, step) -> float:
        """Score a move [higher = preferred]"""
    
    @property
    def weight(self) -> float:
        """Heuristic weight in decision mix"""


# Available Heuristics:

class CuriosityHeuristic(DecisionHeuristic):
    """Prefer unexplored areas (UNKNOWN=1.0, EXPLORED=0.0)"""

class RecencyHeuristic(DecisionHeuristic):
    """Prefer recently seen paths (spatial memory)"""

class SafetyHeuristic(DecisionHeuristic):
    """Avoid enemies and threats (-1.0 for danger)"""

class GoalSeekingHeuristic(DecisionHeuristic):
    """Move toward remembered goal (A*-like when goal known)"""

class ItemSeekingHeuristic(DecisionHeuristic):
    """Move toward remembered items (keys, bombs)"""
```

---

## Algorithm Pseudocode

### Main CBS Loop

```
ALGORITHM: CognitiveBoundedSearch.solve()

INPUT:
    env: ZeldaLogicEnv (semantic grid + game logic)
    persona: PersonaConfig (memory, vision, heuristic weights)
    timeout: int (max steps)

OUTPUT:
    success: bool
    path: List[Position]
    states_explored: int
    metrics: CBSMetrics

1. INITIALIZE:
    belief_map ← BeliefMap(grid_shape, decay_rate=persona.decay_rate)
    memory ← WorkingMemory(capacity=persona.capacity, decay_rate=persona.decay_rate)
    vision ← VisionSystem(radius=persona.radius, cone=persona.cone)
    heuristics ← build_heuristics(persona.weights)
    
    cog_state ← CognitiveState(
        game_state=env.reset(),
        belief_map, memory, step=0
    )
    path ← [cog_state.position]

2. MAIN LOOP (while step < timeout):
    
    a) PERCEIVE:
        visible_tiles ← vision.get_visible_tiles(position, direction, grid)
        FOR each tile_pos in visible_tiles:
            tile_type ← grid[tile_pos]
            belief_map.observe(tile_pos, tile_type, step, is_visit=(tile_pos==position))
            
            IF tile_type == TRIFORCE:
                memory.remember(GOAL, tile_pos, step, salience_boost=0.5)
            ELIF tile_type in PICKUP_IDS:
                memory.remember(ITEM, tile_pos, step)
            ELIF tile_type in {ENEMY, BOSS}:
                memory.remember(THREAT, tile_pos, step)
            ELIF tile_type in CONDITIONAL_IDS:
                memory.remember(DOOR, tile_pos, step)
    
    b) CHECK WIN:
        IF position == goal_pos:
            RETURN (True, path, step, compute_metrics())
    
    c) MEMORY DECAY:
        belief_map.apply_decay(step)
        memory.apply_decay(step)
    
    d) DECIDE:
        candidates ← get_candidate_moves(cog_state, grid)
        
        IF candidates is empty:
            BREAK  # Stuck
        
        scored_moves ← []
        FOR (target_pos, target_tile) in candidates:
            score ← 0.0
            total_weight ← 0.0
            FOR heuristic in heuristics:
                s ← heuristic.score(position, target_pos, target_tile,
                                    belief_map, memory, goal_pos, step)
                score += s * heuristic.weight
                total_weight += heuristic.weight
            
            normalized_score ← score / total_weight
            scored_moves.append((normalized_score, target_pos, target_tile))
        
        SORT scored_moves by score (descending)
        
        # Satisficing: accept "good enough" moves
        best_score, best_pos, best_tile ← scored_moves[0]
        threshold ← best_score * persona.satisficing_threshold
        acceptable ← [m for m in scored_moves if m[0] >= threshold]
        
        # Random tiebreaker
        IF random() < persona.random_tiebreaker AND len(acceptable) > 1:
            chosen ← random.choice(acceptable)
        ELSE:
            chosen ← (best_score, best_pos, best_tile)
    
    e) EXECUTE:
        moved, new_game_state ← try_move(game_state, best_pos, best_tile)
        
        IF moved:
            cog_state.game_state ← new_game_state
            cog_state.direction ← (best_pos - position)
            cog_state.step += 1
            path.append(best_pos)
            step += 1

3. RETURN (False, path, step, compute_metrics())
```

### Heuristic Scoring

```
FUNCTION score_curiosity(target_pos, belief_map):
    knowledge ← belief_map.get_knowledge_state(target_pos)
    SWITCH knowledge:
        UNKNOWN:   RETURN 1.0   # Maximum curiosity
        GLIMPSED:  RETURN 0.7
        OBSERVED:  RETURN 0.3
        EXPLORED:  RETURN 0.0   # No curiosity for visited

FUNCTION score_safety(target_tile, memory, step):
    IF target_tile in {ENEMY, BOSS, ELEMENT}:
        RETURN -1.0  # Strongly avoid
    
    FOR threat in memory.recall(THREAT, step):
        IF manhattan_distance(threat.position, target_pos) <= 2:
            RETURN -0.5 * threat.salience
    
    RETURN 0.1  # Slight preference for safe tiles

FUNCTION score_goal_seeking(current_pos, target_pos, memory, goal_pos, step):
    remembered_goal ← memory.recall_nearest(current_pos, GOAL, step)
    goal ← remembered_goal.position IF remembered_goal ELSE goal_pos
    
    IF goal is None:
        RETURN 0.0  # No goal-seeking if goal unknown
    
    current_dist ← manhattan(current_pos, goal)
    target_dist ← manhattan(target_pos, goal)
    
    IF current_dist > 0:
        RETURN (current_dist - target_dist) / current_dist
    RETURN 0.0
```

---

## Agent Personas

| Persona | Memory | Decay | Vision | Key Weights | Behavior |
|---------|--------|-------|--------|-------------|----------|
| **Speedrunner** | 10 | 0.99 | 10/360° | goal=2.0, safety=0 | Optimal A*, ignores danger |
| **Explorer** | 7 | 0.95 | 5/360° | curiosity=2.0, goal=0.3 | Clears all rooms |
| **Cautious** | 7 | 0.95 | 5/120° | safety=2.0, recency=1.0 | Avoids enemies, known paths |
| **Forgetful** | 4 | 0.80 | 4/120° | recency=1.5, random=0.3 | Gets lost, backtracks |
| **Completionist** | 10 | 0.98 | 6/360° | items=2.0, curiosity=1.5 | Collects everything |
| **Balanced** | 7 | 0.95 | 5/360° | all≈1.0 | Average human player |

---

## Cognitive Metrics

CBS outputs rich metrics for analyzing human-like navigation:

```python
@dataclass
class CBSMetrics:
    # PRIMARY METRICS
    confusion_index: float      # revisits / unique_visits (high = lost)
    navigation_entropy: float   # -Σ p(dir) log p(dir) (high = random)
    cognitive_load: float       # memory × uncertainty (high = mentally taxing)
    aha_latency: int           # steps from see_goal to reach_goal
    
    # SECONDARY METRICS
    unique_tiles_visited: int
    total_steps: int
    peak_memory_usage: int
    goal_first_seen_step: int
    decisions_made: int
    suboptimal_decisions: int
    exploration_efficiency: float  # unique / total
    
    # DETAILED DATA
    room_visit_counts: Dict[Tuple, int]
    direction_distribution: Dict[str, int]
    memory_timeline: List[int]
```

### Metric Interpretation

| Metric | Good | Bad | Interpretation |
|--------|------|-----|----------------|
| Confusion Index | < 0.5 | > 2.0 | How often agent backtracks |
| Navigation Entropy | 0.5-1.5 | < 0.3 or > 1.8 | Decision randomness |
| Cognitive Load | < 1.0 | > 2.0 | Mental effort required |
| Aha Latency | < 20 | > 100 | Speed of goal exploitation |
| Exploration Efficiency | > 0.8 | < 0.3 | Wasted movement |

---

## Integration with Validator

### File Structure

```
src/simulation/
├── validator.py                    # Original A* solver
├── cognitive_bounded_search.py     # NEW: CBS implementation
├── __init__.py                     # Update exports
```

### Updated `__init__.py`

```python
# src/simulation/__init__.py

from .validator import (
    ZeldaLogicEnv,
    StateSpaceAStar,
    GameState,
    ValidationResult,
    SolverOptions,
    SolverDiagnostics,
    SEMANTIC_PALETTE,
)

from .cognitive_bounded_search import (
    CognitiveBoundedSearch,
    CBSMetrics,
    BeliefMap,
    VisionSystem,
    WorkingMemory,
    AgentPersona,
    PersonaConfig,
    solve_with_cbs,
    compare_personas,
)

__all__ = [
    # Validator
    'ZeldaLogicEnv',
    'StateSpaceAStar',
    'GameState',
    'ValidationResult',
    'SolverOptions',
    'SolverDiagnostics',
    
    # CBS
    'CognitiveBoundedSearch',
    'CBSMetrics',
    'BeliefMap',
    'VisionSystem',
    'WorkingMemory',
    'AgentPersona',
    'PersonaConfig',
    'solve_with_cbs',
    'compare_personas',
]
```

### Usage in Validation Pipeline

```python
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, compare_personas
)

# Load dungeon grid
grid = np.load('dungeon.npy')

# 1. Optimal validation (is it solvable?)
env = ZeldaLogicEnv(semantic_grid=grid)
solver = StateSpaceAStar(env)
optimal_success, optimal_path, optimal_states = solver.solve()

# 2. Human playability validation (is it enjoyable?)
cbs = CognitiveBoundedSearch(env, persona='balanced')
human_success, human_path, human_states, metrics = cbs.solve()

# 3. Compare multiple personas
comparison = compare_personas(grid, personas=['explorer', 'cautious', 'forgetful'])

for persona, (success, path_len, metrics) in comparison.items():
    print(f"{persona}: success={success}, steps={path_len}")
    print(f"  Confusion: {metrics.confusion_index:.2f}")
    print(f"  Entropy: {metrics.navigation_entropy:.2f}")
```

---

## Usage Examples

### Basic Usage

```python
from src.simulation.validator import ZeldaLogicEnv
from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch, AgentPersona

# Create environment
env = ZeldaLogicEnv(semantic_grid=grid)

# Create CBS solver with explorer persona
cbs = CognitiveBoundedSearch(
    env,
    persona=AgentPersona.EXPLORER,
    timeout=50000,
    seed=42
)

# Solve
success, path, states, metrics = cbs.solve()

# Analyze results
print(f"Success: {success}")
print(f"Path length: {len(path)}")
print(f"States explored: {states}")
print(metrics.summary())
```

### Custom Persona

```python
from src.simulation.cognitive_bounded_search import (
    CognitiveBoundedSearch, PersonaConfig
)

# Create custom "anxious player" persona
anxious_config = PersonaConfig(
    name="Anxious",
    memory_capacity=5,        # Poor memory under stress
    memory_decay_rate=0.85,   # Fast forgetting
    vision_radius=4,          # Tunnel vision
    vision_cone=90.0,         # Narrow FOV
    heuristic_weights={
        'safety': 3.0,        # Very safety-conscious
        'recency': 2.0,       # Clings to known paths
        'goal_seeking': 0.5,  # Slow progress
        'curiosity': 0.2,     # Avoids exploration
        'item_seeking': 0.3,
    },
    satisficing_threshold=0.95,  # Needs near-perfect moves
    random_tiebreaker=0.05,
)

cbs = CognitiveBoundedSearch(env, custom_config=anxious_config)
success, path, states, metrics = cbs.solve()
```

### Batch Validation

```python
from src.simulation.cognitive_bounded_search import solve_with_cbs
import json

# Validate multiple dungeons
results = []
for dungeon_file in dungeon_files:
    grid = np.load(dungeon_file)
    
    # Test each persona
    for persona in ['speedrunner', 'explorer', 'cautious', 'forgetful']:
        success, path, states, metrics = solve_with_cbs(
            grid, persona=persona, timeout=50000, seed=42
        )
        
        results.append({
            'dungeon': dungeon_file,
            'persona': persona,
            'success': success,
            'path_length': len(path),
            'confusion': metrics.confusion_index,
            'entropy': metrics.navigation_entropy,
            'cognitive_load': metrics.cognitive_load,
            'aha_latency': metrics.aha_latency,
        })

# Save results
with open('cbs_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Performance Considerations

### Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Vision (get_visible_tiles) | O(r²) where r=radius | O(r²) |
| Belief Map (observe) | O(1) | O(tiles_seen) |
| Memory (remember) | O(capacity × log capacity) | O(capacity) |
| Heuristic scoring | O(H) where H=num_heuristics | O(1) |
| Overall per step | O(r² + candidates × H) | O(grid + memory) |

### Optimization Tips

1. **Vision Caching**: Pre-compute visibility offsets for fixed radius
2. **Belief Map Pruning**: Remove very low-confidence observations periodically
3. **Lazy Decay**: Only apply decay when tile is accessed
4. **Heuristic Short-circuit**: Skip remaining heuristics if score already decisive

### Comparison with StateSpaceAStar

| Aspect | StateSpaceAStar | CBS |
|--------|-----------------|-----|
| Optimality | Guaranteed optimal | Satisficing |
| Information | Perfect (full grid) | Limited (vision) |
| Memory | State-space expansion | Bounded working memory |
| Speed | Very fast (pruning) | Slower (simulation) |
| Output | Path only | Path + cognitive metrics |
| Use Case | Solvability check | Playability analysis |

---

## Future Extensions

### Planned Enhancements

1. **Learning Agents**: Memory that persists across dungeon attempts
2. **Anxiety Modeling**: Performance degradation under threat
3. **Fatigue Simulation**: Increasing errors over time
4. **Multi-Agent**: Coop navigation with shared memory
5. **Procedural Persona**: Generate personas from player data

### Research Directions

- **Calibration**: Tune parameters against human playtest data
- **Difficulty Prediction**: Use CBS metrics to predict perceived difficulty
- **Level Generation**: Optimize dungeons for specific persona profiles
- **Accessibility**: Identify dungeons that fail for low-skill personas

---

## References

1. Miller, G.A. (1956). "The magical number seven, plus or minus two"
2. Simon, H.A. (1955). "A behavioral model of rational choice"
3. Kahneman, D. (2011). "Thinking, Fast and Slow"
4. Tolman, E.C. (1948). "Cognitive maps in rats and men"
5. Berlyne, D.E. (1966). "Curiosity and exploration"
6. Cowan, N. (2001). "The magical number 4 in short-term memory"
7. Ebbinghaus, H. (1885). "Über das Gedächtnis"

---

## Changelog

### v1.0.0 (2026-02-06)
- Initial CBS architecture design
- Core components: BeliefMap, VisionSystem, WorkingMemory
- 6 agent personas with configurable heuristics
- CBSMetrics with 4 primary and 6 secondary metrics
- Integration with validator.py via ZeldaLogicEnv
- Convenience functions: solve_with_cbs, compare_personas
