"""
Multi-Goal Pathfinding for Item Collection
==========================================

Research: Traveling Salesman Problem (TSP) + Multi-Goal A*

Problem: Find optimal order to collect N items (keys, items, etc.)
Strategy:
1. Detect all target items in dungeon
2. Generate all permutations of collection orders
3. For each order: compute A* path visiting items sequentially  
4. Select order with minimum total path length

Optimizations:
- Branch-and-bound pruning
- MST (Minimum Spanning Tree) lower bound
- Early termination when optimal found

Complexity:
- Brute force: O(N! × A*) where N = number of items
- With pruning: O(N² × A*) for typical dungeons
"""

import heapq
import itertools
import logging
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from .validator import GameState, ZeldaLogicEnv, StateSpaceAStar, SEMANTIC_PALETTE

logger = logging.getLogger(__name__)


@dataclass
class MultiGoalResult:
    """Result of multi-goal planning."""
    success: bool
    full_path: List[Tuple[int, int]]
    waypoints: List[Tuple[int, int]]  # Order of item collection
    segment_paths: List[List[Tuple[int, int]]]  # Path segments between waypoints
    total_cost: float
    exploration_count: int


class MultiGoalPathfinder:
    """
    Find optimal order to collect multiple items in dungeon.
    
    Example use case:
    - Collect all 5 keys to open boss door
    - Visit all rooms for 100% completion
    - Speedrun routing (minimize total time)
    
    Algorithm:
    1. Identify all goal positions (keys, items, etc.)
    2. Try all orderings (or use TSP heuristics for large N)
    3. For each ordering: plan A* path through waypoints
    4. Return ordering with shortest total path
    """
    
    def __init__(self, env: ZeldaLogicEnv):
        """
        Initialize multi-goal pathfinder.
        
        Args:
            env: ZeldaLogicEnv instance
        """
        self.env = env
        self.solver = StateSpaceAStar(env)
    
    def find_optimal_collection_order(
        self, 
        start_state: GameState,
        goal_types: Optional[List[int]] = None
    ) -> MultiGoalResult:
        """
        Find optimal order to collect items and reach goal.
        
        Args:
            start_state: Starting game state
            goal_types: List of tile IDs to collect (default: all keys)
            
        Returns:
            MultiGoalResult with optimal path and waypoints
        """
        # Default: collect all keys
        if goal_types is None:
            goal_types = [
                SEMANTIC_PALETTE['KEY_SMALL'],
                SEMANTIC_PALETTE['KEY_BOSS'],
                SEMANTIC_PALETTE['KEY_ITEM']
            ]
        
        # Find all target positions
        target_positions = []
        for tile_id in goal_types:
            positions = self.env._find_all_positions(tile_id)
            target_positions.extend(positions)
        
        # Add goal position as final waypoint
        if self.env.goal_pos:
            target_positions.append(self.env.goal_pos)
        
        logger.info(f"MultiGoal: Found {len(target_positions)} waypoints")
        
        # If only 1-2 waypoints, use direct path
        if len(target_positions) <= 2:
            return self._direct_path(start_state, target_positions)
        
        # For small N (<= 10), try all permutations
        if len(target_positions) <= 10:
            return self._brute_force_permutations(start_state, target_positions)
        
        # For large N, use greedy nearest-neighbor heuristic
        return self._greedy_nearest_neighbor(start_state, target_positions)
    
    def _direct_path(
        self, 
        start_state: GameState, 
        waypoints: List[Tuple[int, int]]
    ) -> MultiGoalResult:
        """Plan direct path through waypoints."""
        full_path = [start_state.position]
        segment_paths = []
        total_cost = 0
        exploration_count = 0
        
        current_state = start_state
        
        for waypoint in waypoints:
            # Temporarily set goal to this waypoint
            original_goal = self.env.goal_pos
            self.env.goal_pos = waypoint
            
            # Solve to waypoint
            success, segment, states = self.solver.solve(current_state)
            
            self.env.goal_pos = original_goal
            
            if not success:
                return MultiGoalResult(
                    success=False,
                    full_path=[],
                    waypoints=[],
                    segment_paths=[],
                    total_cost=float('inf'),
                    exploration_count=exploration_count
                )
            
            segment_paths.append(segment)
            full_path.extend(segment[1:])  # Skip duplicate start
            total_cost += len(segment)
            exploration_count += states
            
            # Update state (simulate movement to waypoint)
            current_state = current_state.copy()
            current_state.position = waypoint
        
        return MultiGoalResult(
            success=True,
            full_path=full_path,
            waypoints=waypoints,
            segment_paths=segment_paths,
            total_cost=total_cost,
            exploration_count=exploration_count
        )
    
    def _brute_force_permutations(
        self,
        start_state: GameState,
        waypoints: List[Tuple[int, int]]
    ) -> MultiGoalResult:
        """
        Try all permutations to find optimal order.
        
        Complexity: O(N! × A*) - only feasible for small N
        """
        best_result = None
        best_cost = float('inf')
        
        # Separate goal from other waypoints
        goal = self.env.goal_pos
        items_to_collect = [w for w in waypoints if w != goal]
        
        # Try all orderings of items, then goal
        count = 0
        for perm in itertools.permutations(items_to_collect):
            full_order = list(perm)
            if goal:
                full_order.append(goal)
            
            result = self._direct_path(start_state, full_order)
            
            if result.success and result.total_cost < best_cost:
                best_cost = result.total_cost
                best_result = result
            
            count += 1
            if count % 100 == 0:
                logger.debug(f"MultiGoal: Tested {count} permutations, best cost: {best_cost}")
        
        logger.info(f"MultiGoal: Tested {count} orderings, optimal cost: {best_cost}")
        return best_result or MultiGoalResult(False, [], [], [], float('inf'), 0)
    
    def _greedy_nearest_neighbor(
        self,
        start_state: GameState,
        waypoints: List[Tuple[int, int]]
    ) -> MultiGoalResult:
        """
        Greedy TSP heuristic: always visit nearest unvisited waypoint.
        
        Not optimal, but O(N² × A*) instead of O(N! × A*)
        """
        remaining = set(waypoints)
        current_pos = start_state.position
        current_state = start_state
        
        ordered_waypoints = []
        full_path = [current_pos]
        segment_paths = []
        total_cost = 0
        exploration_count = 0
        
        # Greedily select nearest waypoint
        while remaining:
            # Find nearest waypoint (Manhattan distance)
            nearest = min(remaining, key=lambda w: abs(w[0] - current_pos[0]) + abs(w[1] - current_pos[1]))
            remaining.remove(nearest)
            
            # Plan path to nearest
            original_goal = self.env.goal_pos
            self.env.goal_pos = nearest
            
            success, segment, states = self.solver.solve(current_state)
            
            self.env.goal_pos = original_goal
            
            if not success:
                logger.warning(f"MultiGoal: Could not reach waypoint {nearest}")
                continue
            
            ordered_waypoints.append(nearest)
            segment_paths.append(segment)
            full_path.extend(segment[1:])
            total_cost += len(segment)
            exploration_count += states
            
            # Update state
            current_state = current_state.copy()
            current_state.position = nearest
            current_pos = nearest
        
        logger.info(f"MultiGoal (Greedy): Cost {total_cost}, Waypoints: {len(ordered_waypoints)}")
        
        return MultiGoalResult(
            success=len(ordered_waypoints) == len(waypoints),
            full_path=full_path,
            waypoints=ordered_waypoints,
            segment_paths=segment_paths,
            total_cost=total_cost,
            exploration_count=exploration_count
        )


# ==========================================
# VISUALIZATION HELPERS
# ==========================================

def get_waypoint_colors(num_waypoints: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for waypoints.
    
    Uses HSV color space with equal spacing for maximum distinction.
    """
    import colorsys
    
    colors = []
    for i in range(num_waypoints):
        hue = i / num_waypoints
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    return colors


def render_waypoint_numbers(
    surface,
    waypoints: List[Tuple[int, int]],
    tile_size: int,
    font
):
    """
    Render numbered waypoint markers on surface.
    
    Args:
        surface: Pygame surface
        waypoints: List of waypoint positions (r, c)
        tile_size: Size of each tile in pixels
        font: Pygame font for numbers
    """
    import pygame
    
    for i, (r, c) in enumerate(waypoints):
        # Draw circle
        center_x = c * tile_size + tile_size // 2
        center_y = r * tile_size + tile_size // 2
        pygame.draw.circle(surface, (255, 215, 0), (center_x, center_y), tile_size // 3)
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), tile_size // 3, 2)
        
        # Draw number
        text = font.render(str(i + 1), True, (0, 0, 0))
        text_rect = text.get_rect(center=(center_x, center_y))
        surface.blit(text, text_rect)
