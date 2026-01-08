"""
BLOCK VI: EXTERNAL VALIDATOR
============================
Automated Playtesting Suite for Zelda AI Validation.

This module provides:
1. ZELDA LOGIC ENVIRONMENT - State machine simulator
2. STATE-SPACE A* SOLVER - Intelligent pathfinding with inventory state
3. SANITY CHECKER - Pre-validation structural checks
4. METRICS ENGINE - Solvability, reachability, diversity metrics
5. DIVERSITY EVALUATOR - Mode collapse detection

Author: KLTN Thesis Project
"""

import os
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import IntEnum

# Import semantic palette from adapter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.adapter import SEMANTIC_PALETTE, ID_TO_NAME
except ImportError:
    # Fallback definitions if adapter not available
    SEMANTIC_PALETTE = {
        'VOID': 0, 'FLOOR': 1, 'WALL': 2, 'BLOCK': 3,
        'DOOR_OPEN': 10, 'DOOR_LOCKED': 11, 'DOOR_BOMB': 12,
        'DOOR_PUZZLE': 13, 'DOOR_BOSS': 14, 'DOOR_SOFT': 15,
        'ENEMY': 20, 'START': 21, 'TRIFORCE': 22, 'BOSS': 23,
        'KEY_SMALL': 30, 'KEY_BOSS': 31, 'KEY_ITEM': 32, 'ITEM_MINOR': 33,
        'ELEMENT': 40, 'ELEMENT_FLOOR': 41, 'STAIR': 42, 'PUZZLE': 43,
    }
    ID_TO_NAME = {v: k for k, v in SEMANTIC_PALETTE.items()}


# ==========================================
# CONSTANTS
# ==========================================

# Tile categories for movement logic
WALKABLE_IDS = {
    SEMANTIC_PALETTE['FLOOR'],
    SEMANTIC_PALETTE['DOOR_OPEN'],
    SEMANTIC_PALETTE['DOOR_SOFT'],  # One-way passable
    SEMANTIC_PALETTE['START'],
    SEMANTIC_PALETTE['TRIFORCE'],
    SEMANTIC_PALETTE['KEY_SMALL'],
    SEMANTIC_PALETTE['KEY_BOSS'],
    SEMANTIC_PALETTE['KEY_ITEM'],
    SEMANTIC_PALETTE['ITEM_MINOR'],
    SEMANTIC_PALETTE['ELEMENT_FLOOR'],
    SEMANTIC_PALETTE['STAIR'],
}

BLOCKING_IDS = {
    SEMANTIC_PALETTE['VOID'],
    SEMANTIC_PALETTE['WALL'],
    SEMANTIC_PALETTE['BLOCK'],
    SEMANTIC_PALETTE['ELEMENT'],
}

CONDITIONAL_IDS = {
    SEMANTIC_PALETTE['DOOR_LOCKED'],   # Needs key
    SEMANTIC_PALETTE['DOOR_BOMB'],     # Needs bomb
    SEMANTIC_PALETTE['DOOR_BOSS'],     # Needs boss key
    SEMANTIC_PALETTE['DOOR_PUZZLE'],   # Needs puzzle solved
}

PICKUP_IDS = {
    SEMANTIC_PALETTE['KEY_SMALL'],
    SEMANTIC_PALETTE['KEY_BOSS'],
    SEMANTIC_PALETTE['KEY_ITEM'],
    SEMANTIC_PALETTE['ITEM_MINOR'],
}

# Action enumeration
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class GameState:
    """Represents the complete state of the game at a point in time."""
    position: Tuple[int, int]
    keys: int = 0
    has_bomb: bool = False
    has_boss_key: bool = False
    has_item: bool = False
    opened_doors: Set[Tuple[int, int]] = field(default_factory=set)
    collected_items: Set[Tuple[int, int]] = field(default_factory=set)
    
    def __hash__(self):
        return hash((
            self.position,
            self.keys,
            self.has_bomb,
            self.has_boss_key,
            self.has_item,
            frozenset(self.opened_doors),
            frozenset(self.collected_items)
        ))
    
    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return (
            self.position == other.position and
            self.keys == other.keys and
            self.has_bomb == other.has_bomb and
            self.has_boss_key == other.has_boss_key and
            self.has_item == other.has_item and
            self.opened_doors == other.opened_doors and
            self.collected_items == other.collected_items
        )
    
    def copy(self) -> 'GameState':
        return GameState(
            position=self.position,
            keys=self.keys,
            has_bomb=self.has_bomb,
            has_boss_key=self.has_boss_key,
            has_item=self.has_item,
            opened_doors=self.opened_doors.copy(),
            collected_items=self.collected_items.copy()
        )


@dataclass
class ValidationResult:
    """Results from validating a single map."""
    is_solvable: bool
    is_valid_syntax: bool
    reachability: float
    path_length: int
    backtracking_score: float
    logical_errors: List[str]
    path: List[Tuple[int, int]] = field(default_factory=list)
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'is_solvable': self.is_solvable,
            'is_valid_syntax': self.is_valid_syntax,
            'reachability': self.reachability,
            'path_length': self.path_length,
            'backtracking_score': self.backtracking_score,
            'logical_errors': self.logical_errors,
            'error_message': self.error_message
        }


@dataclass
class BatchValidationResult:
    """Results from validating a batch of maps."""
    total_maps: int
    valid_syntax_count: int
    solvable_count: int
    solvability_rate: float
    avg_reachability: float
    avg_path_length: float
    avg_backtracking: float
    diversity_score: float
    individual_results: List[ValidationResult] = field(default_factory=list)
    
    def summary(self) -> str:
        return f"""
=== Batch Validation Summary ===
Total Maps: {self.total_maps}
Valid Syntax: {self.valid_syntax_count} ({100*self.valid_syntax_count/max(1,self.total_maps):.1f}%)
Solvable: {self.solvable_count} ({100*self.solvability_rate:.1f}%)
Avg Reachability: {100*self.avg_reachability:.1f}%
Avg Path Length: {self.avg_path_length:.1f}
Avg Backtracking: {self.avg_backtracking:.2f}
Diversity Score: {self.diversity_score:.3f}
================================
"""


# ==========================================
# MODULE 1: ZELDA LOGIC ENVIRONMENT
# ==========================================

class ZeldaLogicEnv:
    """
    Discrete state simulator for Zelda dungeon logic.
    
    Handles:
    - Movement with collision detection
    - Item pickup and inventory management
    - Door unlocking (key, bomb, boss key)
    - Win/lose conditions
    
    This is a "headless" environment - no graphics, just logic.
    """
    
    def __init__(self, semantic_grid: np.ndarray, render_mode: bool = False):
        """
        Initialize the environment.
        
        Args:
            semantic_grid: 2D numpy array of semantic IDs
            render_mode: If True, enables Pygame rendering (optional)
        """
        self.original_grid = np.array(semantic_grid, dtype=np.int64)
        self.grid = self.original_grid.copy()
        self.height, self.width = self.grid.shape
        self.render_mode = render_mode
        
        # Find start and goal positions
        self.start_pos = self._find_position(SEMANTIC_PALETTE['START'])
        self.goal_pos = self._find_position(SEMANTIC_PALETTE['TRIFORCE'])
        
        # Initialize game state
        self.state = GameState(position=self.start_pos if self.start_pos else (0, 0))
        self.done = False
        self.won = False
        self.step_count = 0
        self.max_steps = 10000  # Prevent infinite loops
        
        # Initialize rendering if needed
        self._screen = None
        self._images = {}
        if render_mode:
            self._init_render()
    
    def _find_position(self, target_id: int) -> Optional[Tuple[int, int]]:
        """Find the first occurrence of a tile ID."""
        positions = np.where(self.grid == target_id)
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None
    
    def _find_all_positions(self, target_id: int) -> List[Tuple[int, int]]:
        """Find all occurrences of a tile ID."""
        positions = np.where(self.grid == target_id)
        return list(zip(positions[0].tolist(), positions[1].tolist()))
    
    def reset(self) -> GameState:
        """Reset the environment to initial state."""
        self.grid = self.original_grid.copy()
        self.state = GameState(position=self.start_pos if self.start_pos else (0, 0))
        self.done = False
        self.won = False
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            state: New game state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            return self.state.copy(), 0.0, True, {'msg': 'Episode already done'}
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            return self.state.copy(), -100.0, True, {'msg': 'Max steps exceeded'}
        
        # Get movement delta
        dr, dc = ACTION_DELTAS.get(Action(action), (0, 0))
        current_r, current_c = self.state.position
        new_r, new_c = current_r + dr, current_c + dc
        
        # Check bounds
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            return self.state.copy(), -1.0, False, {'msg': 'Out of bounds'}
        
        target_tile = self.grid[new_r, new_c]
        info = {'msg': ''}
        reward = -0.1  # Small step penalty
        
        # Check if movement is possible
        can_move, new_state, step_reward, step_info = self._try_move(
            (new_r, new_c), target_tile
        )
        
        if can_move:
            self.state = new_state
            reward += step_reward
            info.update(step_info)
            
            # Check win condition
            if target_tile == SEMANTIC_PALETTE['TRIFORCE']:
                self.done = True
                self.won = True
                reward = 100.0
                info['msg'] = 'Victory!'
        else:
            reward = -1.0
            info['msg'] = step_info.get('msg', 'Blocked')
        
        return self.state.copy(), reward, self.done, info
    
    def _try_move(self, target_pos: Tuple[int, int], target_tile: int
                 ) -> Tuple[bool, GameState, float, Dict]:
        """
        Attempt to move to target position.
        
        Returns:
            can_move: Whether movement is possible
            new_state: Updated state if movement succeeds
            reward: Reward for this action
            info: Additional information
        """
        new_state = self.state.copy()
        reward = 0.0
        info = {}
        
        # Blocking tiles - cannot pass
        if target_tile in BLOCKING_IDS:
            return False, self.state, 0.0, {'msg': 'Blocked by wall'}
        
        # Walkable tiles - free movement
        if target_tile in WALKABLE_IDS:
            new_state.position = target_pos
            
            # Handle item pickup
            if target_tile in PICKUP_IDS and target_pos not in new_state.collected_items:
                new_state, pickup_reward, pickup_info = self._pickup_item(
                    new_state, target_pos, target_tile
                )
                reward += pickup_reward
                info.update(pickup_info)
            
            return True, new_state, reward, info
        
        # Conditional tiles - require inventory items
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if target_pos in new_state.opened_doors:
                new_state.position = target_pos
                return True, new_state, 0.0, {'msg': 'Door already open'}
            elif new_state.keys > 0:
                new_state.keys -= 1
                new_state.opened_doors.add(target_pos)
                new_state.position = target_pos
                # Update grid to show door is open
                self.grid[target_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
                return True, new_state, 10.0, {'msg': 'Unlocked door with key'}
            else:
                return False, self.state, 0.0, {'msg': 'Door locked - need key'}
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
            if target_pos in new_state.opened_doors:
                new_state.position = target_pos
                return True, new_state, 0.0, {'msg': 'Wall already bombed'}
            elif new_state.has_bomb:
                new_state.opened_doors.add(target_pos)
                new_state.position = target_pos
                self.grid[target_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
                return True, new_state, 10.0, {'msg': 'Bombed wall'}
            else:
                return False, self.state, 0.0, {'msg': 'Need bombs'}
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if target_pos in new_state.opened_doors:
                new_state.position = target_pos
                return True, new_state, 0.0, {'msg': 'Boss door already open'}
            elif new_state.has_boss_key:
                new_state.opened_doors.add(target_pos)
                new_state.position = target_pos
                self.grid[target_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
                return True, new_state, 20.0, {'msg': 'Opened boss door'}
            else:
                return False, self.state, 0.0, {'msg': 'Need boss key'}
        
        if target_tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
            # For now, assume puzzle doors can be passed
            new_state.position = target_pos
            return True, new_state, 0.0, {'msg': 'Passed puzzle door'}
        
        # Default: allow movement
        new_state.position = target_pos
        return True, new_state, 0.0, info
    
    def _pickup_item(self, state: GameState, pos: Tuple[int, int], tile: int
                    ) -> Tuple[GameState, float, Dict]:
        """Handle item pickup."""
        state.collected_items.add(pos)
        
        if tile == SEMANTIC_PALETTE['KEY_SMALL']:
            state.keys += 1
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 5.0, {'msg': 'Picked up key', 'item': 'key'}
        
        if tile == SEMANTIC_PALETTE['KEY_BOSS']:
            state.has_boss_key = True
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 15.0, {'msg': 'Picked up boss key', 'item': 'boss_key'}
        
        if tile == SEMANTIC_PALETTE['KEY_ITEM']:
            state.has_item = True
            # Assume key items often grant bombs (like in Zelda)
            state.has_bomb = True
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 10.0, {'msg': 'Picked up key item', 'item': 'key_item'}
        
        if tile == SEMANTIC_PALETTE['ITEM_MINOR']:
            self.grid[pos] = SEMANTIC_PALETTE['FLOOR']
            return state, 1.0, {'msg': 'Picked up item', 'item': 'minor'}
        
        return state, 0.0, {}
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions from current state."""
        valid = []
        r, c = self.state.position
        
        for action in Action:
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < self.height and 0 <= nc < self.width:
                tile = self.grid[nr, nc]
                if tile not in BLOCKING_IDS:
                    # Check if we can actually pass this tile
                    if tile in WALKABLE_IDS:
                        valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
                        if self.state.keys > 0 or (nr, nc) in self.state.opened_doors:
                            valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_BOMB']:
                        if self.state.has_bomb or (nr, nc) in self.state.opened_doors:
                            valid.append(int(action))
                    elif tile == SEMANTIC_PALETTE['DOOR_BOSS']:
                        if self.state.has_boss_key or (nr, nc) in self.state.opened_doors:
                            valid.append(int(action))
                    else:
                        valid.append(int(action))
        
        return valid
    
    # ==========================================
    # RENDERING (OPTIONAL - PYGAME)
    # ==========================================
    
    def _init_render(self):
        """Initialize Pygame rendering."""
        try:
            import pygame
            pygame.init()
            
            self.TILE_SIZE = 32
            screen_w = self.width * self.TILE_SIZE
            screen_h = self.height * self.TILE_SIZE + 60  # Extra space for HUD
            
            self._screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("ZAVE: Zelda Validation Environment")
            self._font = pygame.font.SysFont('Arial', 18, bold=True)
            
            self._load_images()
        except ImportError:
            print("Warning: Pygame not available. Rendering disabled.")
            self.render_mode = False
    
    def _load_images(self):
        """Load tile images or create colored fallbacks."""
        import pygame
        
        TILE_SIZE = self.TILE_SIZE
        
        # Color fallbacks for each tile type
        color_map = {
            SEMANTIC_PALETTE['VOID']: (0, 0, 0),
            SEMANTIC_PALETTE['FLOOR']: (200, 180, 140),
            SEMANTIC_PALETTE['WALL']: (70, 70, 150),
            SEMANTIC_PALETTE['BLOCK']: (139, 90, 43),
            SEMANTIC_PALETTE['DOOR_OPEN']: (50, 50, 50),
            SEMANTIC_PALETTE['DOOR_LOCKED']: (139, 69, 19),
            SEMANTIC_PALETTE['DOOR_BOMB']: (100, 100, 100),
            SEMANTIC_PALETTE['DOOR_BOSS']: (200, 50, 50),
            SEMANTIC_PALETTE['DOOR_PUZZLE']: (150, 100, 200),
            SEMANTIC_PALETTE['ENEMY']: (200, 50, 50),
            SEMANTIC_PALETTE['START']: (100, 200, 100),
            SEMANTIC_PALETTE['TRIFORCE']: (255, 215, 0),
            SEMANTIC_PALETTE['KEY_SMALL']: (255, 200, 50),
            SEMANTIC_PALETTE['KEY_BOSS']: (200, 100, 50),
            SEMANTIC_PALETTE['ELEMENT']: (50, 50, 200),
            SEMANTIC_PALETTE['ELEMENT_FLOOR']: (100, 100, 200),
        }
        
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
        
        for tile_id, color in color_map.items():
            # Try to load image
            tile_name = ID_TO_NAME.get(tile_id, 'unknown').lower()
            img_path = os.path.join(assets_dir, f'{tile_name}.png')
            
            if os.path.exists(img_path):
                try:
                    img = pygame.image.load(img_path)
                    self._images[tile_id] = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
                    continue
                except:
                    pass
            
            # Fallback: colored square
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
            surf.fill(color)
            self._images[tile_id] = surf
        
        # Create Link sprite
        link_path = os.path.join(assets_dir, 'link.png')
        if os.path.exists(link_path):
            try:
                img = pygame.image.load(link_path)
                self._link_img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
            except:
                self._link_img = None
        else:
            self._link_img = None
    
    def render(self):
        """Render current state to screen."""
        if not self.render_mode or self._screen is None:
            return
        
        import pygame
        
        # Clear screen
        self._screen.fill((30, 30, 30))
        
        # Draw grid
        for r in range(self.height):
            for c in range(self.width):
                tile_id = self.grid[r, c]
                img = self._images.get(tile_id, self._images.get(SEMANTIC_PALETTE['FLOOR']))
                self._screen.blit(img, (c * self.TILE_SIZE, r * self.TILE_SIZE))
        
        # Draw agent (Link)
        r, c = self.state.position
        x, y = c * self.TILE_SIZE, r * self.TILE_SIZE
        
        if self._link_img:
            self._screen.blit(self._link_img, (x, y))
        else:
            pygame.draw.rect(self._screen, (0, 255, 0), 
                           (x + 4, y + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8))
        
        # Draw HUD
        hud_y = self.height * self.TILE_SIZE
        pygame.draw.rect(self._screen, (0, 0, 0), 
                        (0, hud_y, self.width * self.TILE_SIZE, 60))
        
        hud_text = f"Keys: {self.state.keys} | Bomb: {'Y' if self.state.has_bomb else 'N'} | Boss Key: {'Y' if self.state.has_boss_key else 'N'} | Steps: {self.step_count}"
        text_surf = self._font.render(hud_text, True, (255, 255, 255))
        self._screen.blit(text_surf, (10, hud_y + 10))
        
        status = "WON!" if self.won else ("DONE" if self.done else "Playing...")
        status_surf = self._font.render(status, True, (255, 255, 0) if self.won else (255, 255, 255))
        self._screen.blit(status_surf, (10, hud_y + 35))
        
        pygame.display.flip()
    
    def close(self):
        """Clean up resources."""
        if self.render_mode:
            try:
                import pygame
                pygame.quit()
            except:
                pass


# ==========================================
# MODULE 2: STATE-SPACE A* SOLVER
# ==========================================

class StateSpaceAStar:
    """
    A* pathfinder that operates on game state space, not just positions.
    
    This allows finding solutions that require:
    - Picking up keys before opening doors
    - Getting bombs before bombing walls
    - Proper sequencing of item collection
    """
    
    def __init__(self, env: ZeldaLogicEnv, timeout: int = 100000):
        """
        Initialize the solver.
        
        Args:
            env: ZeldaLogicEnv instance to solve
            timeout: Maximum states to explore
        """
        self.env = env
        self.timeout = timeout
    
    def solve(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Find a solution path using A* on state space.
        
        OPTIMIZED VERSION:
        - No grid copies during search (read-only grid)
        - State-only tracking for doors and items
        - Significant memory and CPU savings
        
        Returns:
            success: Whether a solution was found
            path: List of positions visited
            states_explored: Number of states explored
        """
        self.env.reset()
        
        if self.env.goal_pos is None:
            return False, [], 0
        
        if self.env.start_pos is None:
            return False, [], 0
        
        # Use read-only grid reference (no copies!)
        grid = self.env.original_grid
        height, width = grid.shape
        
        # Priority queue: (f_score, counter, state_hash, state, path)
        start_state = self.env.state.copy()
        start_h = self._heuristic(start_state)
        
        open_set = [(start_h, 0, hash(start_state), start_state, [start_state.position])]
        heapq.heapify(open_set)
        
        closed_set = set()
        g_scores = {hash(start_state): 0}
        
        states_explored = 0
        counter = 1  # Tie-breaker for heap
        
        # Movement deltas: UP, DOWN, LEFT, RIGHT
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set and states_explored < self.timeout:
            _, _, state_hash, current_state, path = heapq.heappop(open_set)
            
            if state_hash in closed_set:
                continue
            
            closed_set.add(state_hash)
            states_explored += 1
            
            # Check win condition
            if current_state.position == self.env.goal_pos:
                return True, path, states_explored
            
            # Explore neighbors using pure state-based logic (NO grid copies)
            curr_r, curr_c = current_state.position
            
            for dr, dc in deltas:
                new_r, new_c = curr_r + dr, curr_c + dc
                
                # Bounds check
                if not (0 <= new_r < height and 0 <= new_c < width):
                    continue
                
                target_pos = (new_r, new_c)
                target_tile = grid[new_r, new_c]
                
                # Determine if move is possible and what state changes occur
                can_move, new_state = self._try_move_pure(
                    current_state, target_pos, target_tile
                )
                
                if not can_move:
                    continue
                
                new_hash = hash(new_state)
                
                if new_hash in closed_set:
                    continue
                
                # Calculate scores
                g_score = g_scores[state_hash] + 1
                
                if new_hash in g_scores and g_score >= g_scores[new_hash]:
                    continue
                
                g_scores[new_hash] = g_score
                f_score = g_score + self._heuristic(new_state)
                
                new_path = path + [new_state.position]
                heapq.heappush(open_set, (f_score, counter, new_hash, new_state, new_path))
                counter += 1
        
        return False, [], states_explored
    
    def _try_move_pure(self, state: GameState, target_pos: Tuple[int, int], 
                       target_tile: int) -> Tuple[bool, GameState]:
        """
        Pure state-based move attempt (no grid modifications).
        
        Returns:
            can_move: Whether the move is valid
            new_state: Updated state if move is valid
        """
        # Blocking tiles - cannot pass
        if target_tile in BLOCKING_IDS:
            return False, state
        
        new_state = state.copy()
        new_state.position = target_pos
        
        # Handle special tiles based on STATE, not grid modifications
        
        # Check if this door was already opened (in state)
        if target_pos in state.opened_doors:
            # Door is open, can pass freely
            return True, new_state
        
        # Check if this item was already collected (in state)
        if target_pos in state.collected_items:
            # Item already collected, treat as floor
            return True, new_state
        
        # Walkable tiles - free movement
        if target_tile in WALKABLE_IDS:
            # Handle item pickup (add to collected_items)
            if target_tile in PICKUP_IDS:
                new_state.collected_items = state.collected_items | {target_pos}
                
                if target_tile == SEMANTIC_PALETTE['KEY_SMALL']:
                    new_state.keys = state.keys + 1
                elif target_tile == SEMANTIC_PALETTE['KEY_BOSS']:
                    new_state.has_boss_key = True
                elif target_tile == SEMANTIC_PALETTE['KEY_ITEM']:
                    new_state.has_item = True
                    new_state.has_bomb = True
            
            return True, new_state
        
        # Conditional tiles - require inventory items
        if target_tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if state.keys > 0:
                new_state.keys = state.keys - 1
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOMB']:
            if state.has_bomb:
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_BOSS']:
            if state.has_boss_key:
                new_state.opened_doors = state.opened_doors | {target_pos}
                return True, new_state
            return False, state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_PUZZLE']:
            # Puzzle doors can be passed (simplified)
            return True, new_state
        
        if target_tile == SEMANTIC_PALETTE['DOOR_OPEN']:
            return True, new_state
        
        # TRIFORCE - goal tile
        if target_tile == SEMANTIC_PALETTE['TRIFORCE']:
            return True, new_state
        
        # Default: allow movement for unknown walkable types
        return True, new_state
    
    def _heuristic(self, state: GameState) -> float:
        """
        Heuristic function for A*.
        
        Uses Manhattan distance to goal, with adjustments for:
        - Missing keys when locked doors are on path
        - Missing items needed for progression
        """
        if self.env.goal_pos is None:
            return float('inf')
        
        pos = state.position
        goal = self.env.goal_pos
        
        # Base: Manhattan distance
        h = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Penalty for missing items needed for doors
        # This is a simplified heuristic
        locked_doors = self.env._find_all_positions(SEMANTIC_PALETTE['DOOR_LOCKED'])
        boss_doors = self.env._find_all_positions(SEMANTIC_PALETTE['DOOR_BOSS'])
        
        # Count unvisited locked doors not yet opened
        unopened_locked = sum(1 for d in locked_doors if d not in state.opened_doors)
        unopened_boss = sum(1 for d in boss_doors if d not in state.opened_doors)
        
        # Add penalty if we don't have enough keys
        if unopened_locked > state.keys:
            h += (unopened_locked - state.keys) * 10
        
        if unopened_boss > 0 and not state.has_boss_key:
            h += 20
        
        return h


# ==========================================
# MODULE 3: SANITY CHECKER
# ==========================================

class SanityChecker:
    """
    Pre-validation checks for map structural validity.
    
    Catches obvious errors before running expensive A* search:
    - Missing start position
    - Missing goal (Triforce)
    - Unreachable items (basic check)
    """
    
    def __init__(self, semantic_grid: np.ndarray):
        self.grid = semantic_grid
        self.height, self.width = self.grid.shape
    
    def check_all(self) -> Tuple[bool, List[str]]:
        """
        Run all sanity checks.
        
        Returns:
            is_valid: Whether map passes all checks
            errors: List of error messages
        """
        errors = []
        
        # Check for start position
        starts = np.where(self.grid == SEMANTIC_PALETTE['START'])
        if len(starts[0]) == 0:
            errors.append("No start position (S) found")
        elif len(starts[0]) > 1:
            errors.append(f"Multiple start positions found: {len(starts[0])}")
        
        # Check for goal (Triforce)
        goals = np.where(self.grid == SEMANTIC_PALETTE['TRIFORCE'])
        if len(goals[0]) == 0:
            errors.append("No goal (Triforce) found")
        
        # Check for completely blocked map
        # FIX: Exclude VOID tiles from total since stitched dungeons have padding
        # VOID represents empty space, not blocked terrain
        walkable_count = np.sum(np.isin(self.grid, list(WALKABLE_IDS)))
        void_count = np.sum(self.grid == SEMANTIC_PALETTE['VOID'])
        total_cells = self.height * self.width
        non_void_cells = total_cells - void_count
        
        # Only check ratio against non-void cells
        if non_void_cells > 0 and walkable_count < 0.05 * non_void_cells:
            errors.append(f"Map is mostly blocked ({walkable_count}/{non_void_cells} walkable, excluding void)")
        
        # Check for doors without possible keys
        locked_doors = np.sum(self.grid == SEMANTIC_PALETTE['DOOR_LOCKED'])
        keys = np.sum(self.grid == SEMANTIC_PALETTE['KEY_SMALL'])
        if locked_doors > 0 and keys == 0:
            errors.append(f"Locked doors ({locked_doors}) but no keys")
        
        boss_doors = np.sum(self.grid == SEMANTIC_PALETTE['DOOR_BOSS'])
        boss_keys = np.sum(self.grid == SEMANTIC_PALETTE['KEY_BOSS'])
        if boss_doors > 0 and boss_keys == 0:
            errors.append(f"Boss door present but no boss key")
        
        return len(errors) == 0, errors
    
    def count_elements(self) -> Dict[str, int]:
        """Count occurrences of each semantic element."""
        counts = {}
        for name, id_val in SEMANTIC_PALETTE.items():
            count = int(np.sum(self.grid == id_val))
            if count > 0:
                counts[name] = count
        return counts


# ==========================================
# MODULE 4: METRICS ENGINE
# ==========================================

class MetricsEngine:
    """
    Calculate validation metrics for a solved map.
    """
    
    @staticmethod
    def calculate_reachability(env: ZeldaLogicEnv, path: List[Tuple[int, int]]) -> float:
        """
        Calculate what fraction of walkable tiles were visited.
        """
        visited = set(path)
        
        # Count total walkable tiles
        walkable = 0
        for r in range(env.height):
            for c in range(env.width):
                if env.original_grid[r, c] in WALKABLE_IDS:
                    walkable += 1
        
        if walkable == 0:
            return 0.0
        
        return len(visited) / walkable
    
    @staticmethod
    def calculate_backtracking(path: List[Tuple[int, int]]) -> float:
        """
        Calculate backtracking score.
        
        Higher score = more revisiting of positions.
        0 = no backtracking (each tile visited once)
        """
        if len(path) <= 1:
            return 0.0
        
        unique_positions = len(set(path))
        total_steps = len(path)
        
        # Backtracking ratio: how many extra steps over unique positions
        return (total_steps - unique_positions) / total_steps
    
    @staticmethod
    def calculate_linearity(path: List[Tuple[int, int]], start: Tuple[int, int], 
                           goal: Tuple[int, int]) -> float:
        """
        Calculate linearity score.
        
        1.0 = perfectly linear (Manhattan distance = path length)
        0.0 = highly non-linear (lots of detours)
        """
        if len(path) <= 1:
            return 1.0
        
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if manhattan == 0:
            return 1.0
        
        return min(1.0, manhattan / len(path))
    
    @staticmethod
    def find_logical_errors(env: ZeldaLogicEnv, path: List[Tuple[int, int]]) -> List[str]:
        """
        Find logical consistency errors.
        
        - Keys behind locked doors (impossible without other routes)
        - Unreachable items
        """
        errors = []
        visited = set(path)
        
        # Check for unvisited keys
        key_positions = env._find_all_positions(SEMANTIC_PALETTE['KEY_SMALL'])
        for kp in key_positions:
            if kp not in visited and kp not in env.state.collected_items:
                errors.append(f"Unreachable key at {kp}")
        
        # Check for unvisited boss key
        boss_key_positions = env._find_all_positions(SEMANTIC_PALETTE['KEY_BOSS'])
        for bp in boss_key_positions:
            if bp not in visited and not env.state.has_boss_key:
                errors.append(f"Unreachable boss key at {bp}")
        
        return errors


# ==========================================
# MODULE 5: DIVERSITY EVALUATOR
# ==========================================

class DiversityEvaluator:
    """
    Evaluate diversity across a batch of generated maps.
    
    Detects mode collapse where generator produces nearly identical outputs.
    """
    
    @staticmethod
    def hamming_distance(grid1: np.ndarray, grid2: np.ndarray) -> float:
        """
        Calculate normalized Hamming distance between two grids.
        
        Returns value in [0, 1] where 0 = identical, 1 = completely different.
        """
        if grid1.shape != grid2.shape:
            return 1.0
        
        total_cells = grid1.size
        different_cells = np.sum(grid1 != grid2)
        
        return different_cells / total_cells
    
    @staticmethod
    def batch_diversity(grids: List[np.ndarray]) -> float:
        """
        Calculate average pairwise Hamming distance for a batch.
        
        Higher score = more diverse outputs.
        """
        n = len(grids)
        if n < 2:
            return 0.0
        
        total_dist = 0.0
        pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += DiversityEvaluator.hamming_distance(grids[i], grids[j])
                pairs += 1
        
        return total_dist / pairs if pairs > 0 else 0.0
    
    @staticmethod
    def structural_diversity(paths: List[List[Tuple[int, int]]]) -> float:
        """
        Calculate diversity based on solution paths.
        
        Different paths suggest different level structures.
        """
        if len(paths) < 2:
            return 0.0
        
        # Convert paths to sets of visited positions
        path_sets = [set(p) for p in paths if p]
        
        if len(path_sets) < 2:
            return 0.0
        
        # Calculate Jaccard distance between paths
        total_dist = 0.0
        pairs = 0
        
        for i in range(len(path_sets)):
            for j in range(i + 1, len(path_sets)):
                intersection = len(path_sets[i] & path_sets[j])
                union = len(path_sets[i] | path_sets[j])
                if union > 0:
                    jaccard = intersection / union
                    total_dist += (1 - jaccard)  # Distance = 1 - similarity
                    pairs += 1
        
        return total_dist / pairs if pairs > 0 else 0.0


# ==========================================
# MODULE 6: MAIN VALIDATOR
# ==========================================

class ZeldaValidator:
    """
    Main validation orchestrator.
    
    Coordinates sanity checking, solving, and metrics calculation.
    """
    
    def __init__(self, calibration_map: np.ndarray = None):
        """
        Initialize validator.
        
        Args:
            calibration_map: Known-solvable map for calibration test
        """
        self.calibration_map = calibration_map
        self.is_calibrated = False
        
        if calibration_map is not None:
            self._run_calibration()
    
    def _run_calibration(self) -> bool:
        """
        Run calibration test on known-solvable map.
        
        This verifies the solver is working correctly.
        """
        if self.calibration_map is None:
            return True
        
        print("Running calibration test...")
        result = self.validate_single(self.calibration_map)
        
        if not result.is_solvable:
            raise RuntimeError(
                f"CALIBRATION FAILED: Known-solvable map was not solved! "
                f"Error: {result.error_message}"
            )
        
        print(f"Calibration passed. Path length: {result.path_length}")
        self.is_calibrated = True
        return True
    
    def validate_single(self, semantic_grid: np.ndarray, 
                       render: bool = False) -> ValidationResult:
        """
        Validate a single map.
        
        Args:
            semantic_grid: 2D numpy array of semantic IDs
            render: If True, show Pygame visualization
            
        Returns:
            ValidationResult with all metrics
        """
        # Step 1: Sanity Check
        checker = SanityChecker(semantic_grid)
        is_valid, errors = checker.check_all()
        
        if not is_valid:
            return ValidationResult(
                is_solvable=False,
                is_valid_syntax=False,
                reachability=0.0,
                path_length=0,
                backtracking_score=0.0,
                logical_errors=errors,
                error_message="; ".join(errors)
            )
        
        # Step 2: Create Environment
        env = ZeldaLogicEnv(semantic_grid, render_mode=render)
        
        # Step 3: Run A* Solver
        solver = StateSpaceAStar(env)
        success, path, states_explored = solver.solve()
        
        if not success:
            return ValidationResult(
                is_solvable=False,
                is_valid_syntax=True,
                reachability=0.0,
                path_length=0,
                backtracking_score=0.0,
                logical_errors=["A* solver failed to find path"],
                error_message=f"No solution found after exploring {states_explored} states"
            )
        
        # Step 4: Calculate Metrics
        reachability = MetricsEngine.calculate_reachability(env, path)
        backtracking = MetricsEngine.calculate_backtracking(path)
        logical_errors = MetricsEngine.find_logical_errors(env, path)
        
        # Step 5: Render if requested
        if render:
            self._visualize_solution(env, path)
        
        env.close()
        
        return ValidationResult(
            is_solvable=True,
            is_valid_syntax=True,
            reachability=reachability,
            path_length=len(path),
            backtracking_score=backtracking,
            logical_errors=logical_errors,
            path=path
        )
    
    def validate_batch(self, grids: List[np.ndarray], 
                      verbose: bool = True) -> BatchValidationResult:
        """
        Validate a batch of maps.
        
        Args:
            grids: List of semantic grids to validate
            verbose: Print progress
            
        Returns:
            BatchValidationResult with aggregate metrics
        """
        results = []
        solvable_count = 0
        valid_count = 0
        
        total_reachability = 0.0
        total_path_length = 0
        total_backtracking = 0.0
        
        paths = []
        
        for i, grid in enumerate(grids):
            if verbose and (i + 1) % 10 == 0:
                print(f"Validating {i + 1}/{len(grids)}...")
            
            result = self.validate_single(grid)
            results.append(result)
            
            if result.is_valid_syntax:
                valid_count += 1
            
            if result.is_solvable:
                solvable_count += 1
                total_reachability += result.reachability
                total_path_length += result.path_length
                total_backtracking += result.backtracking_score
                paths.append(result.path)
        
        # Calculate averages
        n = len(grids)
        n_solvable = max(1, solvable_count)
        
        # Calculate diversity
        diversity = DiversityEvaluator.batch_diversity(grids)
        
        return BatchValidationResult(
            total_maps=n,
            valid_syntax_count=valid_count,
            solvable_count=solvable_count,
            solvability_rate=solvable_count / n if n > 0 else 0.0,
            avg_reachability=total_reachability / n_solvable,
            avg_path_length=total_path_length / n_solvable,
            avg_backtracking=total_backtracking / n_solvable,
            diversity_score=diversity,
            individual_results=results
        )
    
    def _visualize_solution(self, env: ZeldaLogicEnv, path: List[Tuple[int, int]]):
        """Show animated solution using Pygame."""
        import time
        
        try:
            import pygame
        except ImportError:
            print("Pygame not available for visualization")
            return
        
        env.reset()
        env.render()
        time.sleep(0.5)
        
        for i, pos in enumerate(path[1:], 1):
            # Determine action
            prev = path[i - 1]
            dr = pos[0] - prev[0]
            dc = pos[1] - prev[1]
            
            if dr == -1:
                action = 0
            elif dr == 1:
                action = 1
            elif dc == -1:
                action = 2
            else:
                action = 3
            
            env.step(action)
            env.render()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            time.sleep(0.1)
        
        # Show final state
        time.sleep(2)


# ==========================================
# MODULE 7: GRAPH-GUIDED VALIDATOR
# ==========================================

class GraphGuidedValidator:
    """
    Validator that uses graph topology to determine dungeon solvability.
    
    Instead of pathfinding through a stitched grid (which fails when rooms are missing),
    this validator:
    1. Uses the graph to understand logical room connectivity
    2. Validates that paths exist WITHIN each room
    3. Verifies that connected rooms have traversable doorways
    4. Uses graph-based BFS to determine if START can reach TRIFORCE
    
    This approach handles the VGLC dataset limitation where some logical rooms
    are missing from the physical room data.
    """
    
    def __init__(self):
        """Initialize the graph-guided validator."""
        self.validation_cache = {}
    
    def validate_dungeon_with_graph(self, dungeon_data, stitched_result=None) -> 'GraphValidationResult':
        """
        Validate a dungeon using its graph topology.
        
        Args:
            dungeon_data: DungeonData object with rooms and graph
            stitched_result: Optional StitchedDungeon (for visualization)
            
        Returns:
            GraphValidationResult with detailed analysis
        """
        import networkx as nx
        
        graph = dungeon_data.graph
        rooms = dungeon_data.rooms
        existing_room_ids = set(int(k) for k in rooms.keys())
        
        # Step 1: Find START and TRIFORCE nodes from graph
        start_node = None
        triforce_node = None
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if node_data.get('is_start', False):
                start_node = node_id
            if node_data.get('has_triforce', False):
                triforce_node = node_id
        
        if start_node is None or triforce_node is None:
            return GraphValidationResult(
                is_solvable=False,
                graph_path=[],
                missing_rooms=[],
                room_validations={},
                error_message="No START or TRIFORCE node found in graph"
            )
        
        # Step 2: Find shortest path in graph from START to TRIFORCE
        try:
            undirected = graph.to_undirected()
            graph_path = nx.shortest_path(undirected, source=start_node, target=triforce_node)
        except nx.NetworkXNoPath:
            return GraphValidationResult(
                is_solvable=False,
                graph_path=[],
                missing_rooms=[],
                room_validations={},
                error_message="No path exists in graph from START to TRIFORCE"
            )
        
        # Step 3: Check which rooms in the path exist
        missing_rooms = [n for n in graph_path if n not in existing_room_ids]
        existing_in_path = [n for n in graph_path if n in existing_room_ids]
        
        # Step 4: Validate each existing room is internally traversable
        room_validations = {}
        for room_id in existing_in_path:
            room_key = str(room_id)
            if room_key in rooms:
                room_grid = rooms[room_key].grid
                is_traversable, floor_count = self._validate_room_traversability(room_grid)
                room_validations[room_id] = {
                    'is_traversable': is_traversable,
                    'floor_count': floor_count,
                    'shape': room_grid.shape
                }
        
        # Step 5: Determine solvability based on graph analysis
        # A dungeon is "graph-solvable" if:
        # - All existing rooms in the path are internally traversable
        # - OR we can find an alternate path using only existing rooms
        
        all_existing_traversable = all(
            rv['is_traversable'] for rv in room_validations.values()
        )
        
        # Try to find a path using only existing rooms
        subgraph_path = self._find_path_in_existing_rooms(
            graph, start_node, triforce_node, existing_room_ids
        )
        
        is_solvable = (
            len(missing_rooms) == 0 and all_existing_traversable
        ) or (
            subgraph_path is not None and len(subgraph_path) > 0
        )
        
        # Calculate graph-based metrics
        connectivity_score = len(existing_in_path) / len(graph_path) if graph_path else 0
        
        return GraphValidationResult(
            is_solvable=is_solvable,
            graph_path=graph_path,
            subgraph_path=subgraph_path or [],
            missing_rooms=missing_rooms,
            room_validations=room_validations,
            connectivity_score=connectivity_score,
            start_node=start_node,
            triforce_node=triforce_node,
            error_message="" if is_solvable else f"Path requires {len(missing_rooms)} missing rooms"
        )
    
    def _validate_room_traversability(self, room_grid: np.ndarray) -> Tuple[bool, int]:
        """
        Check if a room is internally traversable.
        
        A room is traversable if:
        - It has floor tiles
        - Floor tiles form a connected region
        """
        floor_mask = np.isin(room_grid, list(WALKABLE_IDS))
        floor_count = np.sum(floor_mask)
        
        if floor_count == 0:
            return False, 0
        
        # Check connectivity using flood fill
        visited = np.zeros_like(floor_mask, dtype=bool)
        positions = np.argwhere(floor_mask)
        
        if len(positions) == 0:
            return False, 0
        
        # Start flood fill from first floor position
        start = tuple(positions[0])
        stack = [start]
        connected_count = 0
        
        while stack:
            r, c = stack.pop()
            if visited[r, c]:
                continue
            if not floor_mask[r, c]:
                continue
            
            visited[r, c] = True
            connected_count += 1
            
            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < room_grid.shape[0] and 0 <= nc < room_grid.shape[1]:
                    if not visited[nr, nc] and floor_mask[nr, nc]:
                        stack.append((nr, nc))
        
        # Room is traversable if most floor tiles are connected
        is_traversable = connected_count >= 0.5 * floor_count
        return is_traversable, int(floor_count)
    
    def _find_path_in_existing_rooms(self, graph, start_node, end_node, 
                                     existing_room_ids: Set[int]) -> Optional[List[int]]:
        """
        Try to find a path that only uses existing rooms.
        
        Uses BFS on the subgraph of existing rooms.
        """
        import networkx as nx
        
        # Create subgraph with only existing rooms
        existing_nodes = [n for n in graph.nodes() if n in existing_room_ids]
        
        if start_node not in existing_nodes or end_node not in existing_nodes:
            return None
        
        subgraph = graph.subgraph(existing_nodes).copy()
        
        try:
            undirected = subgraph.to_undirected()
            path = nx.shortest_path(undirected, source=start_node, target=end_node)
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    
    def validate_with_edge_types(self, dungeon_data, inventory_start: Dict = None) -> 'GraphValidationResult':
        """
        Validate considering edge types (locked doors, bombable walls, etc.)
        
        This performs a state-space search on the GRAPH, where:
        - States = (current_node, keys_held, bombs_held, doors_opened)
        - Edges are traversable based on their type and current inventory
        
        Args:
            dungeon_data: DungeonData with rooms and graph
            inventory_start: Initial inventory (default: no keys, no bombs)
            
        Returns:
            GraphValidationResult with solution path
        """
        import networkx as nx
        from collections import deque
        
        if inventory_start is None:
            inventory_start = {'keys': 0, 'bombs': 0, 'boss_key': False}
        
        graph = dungeon_data.graph
        rooms = dungeon_data.rooms
        existing_room_ids = set(int(k) for k in rooms.keys())
        
        # Find START and TRIFORCE
        start_node = None
        triforce_node = None
        key_nodes = []
        bomb_nodes = []
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if node_data.get('is_start', False):
                start_node = node_id
            if node_data.get('has_triforce', False):
                triforce_node = node_id
            if node_data.get('has_key', False):
                key_nodes.append(node_id)
            if 'bomb' in str(node_data.get('contents', [])).lower():
                bomb_nodes.append(node_id)
        
        if start_node is None or triforce_node is None:
            return GraphValidationResult(
                is_solvable=False,
                graph_path=[],
                missing_rooms=[],
                room_validations={},
                error_message="No START or TRIFORCE in graph"
            )
        
        # State-space BFS on the graph
        # State: (current_node, frozenset(collected_keys), frozenset(opened_doors))
        initial_state = (
            start_node,
            frozenset(),  # collected items
            frozenset(),  # opened doors (edge tuples)
            inventory_start['keys']  # initial key count
        )
        
        queue = deque([(initial_state, [start_node])])
        visited = {initial_state}
        
        while queue:
            (current_node, collected, opened, keys), path = queue.popleft()
            
            # Check win
            if current_node == triforce_node:
                return GraphValidationResult(
                    is_solvable=True,
                    graph_path=path,
                    subgraph_path=path,
                    missing_rooms=[n for n in path if n not in existing_room_ids],
                    room_validations={},
                    connectivity_score=1.0,
                    start_node=start_node,
                    triforce_node=triforce_node,
                    error_message=""
                )
            
            # Collect items at current node
            new_collected = collected
            new_keys = keys
            if current_node in key_nodes and current_node not in collected:
                new_collected = collected | {current_node}
                new_keys = keys + 1
            
            # Explore edges
            for neighbor in graph.neighbors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor)
                edge_type = edge_data.get('type', 'open') if edge_data else 'open'
                edge_key = (min(current_node, neighbor), max(current_node, neighbor))
                
                can_traverse = False
                new_opened = opened
                use_key = False
                
                if edge_type == 'open' or edge_type == '':
                    can_traverse = True
                elif edge_type == 'locked':
                    if new_keys > 0 or edge_key in opened:
                        can_traverse = True
                        if edge_key not in opened:
                            new_opened = opened | {edge_key}
                            use_key = True
                elif edge_type == 'soft_locked':
                    can_traverse = True  # One-way but passable
                elif edge_type == 'bombable':
                    # Would need bombs - skip for now unless we have them
                    can_traverse = edge_key in opened
                else:
                    can_traverse = True  # Default: allow passage
                
                if can_traverse:
                    final_keys = new_keys - 1 if use_key else new_keys
                    final_keys = max(0, final_keys)
                    
                    new_state = (neighbor, new_collected, new_opened, final_keys)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [neighbor]))
        
        # No path found
        return GraphValidationResult(
            is_solvable=False,
            graph_path=[],
            subgraph_path=[],
            missing_rooms=[],
            room_validations={},
            connectivity_score=0.0,
            start_node=start_node,
            triforce_node=triforce_node,
            error_message="No valid path considering locked doors and keys"
        )


@dataclass
class GraphValidationResult:
    """Result of graph-guided validation."""
    is_solvable: bool
    graph_path: List[int]  # Path through graph nodes
    subgraph_path: List[int] = field(default_factory=list)  # Path using only existing rooms
    missing_rooms: List[int] = field(default_factory=list)
    room_validations: Dict[int, Dict] = field(default_factory=dict)
    connectivity_score: float = 0.0
    start_node: Optional[int] = None
    triforce_node: Optional[int] = None
    error_message: str = ""


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def create_test_map() -> np.ndarray:
    """Create a simple test map for validation testing."""
    W = SEMANTIC_PALETTE['WALL']
    F = SEMANTIC_PALETTE['FLOOR']
    S = SEMANTIC_PALETTE['START']
    T = SEMANTIC_PALETTE['TRIFORCE']
    K = SEMANTIC_PALETTE['KEY_SMALL']
    L = SEMANTIC_PALETTE['DOOR_LOCKED']
    
    # 11x16 test map - Simple solvable layout
    # Player starts at top-left, gets key, unlocks door, reaches triforce
    # Path: Start -> Key -> Door -> Triforce
    test_map = np.array([
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, S, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, K, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, F, F, F, F, F, F, F, F, F, W],
        [W, F, F, F, F, F, W, W, L, W, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, F, F, F, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, F, T, F, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, F, F, F, W, F, F, F, F, W],
        [W, F, F, F, F, F, W, W, W, W, W, F, F, F, F, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],
    ], dtype=np.int64)
    
    return test_map


# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("=== ZAVE: Zelda AI Validation Environment ===\n")
    
    # Create test map
    test_map = create_test_map()
    print("Created test map (11x16)")
    print(f"Start: {np.where(test_map == SEMANTIC_PALETTE['START'])}")
    print(f"Goal: {np.where(test_map == SEMANTIC_PALETTE['TRIFORCE'])}")
    
    # Run validation
    validator = ZeldaValidator()
    
    print("\n--- Validating Test Map ---")
    result = validator.validate_single(test_map, render=False)
    
    print(f"Solvable: {result.is_solvable}")
    print(f"Valid Syntax: {result.is_valid_syntax}")
    print(f"Path Length: {result.path_length}")
    print(f"Reachability: {result.reachability:.2%}")
    print(f"Backtracking: {result.backtracking_score:.2%}")
    
    if result.logical_errors:
        print(f"Logical Errors: {result.logical_errors}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    print("\n--- Test Complete ---")
