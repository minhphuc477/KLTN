"""
ZAVE - Zelda AI Validation Environment
=======================================

Interactive GUI that visualizes dungeon solving step-by-step.

Controls:
- SPACE: Start/Pause auto-solve animation
- R: Reset current dungeon
- N: Next dungeon
- P: Previous dungeon  
- Arrow Keys: Manual movement
- +/-: Speed up/slow down animation
- ESC: Quit
"""

import sys
import os
import time
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
    from pygame import Surface
except ImportError:
    print("Pygame not installed. Run: pip install pygame")
    sys.exit(1)

from Data.zelda_core import (
    ZeldaDungeonAdapter, 
    DungeonSolver, 
    SEMANTIC_PALETTE,
    ROOM_HEIGHT, 
    ROOM_WIDTH
)


# ==========================================
# COLOR PALETTE
# ==========================================
COLORS = {
    0: (20, 20, 30),       # VOID - dark
    1: (139, 119, 101),    # FLOOR - tan
    2: (60, 60, 80),       # WALL - dark gray
    3: (255, 215, 0),      # TRIFORCE - gold
    10: (101, 67, 33),     # DOOR_OPEN - brown
    20: (80, 80, 100),     # BLOCK - gray
    21: (50, 205, 50),     # START - green
    22: (255, 215, 0),     # TRIFORCE - gold
    30: (173, 216, 230),   # STAIR - light blue
    33: (139, 90, 43),     # DOOR_LOCKED - darker brown
    40: (255, 255, 0),     # KEY - yellow
    42: (255, 165, 0),     # ITEM - orange
    50: (255, 0, 0),       # ENEMY - red
    51: (139, 0, 0),       # BOSS - dark red
}

# Path visualization colors
PATH_COLOR = (100, 200, 255)       # Light blue for path
VISITED_COLOR = (70, 70, 100)      # Dark blue for visited
CURRENT_COLOR = (255, 100, 100)    # Red for current position
GOAL_COLOR = (255, 215, 0)         # Gold for goal


@dataclass
class SolveState:
    """State of the solving animation."""
    path: List[Tuple[int, int]]
    visited: set
    current_idx: int
    is_solving: bool
    is_complete: bool
    speed: float  # seconds per step


class DungeonVisualizer:
    """
    Main visualizer class that renders dungeons and solving animations.
    """
    
    def __init__(self, tile_size: int = 8):
        pygame.init()
        pygame.display.set_caption("ZAVE - Zelda AI Validation Environment")
        
        self.tile_size = tile_size
        self.screen = None
        self.clock = pygame.time.Clock()
        
        # Data
        self.adapter = None
        self.dungeons = []  # List of (dungeon_id, stitched_grid, start, goal, path)
        self.current_idx = 0
        
        # Animation state
        self.solve_state = None
        self.last_step_time = 0
        
        # Font
        pygame.font.init()
        self.font = pygame.font.SysFont('consolas', 14)
        self.large_font = pygame.font.SysFont('consolas', 24)
        
    def load_dungeons(self, data_root: str):
        """Load all 18 dungeon variants."""
        self.adapter = ZeldaDungeonAdapter(data_root)
        solver = DungeonSolver()
        
        print("Loading dungeons...")
        self.dungeons = []
        
        for d in range(1, 10):
            for v in [1, 2]:
                try:
                    dungeon = self.adapter.load_dungeon(d, variant=v)
                    stitched = self.adapter.stitch_dungeon(dungeon)
                    
                    # Compute path
                    result = solver.solve(stitched)
                    path = self._compute_path(stitched) if result['solvable'] else []
                    
                    self.dungeons.append({
                        'id': f"D{d}-{v}",
                        'grid': stitched.global_grid,
                        'start': stitched.start_global,
                        'goal': stitched.triforce_global,
                        'path': path,
                        'solvable': result['solvable'],
                        'rooms': len(dungeon.rooms)
                    })
                    
                    status = "✓" if result['solvable'] else "✗"
                    print(f"  {status} D{d}-{v}: {len(path)} steps")
                    
                except Exception as e:
                    print(f"  ✗ D{d}-{v}: Error - {e}")
        
        print(f"Loaded {len(self.dungeons)} dungeons")
        
    def _compute_path(self, stitched) -> List[Tuple[int, int]]:
        """Compute the full path from START to TRIFORCE using A* pathfinding."""
        if stitched.start_global is None or stitched.triforce_global is None:
            return []
        
        # Only truly walkable tiles (floor-like)
        WALKABLE = {1, 10, 21, 22, 30, 33, 40, 42}  # FLOOR, DOOR, START, TRIFORCE, STAIR, KEY, ITEM
        
        grid = stitched.global_grid
        start = stitched.start_global
        goal = stitched.triforce_global
        
        # A* pathfinding for better paths
        import heapq
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(heuristic(start, goal), 0, start, [start])]
        visited = {start}
        
        while open_set:
            f, g, pos, path = heapq.heappop(open_set)
            
            if pos == goal:
                return path
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                
                if (nr, nc) in visited:
                    continue
                
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    tile = grid[nr, nc]
                    if tile in WALKABLE:
                        visited.add((nr, nc))
                        new_g = g + 1
                        new_f = new_g + heuristic((nr, nc), goal)
                        heapq.heappush(open_set, (new_f, new_g, (nr, nc), path + [(nr, nc)]))
        
        return []
    
    def _init_screen(self, grid_shape: Tuple[int, int]):
        """Initialize screen for current dungeon size."""
        h, w = grid_shape
        screen_w = w * self.tile_size + 300  # Extra space for info panel
        screen_h = max(h * self.tile_size + 100, 400)
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        
    def _draw_grid(self, grid: np.ndarray, offset: Tuple[int, int] = (10, 50)):
        """Draw the dungeon grid."""
        ox, oy = offset
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                tile = grid[r, c]
                color = COLORS.get(tile, (128, 128, 128))
                
                rect = pygame.Rect(
                    ox + c * self.tile_size,
                    oy + r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                pygame.draw.rect(self.screen, color, rect)
                
    def _draw_path(self, path: List[Tuple[int, int]], 
                   current_idx: int,
                   offset: Tuple[int, int] = (10, 50)):
        """Draw the solution path with animation."""
        ox, oy = offset
        ts = self.tile_size
        
        if not path:
            return
        
        # Draw the full path as a faint line first (preview)
        if len(path) > 1:
            points = [(ox + c * ts + ts//2, oy + r * ts + ts//2) for r, c in path]
            pygame.draw.lines(self.screen, (50, 80, 120), False, points, 1)
        
        # Draw visited portion of path (already traversed) as bright trail
        for i in range(max(0, current_idx - 20), min(current_idx, len(path))):
            r, c = path[i]
            # Fade effect based on recency
            alpha = 150 + int(105 * (i - (current_idx - 20)) / 20) if current_idx > 20 else 255
            rect = pygame.Rect(
                ox + c * ts + 1,
                oy + r * ts + 1,
                ts - 2,
                ts - 2
            )
            # Draw trail
            s = pygame.Surface((ts - 2, ts - 2))
            s.fill(PATH_COLOR)
            s.set_alpha(min(255, alpha))
            self.screen.blit(s, (rect.x, rect.y))
        
        # Draw current position (Link) as a bright green square
        if current_idx < len(path):
            r, c = path[current_idx]
            
            # Link body (bright green)
            link_rect = pygame.Rect(
                ox + c * ts,
                oy + r * ts,
                ts,
                ts
            )
            pygame.draw.rect(self.screen, (0, 255, 0), link_rect)
            pygame.draw.rect(self.screen, (0, 150, 0), link_rect, 1)
            
            # Direction indicator (show where Link came from)
            if current_idx > 0:
                pr, pc = path[current_idx - 1]
                dr, dc = r - pr, c - pc
                cx = ox + c * ts + ts // 2
                cy = oy + r * ts + ts // 2
                # Draw arrow showing movement direction
                if dr != 0 or dc != 0:
                    pygame.draw.circle(self.screen, (255, 255, 0), 
                                       (cx + dc * ts // 4, cy + dr * ts // 4), 
                                       max(1, ts // 4))
            
    def _draw_markers(self, start: Tuple[int, int], goal: Tuple[int, int],
                      offset: Tuple[int, int] = (10, 50)):
        """Draw START and GOAL markers."""
        ox, oy = offset
        
        # START marker (green S)
        if start:
            sr, sc = start
            text = self.font.render("S", True, (255, 255, 255))
            self.screen.blit(text, (ox + sc * self.tile_size + 2, oy + sr * self.tile_size))
        
        # GOAL marker (gold T for Triforce)
        if goal:
            gr, gc = goal
            text = self.font.render("T", True, (255, 255, 255))
            self.screen.blit(text, (ox + gc * self.tile_size + 2, oy + gr * self.tile_size))
            
    def _draw_info_panel(self, dungeon_data: dict, offset: Tuple[int, int]):
        """Draw information panel on the right side."""
        ox, oy = offset
        
        # Title
        title = self.large_font.render(dungeon_data['id'], True, (255, 255, 255))
        self.screen.blit(title, (ox, oy))
        
        # Stats
        stats = [
            f"Rooms: {dungeon_data['rooms']}",
            f"Grid: {dungeon_data['grid'].shape[0]}×{dungeon_data['grid'].shape[1]}",
            f"Path: {len(dungeon_data['path'])} steps",
            f"Solvable: {'Yes' if dungeon_data['solvable'] else 'No'}",
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (ox, oy + 40 + i * 20))
        
        # Animation status
        if self.solve_state:
            status = "SOLVING..." if self.solve_state.is_solving else "PAUSED"
            if self.solve_state.is_complete:
                status = "COMPLETE!"
            color = (0, 255, 0) if self.solve_state.is_complete else (255, 255, 0)
            text = self.font.render(status, True, color)
            self.screen.blit(text, (ox, oy + 140))
            
            # Progress
            progress = f"Step: {self.solve_state.current_idx}/{len(dungeon_data['path'])}"
            text = self.font.render(progress, True, (200, 200, 200))
            self.screen.blit(text, (ox, oy + 160))
            
            # Speed
            speed_text = f"Speed: {1/self.solve_state.speed:.1f} steps/sec"
            text = self.font.render(speed_text, True, (200, 200, 200))
            self.screen.blit(text, (ox, oy + 180))
        
        # Controls
        controls = [
            "",
            "CONTROLS:",
            "SPACE: Start/Pause",
            "R: Reset",
            "N/P: Next/Prev dungeon",
            "+/-: Speed up/down",
            "ESC: Quit"
        ]
        
        for i, ctrl in enumerate(controls):
            color = (150, 150, 150) if ctrl else (100, 100, 100)
            text = self.font.render(ctrl, True, color)
            self.screen.blit(text, (ox, oy + 220 + i * 18))
            
    def run(self):
        """Main game loop."""
        if not self.dungeons:
            print("No dungeons loaded!")
            return
        
        running = True
        
        while running:
            # Get current dungeon
            dungeon = self.dungeons[self.current_idx]
            
            # Initialize screen if needed
            if self.screen is None:
                self._init_screen(dungeon['grid'].shape)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
                    elif event.key == pygame.K_SPACE:
                        # Toggle solving animation
                        if self.solve_state is None:
                            self.solve_state = SolveState(
                                path=dungeon['path'],
                                visited=set(),
                                current_idx=0,
                                is_solving=True,
                                is_complete=False,
                                speed=0.05  # 20 steps per second
                            )
                        else:
                            self.solve_state.is_solving = not self.solve_state.is_solving
                            
                    elif event.key == pygame.K_r:
                        # Reset
                        self.solve_state = None
                        
                    elif event.key == pygame.K_n:
                        # Next dungeon
                        self.current_idx = (self.current_idx + 1) % len(self.dungeons)
                        self.solve_state = None
                        self.screen = None
                        
                    elif event.key == pygame.K_p:
                        # Previous dungeon
                        self.current_idx = (self.current_idx - 1) % len(self.dungeons)
                        self.solve_state = None
                        self.screen = None
                        
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        # Speed up
                        if self.solve_state:
                            self.solve_state.speed = max(0.01, self.solve_state.speed / 2)
                            
                    elif event.key == pygame.K_MINUS:
                        # Slow down
                        if self.solve_state:
                            self.solve_state.speed = min(1.0, self.solve_state.speed * 2)
            
            # Update animation
            if self.solve_state and self.solve_state.is_solving and not self.solve_state.is_complete:
                current_time = time.time()
                if current_time - self.last_step_time >= self.solve_state.speed:
                    self.solve_state.current_idx += 1
                    self.last_step_time = current_time
                    
                    if self.solve_state.current_idx >= len(self.solve_state.path):
                        self.solve_state.is_complete = True
                        self.solve_state.is_solving = False
            
            # Re-initialize screen if needed (after changing dungeon)
            if self.screen is None:
                self._init_screen(dungeon['grid'].shape)
            
            # Draw
            self.screen.fill((30, 30, 40))
            
            # Title bar
            title = f"ZAVE - Dungeon {self.current_idx + 1}/{len(self.dungeons)}"
            title_text = self.large_font.render(title, True, (255, 255, 255))
            self.screen.blit(title_text, (10, 10))
            
            # Draw dungeon grid
            self._draw_grid(dungeon['grid'])
            
            # Draw path animation
            if self.solve_state:
                self._draw_path(self.solve_state.path, self.solve_state.current_idx)
            
            # Draw markers
            self._draw_markers(dungeon['start'], dungeon['goal'])
            
            # Draw info panel
            grid_width = dungeon['grid'].shape[1] * self.tile_size
            self._draw_info_panel(dungeon, (grid_width + 30, 50))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main entry point."""
    print("="*60)
    print("ZAVE - Zelda AI Validation Environment")
    print("="*60)
    
    # Initialize visualizer
    viz = DungeonVisualizer(tile_size=6)
    
    # Load dungeons
    data_root = os.path.join(os.path.dirname(__file__), "Data", "The Legend of Zelda")
    viz.load_dungeons(data_root)
    
    # Run GUI
    print("\nStarting GUI...")
    print("Press SPACE to start solving animation")
    viz.run()


if __name__ == "__main__":
    main()
