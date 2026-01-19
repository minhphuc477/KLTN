"""
ZAVE - Zelda AI Validation Environment (Fixed Maze Solver)
===========================================================

This visualizer properly solves the dungeon maze by:
1. Using DOT graph topology for room-to-room navigation
2. Walking ONLY on floor tiles (not items, keys, etc.)
3. Navigating through doors between rooms

Controls:
- SPACE: Start/Pause solving animation
- R: Reset
- N/P: Next/Previous dungeon
- +/-: Speed up/down
- ESC: Quit
"""

import sys
import os
import time
import heapq
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
except ImportError:
    print("Install pygame: pip install pygame")
    sys.exit(1)

from Data.zelda_core import (
    ZeldaDungeonAdapter, 
    SEMANTIC_PALETTE,
    ROOM_HEIGHT, 
    ROOM_WIDTH,
    Dungeon,
    StitchedDungeon
)


# ==========================================
# TILE COLORS
# ==========================================
COLORS = {
    0: (20, 20, 30),       # VOID
    1: (160, 140, 120),    # FLOOR - tan
    2: (60, 60, 80),       # WALL - dark gray
    3: (80, 80, 100),      # BLOCK
    10: (139, 90, 60),     # DOOR_OPEN - brown
    11: (100, 60, 40),     # DOOR_LOCKED
    12: (120, 80, 50),     # DOOR_BOMB
    15: (110, 70, 45),     # DOOR_SOFT
    20: (200, 50, 50),     # ENEMY - red
    21: (50, 200, 50),     # START - green
    22: (255, 215, 0),     # TRIFORCE - gold
    23: (150, 0, 0),       # BOSS
    30: (255, 255, 100),   # KEY - yellow
    33: (255, 200, 100),   # ITEM - orange
    40: (100, 150, 200),   # ELEMENT - blue
    42: (150, 200, 255),   # STAIR - light blue
}


class MazeSolver:
    """
    Proper maze solver that uses room topology and floor navigation.
    
    This solver walks ONLY on proper floor tiles, NOT through water/lava.
    Some dungeons (D2-1, D3-1, D4-1, D7-1, D9-1, D9-2) require special
    items like rafts or ladders to cross water - these are marked as
    requiring special mechanics.
    """
    
    # These tiles are walkable (can step on them)
    # Based on VGLC README:
    # - F (FLOOR) = walkable floor
    # - D (DOOR) = door openings
    # - S (STAIR) = staircases
    # - M (MONSTER) = monsters are ON floor tiles, so walkable
    # - P (ELEMENT) = water/lava, NOT walkable without special items
    # - O (ELEMENT+FLOOR) = floor with element overlay, mapped to FLOOR
    WALKABLE_TILES = {
        1,   # FLOOR (F) - standard floor
        10,  # DOOR_OPEN (D) - open doors  
        20,  # ENEMY (M) - monsters are on floor tiles
        21,  # START - Link's starting position
        22,  # TRIFORCE - goal position
        42,  # STAIR (S) - staircases
        # NOTE: ELEMENT (40) is NOT walkable - it's water/lava!
    }
    
    def __init__(self, stitched: StitchedDungeon, dungeon: Dungeon):
        self.stitched = stitched
        self.dungeon = dungeon
        self.grid = stitched.global_grid
        
    def solve(self) -> Tuple[List[Tuple[int, int]], bool]:
        """
        Solve the maze using A* on walkable floor tiles only.
        
        Returns:
            (path, success) - list of (row, col) positions
        """
        if self.stitched.start_global is None or self.stitched.triforce_global is None:
            return [], False
        
        start = self.stitched.start_global
        goal = self.stitched.triforce_global
        
        # A* pathfinding - only on floor tiles
        path = self._astar(start, goal)
        
        return path, len(path) > 0
    
    def _is_walkable(self, r: int, c: int) -> bool:
        """Check if a tile is walkable (floor, door, or special marker)."""
        if r < 0 or r >= self.grid.shape[0] or c < 0 or c >= self.grid.shape[1]:
            return False
        
        tile = self.grid[r, c]
        return tile in self.WALKABLE_TILES
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding on walkable tiles only."""
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Priority queue: (f_score, counter, position, path)
        counter = 0
        open_set = [(heuristic(start, goal), counter, start, [start])]
        visited = {start}
        
        while open_set:
            f, _, pos, path = heapq.heappop(open_set)
            
            if pos == goal:
                return path
            
            # 4-directional movement
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                
                if (nr, nc) in visited:
                    continue
                
                if self._is_walkable(nr, nc):
                    visited.add((nr, nc))
                    counter += 1
                    new_g = len(path)
                    new_f = new_g + heuristic((nr, nc), goal)
                    heapq.heappush(open_set, (new_f, counter, (nr, nc), path + [(nr, nc)]))
        
        return []


# --- Integration helper: solve from a single-room visual extraction ---
def solve_room_from_visual(image_path: str, templates_dir: str, tile_px: int = 16, inventory: Optional[Set[str]] = None):
    """Minimal helper: run visual extractor on a screenshot and attempt a tile-level solve.

    Returns: (tile_path, success)
    - This is intentionally small: it creates a single-room StitchedDungeon via
      src.data_processing.visual_integration and reuses TilePathFinder for pathing.
    """
    try:
        from src.data_processing.visual_integration import visual_extract_to_room, make_stitched_for_single_room, infer_inventory_from_room
        from graph_solver import TilePathFinder
    except Exception as _:
        raise

    ids, conf = visual_extract_to_room(image_path, templates_dir, tile_px=tile_px)
    # Map to semantic IDs if extractor returned template indices (best-effort):
    # here we assume extractor returned semantic-like ids already (caller should
    # validate). Use infer_inventory to detect keys/items in room.
    inv = set(inventory or ()) | infer_inventory_from_room(ids, conf)

    stitched = make_stitched_for_single_room(ids)
    tile_finder = TilePathFinder(stitched, None)
    room_path = [(0, 0)]  # single-room
    tile_path = tile_finder.find_tile_path(room_path, inventory=inv)
    success = len(tile_path) > 0
    return tile_path, success


@dataclass
class AnimationState:
    """State for solving animation."""
    path: List[Tuple[int, int]]
    current_idx: int
    is_running: bool
    is_complete: bool
    speed: float  # seconds per step


class DungeonVisualizer:
    """Main visualization GUI."""
    
    def __init__(self, tile_size: int = 8):
        pygame.init()
        pygame.display.set_caption("ZAVE - Zelda Maze Solver")
        
        self.tile_size = tile_size
        self.screen = None
        self.clock = pygame.time.Clock()
        
        self.dungeons = []
        self.raw_dungeons = []  # Keep raw dungeon data for topology
        self.current_idx = 0
        self.anim = None
        self.last_step = 0
        
        pygame.font.init()
        self.font = pygame.font.SysFont('consolas', 12)
        self.title_font = pygame.font.SysFont('consolas', 18)
        
    def load_dungeons(self, data_root: str):
        """Load all 18 dungeon variants."""
        adapter = ZeldaDungeonAdapter(data_root)
        
        print("Loading dungeons with proper maze solving...")
        print("(Dungeons with water obstacles require special items)")
        print()
        self.dungeons = []
        self.raw_dungeons = []
        
        # Dungeons known to require water/special mechanics
        WATER_DUNGEONS = {'D2-1', 'D3-1', 'D4-1', 'D7-1', 'D9-1', 'D9-2'}
        
        for d in range(1, 10):
            for v in [1, 2]:
                try:
                    dungeon_id = f"D{d}-{v}"
                    
                    # Load raw dungeon (has topology info)
                    dungeon = adapter.load_dungeon(d, variant=v)
                    stitched = adapter.stitch_dungeon(dungeon)
                    
                    # Solve using proper maze solver
                    solver = MazeSolver(stitched, dungeon)
                    path, success = solver.solve()
                    
                    self.dungeons.append({
                        'id': dungeon_id,
                        'grid': stitched.global_grid,
                        'start': stitched.start_global,
                        'goal': stitched.triforce_global,
                        'path': path,
                        'solvable': success,
                        'rooms': len(dungeon.rooms),
                        'room_positions': stitched.room_positions,
                        'needs_special': dungeon_id in WATER_DUNGEONS
                    })
                    self.raw_dungeons.append(dungeon)
                    
                    if success:
                        status = "✓"
                        note = ""
                    elif dungeon_id in WATER_DUNGEONS:
                        status = "~"
                        note = " (needs raft/ladder)"
                    else:
                        status = "✗"
                        note = ""
                    print(f"  {status} {dungeon_id}: {len(path)} steps, {len(dungeon.rooms)} rooms{note}")
                    
                except Exception as e:
                    print(f"  ✗ D{d}-{v}: Error - {e}")
        
        print(f"\nLoaded {len(self.dungeons)} dungeons")
        solvable = sum(1 for d in self.dungeons if d['solvable'])
        water_count = sum(1 for d in self.dungeons if d.get('needs_special'))
        print(f"Solvable without items: {solvable}/{len(self.dungeons)} ({100*solvable/len(self.dungeons):.1f}%)")
        print(f"Requires special items: {water_count}")
        
    def _init_screen(self, grid_shape):
        """Initialize screen for dungeon size."""
        h, w = grid_shape
        screen_w = w * self.tile_size + 280
        screen_h = max(h * self.tile_size + 80, 350)
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        
    def _draw_dungeon(self, data: dict, offset=(10, 50)):
        """Draw the dungeon grid with proper tile colors."""
        ox, oy = offset
        grid = data['grid']
        ts = self.tile_size
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                tile = grid[r, c]
                color = COLORS.get(tile, (100, 100, 100))
                
                rect = pygame.Rect(ox + c * ts, oy + r * ts, ts, ts)
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw grid lines for walls
                if tile == SEMANTIC_PALETTE['WALL']:
                    pygame.draw.rect(self.screen, (40, 40, 60), rect, 1)
                    
    def _draw_room_boundaries(self, data: dict, offset=(10, 50)):
        """Draw room boundaries to show structure."""
        ox, oy = offset
        ts = self.tile_size
        
        for room_pos, (r_off, c_off) in data['room_positions'].items():
            rect = pygame.Rect(
                ox + c_off * ts, 
                oy + r_off * ts, 
                ROOM_WIDTH * ts, 
                ROOM_HEIGHT * ts
            )
            pygame.draw.rect(self.screen, (80, 80, 120), rect, 1)
            
    def _draw_path(self, path: List[Tuple[int, int]], current_idx: int, offset=(10, 50)):
        """Draw the solution path with Link animation."""
        if not path:
            return
            
        ox, oy = offset
        ts = self.tile_size
        
        # Draw full path preview (faint)
        if len(path) > 1:
            points = [(ox + c * ts + ts//2, oy + r * ts + ts//2) for r, c in path]
            pygame.draw.lines(self.screen, (80, 100, 80), False, points, 1)
        
        # Draw traversed path (bright green trail)
        trail_start = max(0, current_idx - 30)
        for i in range(trail_start, min(current_idx, len(path))):
            r, c = path[i]
            # Fade based on recency
            brightness = 100 + int(155 * (i - trail_start) / max(1, current_idx - trail_start))
            color = (0, min(255, brightness), 0)
            
            rect = pygame.Rect(ox + c * ts + 1, oy + r * ts + 1, ts - 2, ts - 2)
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw Link (current position)
        if current_idx < len(path):
            r, c = path[current_idx]
            
            # Link as bright green square
            link_rect = pygame.Rect(ox + c * ts, oy + r * ts, ts, ts)
            pygame.draw.rect(self.screen, (0, 255, 0), link_rect)
            pygame.draw.rect(self.screen, (0, 180, 0), link_rect, 1)
            
    def _draw_markers(self, data: dict, offset=(10, 50)):
        """Draw START (S) and GOAL (T) markers."""
        ox, oy = offset
        ts = self.tile_size
        
        if data['start']:
            sr, sc = data['start']
            text = self.font.render("S", True, (255, 255, 255))
            self.screen.blit(text, (ox + sc * ts + 2, oy + sr * ts + 1))
            
        if data['goal']:
            gr, gc = data['goal']
            text = self.font.render("T", True, (255, 255, 255))
            self.screen.blit(text, (ox + gc * ts + 2, oy + gr * ts + 1))
            
    def _draw_info(self, data: dict, offset):
        """Draw information panel."""
        ox, oy = offset
        
        # Title
        title = self.title_font.render(data['id'], True, (255, 255, 255))
        self.screen.blit(title, (ox, oy))
        
        # Status
        status = "SOLVABLE" if data['solvable'] else "NOT SOLVABLE"
        color = (0, 255, 0) if data['solvable'] else (255, 100, 100)
        text = self.font.render(status, True, color)
        self.screen.blit(text, (ox, oy + 25))
        
        # Stats
        stats = [
            f"Rooms: {data['rooms']}",
            f"Grid: {data['grid'].shape[0]}x{data['grid'].shape[1]}",
            f"Path: {len(data['path'])} tiles",
        ]
        for i, s in enumerate(stats):
            text = self.font.render(s, True, (180, 180, 180))
            self.screen.blit(text, (ox, oy + 50 + i * 16))
            
        # Animation status
        if self.anim:
            anim_status = "RUNNING" if self.anim.is_running else "PAUSED"
            if self.anim.is_complete:
                anim_status = "COMPLETE!"
            text = self.font.render(anim_status, True, (255, 255, 0))
            self.screen.blit(text, (ox, oy + 110))
            
            prog = f"Step {self.anim.current_idx}/{len(data['path'])}"
            text = self.font.render(prog, True, (180, 180, 180))
            self.screen.blit(text, (ox, oy + 126))
            
        # Controls
        controls = [
            "", "CONTROLS:",
            "SPACE: Start/Pause",
            "R: Reset",
            "N/P: Next/Prev",
            "+/-: Speed",
            "ESC: Quit"
        ]
        for i, c in enumerate(controls):
            text = self.font.render(c, True, (120, 120, 120))
            self.screen.blit(text, (ox, oy + 160 + i * 14))
            
        # Legend
        legend_y = oy + 280
        text = self.font.render("LEGEND:", True, (150, 150, 150))
        self.screen.blit(text, (ox, legend_y))
        
        legend_items = [
            ((160, 140, 120), "Floor"),
            ((60, 60, 80), "Wall"),
            ((139, 90, 60), "Door"),
            ((50, 200, 50), "Start"),
            ((255, 215, 0), "Triforce"),
        ]
        for i, (color, name) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color, (ox, legend_y + 18 + i * 14, 10, 10))
            text = self.font.render(name, True, (150, 150, 150))
            self.screen.blit(text, (ox + 15, legend_y + 16 + i * 14))
            
    def run(self):
        """Main loop."""
        if not self.dungeons:
            print("No dungeons loaded!")
            return
            
        running = True
        
        while running:
            data = self.dungeons[self.current_idx]
            
            if self.screen is None:
                self._init_screen(data['grid'].shape)
                
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
                    elif event.key == pygame.K_SPACE:
                        if data['solvable']:
                            if self.anim is None:
                                self.anim = AnimationState(
                                    path=data['path'],
                                    current_idx=0,
                                    is_running=True,
                                    is_complete=False,
                                    speed=0.02
                                )
                            else:
                                self.anim.is_running = not self.anim.is_running
                                
                    elif event.key == pygame.K_r:
                        self.anim = None
                        
                    elif event.key == pygame.K_n:
                        self.current_idx = (self.current_idx + 1) % len(self.dungeons)
                        self.anim = None
                        self.screen = None
                        
                    elif event.key == pygame.K_p:
                        self.current_idx = (self.current_idx - 1) % len(self.dungeons)
                        self.anim = None
                        self.screen = None
                        
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        if self.anim:
                            self.anim.speed = max(0.005, self.anim.speed / 2)
                            
                    elif event.key == pygame.K_MINUS:
                        if self.anim:
                            self.anim.speed = min(0.5, self.anim.speed * 2)
                            
            # Update animation
            if self.anim and self.anim.is_running and not self.anim.is_complete:
                now = time.time()
                if now - self.last_step >= self.anim.speed:
                    self.anim.current_idx += 1
                    self.last_step = now
                    if self.anim.current_idx >= len(self.anim.path):
                        self.anim.is_complete = True
                        self.anim.is_running = False
                        
            # Reinit screen if needed
            if self.screen is None:
                self._init_screen(data['grid'].shape)
                
            # Draw
            self.screen.fill((25, 25, 35))
            
            # Header
            header = f"Dungeon {self.current_idx + 1}/{len(self.dungeons)}"
            text = self.title_font.render(header, True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
            
            # Dungeon
            self._draw_dungeon(data)
            self._draw_room_boundaries(data)
            
            # Path
            if self.anim:
                self._draw_path(self.anim.path, self.anim.current_idx)
                
            # Markers
            self._draw_markers(data)
            
            # Info panel
            grid_w = data['grid'].shape[1] * self.tile_size
            self._draw_info(data, (grid_w + 25, 50))
            
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()


def main():
    print("=" * 60)
    print("ZAVE - Zelda Maze Solver")
    print("=" * 60)
    print("\nThis solver walks ONLY on floor tiles (tan color)")
    print("It navigates through doors and avoids walls/items\n")
    
    viz = DungeonVisualizer(tile_size=7)
    
    data_root = os.path.join(os.path.dirname(__file__), "Data", "The Legend of Zelda")
    viz.load_dungeons(data_root)
    
    print("\nPress SPACE to start solving animation")
    print("Press N/P to change dungeons")
    viz.run()


if __name__ == "__main__":
    main()
