"""
GUI Runner for ZAVE (Zelda AI Validation Environment)
====================================================

Interactive visual interface for validating Zelda dungeon maps.

Features:
- Real-time visualization of map and agent
- Manual play mode (arrow keys)
- Auto-solve mode (A* pathfinding)
- Map loading from processed data

Controls:
- Arrow Keys: Move Link
- SPACE: Run A* solver (auto-solve)
- R: Reset map
- N: Next map (if multiple loaded)
- P: Previous map
- ESC: Quit

Author: KLTN Thesis Project
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.validator import (
    ZeldaLogicEnv, 
    ZeldaValidator, 
    StateSpaceAStar,
    SanityChecker,
    create_test_map,
    SEMANTIC_PALETTE,
    Action
)

# Try to import Pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: Pygame not installed. Run 'pip install pygame' for GUI support.")


class ZeldaGUI:
    """
    Interactive GUI for Zelda dungeon validation.
    """
    
    def __init__(self, maps: list = None):
        """
        Initialize GUI.
        
        Args:
            maps: List of semantic grids to visualize
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for GUI. Install with: pip install pygame")
        
        self.maps = maps if maps else [create_test_map()]
        self.current_map_idx = 0
        
        # Initialize Pygame
        pygame.init()
        
        # Display settings
        self.TILE_SIZE = 32
        self.HUD_HEIGHT = 80
        
        # Get map dimensions from first map
        h, w = self.maps[0].shape
        self.screen_w = w * self.TILE_SIZE
        self.screen_h = h * self.TILE_SIZE + self.HUD_HEIGHT
        
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("ZAVE: Zelda AI Validation Environment")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Load assets
        self._load_assets()
        
        # Initialize environment
        self.env = None
        self.solver = None
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        self.message = "Press SPACE to auto-solve, Arrow keys to move"
        
        self._load_current_map()
    
    def _load_assets(self):
        """Load tile images or create colored fallbacks."""
        self.images = {}
        
        # Color definitions for fallback rendering
        color_map = {
            SEMANTIC_PALETTE['VOID']: (20, 20, 20),
            SEMANTIC_PALETTE['FLOOR']: (200, 180, 140),
            SEMANTIC_PALETTE['WALL']: (60, 60, 140),
            SEMANTIC_PALETTE['BLOCK']: (139, 90, 43),
            SEMANTIC_PALETTE['DOOR_OPEN']: (40, 40, 40),
            SEMANTIC_PALETTE['DOOR_LOCKED']: (139, 69, 19),
            SEMANTIC_PALETTE['DOOR_BOMB']: (80, 80, 80),
            SEMANTIC_PALETTE['DOOR_BOSS']: (180, 40, 40),
            SEMANTIC_PALETTE['DOOR_PUZZLE']: (140, 80, 180),
            SEMANTIC_PALETTE['DOOR_SOFT']: (100, 100, 60),
            SEMANTIC_PALETTE['ENEMY']: (200, 50, 50),
            SEMANTIC_PALETTE['START']: (80, 180, 80),
            SEMANTIC_PALETTE['TRIFORCE']: (255, 215, 0),
            SEMANTIC_PALETTE['BOSS']: (150, 20, 20),
            SEMANTIC_PALETTE['KEY_SMALL']: (255, 200, 50),
            SEMANTIC_PALETTE['KEY_BOSS']: (200, 100, 50),
            SEMANTIC_PALETTE['KEY_ITEM']: (100, 200, 255),
            SEMANTIC_PALETTE['ITEM_MINOR']: (200, 200, 200),
            SEMANTIC_PALETTE['ELEMENT']: (30, 30, 200),
            SEMANTIC_PALETTE['ELEMENT_FLOOR']: (60, 60, 180),
            SEMANTIC_PALETTE['STAIR']: (150, 150, 100),
            SEMANTIC_PALETTE['PUZZLE']: (180, 100, 180),
        }
        
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        
        # Try to load images, fall back to colored squares
        for tile_id, color in color_map.items():
            surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
            surf.fill(color)
            
            # Add visual indicators for special tiles
            if tile_id == SEMANTIC_PALETTE['DOOR_LOCKED']:
                # Draw keyhole
                pygame.draw.circle(surf, (255, 200, 50), 
                                 (self.TILE_SIZE//2, self.TILE_SIZE//2 - 4), 4)
                pygame.draw.rect(surf, (255, 200, 50),
                               (self.TILE_SIZE//2 - 2, self.TILE_SIZE//2, 4, 8))
            elif tile_id == SEMANTIC_PALETTE['DOOR_BOMB']:
                # Draw crack pattern
                pygame.draw.line(surf, (40, 40, 40), (8, 8), (24, 24), 2)
                pygame.draw.line(surf, (40, 40, 40), (24, 8), (8, 24), 2)
            elif tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                # Draw key shape
                pygame.draw.circle(surf, (200, 150, 0), (16, 10), 6)
                pygame.draw.rect(surf, (200, 150, 0), (14, 10, 4, 16))
            elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                # Draw triangle
                points = [(16, 4), (4, 28), (28, 28)]
                pygame.draw.polygon(surf, (255, 255, 200), points)
            elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                # Draw enemy indicator
                pygame.draw.circle(surf, (255, 100, 100), (16, 16), 10)
                pygame.draw.circle(surf, (0, 0, 0), (12, 12), 3)
                pygame.draw.circle(surf, (0, 0, 0), (20, 12), 3)
            
            self.images[tile_id] = surf
        
        # Create Link sprite
        self.link_img = pygame.Surface((self.TILE_SIZE - 4, self.TILE_SIZE - 4))
        self.link_img.fill((0, 200, 0))
        # Draw face
        pygame.draw.rect(self.link_img, (255, 200, 150), (6, 2, 16, 12))  # Face
        pygame.draw.circle(self.link_img, (0, 0, 0), (10, 8), 2)  # Eye
        pygame.draw.circle(self.link_img, (0, 0, 0), (18, 8), 2)  # Eye
        # Draw body/tunic
        pygame.draw.rect(self.link_img, (0, 150, 0), (4, 14, 20, 12))
    
    def _load_current_map(self):
        """Load and initialize the current map."""
        current_grid = self.maps[self.current_map_idx]
        self.env = ZeldaLogicEnv(current_grid, render_mode=False)
        self.solver = StateSpaceAStar(self.env)
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        
        # Run sanity check
        checker = SanityChecker(current_grid)
        is_valid, errors = checker.check_all()
        
        if not is_valid:
            self.message = f"Map Error: {errors[0]}"
        else:
            self.message = f"Map {self.current_map_idx + 1}/{len(self.maps)} - Press SPACE to solve"
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    elif event.key == pygame.K_SPACE:
                        # Toggle auto-solve
                        self._start_auto_solve()
                    
                    elif event.key == pygame.K_r:
                        # Reset
                        self._load_current_map()
                        self.message = "Map Reset"
                    
                    elif event.key == pygame.K_n:
                        # Next map
                        self.current_map_idx = (self.current_map_idx + 1) % len(self.maps)
                        self._load_current_map()
                    
                    elif event.key == pygame.K_p:
                        # Previous map
                        self.current_map_idx = (self.current_map_idx - 1) % len(self.maps)
                        self._load_current_map()
                    
                    elif not self.auto_mode:
                        # Manual movement
                        action = None
                        if event.key == pygame.K_UP:
                            action = Action.UP
                        elif event.key == pygame.K_DOWN:
                            action = Action.DOWN
                        elif event.key == pygame.K_LEFT:
                            action = Action.LEFT
                        elif event.key == pygame.K_RIGHT:
                            action = Action.RIGHT
                        
                        if action is not None:
                            self._manual_step(action)
            
            # Auto-solve stepping
            if self.auto_mode and not self.env.done:
                self._auto_step()
            
            # Render
            self._render()
            
            # Cap framerate
            self.clock.tick(30 if not self.auto_mode else 10)
        
        pygame.quit()
    
    def _start_auto_solve(self):
        """Start auto-solve mode."""
        if self.env.done:
            self._load_current_map()
        
        self.message = "Solving..."
        self._render()
        pygame.display.flip()
        
        # Run solver
        success, path, states = self.solver.solve()
        
        if success:
            self.auto_path = path
            self.auto_step_idx = 0
            self.auto_mode = True
            self.env.reset()
            self.message = f"Solution found! Path length: {len(path)}"
        else:
            self.auto_mode = False
            self.message = f"No solution found (explored {states} states)"
    
    def _auto_step(self):
        """Execute one step of auto-solve."""
        if self.auto_step_idx >= len(self.auto_path) - 1:
            self.auto_mode = False
            self.message = "Solution complete!"
            return
        
        self.auto_step_idx += 1
        target = self.auto_path[self.auto_step_idx]
        current = self.env.state.position
        
        # Determine action
        dr = target[0] - current[0]
        dc = target[1] - current[1]
        
        if dr == -1:
            action = Action.UP
        elif dr == 1:
            action = Action.DOWN
        elif dc == -1:
            action = Action.LEFT
        else:
            action = Action.RIGHT
        
        state, reward, done, info = self.env.step(int(action))
        
        if done:
            self.auto_mode = False
            if self.env.won:
                self.message = "AUTO-SOLVE: Victory!"
            else:
                self.message = f"AUTO-SOLVE: Failed - {info.get('msg', '')}"
    
    def _manual_step(self, action: Action):
        """Execute manual step."""
        if self.env.done:
            return
        
        state, reward, done, info = self.env.step(int(action))
        
        if done:
            if self.env.won:
                self.message = "YOU WIN!"
            else:
                self.message = f"Game Over: {info.get('msg', '')}"
        else:
            msg = info.get('msg', '')
            if msg:
                self.message = msg
    
    def _render(self):
        """Render the current state."""
        # Clear screen
        self.screen.fill((30, 30, 30))
        
        h, w = self.env.height, self.env.width
        
        # Draw grid
        for r in range(h):
            for c in range(w):
                tile_id = self.env.grid[r, c]
                img = self.images.get(tile_id, self.images.get(SEMANTIC_PALETTE['FLOOR']))
                self.screen.blit(img, (c * self.TILE_SIZE, r * self.TILE_SIZE))
        
        # Draw solution path (if auto-solving)
        if self.auto_mode and self.auto_path:
            for i, pos in enumerate(self.auto_path[:self.auto_step_idx + 1]):
                r, c = pos
                alpha = 100 + int(155 * (i / len(self.auto_path)))
                path_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                path_surf.fill((0, 255, 0, 50))
                self.screen.blit(path_surf, (c * self.TILE_SIZE, r * self.TILE_SIZE))
        
        # Draw Link
        r, c = self.env.state.position
        x = c * self.TILE_SIZE + 2
        y = r * self.TILE_SIZE + 2
        self.screen.blit(self.link_img, (x, y))
        
        # Draw HUD background
        hud_y = h * self.TILE_SIZE
        pygame.draw.rect(self.screen, (20, 20, 40), (0, hud_y, self.screen_w, self.HUD_HEIGHT))
        
        # Draw HUD content
        # Inventory
        inv_text = f"Keys: {self.env.state.keys}  Bomb: {'✓' if self.env.state.has_bomb else '✗'}  Boss Key: {'✓' if self.env.state.has_boss_key else '✗'}"
        inv_surf = self.font.render(inv_text, True, (255, 255, 255))
        self.screen.blit(inv_surf, (10, hud_y + 5))
        
        # Steps
        steps_text = f"Steps: {self.env.step_count}"
        steps_surf = self.font.render(steps_text, True, (200, 200, 200))
        self.screen.blit(steps_surf, (10, hud_y + 25))
        
        # Message
        msg_color = (255, 255, 0) if self.env.won else (200, 200, 200)
        msg_surf = self.font.render(self.message, True, msg_color)
        self.screen.blit(msg_surf, (10, hud_y + 45))
        
        # Controls hint
        hint = "↑↓←→: Move | SPACE: Solve | R: Reset | N/P: Next/Prev | ESC: Quit"
        hint_surf = self.font.render(hint, True, (100, 100, 100))
        self.screen.blit(hint_surf, (10, hud_y + 65))
        
        pygame.display.flip()


def load_maps_from_adapter():
    """Load processed maps from data adapter."""
    try:
        from data.adapter import IntelligentDataAdapter
        from pathlib import Path
        
        data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
        
        if not data_root.exists():
            print(f"Data folder not found: {data_root}")
            return None
        
        adapter = IntelligentDataAdapter(str(data_root))
        dungeons = adapter.process_all_dungeons()
        
        maps = []
        for dungeon_id, dungeon in dungeons.items():
            for room_id, room in dungeon.rooms.items():
                maps.append(room.grid)
        
        return maps if maps else None
        
    except Exception as e:
        print(f"Error loading maps: {e}")
        return None


def main():
    """Main entry point."""
    print("=== ZAVE GUI Runner ===\n")
    
    if not PYGAME_AVAILABLE:
        print("Pygame is not installed. Please run: pip install pygame")
        return
    
    # Try to load processed maps
    maps = load_maps_from_adapter()
    
    if maps:
        print(f"Loaded {len(maps)} maps from data adapter")
    else:
        print("Using test map")
        maps = [create_test_map()]
    
    # Start GUI
    gui = ZeldaGUI(maps)
    gui.run()


if __name__ == "__main__":
    main()
