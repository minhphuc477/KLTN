"""
Dungeon Replay Engine - Scientific Visualization System
=======================================================

The main visualization component for replaying A* pathfinding solutions.
Designed for thesis presentation: shows Link solving the dungeon step-by-step.

Architecture:
------------
This engine takes PRE-COMPUTED solutions (from the validator/solver) and
visualizes them. It does NOT run A* during rendering (which would cause
frame drops). The workflow is:

    1. Solver computes path → List[Tuple[int, int]]
    2. ReplayEngine receives path
    3. ReplayEngine animates agent following path at configurable speed

Key Features:
- Smooth lerp-based agent movement (no teleporting)
- Path overlay showing future trajectory
- HUD displaying inventory state (Keys, Bombs, Steps)
- Configurable replay speed (0.25x to 10x)
- Fog of War (optional) - darkens unvisited areas

Scientific Visualization Goals:
------------------------------
1. CORRECTNESS: Visually verify the solver found a valid path
2. INVENTORY: Show key collection/usage at each step
3. METRICS: Display step count, path length, time elapsed
4. REPRODUCIBILITY: Same path always produces same visualization

Usage:
------
    from src.visualization.replay_engine import DungeonReplayEngine
    
    # After solving
    engine = DungeonReplayEngine(dungeon_grid, solution_path)
    engine.run()  # Blocking - opens Pygame window

Author: KLTN Visualization Module
"""

from __future__ import annotations

import sys
import time
import math
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pygame
    from pygame import Surface
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.error("Pygame not available - install with: pip install pygame")

# Local imports
from src.visualization.asset_manager import AssetManager, SEMANTIC_COLORS, SEMANTIC_NAMES
from src.visualization.camera import Camera, create_camera_for_map


# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class ReplayConfig:
    """Configuration for the replay engine."""
    
    # Window settings
    window_width: int = 1280
    window_height: int = 720
    window_title: str = "ZAVE - Zelda AI Validation Environment"
    
    # Tile rendering
    tile_size: int = 32
    
    # Animation
    agent_speed: float = 8.0  # Lerp speed for smooth movement
    step_delay: float = 0.2  # Seconds between path steps
    
    # Visual options
    show_path_overlay: bool = True
    show_hud: bool = True
    show_minimap: bool = True
    show_grid: bool = False
    fog_of_war: bool = False
    
    # Colors
    path_color: Tuple[int, int, int, int] = (100, 200, 255, 100)
    path_future_color: Tuple[int, int, int, int] = (100, 150, 255, 60)
    visited_color: Tuple[int, int, int, int] = (80, 120, 200, 40)
    
    # HUD
    hud_width: int = 220
    hud_height: int = 100
    
    # Speed control
    speed_levels: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    default_speed_index: int = 2


class ReplayState(Enum):
    """Current state of the replay."""
    IDLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    FINISHED = auto()


# ==========================================
# REPLAY ENGINE
# ==========================================

class DungeonReplayEngine:
    """
    Scientific Replay Engine for visualizing solved dungeon paths.
    
    This is the main entry point for thesis visualization. It renders:
    - The dungeon map (using AssetManager for tiles)
    - The agent (Link) following the solution path
    - A HUD showing inventory and metrics
    - Path overlay showing the complete trajectory
    
    The engine is designed for REPLAY, not real-time solving. Feed it
    a pre-computed path from your A* solver.
    """
    
    def __init__(
        self,
        dungeon_grid: np.ndarray,
        solution_path: Optional[List[Tuple[int, int]]] = None,
        config: Optional[ReplayConfig] = None,
        solver_result: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the replay engine.
        
        Args:
            dungeon_grid: 2D numpy array of semantic tile IDs
            solution_path: List of (row, col) positions for the agent to follow
            config: Optional configuration (uses defaults if None)
            solver_result: Optional metadata from solver (keys_used, edge_types, etc.)
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required. Install with: pip install pygame")
        
        self.grid = dungeon_grid
        self.path = solution_path or []
        self.config = config or ReplayConfig()
        self.solver_result = solver_result or {}
        
        # Grid dimensions
        self.grid_rows, self.grid_cols = dungeon_grid.shape
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption(self.config.window_title)
        
        # Create window
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height),
            pygame.RESIZABLE
        )
        
        # Initialize components
        self.assets = AssetManager(tile_size=self.config.tile_size)
        self.camera = create_camera_for_map(
            self.grid_rows, self.grid_cols,
            tile_size=self.config.tile_size,
            viewport_width=self.config.window_width - self.config.hud_width,
            viewport_height=self.config.window_height - self.config.hud_height
        )
        
        # Clock for frame timing
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_medium = pygame.font.SysFont('Arial', 14, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 12)
        
        # Replay state
        self.state = ReplayState.IDLE
        self.current_step = 0
        self.step_timer = 0.0
        
        # Speed control
        self.speed_index = self.config.default_speed_index
        self.speed_multiplier = self.config.speed_levels[self.speed_index]
        
        # Agent position (smooth interpolation)
        self._agent_x: float = 0.0  # Column (X)
        self._agent_y: float = 0.0  # Row (Y)
        self._agent_target_x: float = 0.0
        self._agent_target_y: float = 0.0
        
        # Initialize agent position
        if self.path:
            start = self.path[0]
            self._agent_y = float(start[0])
            self._agent_x = float(start[1])
            self._agent_target_y = self._agent_y
            self._agent_target_x = self._agent_x
        else:
            # Find START tile
            start_pos = self._find_tile(21)  # START ID
            if start_pos:
                self._agent_y = float(start_pos[0])
                self._agent_x = float(start_pos[1])
                self._agent_target_y = self._agent_y
                self._agent_target_x = self._agent_x
        
        # Inventory tracking (simulated from path)
        self.keys_held = 0
        self.keys_collected = 0
        self.keys_used = 0
        self.has_bomb = False
        self.has_boss_key = False
        self.steps_taken = 0
        
        # Visited tiles (for fog of war)
        self.visited: set = set()
        if self.path:
            self.visited.add(self.path[0])
        
        # Performance tracking
        self.fps_history: List[float] = []
        
        logger.info(f"ReplayEngine initialized: {self.grid_rows}x{self.grid_cols} map, {len(self.path)} step path")
    
    def _find_tile(self, tile_id: int) -> Optional[Tuple[int, int]]:
        """Find the first occurrence of a tile ID."""
        positions = np.where(self.grid == tile_id)
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None
    
    def run(self) -> None:
        """
        Run the replay engine (blocking).
        
        Opens a Pygame window and enters the main loop. Returns when
        the user closes the window.
        """
        running = True
        last_time = time.time()
        
        while running:
            # Delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_scroll(event.y)
            
            # Update
            self._update(dt)
            
            # Render
            self._render()
            
            # Cap framerate
            self.clock.tick(60)
            
            # Track FPS
            fps = self.clock.get_fps()
            self.fps_history.append(fps)
            if len(self.fps_history) > 60:
                self.fps_history.pop(0)
        
        pygame.quit()
    
    def _handle_resize(self, width: int, height: int) -> None:
        """Handle window resize."""
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.camera.set_viewport_size(
            width - self.config.hud_width,
            height - self.config.hud_height
        )
    
    def _handle_key(self, key: int) -> bool:
        """
        Handle key press.
        
        Returns:
            False if should quit, True otherwise
        """
        if key == pygame.K_ESCAPE:
            return False
        
        elif key == pygame.K_SPACE:
            # Toggle play/pause
            if self.state == ReplayState.IDLE or self.state == ReplayState.PAUSED:
                self.state = ReplayState.PLAYING
            elif self.state == ReplayState.PLAYING:
                self.state = ReplayState.PAUSED
            elif self.state == ReplayState.FINISHED:
                # Restart
                self._reset_replay()
                self.state = ReplayState.PLAYING
        
        elif key == pygame.K_r:
            # Reset
            self._reset_replay()
        
        elif key == pygame.K_RIGHT:
            # Step forward
            if self.current_step < len(self.path) - 1:
                self._advance_step()
        
        elif key == pygame.K_LEFT:
            # Step backward
            if self.current_step > 0:
                self.current_step -= 1
                pos = self.path[self.current_step]
                self._agent_y = float(pos[0])
                self._agent_x = float(pos[1])
                self._agent_target_y = self._agent_y
                self._agent_target_x = self._agent_x
        
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            # Increase speed
            if self.speed_index < len(self.config.speed_levels) - 1:
                self.speed_index += 1
                self.speed_multiplier = self.config.speed_levels[self.speed_index]
        
        elif key == pygame.K_MINUS:
            # Decrease speed
            if self.speed_index > 0:
                self.speed_index -= 1
                self.speed_multiplier = self.config.speed_levels[self.speed_index]
        
        elif key == pygame.K_h:
            # Toggle HUD
            self.config.show_hud = not self.config.show_hud
        
        elif key == pygame.K_p:
            # Toggle path overlay
            self.config.show_path_overlay = not self.config.show_path_overlay
        
        elif key == pygame.K_m:
            # Toggle minimap
            self.config.show_minimap = not self.config.show_minimap
        
        elif key == pygame.K_g:
            # Toggle grid
            self.config.show_grid = not self.config.show_grid
        
        elif key == pygame.K_f:
            # Toggle fog of war
            self.config.fog_of_war = not self.config.fog_of_war
        
        elif key == pygame.K_F1:
            # Show help (TODO: implement help overlay)
            pass
        
        return True
    
    def _handle_scroll(self, direction: int) -> None:
        """Handle mouse scroll for zoom."""
        # For now, just log - could implement zoom here
        pass
    
    def _reset_replay(self) -> None:
        """Reset replay to beginning."""
        self.current_step = 0
        self.step_timer = 0.0
        self.state = ReplayState.IDLE
        
        # Reset inventory
        self.keys_held = 0
        self.keys_collected = 0
        self.keys_used = 0
        self.has_bomb = False
        self.has_boss_key = False
        self.steps_taken = 0
        
        # Reset visited
        self.visited.clear()
        if self.path:
            self.visited.add(self.path[0])
            start = self.path[0]
            self._agent_y = float(start[0])
            self._agent_x = float(start[1])
            self._agent_target_y = self._agent_y
            self._agent_target_x = self._agent_x
    
    def _update(self, dt: float) -> None:
        """Update game state."""
        # Apply speed multiplier to delta time for animations
        effective_dt = dt * self.speed_multiplier
        
        # Update camera
        world_x = self._agent_x * self.config.tile_size + self.config.tile_size / 2
        world_y = self._agent_y * self.config.tile_size + self.config.tile_size / 2
        self.camera.set_target(world_x, world_y)
        self.camera.update(dt)  # Camera uses real dt for smooth feel
        
        # Smooth agent movement
        if self._agent_x != self._agent_target_x or self._agent_y != self._agent_target_y:
            lerp_t = min(1.0, effective_dt * self.config.agent_speed)
            self._agent_x += (self._agent_target_x - self._agent_x) * lerp_t
            self._agent_y += (self._agent_target_y - self._agent_y) * lerp_t
            
            # Snap if close enough
            if abs(self._agent_x - self._agent_target_x) < 0.01:
                self._agent_x = self._agent_target_x
            if abs(self._agent_y - self._agent_target_y) < 0.01:
                self._agent_y = self._agent_target_y
        
        # Auto-advance in PLAYING state
        if self.state == ReplayState.PLAYING:
            self.step_timer += effective_dt
            
            # Only advance when agent has reached current target
            agent_at_target = (
                abs(self._agent_x - self._agent_target_x) < 0.1 and
                abs(self._agent_y - self._agent_target_y) < 0.1
            )
            
            if self.step_timer >= self.config.step_delay and agent_at_target:
                self.step_timer = 0.0
                
                if self.current_step < len(self.path) - 1:
                    self._advance_step()
                else:
                    self.state = ReplayState.FINISHED
    
    def _advance_step(self) -> None:
        """Advance to the next step in the path."""
        if self.current_step >= len(self.path) - 1:
            return
        
        self.current_step += 1
        self.steps_taken += 1
        
        pos = self.path[self.current_step]
        self._agent_target_y = float(pos[0])
        self._agent_target_x = float(pos[1])
        
        # Track visited
        self.visited.add(pos)
        
        # Check for item pickups
        tile = self.grid[pos[0], pos[1]]
        if tile == 30:  # KEY_SMALL
            self.keys_held += 1
            self.keys_collected += 1
        elif tile == 31:  # KEY_BOSS
            self.has_boss_key = True
        elif tile == 32:  # KEY_ITEM (gives bomb ability)
            self.has_bomb = True
        
        # Check for door unlocking (simplified)
        if tile == 11:  # DOOR_LOCKED
            if self.keys_held > 0:
                self.keys_held -= 1
                self.keys_used += 1
    
    def _render(self) -> None:
        """Render the current frame."""
        # Clear screen
        self.screen.fill((25, 27, 35))
        
        # Calculate viewport area
        viewport_w = self.screen.get_width() - (self.config.hud_width if self.config.show_hud else 0)
        viewport_h = self.screen.get_height() - self.config.hud_height
        
        # Create map surface
        map_surface = Surface((viewport_w, viewport_h))
        map_surface.fill((20, 22, 30))
        
        # Get visible tile range for culling
        start_row, end_row, start_col, end_col = self.camera.get_visible_tile_range()
        
        # Render tiles
        for row in range(start_row, min(end_row, self.grid_rows)):
            for col in range(start_col, min(end_col, self.grid_cols)):
                tile_id = self.grid[row, col]
                
                # Skip void tiles
                if tile_id == 0:
                    continue
                
                # Get screen position
                screen_x, screen_y = self.camera.grid_to_screen(row, col)
                
                # Apply fog of war
                if self.config.fog_of_war and (row, col) not in self.visited:
                    # Darken unvisited tiles
                    tile_surface = self.assets.get_tile(tile_id).copy()
                    dark_overlay = Surface((self.config.tile_size, self.config.tile_size), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 180))
                    tile_surface.blit(dark_overlay, (0, 0))
                else:
                    tile_surface = self.assets.get_tile(tile_id)
                
                map_surface.blit(tile_surface, (screen_x, screen_y))
        
        # Render grid overlay
        if self.config.show_grid:
            self._render_grid(map_surface)
        
        # Render path overlay
        if self.config.show_path_overlay and self.path:
            self._render_path(map_surface)
        
        # Render agent
        self._render_agent(map_surface)
        
        # Blit map to screen
        self.screen.blit(map_surface, (0, 0))
        
        # Render HUD
        if self.config.show_hud:
            self._render_hud()
        
        # Render status bar
        self._render_status_bar()
        
        # Render minimap
        if self.config.show_minimap:
            self._render_minimap()
        
        pygame.display.flip()
    
    def _render_grid(self, surface: Surface) -> None:
        """Render grid lines."""
        grid_color = (60, 65, 80)
        
        start_row, end_row, start_col, end_col = self.camera.get_visible_tile_range()
        
        for row in range(start_row, min(end_row + 1, self.grid_rows + 1)):
            screen_x, screen_y = self.camera.grid_to_screen(row, start_col)
            end_x, _ = self.camera.grid_to_screen(row, min(end_col, self.grid_cols))
            pygame.draw.line(surface, grid_color, (screen_x, screen_y), (end_x, screen_y), 1)
        
        for col in range(start_col, min(end_col + 1, self.grid_cols + 1)):
            screen_x, screen_y = self.camera.grid_to_screen(start_row, col)
            _, end_y = self.camera.grid_to_screen(min(end_row, self.grid_rows), col)
            pygame.draw.line(surface, grid_color, (screen_x, screen_y), (screen_x, end_y), 1)
    
    def _render_path(self, surface: Surface) -> None:
        """Render the solution path overlay."""
        if not self.path:
            return
        
        # Past path (visited)
        for i in range(self.current_step + 1):
            pos = self.path[i]
            screen_x, screen_y = self.camera.grid_to_screen(pos[0], pos[1])
            
            path_surf = Surface((self.config.tile_size, self.config.tile_size), pygame.SRCALPHA)
            path_surf.fill(self.config.visited_color)
            surface.blit(path_surf, (screen_x, screen_y))
        
        # Future path
        for i in range(self.current_step + 1, len(self.path)):
            pos = self.path[i]
            screen_x, screen_y = self.camera.grid_to_screen(pos[0], pos[1])
            
            path_surf = Surface((self.config.tile_size, self.config.tile_size), pygame.SRCALPHA)
            path_surf.fill(self.config.path_future_color)
            surface.blit(path_surf, (screen_x, screen_y))
        
        # Draw path line
        if len(self.path) > 1:
            points = []
            for pos in self.path:
                screen_x, screen_y = self.camera.grid_to_screen(pos[0], pos[1])
                center_x = screen_x + self.config.tile_size // 2
                center_y = screen_y + self.config.tile_size // 2
                points.append((center_x, center_y))
            
            if len(points) >= 2:
                pygame.draw.lines(surface, (100, 150, 255), False, points, 2)
    
    def _render_agent(self, surface: Surface) -> None:
        """Render the agent (Link)."""
        # Calculate screen position from smooth world position
        world_x = self._agent_x * self.config.tile_size
        world_y = self._agent_y * self.config.tile_size
        screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)
        
        # Get Link sprite
        link = self.assets.get_link_sprite()
        
        # Center the sprite in the tile
        offset = 2  # Sprite is tile_size - 4
        surface.blit(link, (screen_x + offset, screen_y + offset))
    
    def _render_hud(self) -> None:
        """Render the HUD sidebar."""
        hud_x = self.screen.get_width() - self.config.hud_width
        hud_rect = pygame.Rect(hud_x, 0, self.config.hud_width, self.screen.get_height())
        
        # Background
        hud_surface = Surface((self.config.hud_width, self.screen.get_height()), pygame.SRCALPHA)
        pygame.draw.rect(hud_surface, (35, 37, 45, 240), hud_surface.get_rect())
        pygame.draw.line(hud_surface, (60, 65, 80), (0, 0), (0, self.screen.get_height()), 2)
        
        y = 10
        
        # Title
        title = self.font_large.render("ZAVE", True, (100, 200, 255))
        hud_surface.blit(title, (10, y))
        y += 30
        
        # Subtitle
        subtitle = self.font_small.render("Zelda AI Validation", True, (150, 155, 165))
        hud_surface.blit(subtitle, (10, y))
        y += 25
        
        # Divider
        pygame.draw.line(hud_surface, (60, 65, 80), (10, y), (self.config.hud_width - 10, y))
        y += 15
        
        # Inventory
        inv_title = self.font_medium.render("Inventory", True, (255, 220, 100))
        hud_surface.blit(inv_title, (10, y))
        y += 22
        
        # Keys
        keys_text = f"Keys: {self.keys_held}"
        if self.keys_collected > 0:
            keys_text += f" ({self.keys_collected} found, {self.keys_used} used)"
        keys_surf = self.font_small.render(keys_text, True, (255, 220, 100))
        hud_surface.blit(keys_surf, (15, y))
        y += 18
        
        # Boss Key
        boss_text = f"Boss Key: {'✓' if self.has_boss_key else '✗'}"
        boss_color = (100, 255, 100) if self.has_boss_key else (150, 155, 165)
        boss_surf = self.font_small.render(boss_text, True, boss_color)
        hud_surface.blit(boss_surf, (15, y))
        y += 18
        
        # Bomb
        bomb_text = f"Bomb: {'✓' if self.has_bomb else '✗'}"
        bomb_color = (100, 255, 100) if self.has_bomb else (150, 155, 165)
        bomb_surf = self.font_small.render(bomb_text, True, bomb_color)
        hud_surface.blit(bomb_surf, (15, y))
        y += 25
        
        # Divider
        pygame.draw.line(hud_surface, (60, 65, 80), (10, y), (self.config.hud_width - 10, y))
        y += 15
        
        # Metrics
        metrics_title = self.font_medium.render("Metrics", True, (150, 200, 255))
        hud_surface.blit(metrics_title, (10, y))
        y += 22
        
        # Step counter
        step_text = f"Step: {self.current_step + 1}/{len(self.path)}"
        step_surf = self.font_small.render(step_text, True, (200, 205, 215))
        hud_surface.blit(step_surf, (15, y))
        y += 18
        
        # Speed
        speed_text = f"Speed: {self.speed_multiplier}x"
        speed_color = (100, 255, 100) if self.speed_multiplier == 1.0 else (255, 200, 100)
        speed_surf = self.font_small.render(speed_text, True, speed_color)
        hud_surface.blit(speed_surf, (15, y))
        y += 18
        
        # Progress bar
        y += 5
        progress = self.current_step / max(1, len(self.path) - 1) if self.path else 0
        bar_width = self.config.hud_width - 30
        pygame.draw.rect(hud_surface, (50, 55, 65), (15, y, bar_width, 8))
        pygame.draw.rect(hud_surface, (100, 200, 100), (15, y, int(bar_width * progress), 8))
        y += 20
        
        # Divider
        pygame.draw.line(hud_surface, (60, 65, 80), (10, y), (self.config.hud_width - 10, y))
        y += 15
        
        # Controls
        controls_title = self.font_medium.render("Controls", True, (100, 200, 100))
        hud_surface.blit(controls_title, (10, y))
        y += 20
        
        controls = [
            "SPACE  Play/Pause",
            "← →    Step",
            "R      Reset",
            "+/-    Speed",
            "H      Toggle HUD",
            "P      Toggle Path",
            "M      Minimap",
            "G      Grid",
            "F      Fog of War",
            "ESC    Quit",
        ]
        
        for ctrl in controls:
            ctrl_surf = self.font_small.render(ctrl, True, (120, 125, 135))
            hud_surface.blit(ctrl_surf, (15, y))
            y += 15
        
        self.screen.blit(hud_surface, (hud_x, 0))
    
    def _render_status_bar(self) -> None:
        """Render the bottom status bar."""
        bar_height = self.config.hud_height
        bar_y = self.screen.get_height() - bar_height
        bar_width = self.screen.get_width() - (self.config.hud_width if self.config.show_hud else 0)
        
        # Background
        bar_surface = Surface((bar_width, bar_height), pygame.SRCALPHA)
        pygame.draw.rect(bar_surface, (30, 32, 40, 230), bar_surface.get_rect())
        pygame.draw.line(bar_surface, (60, 65, 80), (0, 0), (bar_width, 0), 2)
        
        # State
        state_text = {
            ReplayState.IDLE: "Ready - Press SPACE to play",
            ReplayState.PLAYING: "Playing...",
            ReplayState.PAUSED: "Paused",
            ReplayState.FINISHED: "★ Complete! Press R to restart ★",
        }
        state_color = {
            ReplayState.IDLE: (200, 205, 215),
            ReplayState.PLAYING: (100, 255, 100),
            ReplayState.PAUSED: (255, 200, 100),
            ReplayState.FINISHED: (255, 215, 0),
        }
        
        msg = state_text.get(self.state, "Unknown")
        color = state_color.get(self.state, (200, 205, 215))
        msg_surf = self.font_medium.render(msg, True, color)
        bar_surface.blit(msg_surf, (10, 10))
        
        # Position
        pos_text = f"Position: ({int(self._agent_y)}, {int(self._agent_x)})"
        pos_surf = self.font_small.render(pos_text, True, (150, 155, 165))
        bar_surface.blit(pos_surf, (10, 35))
        
        # FPS
        avg_fps = sum(self.fps_history) / max(1, len(self.fps_history))
        fps_text = f"FPS: {int(avg_fps)}"
        fps_surf = self.font_small.render(fps_text, True, (150, 155, 165))
        bar_surface.blit(fps_surf, (bar_width - 80, 35))
        
        # Map size
        size_text = f"Map: {self.grid_cols}×{self.grid_rows}"
        size_surf = self.font_small.render(size_text, True, (150, 155, 165))
        bar_surface.blit(size_surf, (bar_width - 80, 55))
        
        self.screen.blit(bar_surface, (0, bar_y))
    
    def _render_minimap(self) -> None:
        """Render a minimap in the corner."""
        minimap_size = 150
        margin = 10
        
        # Position in bottom-right of the viewport
        viewport_w = self.screen.get_width() - (self.config.hud_width if self.config.show_hud else 0)
        mm_x = viewport_w - minimap_size - margin
        mm_y = self.screen.get_height() - self.config.hud_height - minimap_size - margin
        
        # Create minimap surface
        mm_surface = Surface((minimap_size, minimap_size), pygame.SRCALPHA)
        pygame.draw.rect(mm_surface, (40, 42, 50, 220), mm_surface.get_rect(), border_radius=8)
        
        # Calculate scale
        scale = min(
            (minimap_size - 20) / self.grid_cols,
            (minimap_size - 20) / self.grid_rows
        )
        
        offset_x = (minimap_size - self.grid_cols * scale) / 2
        offset_y = (minimap_size - self.grid_rows * scale) / 2
        
        # Draw simplified map
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                tile = self.grid[row, col]
                if tile == 0:  # VOID
                    continue
                
                color = SEMANTIC_COLORS.get(tile, (128, 128, 128))
                
                px = int(offset_x + col * scale)
                py = int(offset_y + row * scale)
                pw = max(1, int(scale))
                ph = max(1, int(scale))
                
                pygame.draw.rect(mm_surface, color, (px, py, pw, ph))
        
        # Draw player position
        px = int(offset_x + self._agent_x * scale)
        py = int(offset_y + self._agent_y * scale)
        pygame.draw.circle(mm_surface, (255, 100, 100), (px, py), max(2, int(scale * 1.5)))
        pygame.draw.circle(mm_surface, (255, 255, 255), (px, py), max(3, int(scale * 1.5)), 1)
        
        # Draw border
        pygame.draw.rect(mm_surface, (70, 75, 90), mm_surface.get_rect(), 2, border_radius=8)
        
        self.screen.blit(mm_surface, (mm_x, mm_y))


# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

def replay_solution(
    dungeon_grid: np.ndarray,
    solution_path: List[Tuple[int, int]],
    config: Optional[ReplayConfig] = None
) -> None:
    """
    Convenience function to replay a solution.
    
    Args:
        dungeon_grid: 2D numpy array of semantic tile IDs
        solution_path: List of (row, col) positions
        config: Optional configuration
    """
    engine = DungeonReplayEngine(dungeon_grid, solution_path, config)
    engine.run()
