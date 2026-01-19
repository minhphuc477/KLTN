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
    
    Features:
    - Resizable window (drag corners/edges)
    - Zoom in/out with +/- keys or mouse wheel
    - Pan with middle mouse or WASD when zoomed
    - Fullscreen toggle with F11
    """
    
    # Zoom levels available
    ZOOM_LEVELS = [16, 24, 32, 48, 64]
    DEFAULT_ZOOM_IDX = 2  # 32px default
    
    # Minimum window size
    MIN_WIDTH = 400
    MIN_HEIGHT = 300
    
    def __init__(self, maps: list = None, map_names: list = None):
        """
        Initialize GUI.
        
        Args:
            maps: List of semantic grids to visualize
            map_names: List of names for each map
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for GUI. Install with: pip install pygame")
        
        self.maps = maps if maps else [create_test_map()]
        self.map_names = map_names if map_names else [f"Map {i+1}" for i in range(len(self.maps))]
        self.current_map_idx = 0
        
        # Initialize Pygame
        pygame.init()
        
        # Display settings
        self.zoom_idx = self.DEFAULT_ZOOM_IDX
        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
        self.HUD_HEIGHT = 100
        self.SIDEBAR_WIDTH = 220  # Wider for dungeon names
        
        # Get screen info for smart sizing
        display_info = pygame.display.Info()
        max_screen_w = display_info.current_w - 100
        max_screen_h = display_info.current_h - 100
        
        # Calculate initial window size (fit largest map)
        # Handle both raw grids and StitchedDungeon objects
        max_map_h = max(m.global_grid.shape[0] if hasattr(m, 'global_grid') else m.shape[0] for m in self.maps)
        max_map_w = max(m.global_grid.shape[1] if hasattr(m, 'global_grid') else m.shape[1] for m in self.maps)
        
        # Smart sizing: fit map with some padding, but don't exceed screen
        ideal_w = max_map_w * self.TILE_SIZE + self.SIDEBAR_WIDTH
        ideal_h = max_map_h * self.TILE_SIZE + self.HUD_HEIGHT
        
        self.screen_w = min(ideal_w, max_screen_w)
        self.screen_h = min(ideal_h, max_screen_h)
        
        # Ensure minimum size
        self.screen_w = max(self.screen_w, self.MIN_WIDTH)
        self.screen_h = max(self.screen_h, self.MIN_HEIGHT)
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.screen_w, self.screen_h), 
            pygame.RESIZABLE
        )
        pygame.display.set_caption("ZAVE: Zelda AI Validation Environment")
        
        # View offset for panning
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        
        # Fullscreen state
        self.fullscreen = False
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # Load assets
        self._load_assets()
        
        # Initialize environment
        self.env = None
        self.solver = None
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        self.message = "Press SPACE to auto-solve, Arrow keys to move"
        self.show_help = False  # Toggle help overlay
        
        # State-space solver tracking (inventory/edge info)
        self.solver_result = None  # Stores keys_available, keys_used, edge_types etc.
        self.current_keys_held = 0  # Keys currently held during auto-solve
        self.current_keys_used = 0  # Keys used so far during auto-solve
        self.current_edge_types = []  # Edge types traversed so far
        
        self._load_current_map()
        self._center_view()  # Center the map in view
    
    def _load_assets(self):
        """Load tile images - using colored squares for reliability."""
        self.images = {}
        
        # Color definitions for tile rendering
        color_map = {
            SEMANTIC_PALETTE['VOID']: (20, 20, 20),
            SEMANTIC_PALETTE['FLOOR']: (200, 180, 140),
            SEMANTIC_PALETTE['WALL']: (60, 60, 140),
            SEMANTIC_PALETTE['BLOCK']: (139, 90, 43),
            SEMANTIC_PALETTE['DOOR_OPEN']: (100, 80, 60),
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
            SEMANTIC_PALETTE['ELEMENT']: (50, 80, 180),
            SEMANTIC_PALETTE['ELEMENT_FLOOR']: (80, 100, 160),
            SEMANTIC_PALETTE['STAIR']: (120, 100, 80),
            SEMANTIC_PALETTE['PUZZLE']: (180, 100, 180),
        }
        
        # Create colored square tiles for each semantic ID
        for tile_id, color in color_map.items():
            surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
            surf.fill(color)
            
            # Add visual indicators for special tiles
            if tile_id == SEMANTIC_PALETTE['DOOR_LOCKED']:
                # Draw keyhole indicator
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
                # Draw golden triangle
                points = [(16, 4), (4, 28), (28, 28)]
                pygame.draw.polygon(surf, (255, 255, 200), points)
                pygame.draw.polygon(surf, (200, 180, 0), points, 2)
            elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                # Draw enemy indicator (red circle with eyes)
                pygame.draw.circle(surf, (255, 100, 100), (16, 16), 10)
                pygame.draw.circle(surf, (0, 0, 0), (12, 12), 3)
                pygame.draw.circle(surf, (0, 0, 0), (20, 12), 3)
            elif tile_id == SEMANTIC_PALETTE['START']:
                # Draw stair pattern
                pygame.draw.rect(surf, (60, 140, 60), (4, 4, 24, 24))
                for i in range(4):
                    pygame.draw.line(surf, (40, 100, 40), (8, 8+i*6), (24, 8+i*6), 2)
            elif tile_id == SEMANTIC_PALETTE['STAIR']:
                # Draw stair steps
                for i in range(4):
                    pygame.draw.rect(surf, (100, 80, 60), (4+i*4, 20-i*4, 20-i*4, 4))
            elif tile_id == SEMANTIC_PALETTE['WALL']:
                # Add brick pattern to walls
                pygame.draw.rect(surf, (50, 50, 120), (2, 2, 28, 28), 2)
                pygame.draw.line(surf, (70, 70, 150), (0, 16), (32, 16), 1)
                pygame.draw.line(surf, (70, 70, 150), (16, 0), (16, 32), 1)
            elif tile_id == SEMANTIC_PALETTE['BLOCK']:
                # Add block texture
                pygame.draw.rect(surf, (100, 60, 30), (2, 2, 28, 28), 2)
            elif tile_id == SEMANTIC_PALETTE['DOOR_OPEN']:
                # Draw open doorway
                pygame.draw.rect(surf, (40, 30, 20), (8, 0, 16, 32))
            elif tile_id == SEMANTIC_PALETTE['ELEMENT']:
                # Water/lava pattern
                for i in range(4):
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 8, 16, 16), 0, 3.14, 2)
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 16, 16, 16), 3.14, 6.28, 2)
            
            self.images[tile_id] = surf
        
        # Create Link sprite
        self.link_img = self._create_link_sprite()
    
    def _create_link_sprite(self) -> pygame.Surface:
        """Create a detailed Link sprite using pygame drawing."""
        link_img = pygame.Surface((self.TILE_SIZE - 4, self.TILE_SIZE - 4), pygame.SRCALPHA)
        
        # Transparent background
        link_img.fill((0, 0, 0, 0))
        
        # Body colors
        green = (0, 168, 0)
        skin = (252, 216, 168)
        brown = (136, 112, 0)
        dark_green = (0, 120, 0)
        
        # Draw Link's body (green tunic)
        pygame.draw.rect(link_img, green, (8, 12, 12, 12))  # Torso
        pygame.draw.rect(link_img, dark_green, (6, 18, 4, 8))  # Left arm
        pygame.draw.rect(link_img, dark_green, (18, 18, 4, 8))  # Right arm
        
        # Draw head
        pygame.draw.rect(link_img, skin, (8, 2, 12, 10))  # Face
        pygame.draw.circle(link_img, (0, 0, 0), (11, 6), 2)  # Left eye
        pygame.draw.circle(link_img, (0, 0, 0), (17, 6), 2)  # Right eye
        
        # Draw hair/cap (brown)
        pygame.draw.rect(link_img, brown, (6, 0, 16, 4))  # Hair top
        pygame.draw.rect(link_img, brown, (4, 2, 4, 6))  # Hair left
        pygame.draw.rect(link_img, brown, (20, 2, 4, 6))  # Hair right
        
        # Draw shield (brown rectangle on left side)
        pygame.draw.rect(link_img, brown, (2, 14, 6, 10))
        pygame.draw.rect(link_img, (200, 150, 50), (3, 15, 4, 8))  # Shield front
        
        # Draw sword (on right side)
        pygame.draw.rect(link_img, (180, 180, 180), (22, 12, 4, 14))  # Blade
        pygame.draw.rect(link_img, brown, (22, 10, 4, 4))  # Hilt
        
        return link_img

    def load_visual_assets(self, templates_dir: str = None, link_sprite_path: str = None):
        """Optional: override GUI assets with extracted visual tiles/sprites.

        Usage (copy-paste into startup code):
            gui = ZeldaGUI(maps)
            gui.load_visual_assets('data/tileset.png', 'data/link_sprite.png')

        Behaviour:
        - If `templates_dir` is a folder of tile images, create pygame surfaces from them
          and assign to `self.images` keyed by semantic id (best-effort).
        - If `link_sprite_path` is provided, attempt to cut a Link sprite and replace `self.link_img`.
        """
        try:
            from src.data_processing.visual_extractor import extract_grid
            from PIL import Image
        except Exception:
            return

        if templates_dir and os.path.isdir(templates_dir):
            # load any PNGs found and map into fallback image slots
            for fn in sorted(os.listdir(templates_dir)):
                if not fn.lower().endswith('.png'):
                    continue
                name = os.path.splitext(fn)[0]
                try:
                    im = Image.open(os.path.join(templates_dir, fn)).convert('RGBA')
                    surf = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
                    # best-effort: if filename contains 'floor','wall','door','key' map to semantic ids
                    ln = name.lower()
                    if 'floor' in ln or 'f' == ln:
                        self.images[SEMANTIC_PALETTE['FLOOR']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'wall' in ln:
                        self.images[SEMANTIC_PALETTE['WALL']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'door' in ln:
                        self.images[SEMANTIC_PALETTE['DOOR_OPEN']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'key' in ln:
                        self.images[SEMANTIC_PALETTE['KEY']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                except Exception:
                    continue

        if link_sprite_path and os.path.exists(link_sprite_path):
            try:
                im = Image.open(link_sprite_path).convert('RGBA')
                im = im.resize((self.TILE_SIZE - 4, self.TILE_SIZE - 4), Image.NEAREST)
                self.link_img = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
            except Exception as e:
                print(f"Warning: failed to load link sprite from {link_sprite_path}: {e}")
        
        print(f"Loaded {len([k for k in self.images if k in (1,2,10,12)])} visual assets")
        return True

    def load_visual_map(self, image_path: str, templates_dir: str | None = None):
        """Public API: create a GUI map from a screenshot and switch to it.

        - `image_path` can be a full screenshot (HUD allowed).
        - `templates_dir` is passed to the visual extractor (tileset or folder).

        This method is intentionally permissive and returns a bool for success
        so automated tests can call it without a file dialog.
        """
        try:
            from src.data_processing.visual_integration import visual_extract_to_room, make_stitched_for_single_room, infer_inventory_from_room
        except Exception:
            return False

        try:
            ids, conf = visual_extract_to_room(image_path, templates_dir or '')
            # convert single-room semantic -> environment grid expected by GUI
            stitched = make_stitched_for_single_room(ids)
            self.maps = [stitched.global_grid]
            self.current_map_idx = 0
            self._load_current_map()
            self.message = f"Loaded visual map: {image_path}"
            return True
        except Exception as e:
            self.message = f"Visual load failed: {e}"
            return False

    def _load_current_map(self):
        """Load and initialize the current map."""
        current_dungeon = self.maps[self.current_map_idx]
        # Extract grid and graph info from StitchedDungeon
        if hasattr(current_dungeon, 'global_grid'):
            # It's a StitchedDungeon object
            grid = current_dungeon.global_grid
            graph = current_dungeon.graph
            room_to_node = current_dungeon.room_to_node
            room_positions = current_dungeon.room_positions
        else:
            # Legacy: just a grid array
            grid = current_dungeon
            graph = None
            room_to_node = None
            room_positions = None
        
        self.env = ZeldaLogicEnv(grid, render_mode=False, graph=graph, 
                                  room_to_node=room_to_node, room_positions=room_positions)
        self.solver = StateSpaceAStar(self.env)
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        
        # Clear solver result when loading new map
        self.solver_result = None
        self.current_keys_held = 0
        self.current_keys_used = 0
        self.current_edge_types = []
        
        # Run sanity check
        checker = SanityChecker(grid)
        is_valid, errors = checker.check_all()
        
        if not is_valid:
            self.message = f"Map Error: {errors[0]}"
        else:
            self.message = f"Map {self.current_map_idx + 1}/{len(self.maps)} - Press SPACE to solve"
        
        # Auto-fit zoom to show entire map
        self._auto_fit_zoom()
    
    def _center_view(self):
        """Center the current map in the view."""
        if self.env is None:
            return
        map_w = self.env.width * self.TILE_SIZE
        map_h = self.env.height * self.TILE_SIZE
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        self.view_offset_x = max(0, (map_w - view_w) // 2)
        self.view_offset_y = max(0, (map_h - view_h) // 2)
    
    def _auto_fit_zoom(self):
        """Automatically set zoom level to fit the entire map in view."""
        if self.env is None:
            return
        
        view_w = self.screen_w - self.SIDEBAR_WIDTH - 20  # padding
        view_h = self.screen_h - self.HUD_HEIGHT - 20
        
        map_h = self.env.height
        map_w = self.env.width
        
        # Find the largest zoom level that fits
        best_zoom_idx = 0
        for idx, tile_size in enumerate(self.ZOOM_LEVELS):
            if map_w * tile_size <= view_w and map_h * tile_size <= view_h:
                best_zoom_idx = idx
            else:
                break
        
        # Apply the zoom
        if best_zoom_idx != self.zoom_idx:
            self.zoom_idx = best_zoom_idx
            self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
            self._load_assets()
        
        self._center_view()
    
    def _change_zoom(self, delta: int):
        """Change zoom level by delta steps."""
        old_idx = self.zoom_idx
        self.zoom_idx = max(0, min(len(self.ZOOM_LEVELS) - 1, self.zoom_idx + delta))
        if self.zoom_idx != old_idx:
            self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
            self._load_assets()  # Reload assets at new size
            self._center_view()
            self.message = f"Zoom: {self.TILE_SIZE}px"
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_w, self.screen_h = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode(
                (800, 600), pygame.RESIZABLE
            )
            self.screen_w, self.screen_h = 800, 600
        self._center_view()
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.screen_w = max(event.w, self.MIN_WIDTH)
                    self.screen_h = max(event.h, self.MIN_HEIGHT)
                    if not self.fullscreen:
                        self.screen = pygame.display.set_mode(
                            (self.screen_w, self.screen_h), pygame.RESIZABLE
                        )
                
                elif event.type == pygame.MOUSEWHEEL:
                    # Zoom with mouse wheel
                    self._change_zoom(event.y)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 2:  # Middle mouse
                        self.dragging = True
                        self.drag_start = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 2:
                        self.dragging = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = self.drag_start[0] - event.pos[0]
                        dy = self.drag_start[1] - event.pos[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.drag_start = event.pos
                        # Clamp offsets
                        self._clamp_view_offset()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.fullscreen:
                            self._toggle_fullscreen()
                        else:
                            running = False
                    
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    
                    elif event.key == pygame.K_h or event.key == pygame.K_F1:
                        self.show_help = not self.show_help
                    
                    elif event.key == pygame.K_SPACE:
                        self._start_auto_solve()
                    
                    elif event.key == pygame.K_r:
                        self._load_current_map()
                        self._center_view()
                        self.message = "Map Reset"
                    
                    elif event.key == pygame.K_n:
                        self.current_map_idx = (self.current_map_idx + 1) % len(self.maps)
                        self._load_current_map()
                        self._center_view()
                    
                    elif event.key == pygame.K_p:
                        self.current_map_idx = (self.current_map_idx - 1) % len(self.maps)
                        self._load_current_map()
                        self._center_view()
                    
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self._change_zoom(1)
                    
                    elif event.key == pygame.K_MINUS:
                        self._change_zoom(-1)
                    
                    elif event.key == pygame.K_0:
                        # Reset zoom to default
                        self.zoom_idx = self.DEFAULT_ZOOM_IDX
                        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
                        self._load_assets()
                        self._center_view()
                        self.message = "Zoom reset to default"
                    
                    elif event.key == pygame.K_f:
                        # Auto-fit zoom to show entire map
                        self._auto_fit_zoom()
                        self.message = f"Auto-fit: {self.TILE_SIZE}px"
                    
                    elif event.key == pygame.K_c:
                        # Center view on player
                        self._center_on_player()
                    
                    elif event.key == pygame.K_l:
                        ok = self.load_visual_map(os.path.join(os.getcwd(), 'screenshot.png'))
                        if not ok:
                            self.message = "Failed to load ./screenshot.png"
                        
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
                            self._center_on_player()
            
            # Auto-solve stepping
            if self.auto_mode and not self.env.done:
                self._auto_step()
                self._center_on_player()
            
            # Render
            self._render()
            
            # Cap framerate
            self.clock.tick(30 if not self.auto_mode else 10)
        
        pygame.quit()
    
    def _clamp_view_offset(self):
        """Clamp view offset to valid range."""
        if self.env is None:
            return
        map_w = self.env.width * self.TILE_SIZE
        map_h = self.env.height * self.TILE_SIZE
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        max_offset_x = max(0, map_w - view_w)
        max_offset_y = max(0, map_h - view_h)
        
        self.view_offset_x = max(0, min(self.view_offset_x, max_offset_x))
        self.view_offset_y = max(0, min(self.view_offset_y, max_offset_y))
    
    def _center_on_player(self):
        """Center the view on the player position."""
        if self.env is None:
            return
        r, c = self.env.state.position
        player_x = c * self.TILE_SIZE
        player_y = r * self.TILE_SIZE
        
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        self.view_offset_x = player_x - view_w // 2
        self.view_offset_y = player_y - view_h // 2
        self._clamp_view_offset()
    
    def _start_auto_solve(self):
        """Start auto-solve mode using state-space solver with inventory tracking."""
        if self.env.done:
            self._load_current_map()
        
        self.message = "Solving..."
        self._render()
        pygame.display.flip()
        
        # Use state-space graph solver with full inventory tracking
        from Data.zelda_core import DungeonSolver, ValidationMode
        current_dungeon = self.maps[self.current_map_idx]
        
        if hasattr(current_dungeon, 'graph') and current_dungeon.graph:
            solver = DungeonSolver()
            
            # Solve with FULL mode (tracks keys, bombs, etc.)
            result = solver.solve(current_dungeon, mode=ValidationMode.FULL)
            
            if result['solvable']:
                # Store solver result for display
                self.solver_result = result
                
                # Get edge type breakdown
                edge_types = result.get('edge_types', [])
                keys_avail = result.get('keys_available', 0)
                keys_used = result.get('keys_used', 0)
                
                # Count edge types for display
                type_counts = {}
                for et in edge_types:
                    type_counts[et] = type_counts.get(et, 0) + 1
                
                # Try grid pathfinding first
                success, path, teleports = self._smart_grid_path()
                if success:
                    self.auto_path = path
                    self.auto_step_idx = 0
                    self.auto_mode = True
                    self.env.reset()
                    
                    # Build informative message
                    key_info = f"Keys: {keys_avail}→{keys_avail - keys_used}" if keys_used > 0 else ""
                    if teleports > 0:
                        self.message = f"Path: {len(path)} ({teleports} warps) {key_info}"
                    else:
                        self.message = f"Path: {len(path)} (walk) {key_info}"
                else:
                    # Fall back to graph-guided path
                    success2, path2, teleports2 = self._graph_guided_path()
                    if success2:
                        self.auto_path = path2
                        self.auto_step_idx = 0
                        self.auto_mode = True
                        self.env.reset()
                        
                        # Show edge type breakdown
                        edge_summary = " ".join([f"{k[0].upper()}:{v}" for k, v in type_counts.items() if v > 0])
                        key_info = f"Keys: {keys_used}/{keys_avail}" if keys_avail > 0 else ""
                        self.message = f"Graph: {len(path2)} steps | {key_info} | {edge_summary}"
                    else:
                        self.auto_mode = False
                        self.message = f"Graph OK but grid blocked"
            else:
                self.auto_mode = False
                reason = result.get('reason', 'Unknown')
                self.message = f"Not solvable: {reason}"
        else:
            # Fallback to original solver
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
    
    def _smart_grid_path(self):
        """
        Smart pathfinding that prioritizes walking and only warps via STAIRs.
        Returns (success, path, teleport_count).
        """
        from collections import deque
        import networkx as nx
        
        current_dungeon = self.maps[self.current_map_idx]
        grid = current_dungeon.global_grid
        H, W = grid.shape
        
        # Constants
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        WALL = 2
        BLOCK = 3
        VOID = 0
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        if not start or not goal:
            return False, [], 0
        
        # Helper: get room for a position
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        # Helper: find all STAIR tiles in a room
        def get_stairs_in_room(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return []
            ry, rx = current_dungeon.room_positions[room_pos]
            stairs = []
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        stairs.append((y, x))
            return stairs
        
        # Helper: find entry point in room
        def find_entry(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            # Prefer stair, then center, then any walkable
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        # Get graph info for stair connections
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        graph = getattr(current_dungeon, 'graph', None)
        
        # Find stair destinations based on graph connectivity
        def get_stair_destinations(pos):
            """Find where stairs lead to based on graph."""
            if grid[pos[0], pos[1]] != STAIR:
                return []
            
            if not graph:
                return []
            
            current_room = get_room(pos)
            if not current_room:
                return []
            
            current_node = room_to_node.get(current_room)
            if current_node is None:
                return []
            
            destinations = []
            # Check graph neighbors
            for neighbor_node in graph.neighbors(current_node):
                neighbor_room = node_to_room.get(neighbor_node)
                if neighbor_room and neighbor_room in current_dungeon.room_positions:
                    # Find stairs in neighbor room, or entry point
                    neighbor_stairs = get_stairs_in_room(neighbor_room)
                    if neighbor_stairs:
                        destinations.extend(neighbor_stairs)
                    else:
                        entry = find_entry(neighbor_room)
                        if entry:
                            destinations.append(entry)
            
            return destinations
        
        # BFS with stair teleportation
        # State: (position, frozenset of visited stairs to avoid loops)
        initial = (start, frozenset())
        visited = {start}
        queue = deque([(start, [start], 0)])  # pos, path, teleport_count
        
        max_iterations = 100000
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            pos, path, teleports = queue.popleft()
            y, x = pos
            
            # Check if reached goal
            if pos == goal:
                return True, path, teleports
            
            # Get neighbors
            neighbors = []
            
            # 4-directional walking
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    tile = grid[ny, nx]
                    if tile in WALKABLE:
                        neighbors.append(((ny, nx), False))  # (pos, is_teleport)
            
            # Stair teleportation (only if standing on STAIR)
            if grid[y, x] == STAIR:
                for dest in get_stair_destinations(pos):
                    if dest not in visited:
                        neighbors.append((dest, True))
            
            # Expand neighbors
            for (npos, is_teleport) in neighbors:
                if npos not in visited:
                    visited.add(npos)
                    new_teleports = teleports + (1 if is_teleport else 0)
                    queue.append((npos, path + [npos], new_teleports))
        
        # No path found with walking + stairs
        # Fall back to graph-guided teleportation
        return self._graph_guided_path()
    
    def _graph_guided_path(self):
        """Fallback: follow graph path with teleportation when needed."""
        import networkx as nx
        from collections import deque
        
        current_dungeon = self.maps[self.current_map_idx]
        grid = current_dungeon.global_grid
        H, W = grid.shape
        
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        def find_entry(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        def local_bfs(from_pos, to_pos, max_steps=5000):
            if from_pos == to_pos:
                return [from_pos]
            visited = {from_pos}
            queue = deque([(from_pos, [from_pos])])
            while queue and len(visited) < max_steps:
                pos, path = queue.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = pos[0] + dy, pos[1] + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                        if grid[ny, nx] in WALKABLE:
                            if (ny, nx) == to_pos:
                                return path + [(ny, nx)]
                            visited.add((ny, nx))
                            queue.append(((ny, nx), path + [(ny, nx)]))
            return None
        
        # Get graph path
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        graph = current_dungeon.graph
        
        start_room = get_room(start)
        goal_room = get_room(goal)
        start_node = room_to_node.get(start_room)
        goal_node = room_to_node.get(goal_room)
        
        if start_node is None or goal_node is None:
            return False, [], 0
        
        try:
            graph_path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            return False, [], 0
        
        # Build room sequence
        room_sequence = []
        for node in graph_path:
            room = node_to_room.get(node)
            if room and room in current_dungeon.room_positions:
                room_sequence.append(room)
        
        # Build path with teleportation
        full_path = [start]
        current_pos = start
        teleports = 0
        
        for target_room in room_sequence:
            target_pos = find_entry(target_room)
            if not target_pos or current_pos == target_pos:
                continue
            
            segment = local_bfs(current_pos, target_pos)
            if segment:
                full_path.extend(segment[1:])
                current_pos = segment[-1]
            else:
                full_path.append(target_pos)
                current_pos = target_pos
                teleports += 1
        
        # Final to goal
        if current_pos != goal:
            segment = local_bfs(current_pos, goal)
            if segment:
                full_path.extend(segment[1:])
            else:
                full_path.append(goal)
                teleports += 1
        
        if full_path[-1] != goal:
            full_path.append(goal)
        
        return True, full_path, teleports

    def _hybrid_graph_grid_path(self):
        """
        Hybrid pathfinding: use graph to find room sequence, 
        then BFS within each room and teleport between disconnected clusters.
        """
        import networkx as nx
        from collections import deque
        
        current_dungeon = self.maps[self.current_map_idx]
        grid = current_dungeon.global_grid
        H, W = grid.shape
        
        # Constants
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        WALL = 2
        BLOCK = 3
        VOID = 0
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        if not start or not goal:
            return False, []
        
        # Helper: find room containing position
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        # Helper: find passable position in room (prefer stairs, then center)
        def find_entry_point(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            
            # First try to find a stair
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            
            # Then try center
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            
            # Then any walkable tile
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        # Helper: BFS within connected tiles (allows room-to-room through doors)
        def local_bfs(from_pos, to_pos, max_steps=5000):
            if from_pos == to_pos:
                return [from_pos]
            
            visited = {from_pos}
            queue = deque([(from_pos, [from_pos])])
            
            while queue and len(visited) < max_steps:
                pos, path = queue.popleft()
                y, x = pos
                
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                        tile = grid[ny, nx]
                        if tile in WALKABLE:
                            if (ny, nx) == to_pos:
                                return path + [(ny, nx)]
                            visited.add((ny, nx))
                            queue.append(((ny, nx), path + [(ny, nx)]))
            return None
        
        # Step 1: Get graph path from start room to goal room
        start_room = get_room(start)
        goal_room = get_room(goal)
        
        if not start_room or not goal_room:
            return False, []
        
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        
        start_node = room_to_node.get(start_room)
        goal_node = room_to_node.get(goal_room)
        
        if start_node is None or goal_node is None:
            # Fallback to direct BFS
            path = local_bfs(start, goal, max_steps=50000)
            return (True, path) if path else (False, [])
        
        # Get graph path
        graph = current_dungeon.graph
        try:
            graph_path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            return False, []
        
        # Step 2: Build room sequence (skip unmapped nodes)
        room_sequence = []
        for node in graph_path:
            room = node_to_room.get(node)
            if room and room in current_dungeon.room_positions:
                room_sequence.append(room)
        
        if not room_sequence:
            return False, []
        
        # Step 3: Build full path by connecting rooms
        full_path = [start]
        current_pos = start
        
        for target_room in room_sequence:
            target_pos = find_entry_point(target_room)
            if not target_pos:
                continue
            
            # Skip if we're already at this target
            if current_pos == target_pos:
                continue
            
            # Try BFS to target
            path_segment = local_bfs(current_pos, target_pos)
            
            if path_segment:
                # Connected physically - add path
                full_path.extend(path_segment[1:])
                current_pos = path_segment[-1]
            else:
                # Not connected physically - teleport!
                full_path.append(target_pos)
                current_pos = target_pos
        
        # Final segment to goal
        if current_pos != goal:
            final_segment = local_bfs(current_pos, goal)
            if final_segment:
                full_path.extend(final_segment[1:])
            else:
                # Teleport to goal as last resort
                full_path.append(goal)
        
        # Validate path ends at goal
        if full_path[-1] != goal:
            full_path.append(goal)
        
        return True, full_path
    
    def _auto_step(self):
        """Execute one step of auto-solve, handling teleportation."""
        if self.auto_step_idx >= len(self.auto_path) - 1:
            self.auto_mode = False
            self.message = "Solution complete!"
            return
        
        self.auto_step_idx += 1
        target = self.auto_path[self.auto_step_idx]
        current = self.env.state.position
        
        # Check if this is a teleport (non-adjacent move)
        dr = target[0] - current[0]
        dc = target[1] - current[1]
        
        if abs(dr) > 1 or abs(dc) > 1 or (abs(dr) == 1 and abs(dc) == 1):
            # Teleport - directly set position
            self.env.state.position = target
            self.message = f"Teleport! {current} → {target}"
            
            # Check if reached goal
            if target == self.env.goal_pos:
                self.env.won = True
                self.env.done = True
                self.auto_mode = False
                self.message = "AUTO-SOLVE: Victory!"
        else:
            # Normal move
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
        self.screen.fill((25, 25, 35))
        
        h, w = self.env.height, self.env.width
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        # Create map surface for the main view area
        map_surface = pygame.Surface((view_w, view_h))
        map_surface.fill((20, 20, 30))
        
        # Draw grid (only visible tiles for performance)
        start_c = max(0, self.view_offset_x // self.TILE_SIZE)
        start_r = max(0, self.view_offset_y // self.TILE_SIZE)
        end_c = min(w, start_c + (view_w // self.TILE_SIZE) + 2)
        end_r = min(h, start_r + (view_h // self.TILE_SIZE) + 2)
        
        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                tile_id = self.env.grid[r, c]
                img = self.images.get(tile_id, self.images.get(SEMANTIC_PALETTE['FLOOR']))
                screen_x = c * self.TILE_SIZE - self.view_offset_x
                screen_y = r * self.TILE_SIZE - self.view_offset_y
                map_surface.blit(img, (screen_x, screen_y))
        
        # Draw solution path (if auto-solving)
        if self.auto_mode and self.auto_path:
            for i, pos in enumerate(self.auto_path[:self.auto_step_idx + 1]):
                pr, pc = pos
                path_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                path_surf.fill((0, 255, 0, 60))
                screen_x = pc * self.TILE_SIZE - self.view_offset_x
                screen_y = pr * self.TILE_SIZE - self.view_offset_y
                map_surface.blit(path_surf, (screen_x, screen_y))
        
        # Draw Link
        pr, pc = self.env.state.position
        link_x = pc * self.TILE_SIZE - self.view_offset_x + 2
        link_y = pr * self.TILE_SIZE - self.view_offset_y + 2
        map_surface.blit(self.link_img, (link_x, link_y))
        
        # Blit map surface to screen
        self.screen.blit(map_surface, (0, 0))
        
        # Draw sidebar background
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        pygame.draw.rect(self.screen, (35, 35, 50), (sidebar_x, 0, self.SIDEBAR_WIDTH, self.screen_h))
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x, 0), (sidebar_x, self.screen_h), 2)
        
        # Sidebar content
        y_pos = 10
        
        # Title
        title = self.big_font.render("ZAVE", True, (100, 200, 255))
        self.screen.blit(title, (sidebar_x + 10, y_pos))
        y_pos += 28
        
        # Dungeon name
        if self.current_map_idx < len(self.map_names):
            name = self.map_names[self.current_map_idx]
        else:
            name = f"Map {self.current_map_idx + 1}"
        name_surf = self.font.render(name, True, (255, 220, 100))
        self.screen.blit(name_surf, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        # Map number
        map_num = f"({self.current_map_idx + 1}/{len(self.maps)})"
        num_surf = self.small_font.render(map_num, True, (150, 150, 150))
        self.screen.blit(num_surf, (sidebar_x + 10, y_pos))
        y_pos += 18
        
        size_info = f"Size: {w}×{h}"
        size_surf = self.small_font.render(size_info, True, (150, 150, 150))
        self.screen.blit(size_surf, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # Inventory section
        inv_title = self.font.render("Inventory", True, (255, 200, 100))
        self.screen.blit(inv_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        keys_text = f"Keys: {self.env.state.keys}"
        keys_surf = self.small_font.render(keys_text, True, (255, 220, 100))
        self.screen.blit(keys_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        bomb_text = f"Bomb: {'✓' if self.env.state.has_bomb else '✗'}"
        bomb_color = (100, 255, 100) if self.env.state.has_bomb else (150, 150, 150)
        bomb_surf = self.small_font.render(bomb_text, True, bomb_color)
        self.screen.blit(bomb_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        boss_key_text = f"Boss Key: {'✓' if self.env.state.has_boss_key else '✗'}"
        boss_color = (255, 150, 100) if self.env.state.has_boss_key else (150, 150, 150)
        boss_surf = self.small_font.render(boss_key_text, True, boss_color)
        self.screen.blit(boss_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # State-space solver info (if available)
        if self.solver_result:
            y_pos += 5
            pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
            y_pos += 8
            
            solver_title = self.font.render("Path Analysis", True, (100, 200, 255))
            self.screen.blit(solver_title, (sidebar_x + 10, y_pos))
            y_pos += 20
            
            # Keys info from solver
            keys_avail = self.solver_result.get('keys_available', 0)
            keys_used = self.solver_result.get('keys_used', 0)
            key_info = f"Keys: {keys_avail} found, {keys_used} used"
            key_color = (255, 220, 100) if keys_used > 0 else (150, 200, 150)
            key_surf = self.small_font.render(key_info, True, key_color)
            self.screen.blit(key_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            # Edge types breakdown
            edge_types = self.solver_result.get('edge_types', [])
            if edge_types:
                type_counts = {}
                for et in edge_types:
                    type_counts[et] = type_counts.get(et, 0) + 1
                
                # Display each edge type with color
                edge_colors = {
                    'open': (100, 255, 100),       # Green - normal door
                    'key_locked': (255, 220, 100), # Yellow - key door
                    'bombable': (255, 150, 50),    # Orange - bomb wall
                    'soft_locked': (180, 100, 255),# Purple - one-way
                    'stair': (100, 200, 255),      # Cyan - teleport
                }
                
                for etype, count in type_counts.items():
                    color = edge_colors.get(etype, (150, 150, 150))
                    type_name = etype.replace('_', ' ').title()
                    et_text = f"  {type_name}: {count}"
                    et_surf = self.small_font.render(et_text, True, color)
                    self.screen.blit(et_surf, (sidebar_x + 15, y_pos))
                    y_pos += 14
        y_pos += 7
        
        # Steps
        steps_text = f"Steps: {self.env.step_count}"
        steps_surf = self.font.render(steps_text, True, (200, 200, 200))
        self.screen.blit(steps_surf, (sidebar_x + 10, y_pos))
        y_pos += 25
        
        # Zoom info
        zoom_text = f"Zoom: {self.TILE_SIZE}px"
        zoom_surf = self.small_font.render(zoom_text, True, (150, 150, 150))
        self.screen.blit(zoom_surf, (sidebar_x + 10, y_pos))
        y_pos += 25
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # Controls section
        ctrl_title = self.font.render("Controls", True, (100, 200, 100))
        self.screen.blit(ctrl_title, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        controls = [
            "↑↓←→  Move",
            "SPACE Auto-solve",
            "R     Reset map",
            "N/P   Next/Prev",
            "+/-   Zoom",
            "0     Reset zoom",
            "C     Center view",
            "F11   Fullscreen",
            "H     Help",
            "ESC   Quit",
        ]
        for ctrl in controls:
            ctrl_surf = self.small_font.render(ctrl, True, (120, 120, 120))
            self.screen.blit(ctrl_surf, (sidebar_x + 15, y_pos))
            y_pos += 15
        
        # Draw HUD at bottom
        hud_y = self.screen_h - self.HUD_HEIGHT
        pygame.draw.rect(self.screen, (30, 30, 45), (0, hud_y, self.screen_w - self.SIDEBAR_WIDTH, self.HUD_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 80), (0, hud_y), (self.screen_w - self.SIDEBAR_WIDTH, hud_y), 2)
        
        # Status message
        msg_color = (255, 255, 100) if self.env.won else (200, 200, 200)
        msg_surf = self.font.render(self.message, True, msg_color)
        self.screen.blit(msg_surf, (10, hud_y + 10))
        
        # Position info
        pos_text = f"Position: ({pr}, {pc})"
        pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
        self.screen.blit(pos_surf, (10, hud_y + 35))
        
        # Win state
        if self.env.won:
            win_text = "★ VICTORY! ★"
            win_surf = self.big_font.render(win_text, True, (255, 215, 0))
            self.screen.blit(win_surf, (10, hud_y + 55))
        
        # Help overlay
        if self.show_help:
            self._render_help_overlay()
        
        pygame.display.flip()
    
    def _render_help_overlay(self):
        """Render help overlay."""
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        help_lines = [
            "ZAVE - Zelda AI Validation Environment",
            "",
            "Movement:",
            "  Arrow Keys - Move Link",
            "  Mouse Wheel - Zoom in/out",
            "  Middle Mouse Drag - Pan view",
            "",
            "Actions:",
            "  SPACE - Run A* auto-solver",
            "  R - Reset current map",
            "  N - Next map",
            "  P - Previous map",
            "",
            "View:",
            "  +/- - Zoom in/out",
            "  0 - Reset zoom to default",
            "  C - Center view on player",
            "  F11 - Toggle fullscreen",
            "",
            "Press H or ESC to close this help",
        ]
        
        y = 50
        for line in help_lines:
            if line.startswith("ZAVE"):
                surf = self.big_font.render(line, True, (100, 200, 255))
            elif line.endswith(":") and not line.startswith(" "):
                surf = self.font.render(line, True, (255, 200, 100))
            else:
                surf = self.small_font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (50, y))
            y += 22 if line else 10


def load_maps_from_adapter():
    """Load processed maps from data adapter using new zelda_core - ALL 18 variants."""
    try:
        from Data.zelda_core import ZeldaDungeonAdapter, DungeonSolver
        from pathlib import Path
        
        data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
        
        if not data_root.exists():
            print(f"Data folder not found: {data_root}")
            return None, None
        
        adapter = ZeldaDungeonAdapter(str(data_root))
        solver = DungeonSolver()
        
        maps = []  # Store full StitchedDungeon objects
        map_names = []  # Track dungeon names
        print("Loading all 18 dungeon variants (9 dungeons × 2 variants)...")
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                    stitched = adapter.stitch_dungeon(dungeon)
                    
                    # Store the full stitched dungeon (includes graph and room mappings)
                    maps.append(stitched)
                    
                    # Store name
                    quest_name = "Quest 1" if variant == 1 else "Quest 2"
                    map_names.append(f"Dungeon {dungeon_num} ({quest_name})")
                    
                    # Check solvability
                    result = solver.solve(stitched)
                    status = "✓" if result['solvable'] else "✗"
                    print(f"  D{dungeon_num}-{variant}: {status} - {stitched.global_grid.shape}")
                    
                except Exception as e:
                    print(f"  D{dungeon_num}-{variant}: Error - {e}")
        
        return maps if maps else None, map_names if map_names else None
        
    except Exception as e:
        print(f"Error loading maps: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main entry point."""
    print("=== ZAVE GUI Runner ===\n")
    
    if not PYGAME_AVAILABLE:
        print("Pygame is not installed. Please run: pip install pygame")
        return
    
    # Try to load processed maps
    maps, map_names = load_maps_from_adapter()
    
    if maps:
        print(f"Loaded {len(maps)} maps from data adapter")
    else:
        print("Using test map")
        maps = [create_test_map()]
        map_names = ["Test Map"]
    
    # Start GUI
    gui = ZeldaGUI(maps, map_names)
    gui.run()


if __name__ == "__main__":
    main()
