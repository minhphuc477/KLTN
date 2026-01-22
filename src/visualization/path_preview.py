"""
Path Preview Dialog - Feature 5
================================
Modal dialog showing planned path before auto-solve execution.

Research: Game Design Patterns - "Look Before You Leap" (GameDev.net, 2019)
Shows user the AI's plan before execution to build trust and reduce surprise.
"""

import pygame
import time
from typing import Optional, List, Tuple, Dict


class PathPreviewDialog:
    """
    Modal dialog for path planning preview.
    
    Displays:
    - Path length (number of steps)
    - Estimated time (based on animation speed)
    - Keys required vs available
    - Door types traversed
    - Full path overlay on map
    
    User actions:
    - Start Auto-Solve: Begin execution
    - Cancel: Dismiss preview
    """
    
    def __init__(self, path: List[Tuple[int, int]], env, solver_result: Dict = None, speed_multiplier: float = 1.0):
        """
        Initialize path preview dialog.
        
        Args:
            path: List of (row, col) positions in planned path
            env: ZeldaLogicEnv instance (for key counting)
            solver_result: Optional solver metadata (keys_used, edge_types, etc.)
            speed_multiplier: Current animation speed multiplier
        """
        self.path = path
        self.env = env
        self.solver_result = solver_result or {}
        self.speed_multiplier = speed_multiplier
        self.selected = None  # 'start' or 'cancel'
        
        # Calculate metrics
        self.path_length = len(path)
        self.keys_used = self.solver_result.get('keys_used', 0)
        self.keys_avail = self.solver_result.get('keys_available', 0)
        
        # Estimate time (0.1 seconds per step at 1x speed)
        base_time_per_step = 0.1
        self.estimated_time = (self.path_length * base_time_per_step) / speed_multiplier
        
        # Edge types breakdown
        self.edge_types = self.solver_result.get('edge_types', [])
        self.edge_counts = self._count_edge_types()
        
        # UI state
        self.hover_button = None  # 'start' or 'cancel'
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count occurrences of each edge type."""
        counts = {}
        for edge_type in self.edge_types:
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    def render(self, screen: pygame.Surface):
        """
        Render the preview dialog on screen.
        
        Args:
            screen: Pygame surface to render onto
        """
        screen_w, screen_h = screen.get_size()
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((screen_w, screen_h))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Dialog dimensions
        dialog_w, dialog_h = 450, 320
        dialog_x = (screen_w - dialog_w) // 2
        dialog_y = (screen_h - dialog_h) // 2
        
        # Draw dialog box
        pygame.draw.rect(screen, (40, 40, 60), (dialog_x, dialog_y, dialog_w, dialog_h))
        pygame.draw.rect(screen, (100, 150, 255), (dialog_x, dialog_y, dialog_w, dialog_h), 3)
        
        # Fonts
        title_font = pygame.font.SysFont('Arial', 20, bold=True)
        font = pygame.font.SysFont('Arial', 14, bold=True)
        small_font = pygame.font.SysFont('Arial', 12)
        
        # Title
        title_surf = title_font.render("Path Planning Complete!", True, (100, 200, 255))
        title_rect = title_surf.get_rect(center=(screen_w // 2, dialog_y + 25))
        screen.blit(title_surf, title_rect)
        
        # Draw divider
        pygame.draw.line(screen, (100, 150, 255), 
                        (dialog_x + 20, dialog_y + 50), 
                        (dialog_x + dialog_w - 20, dialog_y + 50), 2)
        
        # Metrics section
        y_offset = dialog_y + 70
        
        # Path length
        length_text = f"Path Length: {self.path_length} steps"
        length_surf = font.render(length_text, True, (255, 255, 255))
        screen.blit(length_surf, (dialog_x + 30, y_offset))
        y_offset += 28
        
        # Estimated time
        time_text = f"Estimated Time: {self.estimated_time:.1f} seconds"
        time_surf = font.render(time_text, True, (200, 200, 255))
        screen.blit(time_surf, (dialog_x + 30, y_offset))
        y_offset += 28
        
        # Keys
        if self.keys_avail > 0:
            keys_text = f"Keys Required: {self.keys_used} / {self.keys_avail} available"
            keys_color = (255, 220, 100) if self.keys_used <= self.keys_avail else (255, 100, 100)
        else:
            keys_text = "Keys Required: None"
            keys_color = (150, 200, 150)
        keys_surf = font.render(keys_text, True, keys_color)
        screen.blit(keys_surf, (dialog_x + 30, y_offset))
        y_offset += 28
        
        # Edge types breakdown
        if self.edge_counts:
            doors_text = "Doors: "
            doors_surf = font.render(doors_text, True, (200, 200, 200))
            screen.blit(doors_surf, (dialog_x + 30, y_offset))
            y_offset += 22
            
            edge_colors = {
                'open': (100, 255, 100),
                'key_locked': (255, 220, 100),
                'locked': (255, 220, 100),
                'bombable': (255, 150, 50),
                'soft_locked': (180, 100, 255),
                'stair': (100, 200, 255),
            }
            
            for edge_type, count in self.edge_counts.items():
                color = edge_colors.get(edge_type, (150, 150, 150))
                type_name = edge_type.replace('_', ' ').title()
                et_text = f"  • {type_name}: {count}"
                et_surf = small_font.render(et_text, True, color)
                screen.blit(et_surf, (dialog_x + 40, y_offset))
                y_offset += 18
        
        # Buttons
        button_y = dialog_y + dialog_h - 60
        button_h = 40
        
        # Start button
        start_button_rect = pygame.Rect(dialog_x + 30, button_y, 180, button_h)
        start_color = (50, 180, 50) if self.hover_button == 'start' else (40, 140, 40)
        pygame.draw.rect(screen, start_color, start_button_rect)
        pygame.draw.rect(screen, (100, 255, 100), start_button_rect, 2)
        start_text = font.render("Start Auto-Solve", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=start_button_rect.center)
        screen.blit(start_text, start_text_rect)
        
        # Cancel button
        cancel_button_rect = pygame.Rect(dialog_x + 240, button_y, 180, button_h)
        cancel_color = (140, 40, 40) if self.hover_button == 'cancel' else (100, 30, 30)
        pygame.draw.rect(screen, cancel_color, cancel_button_rect)
        pygame.draw.rect(screen, (255, 100, 100), cancel_button_rect, 2)
        cancel_text = font.render("Cancel", True, (255, 255, 255))
        cancel_text_rect = cancel_text.get_rect(center=cancel_button_rect.center)
        screen.blit(cancel_text, cancel_text_rect)
        
        # Store button rects for click detection
        self.start_button_rect = start_button_rect
        self.cancel_button_rect = cancel_button_rect
    
    def render_path_overlay(self, screen: pygame.Surface, tile_size: int, 
                           view_offset_x: int, view_offset_y: int,
                           sidebar_width: int, hud_height: int):
        """
        Render path as blue translucent overlay with step numbers.
        
        Args:
            screen: Pygame surface
            tile_size: Size of each tile in pixels
            view_offset_x: Camera X offset
            view_offset_y: Camera Y offset
            sidebar_width: Width of sidebar
            hud_height: Height of HUD
        """
        # Create clipping rect for map area (exclude sidebar/HUD)
        view_w = screen.get_width() - sidebar_width
        view_h = screen.get_height() - hud_height
        clip_rect = pygame.Rect(0, 0, view_w, view_h)
        
        for i, (r, c) in enumerate(self.path):
            # Convert to screen coordinates
            x = c * tile_size - view_offset_x
            y = r * tile_size - view_offset_y
            
            # Skip if outside visible area
            if not clip_rect.colliderect(pygame.Rect(x, y, tile_size, tile_size)):
                continue
            
            # Blue translucent square
            s = pygame.Surface((tile_size, tile_size))
            s.set_alpha(100)  # 40% transparent
            s.fill((50, 150, 255))
            screen.blit(s, (x, y))
            
            # Step number every 10 steps
            if i % 10 == 0:
                font = pygame.font.SysFont('Arial', 10, bold=True)
                text = font.render(str(i), True, (255, 255, 255))
                screen.blit(text, (x + 2, y + 2))
    
    def handle_input(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle input events.
        
        Args:
            event: Pygame event
        
        Returns:
            'start' if Start button clicked
            'cancel' if Cancel button clicked
            None otherwise
        """
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            # Check button hover
            if hasattr(self, 'start_button_rect') and self.start_button_rect.collidepoint(mouse_pos):
                self.hover_button = 'start'
            elif hasattr(self, 'cancel_button_rect') and self.cancel_button_rect.collidepoint(mouse_pos):
                self.hover_button = 'cancel'
            else:
                self.hover_button = None
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                if hasattr(self, 'start_button_rect') and self.start_button_rect.collidepoint(mouse_pos):
                    return 'start'
                elif hasattr(self, 'cancel_button_rect') and self.cancel_button_rect.collidepoint(mouse_pos):
                    return 'cancel'
        
        elif event.type == pygame.KEYDOWN:
            # ESC or SPACE to cancel
            if event.key == pygame.K_ESCAPE:
                return 'cancel'
            # RETURN to start
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return 'start'
        
        return None
