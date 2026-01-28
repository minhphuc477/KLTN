"""
KLTN GUI - Interactive Widgets
================================

GUI control widgets for the Zelda GUI including:
- CheckboxWidget: Toggle features on/off
- DropdownWidget: Select from multiple options
- ButtonWidget: Execute actions

All widgets support mouse interaction and visual feedback.
"""

from typing import Tuple, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Provide a lightweight Rect stub for environments without pygame (tests / headless CI).
class _StubRect:
    def __init__(self, x, y, w, h):
        self.x = x; self.y = y; self.width = w; self.height = h
        self.left = x; self.top = y; self.right = x + w; self.bottom = y + h
        self.centerx = x + w // 2
        self.centery = y + h // 2
    def collidepoint(self, p):
        px, py = p
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

# Alias used below so callers can use Rect(...) regardless of pygame availability
Rect = pygame.Rect if PYGAME_AVAILABLE else _StubRect


# ==========================================
# WIDGET BASE CLASS
# ==========================================

class WidgetState(Enum):
    """Visual state of widgets."""
    NORMAL = "normal"
    HOVER = "hover"
    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass
class WidgetTheme:
    """Theme colors for widgets."""
    bg_normal: Tuple[int, int, int] = (45, 45, 60)
    bg_hover: Tuple[int, int, int] = (55, 55, 75)
    bg_active: Tuple[int, int, int] = (70, 130, 180)
    bg_disabled: Tuple[int, int, int] = (30, 30, 40)
    text_normal: Tuple[int, int, int] = (220, 220, 230)
    text_disabled: Tuple[int, int, int] = (100, 100, 110)
    border: Tuple[int, int, int] = (80, 80, 100)
    accent: Tuple[int, int, int] = (100, 200, 255)


class BaseWidget:
    """Base class for all GUI widgets."""
    
    def __init__(self, rect: pygame.Rect, theme: Optional[WidgetTheme] = None):
        self.rect = rect
        self.theme = theme or WidgetTheme()
        self.state = WidgetState.NORMAL
        self.enabled = True
        self.visible = True
    
    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Update widget state based on mouse position."""
        if not self.enabled:
            self.state = WidgetState.DISABLED
            return
        
        if self.rect.collidepoint(mouse_pos):
            if self.state != WidgetState.ACTIVE:
                self.state = WidgetState.HOVER
        else:
            if self.state != WidgetState.ACTIVE:
                self.state = WidgetState.NORMAL
    
    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle mouse down event. Returns True if handled."""
        if not self.enabled or not self.visible:
            return False
        return self.rect.collidepoint(pos)
    
    def handle_mouse_up(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle mouse up event. Returns True if handled."""
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the widget. Override in subclasses."""
        pass


# ==========================================
# CHECKBOX WIDGET
# ==========================================

class CheckboxWidget(BaseWidget):
    """
    Checkbox widget for toggling boolean options.
    
    Features:
    - Visual check mark when enabled
    - Hover feedback
    - Click to toggle
    """
    
    def __init__(self, pos: Tuple[int, int], label: str, 
                 checked: bool = False, theme: Optional[WidgetTheme] = None):
        box_size = 20
        super().__init__(Rect(pos[0], pos[1], box_size, box_size), theme)
        self.label = label
        self.checked = checked
        self._pos = pos
        self.label_rect = Rect(pos[0] + box_size + 8, pos[1] - 2, 200, box_size + 4)
        self.full_rect = Rect(pos[0], pos[1], 250, box_size)
        
        # Font for label
        if PYGAME_AVAILABLE:
            # Ensure the pygame font system is initialized when running in headless test environments
            if not pygame.font.get_init():
                try:
                    pygame.font.init()
                except Exception:
                    # If font init fails, fall back to a simple stub to avoid crashing tests
                    class _StubFont:
                        def render(self, *_args, **_kwargs):
                            # Return a blank surface-like object with blit-safe interface
                            return pygame.Surface((1, 1))
                    self.font = _StubFont()
                else:
                    self.font = pygame.font.SysFont('Arial', 12)
            else:
                self.font = pygame.font.SysFont('Arial', 12)
    
    @property
    def pos(self):
        """Get current position."""
        return self._pos
    
    @pos.setter
    def pos(self, value: Tuple[int, int]):
        """Update position and all related rectangles."""
        self._pos = value
        box_size = 20
        self.rect = Rect(value[0], value[1], box_size, box_size)
        self.label_rect = Rect(value[0] + box_size + 8, value[1] - 2, 200, box_size + 4)
        self.full_rect = Rect(value[0], value[1], 250, box_size)
    
    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Update hover state based on full interactive area."""
        if not self.enabled:
            self.state = WidgetState.DISABLED
            return
        
        if self.full_rect.collidepoint(mouse_pos):
            self.state = WidgetState.HOVER
        else:
            self.state = WidgetState.NORMAL
    
    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle click to toggle checkbox."""
        if not self.enabled or not self.visible:
            return False
        
        if button == 1 and self.full_rect.collidepoint(pos):
            self.checked = not self.checked
            return True
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render checkbox and label."""
        if not self.visible:
            return
        
        # Get colors based on state
        if self.state == WidgetState.DISABLED:
            bg_color = self.theme.bg_disabled
            text_color = self.theme.text_disabled
        elif self.state == WidgetState.HOVER:
            bg_color = self.theme.bg_hover
            text_color = self.theme.text_normal
        else:
            bg_color = self.theme.bg_normal
            text_color = self.theme.text_normal
        
        # Draw checkbox box
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, self.theme.border, self.rect, 2)
        
        # Draw check mark if checked
        if self.checked:
            check_color = self.theme.accent if self.enabled else self.theme.text_disabled
            # Draw checkmark lines
            cx, cy = self.rect.centerx, self.rect.centery
            pygame.draw.line(surface, check_color, 
                           (cx - 6, cy), (cx - 2, cy + 4), 3)
            pygame.draw.line(surface, check_color,
                           (cx - 2, cy + 4), (cx + 6, cy - 4), 3)
        
        # Draw label
        label_surf = self.font.render(self.label, True, text_color)
        surface.blit(label_surf, (self.label_rect.x, self.label_rect.y + 2))


# ==========================================
# DROPDOWN WIDGET
# ==========================================

class DropdownWidget(BaseWidget):
    """
    Dropdown menu widget for selecting from multiple options.
    
    Features:
    - Click to expand/collapse
    - Hover highlighting
    - Scrollable if many options
    """
    
    def __init__(self, pos: Tuple[int, int], label: str, 
                 options: List[str], selected: int = 0,
                 theme: Optional[WidgetTheme] = None,
                 keep_open_on_select: bool = False):
        super().__init__(Rect(pos[0], pos[1], 180, 28), theme)
        self.label = label
        self.options = options
        self.selected = selected
        self.is_open = False
        self._pos = pos
        # If True, keep the dropdown open after selecting an option
        self.keep_open_on_select = bool(keep_open_on_select)
        
        # Calculate dropdown menu rect
        option_height = 24
        self.dropdown_rect = Rect(
            pos[0], pos[1] + 30,
            180, min(len(options) * option_height, 200)
        )
        
        self.hover_option = -1
        
        # Fonts
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont('Arial', 12)
            self.label_font = pygame.font.SysFont('Arial', 11, bold=True)
        # Initialize rects via pos setter to ensure full_rect is created
        try:
            self.pos = self._pos
        except Exception:
            pass
    
    @property
    def pos(self):
        """Get current position."""
        return self._pos
    
    @pos.setter
    def pos(self, value: Tuple[int, int]):
        """Update position and dropdown rect."""
        self._pos = value
        self.rect = Rect(value[0], value[1], 180, 28)
        option_height = 24
        self.dropdown_rect = Rect(
            value[0], value[1] + 30,
            180, min(len(self.options) * option_height, 200)
        )
        # Compute full_rect that includes label area above the control so layout
        # routines can account for the extra height when computing content size.
        # Be robust: always have a non-zero label height using either label_font or font
        try:
            lbl_font = getattr(self, 'label_font', None) or getattr(self, 'font', None)
            if lbl_font is None:
                # Last resort: create a small default font (shouldn't normally be necessary)
                lbl_font = pygame.font.SysFont('Arial', 11, bold=True)
            # Add a bit more padding above the control so the label sits further away
            label_h = max(12, lbl_font.get_height() + 8)
        except Exception:
            label_h = 14
        # Store full_rect including label area
        self.full_rect = Rect(self.rect.x, self.rect.y - label_h, self.rect.width, self.rect.height + label_h)

    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Update hover state."""
        if not self.enabled:
            self.state = WidgetState.DISABLED
            self.is_open = False
            return

        if self.is_open:
            # Check hover over options
            self.hover_option = -1
            if self.dropdown_rect.collidepoint(mouse_pos):
                rel_y = mouse_pos[1] - self.dropdown_rect.y
                # Ensure integer index (mouse positions may be floats)
                option_idx = int(rel_y // 24)
                if 0 <= option_idx < len(self.options):
                    self.hover_option = option_idx
        else:
            # Check hover over main button
            if self.rect.collidepoint(mouse_pos):
                if self.state != WidgetState.ACTIVE:
                    self.state = WidgetState.HOVER
            else:
                if self.state != WidgetState.ACTIVE:
                    self.state = WidgetState.NORMAL
    
    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle click to toggle or select option."""
        if not self.enabled or not self.visible:
            return False
        
        if button != 1:
            return False
        
        if self.is_open:
            # Check if clicking on an option
            if self.dropdown_rect.collidepoint(pos):
                rel_y = pos[1] - self.dropdown_rect.y
                option_idx = int(rel_y // 24)
                if 0 <= option_idx < len(self.options):
                    self.selected = int(option_idx)
                    # Respect keep_open_on_select flag
                    self.is_open = bool(self.keep_open_on_select)
                    return True
            else:
                # Clicked outside, close dropdown
                self.is_open = False
                return True
        else:
            # Check if clicking main button
            if self.rect.collidepoint(pos):
                self.is_open = True
                return True
        
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render dropdown menu."""
        if not self.visible:
            return
        
        # Get colors
        if self.state == WidgetState.DISABLED:
            bg_color = self.theme.bg_disabled
            text_color = self.theme.text_disabled
        elif self.state == WidgetState.HOVER:
            bg_color = self.theme.bg_hover
            text_color = self.theme.text_normal
        else:
            bg_color = self.theme.bg_normal
            text_color = self.theme.text_normal
        
        # Draw label/title above the dropdown if provided
        if getattr(self, 'label', None):
            try:
                lbl_font = getattr(self, 'label_font', None) or getattr(self, 'font', None)
                if lbl_font is None:
                    lbl_font = pygame.font.SysFont('Arial', 11, bold=True)
                label_surf = lbl_font.render(str(self.label), True, self.theme.text_normal)
                # Render label at a slightly larger local offset inside full_rect so it feels spaced out
                # (full_rect is created to include this area)
                label_y_local = 8
                # If widget.render was called with the widget's pos set to (0,0) (as when
                # rendering into a temp surface), this will place the label correctly.
                surface.blit(label_surf, (6, label_y_local))
            except Exception:
                # On error, fallback to render with main font at a safe offset
                try:
                    fallback = getattr(self, 'font', None) or pygame.font.SysFont('Arial', 12)
                    label_surf = fallback.render(str(self.label), True, self.theme.text_normal)
                    surface.blit(label_surf, (4, 4))
                except Exception:
                    # Non-fatal: continue rendering the dropdown even if label paint fails
                    pass

        # Draw main button
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, self.theme.border, self.rect, 2)
        
        # Draw selected option text
        if 0 <= self.selected < len(self.options):
            text = self.options[self.selected]
            text_surf = self.font.render(text, True, text_color)
            surface.blit(text_surf, (self.rect.x + 8, self.rect.y + 7))
        
        # Draw dropdown arrow
        arrow_x = self.rect.right - 20
        arrow_y = self.rect.centery
        if self.is_open:
            # Up arrow
            pygame.draw.polygon(surface, text_color, [
                (arrow_x, arrow_y + 3),
                (arrow_x + 8, arrow_y + 3),
                (arrow_x + 4, arrow_y - 3)
            ])
        else:
            # Down arrow
            pygame.draw.polygon(surface, text_color, [
                (arrow_x, arrow_y - 3),
                (arrow_x + 8, arrow_y - 3),
                (arrow_x + 4, arrow_y + 3)
            ])
        
        # Draw dropdown menu if open
        if self.is_open:
            # Draw dropdown background
            pygame.draw.rect(surface, self.theme.bg_normal, self.dropdown_rect)
            pygame.draw.rect(surface, self.theme.border, self.dropdown_rect, 2)
            
            # Draw options
            for i, option in enumerate(self.options):
                option_y = self.dropdown_rect.y + i * 24
                option_rect = pygame.Rect(self.dropdown_rect.x, option_y, 
                                         self.dropdown_rect.width, 24)
                
                # Highlight hover or selected
                if i == self.hover_option:
                    pygame.draw.rect(surface, self.theme.bg_hover, option_rect)
                elif i == self.selected:
                    pygame.draw.rect(surface, self.theme.bg_active, option_rect)
                
                # Draw option text
                option_surf = self.font.render(option, True, text_color)
                surface.blit(option_surf, (option_rect.x + 8, option_rect.y + 5))

    def render_menu(self, surface: pygame.Surface, alpha: int = 255, scroll_offset: int = 0, panel_rect: pygame.Rect | None = None) -> None:
        """Render only the dropdown menu portion onto the given surface (global coords).
        This allows the GUI to draw the menu on top of other UI elements so it won't be clipped
        by the control panel surface.

        Args:
            surface: global surface to draw onto
            alpha: optional per-menu alpha
            scroll_offset: vertical offset to subtract (used when the menu is in a scrolled panel)
            panel_rect: optional panel rect to constrain and clamp menu position/width
        """
        if not self.visible or not self.is_open:
            return
        # Use same text color rules
        if self.state == WidgetState.DISABLED:
            text_color = self.theme.text_disabled
        else:
            text_color = self.theme.text_normal

        # Base width/height
        w = max(1, self.dropdown_rect.width)
        h = max(1, self.dropdown_rect.height)

        # If panel_rect provided, ensure menu fits inside it horizontally
        if panel_rect is not None:
            max_w = max(40, panel_rect.width - 24)
            if w > max_w:
                w = max_w
            # Compute visible height excluding header
            header_h = getattr(self, 'header_height', None)
        temp = pygame.Surface((w, h), pygame.SRCALPHA)

        # Background + border
        pygame.draw.rect(temp, self.theme.bg_normal, Rect(0, 0, w, h))
        pygame.draw.rect(temp, self.theme.border, Rect(0, 0, w, h), 2)

        # Draw options into temp surface (local coords)
        for i, option in enumerate(self.options):
            option_rect = Rect(0, i * 24, w, 24)
            if i == self.hover_option:
                pygame.draw.rect(temp, self.theme.bg_hover, option_rect)
            elif i == self.selected:
                pygame.draw.rect(temp, self.theme.bg_active, option_rect)
            option_surf = self.font.render(option, True, text_color)
            temp.blit(option_surf, (8, i * 24 + 5))

        if alpha < 255:
            temp.set_alpha(alpha)

        # Compute target position and clamp to panel if provided
        target_x = self.dropdown_rect.x
        target_y = self.dropdown_rect.y - scroll_offset if scroll_offset else self.dropdown_rect.y

        if panel_rect is not None:
            # Clamp X to panel bounds with small padding
            min_x = panel_rect.x + 8
            max_x = panel_rect.x + panel_rect.width - w - 8
            target_x = max(min_x, min(target_x, max_x))
            # If menu would overlap header, nudge it down below header
            header_h = self.font.get_height() + 12
            min_y = panel_rect.y + header_h + 4
            if target_y < min_y:
                target_y = min_y

        surface.blit(temp, (target_x, target_y))


# ==========================================
# BUTTON WIDGET
# ==========================================

class ButtonWidget(BaseWidget):
    """
    Button widget for executing actions.
    
    Features:
    - Click callback
    - Hover and active states
    - Optional icon support
    """
    
    def __init__(self, pos: Tuple[int, int], label: str,
                 callback: Callable[[], None],
                 width: int = 150, height: int = 35,
                 theme: Optional[WidgetTheme] = None):
        super().__init__(Rect(pos[0], pos[1], width, height), theme)
        self.label = label
        self.callback = callback
        self.pressed = False
        self._pos = pos
        self._width = width
        self._height = height
        
        # Font
        if PYGAME_AVAILABLE:
            self.font = pygame.font.SysFont('Arial', 12, bold=True)
    
    @property
    def pos(self):
        """Get current position."""
        return self._pos
    
    @pos.setter
    def pos(self, value: Tuple[int, int]):
        """Update position."""
        self._pos = value
        self.rect = Rect(value[0], value[1], self._width, self._height)
    
    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle mouse down - mark as pressed."""
        if not self.enabled or not self.visible:
            return False
        
        if button == 1 and self.rect.collidepoint(pos):
            self.pressed = True
            self.state = WidgetState.ACTIVE
            return True
        return False
    
    def handle_mouse_up(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle mouse up - execute callback if still over button."""
        if not self.enabled or not self.visible:
            return False
        
        if button == 1 and self.pressed:
            self.pressed = False
            if self.rect.collidepoint(pos):
                # Execute callback
                if self.callback:
                    self.callback()
                return True
            else:
                self.state = WidgetState.NORMAL
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render button."""
        if not self.visible:
            return
        
        # Get colors
        if self.state == WidgetState.DISABLED:
            bg_color = self.theme.bg_disabled
            text_color = self.theme.text_disabled
            border_color = self.theme.border
        elif self.state == WidgetState.ACTIVE or self.pressed:
            bg_color = self.theme.bg_active
            text_color = (255, 255, 255)
            border_color = self.theme.accent
        elif self.state == WidgetState.HOVER:
            bg_color = self.theme.bg_hover
            text_color = self.theme.text_normal
            border_color = self.theme.accent
        else:
            bg_color = self.theme.bg_normal
            text_color = self.theme.text_normal
            border_color = self.theme.border
        
        # Draw button background
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=4)
        
        # Draw label centered
        text_surf = self.font.render(self.label, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


# ==========================================
# WIDGET MANAGER
# ==========================================

class WidgetManager:
    """
    Manages collection of widgets and handles events.
    
    Simplifies widget lifecycle management.
    """
    
    def __init__(self):
        self.widgets: List[BaseWidget] = []
        self.theme = WidgetTheme()
    
    def add_widget(self, widget: BaseWidget) -> None:
        """Add a widget to the manager."""
        self.widgets.append(widget)
    
    def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
        """Update all widgets."""
        for widget in self.widgets:
            widget.update(mouse_pos, dt)
    
    def _close_other_dropdowns(self, keep_widget: Optional[DropdownWidget]) -> None:
        """Close all open dropdowns except for keep_widget (can be None to close all)."""
        import logging
        log = logging.getLogger(__name__)
        for w in self.widgets:
            if isinstance(w, DropdownWidget) and w is not keep_widget:
                if w.is_open:
                    log.debug(f"Closing dropdown {getattr(w,'control_name',None)} (keep {getattr(keep_widget,'control_name',None)})")
                w.is_open = False

    def snapshot_dropdown_state(self) -> dict:
        """Return a snapshot of dropdown states keyed by control_name."""
        state = {}
        for w in self.widgets:
            if isinstance(w, DropdownWidget) and getattr(w, 'control_name', None):
                state[w.control_name] = {
                    'selected': int(getattr(w, 'selected', 0)),
                    'is_open': bool(getattr(w, 'is_open', False)),
                    'keep_open_on_select': bool(getattr(w, 'keep_open_on_select', False))
                }
        return state

    def apply_dropdown_state(self, state: dict) -> None:
        """Apply a previously captured snapshot to current dropdown widgets."""
        for w in self.widgets:
            if isinstance(w, DropdownWidget) and getattr(w, 'control_name', None) in state:
                s = state[w.control_name]
                try:
                    w.selected = int(min(max(0, int(s.get('selected', 0))), len(w.options) - 1))
                except Exception:
                    w.selected = 0
                # Restore is_open exactly as snapshot (user intent persists across rebuild)
                w.is_open = bool(s.get('is_open', False))
                # Restore keep_open flag if present
                w.keep_open_on_select = bool(s.get('keep_open_on_select', getattr(w, 'keep_open_on_select', False)))

    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle mouse down for all widgets. Returns True if any handled."""
        # If any dropdowns are open, handle them first so clicks can interact with
        # the menu or close open menus when clicking outside.
        open_dropdowns = [w for w in self.widgets if isinstance(w, DropdownWidget) and w.is_open]
        if open_dropdowns:
            # If clicking inside any open dropdown (main button OR menu), dispatch to it
            for widget in reversed(self.widgets):
                if isinstance(widget, DropdownWidget) and widget.is_open:
                    if widget.rect.collidepoint(pos) or widget.dropdown_rect.collidepoint(pos):
                        handled = widget.handle_mouse_down(pos, button)
                        # Close other dropdowns regardless of whether this one remained open
                        self._close_other_dropdowns(widget)
                        return bool(handled)
            # Click was outside all open dropdowns: close them and continue so a
            # subsequent widget (e.g., another dropdown button) can handle the same click.
            self._close_other_dropdowns(None)
            # To make handling deterministic, dispatch the same click immediately to
            # the topmost widget under the pointer instead of continuing the full loop.
            import logging
            log = logging.getLogger(__name__)
            log.debug("Dispatching click at %s", pos)
            for target in reversed(self.widgets):
                if target.rect.collidepoint(pos):
                    log.debug("Candidate: %s rect=%s", getattr(target,'control_name',None) or target.__class__.__name__, target.rect)
                    handled = target.handle_mouse_down(pos, button)
                    log.debug("handled=%s, target.is_open=%s", handled, getattr(target,'is_open',None))
                    if isinstance(target, DropdownWidget):
                        self._close_other_dropdowns(target)
                    return bool(handled)
            # Nothing under the pointer to handle the click
            return False

        # Then handle other widgets normally
        for widget in reversed(self.widgets):
            if isinstance(widget, DropdownWidget) and widget.is_open:
                continue  # Already handled above (shouldn't happen)
            if widget.handle_mouse_down(pos, button):
                # If the widget is a dropdown and has just opened, close others
                if isinstance(widget, DropdownWidget):
                    self._close_other_dropdowns(widget)
                return True
        return False
    
    def handle_mouse_up(self, pos: Tuple[int, int], button: int) -> bool:
        """Handle mouse up for all widgets. Returns True if any handled."""
        for widget in reversed(self.widgets):
            if widget.handle_mouse_up(pos, button):
                return True
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render all widgets with proper Z-ordering for dropdowns."""
        # Separate normal widgets from expanded dropdowns
        normal_widgets = []
        expanded_dropdowns = []
        
        for widget in self.widgets:
            if isinstance(widget, DropdownWidget) and widget.is_open:
                expanded_dropdowns.append(widget)
            else:
                normal_widgets.append(widget)
        
        # Render normal widgets first
        for widget in normal_widgets:
            widget.render(surface)
        
        # Render expanded dropdowns last (on top of everything)
        for widget in expanded_dropdowns:
            widget.render(surface)
    
    def clear(self) -> None:
        """Remove all widgets."""
        self.widgets.clear()
