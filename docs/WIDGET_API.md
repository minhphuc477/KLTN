# Widget System API Documentation

## Overview

The widget system provides reusable GUI components for the Zelda GUI. All widgets inherit from `BaseWidget` and support mouse interaction, hover states, and theming.

---

## Core Classes

### BaseWidget

Base class for all widgets.

**Properties:**
- `rect: pygame.Rect` - Widget position and size
- `theme: WidgetTheme` - Color theme
- `state: WidgetState` - Current visual state (NORMAL, HOVER, ACTIVE, DISABLED)
- `enabled: bool` - Whether widget responds to input
- `visible: bool` - Whether widget is rendered

**Methods:**
```python
def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
    """Update widget state based on mouse position."""
    
def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
    """Handle mouse down event. Returns True if handled."""
    
def handle_mouse_up(self, pos: Tuple[int, int], button: int) -> bool:
    """Handle mouse up event. Returns True if handled."""
    
def render(self, surface: pygame.Surface) -> None:
    """Render the widget. Override in subclasses."""
```

---

## Widget Types

### CheckboxWidget

Interactive checkbox with label.

**Constructor:**
```python
CheckboxWidget(pos: Tuple[int, int], label: str, checked: bool = False, theme: Optional[WidgetTheme] = None)
```

**Properties:**
- `checked: bool` - Whether checkbox is checked
- `label: str` - Text label displayed next to checkbox

**Usage Example:**
```python
# Create checkbox
enable_heatmap = CheckboxWidget(
    pos=(20, 50),
    label="Show Heatmap",
    checked=True
)

# Add to widget manager
widget_manager.add_widget(enable_heatmap)

# Later, check state
if enable_heatmap.checked:
    show_heatmap()
```

**Visual States:**
- Normal: Gray background, white text
- Hover: Lighter gray background
- Checked: Blue checkmark displayed
- Disabled: Dark gray, muted text

---

### DropdownWidget

Drop-down menu for selecting from multiple options.

**Constructor:**
```python
DropdownWidget(pos: Tuple[int, int], label: str, options: List[str], selected: int = 0, theme: Optional[WidgetTheme] = None)
```

**Properties:**
- `options: List[str]` - List of option strings
- `selected: int` - Index of currently selected option
- `is_open: bool` - Whether dropdown menu is expanded

**Usage Example:**
```python
# Create dropdown
zoom_selector = DropdownWidget(
    pos=(20, 100),
    label="Zoom",
    options=["25%", "50%", "100%", "150%", "200%"],
    selected=2  # 100%
)

# Add to manager
widget_manager.add_widget(zoom_selector)

# Get selected value
zoom_level = zoom_selector.options[zoom_selector.selected]
```

**Visual States:**
- Closed: Shows selected option with down arrow
- Open: Shows all options in dropdown menu
- Hover: Highlights hovered option
- Selected: Highlights current selection

**Behavior:**
- Click to toggle open/close
- Click option to select
- Click outside to close

---

### ButtonWidget

Clickable button that executes a callback.

**Constructor:**
```python
ButtonWidget(pos: Tuple[int, int], label: str, callback: Callable[[], None], width: int = 150, height: int = 35, theme: Optional[WidgetTheme] = None)
```

**Properties:**
- `label: str` - Button text
- `callback: Callable` - Function called on click
- `pressed: bool` - Whether button is currently pressed

**Usage Example:**
```python
def on_start_clicked():
    print("Starting auto-solve...")
    start_auto_solve()

# Create button
start_button = ButtonWidget(
    pos=(20, 200),
    label="Start Auto-Solve",
    callback=on_start_clicked,
    width=180,
    height=40
)

# Add to manager
widget_manager.add_widget(start_button)
```

**Visual States:**
- Normal: Gray background
- Hover: Lighter gray, blue border
- Active/Pressed: Blue background, white text
- Disabled: Dark gray, no interaction

**Behavior:**
- Click and hold to press
- Release over button to trigger callback
- Release outside button cancels

---

## Widget Manager

Manages collection of widgets and routes events.

**Constructor:**
```python
WidgetManager()
```

**Methods:**
```python
def add_widget(self, widget: BaseWidget) -> None:
    """Add a widget to the manager."""
    
def update(self, mouse_pos: Tuple[int, int], dt: float) -> None:
    """Update all widgets."""
    
def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
    """Handle mouse down for all widgets. Returns True if any handled."""
    
def handle_mouse_up(self, pos: Tuple[int, int], button: int) -> bool:
    """Handle mouse up for all widgets. Returns True if any handled."""
    
def render(self, surface: pygame.Surface) -> None:
    """Render all widgets."""
    
def clear(self) -> None:
    """Remove all widgets."""
```

**Usage Example:**
```python
# Create manager
manager = WidgetManager()

# Add multiple widgets
manager.add_widget(checkbox1)
manager.add_widget(checkbox2)
manager.add_widget(dropdown1)
manager.add_widget(button1)

# In game loop:
mouse_pos = pygame.mouse.get_pos()
manager.update(mouse_pos, delta_time)

# In event handler:
if event.type == pygame.MOUSEBUTTONDOWN:
    if manager.handle_mouse_down(event.pos, event.button):
        # Widget handled the event
        pass

# In render:
manager.render(screen)
```

---

## Theming

### WidgetTheme

Color theme for widgets.

**Properties:**
```python
@dataclass
class WidgetTheme:
    bg_normal: Tuple[int, int, int] = (45, 45, 60)
    bg_hover: Tuple[int, int, int] = (55, 55, 75)
    bg_active: Tuple[int, int, int] = (70, 130, 180)
    bg_disabled: Tuple[int, int, int] = (30, 30, 40)
    text_normal: Tuple[int, int, int] = (220, 220, 230)
    text_disabled: Tuple[int, int, int] = (100, 100, 110)
    border: Tuple[int, int, int] = (80, 80, 100)
    accent: Tuple[int, int, int] = (100, 200, 255)
```

**Usage Example:**
```python
# Create custom theme
dark_theme = WidgetTheme(
    bg_normal=(20, 20, 30),
    bg_hover=(30, 30, 45),
    bg_active=(50, 100, 150),
    accent=(255, 200, 100)
)

# Apply to widget
button = ButtonWidget(
    pos=(10, 10),
    label="Click Me",
    callback=my_function,
    theme=dark_theme
)
```

---

## Creating Custom Widgets

### Step 1: Inherit from BaseWidget

```python
class MyCustomWidget(BaseWidget):
    def __init__(self, pos: Tuple[int, int], custom_param: str, theme: Optional[WidgetTheme] = None):
        super().__init__(pygame.Rect(pos[0], pos[1], 200, 40), theme)
        self.custom_param = custom_param
        self.font = pygame.font.SysFont('Arial', 12)
```

### Step 2: Implement handle_mouse_down

```python
    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        if not self.enabled or not self.visible:
            return False
        
        if button == 1 and self.rect.collidepoint(pos):
            # Handle click
            self.do_something()
            return True
        return False
```

### Step 3: Implement render

```python
    def render(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        
        # Get colors based on state
        if self.state == WidgetState.HOVER:
            bg_color = self.theme.bg_hover
        else:
            bg_color = self.theme.bg_normal
        
        # Draw background
        pygame.draw.rect(surface, bg_color, self.rect)
        
        # Draw content
        text_surf = self.font.render(self.custom_param, True, self.theme.text_normal)
        surface.blit(text_surf, (self.rect.x + 10, self.rect.y + 10))
```

### Step 4: Use your widget

```python
# Create custom widget
my_widget = MyCustomWidget((100, 100), "Hello!")

# Add to manager
widget_manager.add_widget(my_widget)
```

---

## Event Flow

### Mouse Event Handling

1. **Mouse moves** → `WidgetManager.update()` → Each widget's `update()`
   - Updates `state` to HOVER if mouse over widget

2. **Mouse down** → `WidgetManager.handle_mouse_down()` → Widget's `handle_mouse_down()`
   - Returns `True` if widget handled event
   - Stops propagation to other widgets

3. **Mouse up** → `WidgetManager.handle_mouse_up()` → Widget's `handle_mouse_up()`
   - Executes callback if button was pressed
   - Returns `True` if handled

### State Transitions

```
NORMAL ──(mouse enter)──> HOVER
HOVER  ──(mouse exit)───> NORMAL
HOVER  ──(mouse down)───> ACTIVE
ACTIVE ──(mouse up)─────> HOVER (if over widget) or NORMAL
ANY    ──(disabled)─────> DISABLED
```

---

## Best Practices

### 1. Widget Positioning

Use absolute positions for now:
```python
# Position relative to control panel
panel_x = screen_width - 300
button_y = panel_y + 50

button = ButtonWidget((panel_x + 10, button_y), "Click", callback)
```

### 2. Callback Functions

Keep callbacks simple and fast:
```python
# Good
def on_click():
    self.flag = not self.flag

# Avoid
def on_click():
    # Don't do expensive operations in callbacks
    train_neural_network()  # BAD!
```

### 3. Widget Updates

Call `widget_manager.update()` every frame:
```python
# In main loop
while running:
    dt = clock.tick(60) / 1000.0
    mouse_pos = pygame.mouse.get_pos()
    widget_manager.update(mouse_pos, dt)
```

### 4. Event Priority

Handle widget events before other events:
```python
if event.type == pygame.MOUSEBUTTONDOWN:
    if widget_manager.handle_mouse_down(event.pos, event.button):
        continue  # Widget handled it, skip other handlers
    
    # Other event handling...
```

---

## Complete Example

```python
import pygame
from src.gui.widgets import CheckboxWidget, DropdownWidget, ButtonWidget, WidgetManager

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Create widget manager
widgets = WidgetManager()

# Create widgets
enable_feature = CheckboxWidget((20, 20), "Enable Feature", checked=False)
quality_selector = DropdownWidget((20, 60), "Quality", ["Low", "Medium", "High"], selected=1)

def on_apply():
    print(f"Feature enabled: {enable_feature.checked}")
    print(f"Quality: {quality_selector.options[quality_selector.selected]}")

apply_button = ButtonWidget((20, 120), "Apply Settings", on_apply, width=180)

# Add to manager
widgets.add_widget(enable_feature)
widgets.add_widget(quality_selector)
widgets.add_widget(apply_button)

# Main loop
running = True
while running:
    dt = clock.tick(60) / 1000.0
    
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if widgets.handle_mouse_down(event.pos, event.button):
                continue
        elif event.type == pygame.MOUSEBUTTONUP:
            widgets.handle_mouse_up(event.pos, event.button)
    
    # Update
    mouse_pos = pygame.mouse.get_pos()
    widgets.update(mouse_pos, dt)
    
    # Render
    screen.fill((30, 30, 40))
    widgets.render(screen)
    pygame.display.flip()

pygame.quit()
```

---

## Troubleshooting

**Widgets not responding to clicks:**
- Check `enabled` is `True`
- Check `visible` is `True`
- Ensure `handle_mouse_down/up` called in event loop
- Verify widget rect contains click position

**Widgets not rendering:**
- Check `visible` is `True`
- Ensure `render()` called each frame
- Verify pygame surface is correct

**Callbacks not executing:**
- Check callback function exists
- Verify ButtonWidget `handle_mouse_up` called
- Ensure mouse released over button

**Hover state not working:**
- Call `widget_manager.update()` every frame with mouse position
- Check widget `update()` method implemented correctly

---

## Future Extensions

Potential widget types to add:
- **SliderWidget** - Numeric value selection
- **TextInputWidget** - Text entry field
- **RadioButtonGroup** - Mutually exclusive options
- **TabWidget** - Tabbed interface
- **ScrollBarWidget** - Scrollable content
- **TooltipWidget** - Hover tooltips
- **ProgressBarWidget** - Progress indicator
- **ColorPickerWidget** - Color selection

---

End of Widget API Documentation
