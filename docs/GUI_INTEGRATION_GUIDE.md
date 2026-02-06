# GUI INTEGRATION GUIDE
**How to Integrate TIER 2 & 3 Features into gui_runner.py**

This guide provides step-by-step instructions for integrating all new features into the existing GUI.

---

## INTEGRATION CHECKLIST

### ✅ Step 1: Import New Modules

Add these imports at the top of `gui_runner.py`:

```python
# TIER 2 & 3 imports
from simulation.dstar_lite import DStarLiteSolver
from simulation.parallel_astar import ParallelAStarSolver
from simulation.multi_goal import MultiGoalPathfinder
from simulation.solver_comparison import SolverComparison
from src.gui.tier2_components import (
    FloorSelector, MinimapZoom, ItemTooltip, ReplanningIndicator
)
```

---

### ✅ Step 2: Add New Keyboard Controls

Add these to the control mapping in `ZeldaGUI.__init__()`:

```python
# Existing controls
self.controls = {
    'SPACE': 'Auto-solve (A*)',
    'R': 'Reset map',
    # ... existing controls ...
    
    # NEW: TIER 2 controls
    'D': 'Toggle D* Lite replanning',
    'P': 'Toggle Parallel search',
    'M': 'Multi-goal routing',
    'C': 'Solver comparison mode',
    'Z': 'Zoom minimap',
    'G': 'Generate random dungeon',
    'F': 'Cycle floor (multi-floor)',
}
```

---

### ✅ Step 3: Initialize New Components

Add to `ZeldaGUI.__init__()`:

```python
def __init__(self, maps: list = None, map_names: list = None):
    # ... existing initialization ...
    
    # TIER 2 components
    self.use_dstar = False
    self.use_parallel = False
    self.dstar_solver = None
    self.parallel_solver = None
    self.multigoal_finder = None
    
    # GUI components
    self.floor_selector = FloorSelector(self.screen_width, num_floors=1)
    self.minimap_zoom = MinimapZoom(self.minimap_rect)
    self.item_tooltip = ItemTooltip()
    self.replan_indicator = ReplanningIndicator(self.screen_width, self.screen_height)
    
    # Solver comparison
    self.comparison_mode = False
    self.comparison_results = None
```

---

### ✅ Step 4: Handle New Keyboard Events

Add to `handle_events()` method:

```python
def handle_events(self):
    for event in pygame.event.get():
        # ... existing event handling ...
        
        elif event.type == pygame.KEYDOWN:
            # ... existing key handling ...
            
            # TIER 2 controls
            elif event.key == pygame.K_d:
                self.toggle_dstar_mode()
            
            elif event.key == pygame.K_p:
                self.toggle_parallel_mode()
            
            elif event.key == pygame.K_m:
                self.run_multigoal_routing()
            
            elif event.key == pygame.K_c:
                self.toggle_comparison_mode()
            
            elif event.key == pygame.K_g:
                self.generate_random_dungeon()
            
            elif event.key == pygame.K_f:
                self.cycle_floor()
```

---

### ✅ Step 5: Implement Helper Methods

Add these methods to `ZeldaGUI` class:

```python
def toggle_dstar_mode(self):
    """Toggle D* Lite replanning mode."""
    self.use_dstar = not self.use_dstar
    
    if self.use_dstar:
        self.dstar_solver = DStarLiteSolver(self.env)
        print("D* Lite mode: ENABLED")
    else:
        self.dstar_solver = None
        print("D* Lite mode: DISABLED")


def toggle_parallel_mode(self):
    """Toggle parallel A* search."""
    self.use_parallel = not self.use_parallel
    
    if self.use_parallel:
        import multiprocessing as mp
        n_workers = mp.cpu_count()
        self.parallel_solver = ParallelAStarSolver(self.env, n_workers=n_workers)
        print(f"Parallel A* mode: ENABLED ({n_workers} workers)")
    else:
        self.parallel_solver = None
        print("Parallel A* mode: DISABLED")


def run_multigoal_routing(self):
    """Find optimal route to collect all items."""
    if not self.multigoal_finder:
        self.multigoal_finder = MultiGoalPathfinder(self.env)
    
    print("Computing optimal item collection route...")
    start_state = self.env.state.copy()
    result = self.multigoal_finder.find_optimal_collection_order(start_state)
    
    if result.success:
        print(f"✓ Optimal route found!")
        print(f"  Waypoints: {len(result.waypoints)}")
        print(f"  Total cost: {result.total_cost}")
        print(f"  Path length: {len(result.full_path)}")
        
        # Visualize waypoints
        self.visualize_waypoints(result.waypoints)
    else:
        print("✗ No route found")


def toggle_comparison_mode(self):
    """Toggle solver comparison mode."""
    self.comparison_mode = not self.comparison_mode
    
    if self.comparison_mode:
        print("Running solver comparison...")
        comparison = SolverComparison(self.env)
        start_state = self.env.state.copy()
        self.comparison_results = comparison.compare_all(start_state, max_time=10.0)
        
        print("\n=== Comparison Results ===")
        for name, metrics in self.comparison_results.items():
            print(metrics)
    else:
        self.comparison_results = None


def generate_random_dungeon(self):
    """Generate a random dungeon."""
    from src.generation.dungeon_generator import DungeonGenerator, Difficulty
    import random
    
    seed = random.randint(0, 99999)
    difficulty = Difficulty.MEDIUM
    
    print(f"Generating dungeon (seed={seed}, difficulty={difficulty.name})...")
    gen = DungeonGenerator(width=40, height=40, difficulty=difficulty, seed=seed)
    grid = gen.generate()
    
    # Load new dungeon
    self.env = ZeldaLogicEnv(grid)
    self.env.reset()
    print(f"✓ Generated {len(gen.rooms)} rooms, {len(gen.key_positions)} keys")


def cycle_floor(self):
    """Cycle to next floor (multi-floor dungeons)."""
    if self.floor_selector.num_floors > 1:
        self.floor_selector.current_floor = (self.floor_selector.current_floor + 1) % self.floor_selector.num_floors
        print(f"Switched to Floor {self.floor_selector.current_floor + 1}")
```

---

### ✅ Step 6: Update Auto-Solve to Use New Solvers

Modify the auto-solve method:

```python
def auto_solve(self):
    """Run pathfinding solver (choose based on mode)."""
    start_state = self.env.state.copy()
    
    # Choose solver
    if self.use_dstar and self.dstar_solver:
        print("Running D* Lite...")
        success, path, states = self.dstar_solver.solve(start_state)
        solver_name = "D* Lite"
    
    elif self.use_parallel and self.parallel_solver:
        print("Running Parallel A*...")
        success, path, states = self.parallel_solver.solve(start_state)
        solver_name = "Parallel A*"
    
    else:
        print("Running A*...")
        success, path, states = self.solver.solve(start_state)
        solver_name = "A*"
    
    if success:
        print(f"✓ {solver_name}: Found path ({len(path)} steps, {states} states)")
        self.planned_path = path
        self.auto_solve_active = True
    else:
        print(f"✗ {solver_name}: No solution found")
```

---

### ✅ Step 7: Add Replan Detection

Add to the main update loop:

```python
def update(self, dt: float):
    # ... existing update logic ...
    
    # Check if replanning needed (D* Lite)
    if self.use_dstar and self.dstar_solver and self.auto_solve_active:
        current_state = self.env.state
        if self.dstar_solver.needs_replan(current_state):
            print("Environment changed - replanning...")
            success, new_path, updated = self.dstar_solver.replan(current_state)
            
            if success:
                self.planned_path = new_path
                self.replan_indicator.trigger(
                    f"Replanned! ({updated} states updated)",
                    self.clock_time
                )
    
    # Update indicator animation
    self.replan_indicator.update(self.clock_time, dt)
```

---

### ✅ Step 8: Render New GUI Components

Add to `render()` method:

```python
def render(self):
    # ... existing rendering ...
    
    # Render TIER 2 components
    if self.floor_selector.num_floors > 1:
        self.floor_selector.render(self.screen, self.font)
    
    if self.minimap_zoom.state.enabled:
        self.minimap_zoom.render(self.screen, self.minimap_surface, self.tile_size)
    
    if self.item_tooltip.visible:
        self.item_tooltip.render(self.screen)
    
    self.replan_indicator.render(self.screen)
    
    # Comparison mode split-screen
    if self.comparison_mode and self.comparison_results:
        self.render_comparison_split_screen()
```

---

### ✅ Step 9: Handle Mouse Events for Zoom and Tooltips

Add to `handle_events()`:

```python
def handle_events(self):
    for event in pygame.event.get():
        # ... existing events ...
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Floor selector click
            if self.floor_selector.handle_click(mouse_pos):
                print(f"Switched to Floor {self.floor_selector.current_floor + 1}")
            
            # Minimap zoom
            self.minimap_zoom.handle_mouse_down(mouse_pos, event.button)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.minimap_zoom.handle_mouse_up(pygame.mouse.get_pos())
        
        elif event.type == pygame.MOUSEWHEEL:
            self.minimap_zoom.handle_mouse_wheel(event.y, pygame.mouse.get_pos())
        
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            
            # Update floor selector hover
            self.floor_selector.handle_hover(mouse_pos)
            
            # Update tooltip
            from src.gui.tier2_components import get_tile_from_mouse
            tile_pos = get_tile_from_mouse(
                mouse_pos,
                self.minimap_rect,
                self.env.grid.shape,
                self.tile_size
            )
            self.item_tooltip.update(
                mouse_pos,
                tile_pos,
                self.env.grid,
                self.env.state.collected_items,
                self.clock_time
            )
```

---

### ✅ Step 10: Add Waypoint Visualization

Add helper method for multi-goal visualization:

```python
def visualize_waypoints(self, waypoints: list):
    """Visualize waypoints on minimap."""
    from src.gui.tier2_components import render_waypoint_numbers
    
    # Create overlay surface
    overlay = pygame.Surface(self.minimap_surface.get_size(), pygame.SRCALPHA)
    
    # Render waypoint numbers
    render_waypoint_numbers(
        overlay,
        waypoints,
        self.tile_size,
        self.font
    )
    
    # Blit to minimap
    self.minimap_surface.blit(overlay, (0, 0))
    
    print(f"Rendered {len(waypoints)} waypoints on minimap")
```

---

## TESTING INTEGRATION

### Quick Test Commands

After integration, test each feature:

```python
# Test D* Lite
python gui_runner.py  # Press 'D' to toggle D* Lite

# Test Parallel Search
python gui_runner.py  # Press 'P' to toggle Parallel A*

# Test Multi-Goal
python gui_runner.py  # Press 'M' to compute optimal route

# Test Solver Comparison
python gui_runner.py  # Press 'C' to compare algorithms

# Test Procedural Generation
python gui_runner.py  # Press 'G' to generate random dungeon

# Test Minimap Zoom
python gui_runner.py  # Drag on minimap, use mouse wheel

# Test Tooltips
python gui_runner.py  # Hover over items on minimap
```

---

## TROUBLESHOOTING

### Issue: Import errors
**Solution:** Ensure all new files are in correct directories

### Issue: PyTorch not found (ML heuristics)
**Solution:** `pip install torch` or skip ML features

### Issue: Multiprocessing errors
**Solution:** Reduce `n_workers` in ParallelAStarSolver

### Issue: Performance issues
**Solution:** Disable comparison mode for large dungeons

---

## NEXT STEPS

1. ✅ Complete integration following this guide
2. ✅ Run all tests: `pytest tests/test_tier2_features.py -v`
3. ✅ Test each feature interactively in GUI
4. ✅ Record video demonstration
5. ✅ Write user manual with screenshots
6. ✅ Prepare for academic publication

---

**Integration Status: Ready for Implementation**

Follow this guide step-by-step to integrate all TIER 2 & 3 features into the existing GUI!
