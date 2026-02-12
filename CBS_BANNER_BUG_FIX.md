# CBS Banner Display Bug - Root Cause & Fix

## Problem Statement
When user selects "CBS (Balanced)" from the Solver dropdown, the yellow banner shows "🔍 Computing path with A*..." instead of "🔍 Computing path with CBS (Balanced)..."

## Screenshot Evidence
- **Solver dropdown**: "CBS (Balanced)" selected ✓
- **Yellow banner**: "🔍 Computing path with A*..." ✗ WRONG!

## Root Cause Analysis

### The Bug Flow

1. **Map loads** → Preview worker starts automatically (`_start_preview_search()`)
2. **Preview worker** tries quick solvers (graph-based, grid-based), they fail
3. **Preview worker** calls `_schedule_solver()` at line 5907 with `algorithm_idx=0` (A*)
4. **Solver starts** with A*, sets `solver_running=True`, banner shows "Computing path with A*..."
5. **User selects** "CBS (Balanced)" from dropdown → `algorithm_idx=5`
6. **User presses SPACE** to start solving (not knowing solver already running)
7. **`_start_auto_solve()`** calls `_schedule_solver()` again
8. **`_schedule_solver()` blocked** because `solver_running=True` (A* solver still running)
9. **Banner continues** showing "Computing path with A*..." because that's the actual running solver

### Key Code Locations

**Preview worker auto-starts solver:**
```python
# Line 5907 in gui_runner.py
# No quick preview found; schedule heavy solver if allowed
if os.environ.get('KLTN_ALLOW_HEAVY', '1') == '1':
    try:
        # Launch heavy solver asynchronously
        self._schedule_solver()  # ← Uses algorithm_idx=0 (A*) by default
```

**Solver blocks if already running:**
```python
# Line 6320 in gui_runner.py (in _schedule_solver)
if getattr(self, 'solver_running', False):
    self._set_message('Solver already running...')
    logger.warning('SOLVER: _schedule_solver blocked - solver_running already True')
    return  # ← Blocks second solver start
```

**Banner reads algorithm_idx:**
```python
# Line 2773 in gui_runner.py (in _render_solver_status_banner)
alg_idx = getattr(self, 'algorithm_idx', 0)  # ← Reads current dropdown value
alg_name = algorithm_names[alg_idx]  # ← But solver might be using different algorithm!
```

## The Fix

### Three-Part Solution

#### 1. Stop Running Solver When Algorithm Changes
**File:** `gui_runner.py` ~line 3413 (algorithm dropdown handler)

When user changes the algorithm dropdown:
- Terminate the currently running solver process
- Clear all solver state
- Show message: "Switched to [new algorithm] (press SPACE to solve)"
- User must explicitly press SPACE to start new solver

```python
# CRITICAL FIX: If solver is currently running with old algorithm, stop it
if getattr(self, 'solver_running', False):
    # Terminate process, clear preview workers, clear state
    self._clear_solver_state(reason=f"algorithm changed to {algorithm_names[self.algorithm_idx]}")
```

#### 2. Track Which Algorithm the Running Solver Uses
**File:** `gui_runner.py` ~line 6334 (in `_schedule_solver`)

Save the algorithm_idx when solver starts:
```python
# CRITICAL FIX: Save the algorithm_idx that THIS solver is using
self.solver_algorithm_idx = current_alg_idx if current_alg_idx is not None else 0
```

Clear it when solver stops:
```python
def _clear_solver_state(self, reason="cleanup"):
    self.solver_running = False
    self.solver_done = True
    # ... other cleanup ...
    if hasattr(self, 'solver_algorithm_idx'):
        delattr(self, 'solver_algorithm_idx')
```

#### 3. Banner Uses Running Solver's Algorithm, Not Dropdown Value
**File:** `gui_runner.py` ~line 2773 (in `_render_solver_status_banner`)

```python
# CRITICAL FIX: Use solver_algorithm_idx (saved when solver started) instead of algorithm_idx
# This ensures banner shows the algorithm the CURRENTLY RUNNING solver is using,
# even if user changes the dropdown while solver is running
alg_idx = getattr(self, 'solver_algorithm_idx', getattr(self, 'algorithm_idx', 0))
```

### Centralized Cleanup Helper
**File:** `gui_runner.py` ~line 5996

Added `_clear_solver_state()` helper to ensure consistent cleanup:
- Clears `solver_running`, `solver_done`, `solver_proc`, etc.
- **Importantly**: Clears `solver_algorithm_idx` to prevent stale banner display
- Used in all cleanup paths: completion, failure, timeout, algorithm change

## Testing the Fix

### Test Case 1: Change Algorithm Before Solving
1. Open application (algorithm_idx=0, A* by default)
2. Select "CBS (Balanced)" from dropdown → algorithm_idx=5
3. Press SPACE to solve
4. **Expected**: Banner shows "🔍 Computing path with CBS (Balanced)..." ✓

### Test Case 2: Change Algorithm While Solver Running
1. Press SPACE to start A* solver
2. While solver running, select "CBS (Balanced)" from dropdown
3. **Expected**: 
   - A* solver process is terminated
   - Message: "Switched to CBS (Balanced) (press SPACE to solve)"
   - Banner disappears (solver_running=False)
4. Press SPACE again
5. **Expected**: Banner shows "🔍 Computing path with CBS (Balanced)..." ✓

### Test Case 3: Preview Worker Auto-Start
1. Load a map
2. Preview worker starts A* solver automatically
3. User immediately selects "CBS (Balanced)" before A* finishes
4. **Expected**:
   - A* solver is terminated
   - Message: "Switched to CBS (Balanced)"
   - User can start CBS solver with SPACE

## Debug Logging Added

**Dropdown selection:**
```
DROPDOWN: Algorithm changed from 0(A*) to 5(CBS (Balanced))
```

**Solver start:**
```
SOLVER: Acquired solver lock, solver_running=True, algorithm_idx=5, solver_algorithm_idx=5
```

**Banner rendering:**
```
BANNER: Rendering solver banner with solver_algorithm_idx=5, alg_name=CBS (Balanced)
```

**Cleanup:**
```
SOLVER_CLEANUP: Clearing solver state (algorithm changed to CBS (Balanced))
```

## Files Modified
- `gui_runner.py`:
  - Line ~2773: `_render_solver_status_banner()` - use `solver_algorithm_idx`
  - Line ~3413: Algorithm dropdown handler - stop running solver on change
  - Line ~5996: Added `_clear_solver_state()` helper
  - Line ~6334: `_schedule_solver()` - save `solver_algorithm_idx`
  - Multiple locations: Use `_clear_solver_state()` for cleanup

## Related Issues
- Preview worker auto-starting solver before user selects algorithm
- Race condition between dropdown selection and solver start
- Banner showing stale algorithm name

## Notes
- The fix ensures the banner ALWAYS shows the algorithm that's actually running
- User has explicit control over when solver starts (no surprise auto-starts)
- Changing algorithm mid-solve is now safe and predictable
- Debug logging helps diagnose any remaining edge cases
