# CBS Banner Bug Fix - Quick Reference

## ✅ Changes Applied

### 1. Banner Now Uses Running Solver's Algorithm
- **File**: `gui_runner.py` line 2774
- **What**: Banner reads from `solver_algorithm_idx` (set when solver starts) instead of `algorithm_idx` (current dropdown value)
- **Why**: Prevents showing wrong algorithm if user changes dropdown while solver running

```python
# OLD (wrong):
alg_idx = getattr(self, 'algorithm_idx', 0)  # Uses dropdown value

# NEW (correct):
alg_idx = getattr(self, 'solver_algorithm_idx', getattr(self, 'algorithm_idx', 0))  # Uses running solver's value
```

### 2. Solver Saves Its Algorithm Index on Start
- **File**: `gui_runner.py` line 6345
- **What**: When `_schedule_solver()` runs, it saves `solver_algorithm_idx` to lock in which algorithm it's using
- **Why**: Banner can show correct algorithm even if dropdown changes mid-solve

```python
self.solver_algorithm_idx = current_alg_idx if current_alg_idx is not None else 0
```

### 3. Changing Algorithm Stops Running Solver
- **File**: `gui_runner.py` line 3433
- **What**: When user changes algorithm dropdown, if solver is running, it's terminated and state cleared
- **Why**: Prevents confusion - old solver can't finish while banner shows new algorithm

```python
if getattr(self, 'solver_running', False):
    # Terminate old solver, clear state
    self._clear_solver_state(reason=f"algorithm changed to {algorithm_names[self.algorithm_idx]}")
```

### 4. Centralized State Cleanup Helper
- **File**: `gui_runner.py` line 5990
- **What**: Added `_clear_solver_state()` to ensure consistent cleanup everywhere
- **Why**: Previously each cleanup location manually cleared state, easy to miss `solver_algorithm_idx`

```python
def _clear_solver_state(self, reason="cleanup"):
    self.solver_running = False
    self.solver_done = True
    # ... clear all state ...
    if hasattr(self, 'solver_algorithm_idx'):
        delattr(self, 'solver_algorithm_idx')  # ← KEY: Clear saved algorithm
```

### 5. Debug Logging Added
- **Locations**: Dropdown handler, solver start, banner render, cleanup
- **What**: Log current algorithm_idx at each step
- **Why**: Makes debugging similar issues easier in future

```
DROPDOWN: Algorithm changed from 0(A*) to 5(CBS (Balanced))
SOLVER: Acquired solver lock, algorithm_idx=5, solver_algorithm_idx=5
BANNER: Rendering solver banner with solver_algorithm_idx=5, alg_name=CBS (Balanced)
```

---

## 🧪 How to Test

### Test 1: Basic CBS Selection
```
1. Run: python gui_runner.py
2. Select "CBS (Balanced)" from Solver dropdown
3. Press SPACE
4. VERIFY: Banner shows "🔍 Computing path with CBS (Balanced)..."
   (NOT "Computing path with A*...")
```

### Test 2: Change Algorithm While Solving
```
1. Start solver with A* (press SPACE)
2. While A* is running, select "CBS (Balanced)"
3. VERIFY: 
   - A* solver stops
   - Message: "Switched to CBS (Balanced) (press SPACE to solve)"
   - Banner disappears
4. Press SPACE again
5. VERIFY: Banner shows "Computing path with CBS (Balanced)..."
```

### Test 3: Check Logs
```
1. Run with logging enabled
2. Change algorithm dropdown
3. Start solver
4. VERIFY logs show:
   - DROPDOWN: Algorithm changed from X to Y
   - SOLVER: algorithm_idx=Y, solver_algorithm_idx=Y
   - BANNER: solver_algorithm_idx=Y, alg_name=CBS (...)
```

---

## 🐛 What Was The Bug?

**The preview worker auto-started A* solver before user selected CBS:**
1. Map loads → preview worker starts
2. Preview fails fast solvers → launches heavy solver with A* (default)
3. User sees banner "Computing with A*"
4. User thinks "I want CBS!" and selects from dropdown
5. User presses SPACE (not knowing solver already running)
6. New solver blocked ("solver already running")
7. Banner still shows A* because A* is what's actually running!

**Why this is confusing:**
- Dropdown shows "CBS (Balanced)" ✓
- Banner shows "Computing with A*" ✗
- User thinks they're running CBS but they're actually running A*!

---

## 📊 Impact

**Before Fix:**
- User sees wrong algorithm name in banner
- Cannot interrupt auto-started solver to choose different algorithm
- No way to tell which algorithm is actually running

**After Fix:**
- Banner always shows correct running algorithm
- User can change algorithm anytime (stops old solver if needed)
- Clear feedback when algorithm changes
- Debug logs for troubleshooting

---

## 📁 Files Modified

- **`gui_runner.py`**: 
  - ~line 2774: `_render_solver_status_banner()` - use `solver_algorithm_idx`
  - ~line 3433: Algorithm dropdown handler - stop solver on change
  - ~line 5990: Added `_clear_solver_state()` helper
  - ~line 6345: `_schedule_solver()` - save `solver_algorithm_idx`
  - Multiple cleanup locations: Use `_clear_solver_state()`

- **`CBS_BANNER_BUG_FIX.md`**: Detailed analysis document
- **`test_cbs_banner_fix.py`**: Verification test script

---

## ✅ Verification

Run the test script:
```bash
python test_cbs_banner_fix.py
```

Manual verification (recommended):
1. Run GUI: `python gui_runner.py`
2. Select CBS (Balanced)
3. Press SPACE
4. Check banner shows "CBS (Balanced)" not "A*"

---

## 🎯 Summary

**One-line fix:** Banner now reads from `solver_algorithm_idx` (set when solver starts) instead of `algorithm_idx` (current dropdown value), so it always shows the correct running algorithm.

**Why 5 changes for "one-line fix"?**
- Need to SAVE algorithm when solver starts → `solver_algorithm_idx`
- Need to CLEAR saved algorithm on cleanup → `_clear_solver_state()`
- Need to STOP old solver when algorithm changes → dropdown handler
- Need to USE saved algorithm in banner → `_render_solver_status_banner()`
- Add DEBUG logging → all locations

All working together to ensure the banner always matches reality! ✨
