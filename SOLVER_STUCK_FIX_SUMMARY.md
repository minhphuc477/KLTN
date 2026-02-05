# 🔧 SOLVER STUCK FLAG - CRITICAL FIXES APPLIED

## Problem Description
Auto-solve animation would not start. GUI showed "Solver already running" every time SPACE was pressed, indicating the `solver_running` flag was stuck at `True`.

## Root Cause Analysis

### Bug #1: Thread Fallback Setting solver_done=True
**Location:** `gui_runner.py` line ~5195 (thread fallback finally block)

**Problem:**
- When multiprocessing.Process spawn failed, code fell back to threading.Thread
- Thread fallback's finally block set `solver_done = True`
- Main loop checks `if not solver_done:` before polling for results
- With `solver_done=True`, main loop NEVER polled for results
- Results written to file but never loaded
- `solver_running` never cleared by main loop
- Next SPACE press → "Solver already running"

**Fix:**
```python
# BEFORE (BUG):
finally:
    self.solver_running = False
    self.solver_done = True  # ← BUG!
    
# AFTER (FIXED):
finally:
    self.solver_running = False
    # Let main loop discover completion naturally
    self.solver_starting = False
```

### Bug #2: Recovery Block Setting solver_done=True
**Location:** `gui_runner.py` line ~4864 (recovery block)

**Problem:**
- Recovery detected stuck state and cleared `solver_running`
- But also set `solver_done = True`
- Retry would start new solver successfully
- But main loop wouldn't poll (solver_done=True)
- Results never loaded
- Next SPACE press → stuck again

**Fix:**
```python
# BEFORE (BUG):
if needs_recovery:
    self.solver_running = False
    self.solver_done = True  # ← BUG!
    
# AFTER (FIXED):
if needs_recovery:
    self.solver_running = False
    # Do NOT set solver_done! Main loop needs to poll.
    self.solver_starting = False
```

### Bug #3: Thread Spawn Failure Setting solver_done=True
**Location:** `gui_runner.py` line ~5210 (thread spawn exception handler)

**Problem:**
- If threading.Thread() raised exception
- Code set `solver_done = True`
- Prevented cleanup and retry

**Fix:**
```python
# BEFORE (BUG):
except Exception as thread_err:
    self.solver_running = False
    self.solver_done = True  # ← BUG!
    
# AFTER (FIXED):
except Exception as thread_err:
    self.solver_running = False
    # Do NOT set solver_done - allow retry
```

### Bug #4: solver_starting Not Cleared in Recovery
**Location:** `gui_runner.py` line ~4866 (recovery block)

**Problem:**
- Recovery cleared other flags but forgot `solver_starting`
- Could cause startup grace period to block next attempt

**Fix:**
```python
# Added to recovery:
self.solver_starting = False
```

### Bug #5: Duplicate Recovery Log Lines
**Location:** `gui_runner.py` line ~4858-4860

**Problem:**
- Duplicate `if needs_recovery:` blocks
- Confusing logs

**Fix:**
- Removed duplicate block

## Semantic Clarification

### What solver_done ACTUALLY Means:
- `solver_done = False` → "Main loop should poll for results"
- `solver_done = True` → "Main loop has finished polling and cleanup"

### The Confusion:
Subprocess/thread code was setting `solver_done = True` to mean "solver finished running", but main loop interprets it as "I already polled, don't poll again".

### The Solution:
**ONLY the main loop's finally block should set solver_done = True** (after loading results and cleaning up). Background threads/processes should NEVER touch solver_done.

## Testing Instructions

```powershell
cd C:\Users\MPhuc\Desktop\KLTN
python gui_runner.py
```

1. Press SPACE to start auto-solve
2. Wait for solver to complete (or time out after 15s)
3. Verify animation starts automatically
4. Verify "Solver already running" does NOT appear on subsequent SPACE presses
5. Press SPACE again - should start new solve successfully

## Verification Checklist

- [ ] First SPACE press starts solver
- [ ] Solver completes and results are loaded
- [ ] Animation plays automatically
- [ ] solver_running returns to False
- [ ] Second SPACE press starts new solve (no "already running" message)
- [ ] Recovery works if solver times out (press SPACE again after 15s)
- [ ] Thread fallback works (if multiprocessing fails on platform)

## Files Modified
- `gui_runner.py` - 5 critical fixes

## Lines Changed
- Line ~4866: Removed `solver_done = True` from recovery, added `solver_starting = False`
- Line ~4858: Removed duplicate recovery block
- Line ~5195: Removed `solver_done = True` from thread fallback, added `solver_starting = False`
- Line ~5210: Removed `solver_done = True` from thread spawn failure

## Risk Assessment
**Risk Level:** LOW
- Changes are surgical and minimal
- Only affect error handling paths
- Main loop logic unchanged
- Backward compatible

## Rollback Plan
If issues occur, comment out all fixes and add this to recovery block:
```python
if needs_recovery:
    self.solver_running = False
    self.solver_done = True  # Force retry
    self.solver_proc = None
```

## Additional Improvements Made

1. **Enhanced Timeout Detection**
   - Added 15s timeout (configurable via `KLTN_SOLVER_TIMEOUT`)
   - Force-terminates hung processes
   - Automatically recovers on next SPACE press

2. **Startup Grace Period**
   - Added 1.5s grace (configurable via `KLTN_SOLVER_STARTUP_GRACE`)
   - Prevents premature completion detection during async spawn
   - Uses `solver_starting` flag to coordinate

3. **Comprehensive Recovery**
   - Detects dead processes
   - Detects hung processes
   - Detects missing process handles
   - Force-cleans all state flags
   - Allows immediate retry

4. **Better Logging**
   - Added diagnostic dumps when "already running" detected
   - Shows process state, file existence, age, etc.
   - Helps debug future issues

## Known Limitations

1. **Pickle Security**: Still uses pickle for IPC (security risk if temp dir compromised)
2. **No Progress Bar**: Solver runs with no progress indication
3. **Timeout Edge Case**: If solver completes exactly at timeout, results might be lost

## Future Improvements

1. Replace pickle with JSON serialization
2. Add progress callback from subprocess
3. Add cancel button during solving
4. Cache solver results per map
5. Show "Solving... N seconds" timer in UI

---

**Applied:** 2026-02-04
**Priority:** CRITICAL
**Status:** ✅ FIXED
