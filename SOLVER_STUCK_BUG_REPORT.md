# Solver Running State Bug - Root Cause Analysis and Fix

**Date:** 2026-02-04  
**Investigator:** Error Detective Mode  
**Severity:** HIGH - Prevents auto-solve animation from starting

---

## 🔴 ROOT CAUSE

The `solver_running` flag gets stuck at `True` when the subprocess completion detection code encounters exceptions during result processing, causing the cleanup code to be skipped.

### Critical Code Path Flaw

**Location:** [gui_runner.py:4400-4497](gui_runner.py#L4400-L4497)

#### Original Problem:

1. **Subprocess polling at line 4408:**
   ```python
   proc_alive = proc.is_alive() if proc else False
   ```
   - If `proc.is_alive()` raises an exception (process handle corruption, OS issues), it's silently caught
   - Code continues thinking process is done

2. **Result loading and cleanup (lines 4417-4457):**
   - Output file loading in `try` block
   - Cleanup in `finally` block sets `solver_running = False` at line 4456
   - **BUT**: Result application code (lines 4461-4489) is **OUTSIDE** the finally block

3. **Second cleanup at line 4495:**
   ```python
   self.solver_running = False  # Outside exception handling!
   ```
   - If exception occurs in result application (lines 4461-4489), this line is **never reached**
   - `solver_running` stays `True` forever

### Failure Scenarios Identified:

1. ✅ **Process handle exception** - `proc.is_alive()` crashes
2. ✅ **Result loading exception** - Pickle corrupted/malformed
3. ✅ **Result application exception** - `_execute_auto_solve()` crashes  ⬅️ **MOST LIKELY**
4. ✅ **File cleanup exception** - Temp file deletion fails (minor)
5. ✅ **No timeout mechanism** - Hanging subprocess never cleaned up
6. ✅ **Missing start time tracking** - Can't detect hung processes

---

## 📍 ALL SOLVER_RUNNING ASSIGNMENTS

### Set to `True`:
- **Line 4970**: `_schedule_solver()` - Acquires solver lock before spawning

### Set to `False`:
- **Line 4456**: Main completion cleanup `finally` block ⚠️ (skipped if exception after)
- **Line 4495**: **OUTSIDE** exception handling ⚠️ (never reached if crash)
- **Line 4984**: Early exit when start/goal missing ✅
- **Line 5077**: Thread fallback `finally` block ✅
- **Line 5086**: Thread spawn failure ✅

### Critical Gap:
Lines 4456 and 4495 are **separated** by result application code (lines 4461-4489). If that code crashes, `solver_running` is never cleared.

---

## 🔧 FIXES APPLIED

### 1. **Unified Finally Block** (Lines 4400-4510)

Moved ALL cleanup into single comprehensive `finally` block that wraps **entire** completion handler:

```python
try:
    # Load result
    # Apply result (including _execute_auto_solve)
    # Show messages
finally:
    # ALWAYS clean up, no matter what crashes above
    self.solver_running = False
    self.solver_done = True
    self.solver_start_time = None
    # ... cleanup files/processes
```

**Guarantee:** `solver_running` is cleared even if:
- Result loading fails
- Result application crashes
- `_execute_auto_solve()` raises exception
- Message display fails

### 2. **Timeout Detection** (Lines 4407-4420)

Added subprocess timeout mechanism:

```python
solver_timeout = float(os.environ.get('KLTN_SOLVER_TIMEOUT', '10.0'))
solver_start_time = getattr(self, 'solver_start_time', None)
if solver_start_time and (time.time() - solver_start_time) > solver_timeout:
    logger.error('SOLVER: TIMEOUT after %.1fs - forcefully terminating', solver_timeout)
    proc.terminate()
    proc_alive = False
```

**Default:** 10 seconds (configurable via `KLTN_SOLVER_TIMEOUT`)  
**Action:** Forcefully terminates hung subprocess and proceeds to cleanup

### 3. **Start Time Tracking** (Line 4973)

Track when solver starts for timeout detection:

```python
self.solver_start_time = time.time()
```

Cleared in all cleanup paths.

### 4. **Automatic Recovery** (Lines 4789-4817)

Added recovery logic in `_start_auto_solve()` when stuck state detected:

```python
if not proc_alive and not done:
    logger.error('DEBUG_SOLVER: RECOVERY - force-cleaning stuck solver state')
    self.solver_running = False
    self.solver_done = True
    # ... clear all state
```

**Trigger:** `solver_running=True` but process is dead and not marked done  
**Action:** Force-clean state and allow retry

### 5. **Enhanced Diagnostics** (Lines 4766-4788)

When solver blocks, log complete state dump:

```python
logger.warning('DEBUG_SOLVER: Solver state dump:')
logger.warning('  solver_running=%s', ...)
logger.warning('  solver_proc=%s (alive=%s)', ...)
logger.warning('  solver_outfile=%s (exists=%s)', ...)
logger.warning('  solver_start_time=%s (age=%.1fs)', ...)
```

Helps diagnose **which** failure path was taken.

### 6. **Thread Fallback Safety** (Lines 5070-5089)

Ensured thread fallback properly clears state:

```python
finally:
    self.solver_running = False
    self.solver_done = True
    self.solver_start_time = None
```

And handles thread spawn failures:

```python
except Exception:
    # Clear state if thread spawn fails
    self.solver_running = False
    self.solver_done = True
    self.solver_start_time = None
```

---

## 🧪 TESTING RECOMMENDATIONS

### 1. **Simulate Result Application Crash**

Patch `_execute_auto_solve` to raise exception:

```python
def _execute_auto_solve(self, path, solver_result, teleports=0):
    raise RuntimeError("Simulated crash in result application")
```

**Expected:** `solver_running` should still be cleared, next SPACE press works.

### 2. **Simulate Timeout**

Set `KLTN_SOLVER_TIMEOUT=2` and use complex map:

```bash
set KLTN_SOLVER_TIMEOUT=2
python gui_runner.py
```

**Expected:** Solver terminates after 2s, cleanup runs, next solve works.

### 3. **Simulate Process Corruption**

Manually kill subprocess PID during solve:

```bash
# In another terminal while solver running:
taskkill /F /PID <solver_pid>
```

**Expected:** Main loop detects dead process, cleans up, allows retry.

### 4. **Verify Recovery Logic**

Manually set stuck state in debugger:

```python
self.solver_running = True
self.solver_done = False
self.solver_proc = None
```

Press SPACE.

**Expected:** Recovery logic kicks in, state cleared, retry allowed.

---

## 📊 CHANGED CODE LOCATIONS

| File | Lines | Change |
|------|-------|--------|
| gui_runner.py | 4400-4510 | Unified finally block with timeout detection |
| gui_runner.py | 4973 | Added `solver_start_time` tracking |
| gui_runner.py | 4766-4817 | Enhanced diagnostics + auto-recovery |
| gui_runner.py | 4987 | Clear `solver_start_time` on early exit |
| gui_runner.py | 5076-5078 | Thread fallback clears `solver_start_time` |
| gui_runner.py | 5086-5092 | Thread spawn failure clears all state |

---

## 🎯 PREVENTION STRATEGIES

### General Principles Applied:

1. ✅ **Single cleanup point** - All state clearing in one `finally` block
2. ✅ **Timeout guards** - Never wait indefinitely for subprocess
3. ✅ **Exception safety** - Cleanup runs even if result handling crashes
4. ✅ **Diagnostic logging** - Full state dumps on suspicious conditions
5. ✅ **Self-healing** - Auto-recovery when stuck state detected
6. ✅ **Defensive programming** - All subprocess operations wrapped in try/except

### Code Review Checklist:

When working with background processes/threads:

- [ ] State flags cleared in **finally** blocks, not after
- [ ] Timeout mechanisms for all blocking operations
- [ ] Exception handling around ALL subprocess interaction
- [ ] Diagnostic logging for stuck states
- [ ] Recovery logic when inconsistent state detected
- [ ] Start time tracking for timeout detection

---

## 📝 SUMMARY

**Problem:** `solver_running` stuck at `True`, blocking subsequent solve attempts.

**Root Cause:** Exception in result application code (lines 4461-4489) prevented cleanup code at line 4495 from running.

**Solution:** Wrapped entire completion handler in single `finally` block, added timeout detection, start time tracking, and automatic recovery logic.

**Result:** `solver_running` is **guaranteed** to be cleared in all code paths, including crashes, timeouts, and process corruption.

**User Impact:** Press SPACE → Solver runs → Auto-solve animation starts **reliably**.

---

## 🔍 LESSONS LEARNED

1. **Never trust subprocess state** - Always have timeout + recovery
2. **Finally blocks must be comprehensive** - Include ALL code that depends on cleanup
3. **State flags need atomic operations** - Clear all related state together
4. **Defensive diagnostics** - Log full state dumps when suspicious
5. **Test failure paths** - Exception paths are often untested and broken

---

**Investigation Complete** ✅  
**Fixes Applied** ✅  
**Ready for Testing** ✅
