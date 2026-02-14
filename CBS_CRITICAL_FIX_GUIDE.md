# CRITICAL FIX: Duplicate get_tile() Method Resolution
**Issue ID**: BLOCKER #1  
**File**: src/simulation/cognitive_bounded_search.py  
**Severity**: Critical (breaks interface, causes test failures)

---

## Problem Statement

The `BeliefMap` class defines `get_tile()` **twice** with conflicting return types:

1. **Line 376**: Returns `Tuple[int, float]` (tile_type, confidence)
2. **Line 488**: Returns `int` (tile_type only) — **This overwrites the first definition**

Python uses the second definition, breaking code expecting a tuple unpacking:
```python
tile_type, confidence = belief.get_tile((5, 5))  # TypeError!
```

---

## Recommended Fix (Option A): Keep Tuple-Returning Primary Method

**Strategy**: Keep the first definition as `get_tile()`, rename the second to `get_tile_type()`

### Changes Required

**1. BeliefMap class (Lines 488-505)**

**REMOVE** (or rename):
```python
def get_tile(self, row_or_pos: int | Tuple[int, int], col: Optional[int] = None) -> int:
    """
    Get the believed tile type at a position.
    
    Can be called as:
        get_tile((row, col))  or  get_tile(row, col)
    
    Returns:
        Tile type ID (int)
    """
    if col is not None:
        position = (row_or_pos, col)
    else:
        position = row_or_pos
        
    if position in self.known_tiles:
        return self.known_tiles[position].tile_type
    return self.default_assumption
```

**REPLACE WITH**:
```python
def get_tile_type(self, position: Tuple[int, int]) -> int:
    """
    Get the believed tile type at a position (convenience method).
    
    Args:
        position: (row, col) tuple
    
    Returns:
        Tile type ID (int)
    """
    tile_type, _ = self.get_tile(position)
    return tile_type

def get_tile_id(self, position: Tuple[int, int]) -> int:
    """Alias for get_tile_type() for backward compatibility."""
    return self.get_tile_type(position)
```

**2. Update all internal callers**

Search for calls like:
```python
tile_type = self.belief_map.get_tile(pos)  # Returns tuple, expects int
```

**Replace with**:
```python
tile_type = self.belief_map.get_tile_type(pos)  # Returns int
# OR
tile_type, confidence = self.belief_map.get_tile(pos)  # Returns tuple
```

**Known locations to update**:
- Line 1956: `target_tile = grid[...]` (uses grid directly, OK)
- Check if any internal methods call `belief_map.get_tile()` expecting int

---

## Alternative Fix (Option B): Keep Int-Returning Primary Method

**Strategy**: Keep the second definition as `get_tile()`, rename the first to `get_tile_with_confidence()`

### Changes Required

**1. BeliefMap class (Lines 376-387)**

**RENAME**:
```python
def get_tile(self, position: Tuple[int, int]) -> Tuple[int, float]:
```
**TO**:
```python
def get_tile_with_confidence(self, position: Tuple[int, int]) -> Tuple[int, float]:
```

**2. Update test files**

**File**: tests/test_cognitive_bounded_search.py

**Line 127**:
```python
# BEFORE:
tile_type, confidence = belief.get_tile((5, 5))

# AFTER:
tile_type, confidence = belief.get_tile_with_confidence((5, 5))
```

**Line 136**:
```python
# BEFORE:
tile_type, confidence = belief.get_tile((5, 5))

# AFTER:
tile_type, confidence = belief.get_tile_with_confidence((5, 5))
```

**File**: tests/test_cbs_full.py (search for similar patterns)

---

## Recommended Choice: **Option A**

**Rationale**:
1. The tuple-returning version (line 376) is **more informative** (provides confidence)
2. It's **used by tests** (test_cognitive_bounded_search.py expects tuple)
3. Callers who only need `tile_type` can use:
   ```python
   tile_type, _ = belief.get_tile(pos)  # Ignore confidence
   # OR
   tile_type = belief.get_tile_type(pos)  # New convenience method
   ```
4. **Backward compatible** with existing test expectations

---

## Implementation Steps

### Step 1: Edit cognitive_bounded_search.py

```python
# Line 488-505: REPLACE with:

def get_tile_type(self, position: Tuple[int, int]) -> int:
    """
    Get the believed tile type at a position (convenience method).
    
    For callers who don't need confidence information, this is simpler
    than unpacking get_tile().
    
    Args:
        position: (row, col) tuple
    
    Returns:
        Tile type ID (int)
    
    Example:
        >>> tile = belief_map.get_tile_type((5, 5))
        >>> # Instead of:
        >>> tile, _ = belief_map.get_tile((5, 5))
    """
    tile_type, _ = self.get_tile(position)
    return tile_type
```

### Step 2: Search for incorrect callers

Run:
```bash
cd c:\Users\MPhuc\Desktop\KLTN
grep -n "belief_map\.get_tile(" src/simulation/cognitive_bounded_search.py
```

### Step 3: Update callers (if any)

Example:
```python
# BEFORE:
tile = self.belief_map.get_tile(pos)  # Expects int, gets tuple

# AFTER (Option 1):
tile = self.belief_map.get_tile_type(pos)  # Gets int

# AFTER (Option 2):
tile, _ = self.belief_map.get_tile(pos)  # Unpack tuple
```

### Step 4: Run tests

```bash
cd c:\Users\MPhuc\Desktop\KLTN
python -m pytest tests/test_cognitive_bounded_search.py::TestBeliefMap -v
```

**Expected output**:
```
test_initialization PASSED
test_observe_tile PASSED   ← Should now PASS
test_unknown_tile PASSED   ← Should now PASS
test_confidence_decay PASSED
test_confusion_index PASSED

====== 5 passed in X.XXs ======
```

---

## Verification Checklist

After applying the fix:

- [ ] `get_tile()` returns `Tuple[int, float]` (line 376)
- [ ] `get_tile_type()` returns `int` (new method)
- [ ] No duplicate method definitions
- [ ] All tests in `TestBeliefMap` pass
- [ ] No regressions in CBS solver integration test
- [ ] Run full test suite: `pytest tests/ -v`

---

## Estimated Time

- **Implementation**: 15 minutes
- **Testing**: 10 minutes
- **Validation**: 5 minutes
- **Total**: 30 minutes

---

## Risk Assessment

**Risk Level**: Low  
**Reason**: Isolated change, well-defined scope, comprehensive tests available  
**Rollback Plan**: Revert commit if tests fail

---

## Additional Notes

### Why This Happened

Likely causes:
1. Refactoring to add convenience overload (row, col separate args)
2. Forgot to rename one of the methods
3. No static type checker run (Pyright would catch this as "method already defined")

### Prevention

1. **Enable Pyright**: Run `pyright src/simulation/cognitive_bounded_search.py`
2. **Pre-commit hook**: Add type checking to CI pipeline
3. **Test-driven development**: Write tests before refactoring

---

**Fix Author**: AI Code Reviewer  
**Review Status**: Ready for implementation  
**Approval**: Required before production deployment
