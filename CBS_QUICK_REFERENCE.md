# CBS Quick Reference Guide

## Quick Start (3 Steps)

1. **Select CBS Solver**
   - Open Solver dropdown
   - Choose: CBS (Balanced), CBS (Explorer), CBS (Cautious), etc.

2. **Start Solving**
   - Click "Start Auto-Solve" button
   - Watch CBS navigate with human-like behavior

3. **View Metrics**
   - Status bar shows: `CBS (Explorer): 42 steps | Confusion: 0.15 | Cognitive Load: 0.82`
   - Toast notification displays detailed metrics

---

## CBS Persona Cheat Sheet

```
┌─────────────────┬──────────────────┬─────────────────────────────┐
│ Persona         │ Best For         │ Key Characteristics         │
├─────────────────┼──────────────────┼─────────────────────────────┤
│ Balanced        │ General use      │ Default, well-rounded       │
│ Explorer        │ Coverage testing │ High curiosity, explores    │
│ Cautious        │ Safety analysis  │ Avoids enemies, safer paths │
│ Forgetful       │ UX stress test   │ Gets lost easily, high conf.│
│ Speedrunner     │ A* comparison    │ Optimal baseline, like A*   │
│ Greedy          │ Perfect memory   │ Never forgets, low conf.    │
└─────────────────┴──────────────────┴─────────────────────────────┘
```

---

## Metrics Quick Reference

### Confusion Index
```
0.0 - 0.5   ✓ Excellent (minimal backtracking)
0.5 - 1.0   ✓ Good (some exploration)
1.0 - 2.0   ⚠ Moderate (noticeable confusion)
> 2.0       ✗ High (agent very lost)
```

### Navigation Entropy
```
0.0 - 0.5   Single-minded (very directed)
0.5 - 1.0   Focused (mostly directed)
1.0 - 1.5   Mixed (some wandering)
1.5 - 2.0   Random (chaotic movement)
```

### Cognitive Load
```
0.0 - 0.5   Low effort (simple navigation)
0.5 - 1.0   Moderate effort (typical)
1.0 - 1.5   High effort (complex decisions)
> 1.5       Very high (cognitive strain)
```

---

## Common Use Cases

### "Which persona should I use?"

**For realistic player behavior:**
→ CBS (Balanced)

**To test if dungeon is explorable:**
→ CBS (Explorer)

**To check enemy placement safety:**
→ CBS (Cautious)

**To stress-test confusing layouts:**
→ CBS (Forgetful)

**To compare with optimal A* path:**
→ CBS (Speedrunner)

**To test with no memory constraints:**
→ CBS (Greedy)

---

## Interpreting Results

### Example Output:
```
CBS (Explorer): 52 steps | Confusion: 1.23 | Cognitive Load: 0.67
```

**Analysis:**
- **52 steps:** Path length (may be > A* optimal)
- **Confusion 1.23:** Moderate backtracking (explored multiple paths)
- **Cognitive Load 0.67:** Moderate mental effort required

**Interpretation:**
Explorer persona found solution but needed to explore. Higher confusion is expected for Explorer. Cognitive load suggests reasonable navigation difficulty.

---

## Comparing CBS vs A*

### When to Use Each:

**Use A***
- Need optimal path length
- Pure pathfinding benchmark
- Performance testing
- Speedrun validation

**Use CBS**
- Playability testing
- UX evaluation
- Cognitive difficulty assessment
- Human-like navigation validation

### Typical Differences:
```
A* Path:        20 steps, direct
CBS Balanced:   24 steps, explored 2 dead ends
CBS Explorer:   35 steps, visited 80% of map
CBS Cautious:   28 steps, avoided enemy room
```

---

## Troubleshooting

### "CBS takes too long"
- **Cause:** Complex maze, high exploration
- **Solution:** Try CBS (Speedrunner) or reduce timeout

### "Confusion index very high (> 3.0)"
- **Cause:** Dungeon has confusing layout OR persona is Forgetful
- **Solution:** Good for UX testing! Shows players may struggle.

### "CBS finds no solution"
- **Cause:** Persona constraints too strict OR unsolvable dungeon
- **Solution:** Try CBS (Balanced) or check dungeon validity

### "Metrics not showing"
- **Cause:** GUI display issue
- **Solution:** Check status bar and toast notifications; metrics logged to console

---

## Keyboard Shortcuts (When CBS Selected)

```
Space     - Start/Stop auto-solve with CBS
R         - Reset dungeon
S         - Switch to next solver (cycles through CBS personas)
M         - Toggle metrics display
```

---

## Advanced: Custom CBS Tuning

### Modifying CBS Behavior
Edit `src/simulation/cognitive_bounded_search.py`:

```python
# Line ~1720: PersonaConfig parameters
config = PersonaConfig(
    memory_capacity=7,        # Working memory size
    memory_decay_rate=0.95,   # Forgetting rate
    vision_radius=5,          # How far agent sees
    ...
)
```

### Creating Custom Personas
```python
custom = PersonaConfig(
    memory_capacity=10,       # Better memory
    memory_decay_rate=0.98,   # Slower forgetting
    heuristic_weights={
        'curiosity': 0.5,     # Moderate exploration
        'goal_seeking': 0.8,  # Strong goal focus
        'safety': 0.3,        # Some caution
    }
)

cbs = CognitiveBoundedSearch(env, custom_config=custom)
```

---

## Performance Notes

### Typical Solve Times (Intel i7, 2.6GHz):
```
Simple dungeon (10x10):     < 1 second
Medium dungeon (20x20):     1-3 seconds
Complex dungeon (30x30):    3-8 seconds
Large dungeon (50x50):      8-20 seconds
```

### Memory Usage:
```
A*:            ~5 MB (minimal state)
CBS Balanced:  ~15 MB (belief map + memory)
CBS Explorer:  ~25 MB (extensive exploration)
```

---

## Scientific Background

CBS implements cognitive constraints based on:
- **Miller (1956):** Working memory ~7±2 items
- **Kahneman (2011):** Dual-process decision-making
- **Simon (1955):** Bounded rationality theory

Result: More realistic agent behavior vs. omniscient A*

---

## Getting Help

### Check Logs
```bash
tail -f gui_runner.log  # Watch solver execution
```

### Debug Mode
```python
# Set in gui_runner.py line ~100
DEBUG_SYNC_SOLVER = True  # Run CBS synchronously
```

### Report Issues
Include:
- CBS persona used
- Dungeon size/complexity
- Metrics output
- Error messages (if any)

---

## Credits

**Implementation:** KLTN Team
**Scientific Foundation:**
- Miller's Law (1956)
- Newell & Simon (1972)
- Kahneman (2011)

**Integration:** AI-Engineer Mode
**Version:** 1.0.0
