# Work items for feat/jps-productionization

- [ ] Rework `bench/grid_solvers.jps` to follow canonical forced-neighbor rules (parent-direction-based).
- [ ] Ensure diagonal support is correct and add `allow_corner_cutting` parameter.
- [ ] Add `trace` output containing: expanded nodes, jump points, and jump segments (start->jp).
- [ ] Add GUI overlay support: `show_jps_overlay` toggle and rendering primitives (segments in blue, jump points highlighted).
- [ ] Add tests: `test_jps_cornercut_and_consistency.py`, `test_jps_trace.py`, randomized JPS vs A* cost parity.
- [ ] Add micro-bench smoke to bench harness that runs JPS vs A* on corridor/open maps.
- [ ] Validate and add docs describing trace format and GUI options.
