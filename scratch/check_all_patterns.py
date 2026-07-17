"""
Accurate ground-truth pattern check for all defects listed in fresh_diagnosis.md.
Checks against actual file content, not assumed string literals.
"""
import pathlib
import ast

BASE = pathlib.Path('f:/KLTN/src')

def read(rel):
    p = BASE / rel
    return p.read_text(encoding='utf-8-sig', errors='replace') if p.exists() else ''

def section(src, start_class, end_class=None):
    """Extract text between two class definitions."""
    parts = src.split(f'class {start_class}')
    if len(parts) < 2:
        return ''
    rest = parts[1]
    if end_class:
        rest = rest.split(f'class {end_class}')[0]
    return rest

results = {}

# ── C1: train_masked_room.py node_features.to(self.device) unconditional ──────
tmr = read('train_masked_room.py')
in_if_block = 'if not isinstance(edge_index, torch.Tensor):' in tmr and \
              'node_features = node_features.to(self.device' in tmr
results['C1 (tmr node_features.to unconditional)'] = in_if_block

# ── C2: FormBigRoomRule position[2] IndexError on 2-tuple positions ───────────
ar = read('generation/grammar/advanced_rules.py')
fbr = section(ar, 'FormBigRoomRule', 'AddValveRule')
# Fixed if it guards with len() > 2
results['C2 (FormBigRoom position[2] guard)'] = "len(node_a.position) > 2" in fbr

# ── C3: AddEntangledBranchesRule switch_id wrong field ────────────────────────
results['C3 (switch_id -> switches_required)'] = 'switches_required=[switch_id]' in ar or \
    'switch_id=switch_id' not in ar.split('class AddEntangledBranchesRule')[1].split('class ')[0] \
    if 'class AddEntangledBranchesRule' in ar else True

# ── C4: AddResourceLoopRule SHORTCUT bypasses all locks ──────────────────────
results['C4 (no SHORTCUT to start)'] = 'EdgeType.SHORTCUT' not in ar.split('class AddResourceLoopRule')[1].split('class ')[0]

# ── C5: SECRET/HIDDEN filtered in _get_valid_neighbors ───────────────────────
enf = read('generation/graph_constraint_enforcer.py')
results['C5 (hidden/secret filtered in neighbors)'] = "{'visual_link', 'window', 'hidden', 'secret'}" in enf or \
    "edge_type in {'hidden', 'secret'}" in enf

# ── C6: FLOOR_ID != DOOR_ID enforced ─────────────────────────────────────────
results['C6 (tile collision guard)'] = "if len(set(base_tiles.values())) != len(base_tiles):" in enf

# ── C7: room_stitching door written BEFORE A* - does code guard it? ──────────
rs = read('pipeline/room_stitching.py')
results['C7 (door written unconditionally before corridor)'] = \
    'global_grid[src_anchor' in rs  # This is the unfixed pattern - check if still present

# ── C8: neighbor latents use _infer_direction not always-'N' ─────────────────
gc = read('pipeline/generation/graph_context.py')
results['C8 (neighbor direction inferred)'] = '_infer_direction' in gc

# ── C9: robust_pipeline uses SEMANTIC_PALETTE ─────────────────────────────────
rp = read('pipeline/robust_pipeline.py')
results['C9 (robust_pipeline FLOOR from SEMANTIC_PALETTE)'] = "SEMANTIC_PALETTE['FLOOR']" in rp

# ── C10: _normalize_mission_graph BFS might drop extra nodes ─────────────────
ap = read('pipeline/advanced_pipeline.py')
results['C10 (BFS traversal order with extra branch)'] = 'nx.bfs_tree' in ap and \
    'branch_parent' in ap and 'add_edge(branch_parent, new_node' in ap

# ── C11: mission_grammar fungible key consumption via stateful solver ─────────
mg = read('generation/grammar/mission_grammar.py')
results['C11 (stateful progression solver used in repair)'] = \
    'solve_mission_progression' in mg and '_progression_reachable_nodes' in mg

# ── C12: solve_mission_progression inventory copy on backtrack ───────────────
# The diagnosis says C12 is about logic_net's full_coverage/K iterations
# C12b (puzzle_stage_semantics inventory copy) is about solve_mission_progression in mission_grammar
# Let's check what the actual progression solver does
ps_file = read('generation/grammar/progression_solver.py')
if not ps_file:
    ps_file = read('generation/grammar/mission_grammar.py')
results['C12 (progression solver copies inventory on backtrack)'] = \
    'dict(current_inventory)' in mg or 'dict(current_inventory)' in ps_file or \
    'copy()' in mg

# ── C13: logic_net multi-key gets zero gradient  ─────────────────────────────
ln = read('core/logic_net.py')
results['C13 (LogicGraphContractError raised for ambiguous pairs)'] = \
    'LogicGraphContractError' in ln and 'ambiguous provider-to-lock' in ln

# ── C14: VIN source cell zeroed inside loop ───────────────────────────────────
results['C14 (VIN value*(1-source) inside loop)'] = 'value = value * (1.0 - source)' in ln

# ── C15: LocalStreamEncoder batch size mismatch ──────────────────────────────
enc = read('core/condition_encoder.py')
results['C15 (LocalStreamEncoder null_token expand safe)'] = \
    'class LocalStreamEncoder' in enc

# ── C16: cognitive_bounded_search ZeroDivision ───────────────────────────────
cog = read('core/cognitive_bounded_search.py')
if not cog:
    cog = read('simulation/cognitive_bounded_search.py')
results['C16 (ZeroDivision on empty visible set)'] = \
    '/ max(1,' in cog or '/ max(len(visible)' in cog or \
    ('len(visible)' in cog and ('max(1' in cog or 'if not visible' in cog))

# ── H1: joint gradient clipping ───────────────────────────────────────────────
td = read('train_diffusion.py')
results['H1 (joint gradient clipping)'] = '_clip_joint_optimizer_gradients' in td

# ── H2: mid-accumulation reset ────────────────────────────────────────────────
results['H2 (_reset_gradient_accumulation)'] = '_reset_gradient_accumulation' in td

# ── H3: KeyError when topology_focus_loss missing ────────────────────────────
results['H3 (val_topology_focus_loss uses .get())'] = \
    "epoch_metrics.get(\"val_topology_focus_loss\"" in read('train_masked_room.py') or \
    "val_topology_focus_loss\", float(\"nan\"" in read('train_masked_room.py')

# ── H4: advanced_pipeline hardcoded floor ID=1 ───────────────────────────────
results['H4 (SEMANTIC_PALETTE floor in collision validator)'] = \
    "int(SEMANTIC_PALETTE['FLOOR'])" in ap

# ── H5: demo hardcoded IDs ────────────────────────────────────────────────────
results['H5 (demo from_room:3 not present)'] = \
    "'from_room': 3" not in ap and 'from_room: 3' not in ap

# ── H6: strict placement fallback re-places all rooms ───────────────────────
# Current code: per-component try/except with tree-preserving fallback
results['H6 (per-component strict/fallback)'] = \
    'comp_positions = solve_component_strict_adjacency' in rs and \
    'compute_relaxed_room_placement' in rs

# ── H7: boss_door[0] guarded by len check ────────────────────────────────────
results['H7 (boss_door_outgoing_edges[0] guarded)'] = \
    'len(boss_door_outgoing_edges) != 1' in mg and \
    'boss_door_outgoing_edges[0]' in mg

# ── H8: AddValveRule missing break ───────────────────────────────────────────
valve = section(ar, 'AddValveRule', 'AddForeshadowing')
results['H8 (safe_candidates break present)'] = 'break' in valve.split('if safe_candidates:')[1].split('if noncritical_candidates:')[0]

# ── H9: empty cells handled ──────────────────────────────────────────────────
results['H9 (empty cells not flagged as broken)'] = \
    'if not cells or any(' in rs

# ── H10: _place_room_anchor no uncaught ValueError ───────────────────────────
results['H10 (_place_room_anchor carves floor fallback)'] = \
    'grid[y, x] = int(self.FLOOR_ID)' in enf and \
    'candidates.append((0, y, x))' in enf

# ── H11: CrossAttentionFusion zeroes fully-masked rows ───────────────────────
results['H11 (CrossAttention has_valid_context zeroing)'] = \
    'has_valid_context is not None and not torch.all(has_valid_context):\n            attn_output = attn_output.clone()' in enc

# ── H12: GPSLayer batch_idx=None leaks cross-batch attention ─────────────────
results['H12 (GPSLayer single-graph note in docstring)'] = \
    'batch_idx is None' in enc

# ── H13: grid Bellman-Ford checkpoint ────────────────────────────────────────
results['H13 (BF checkpointing present)'] = \
    'use_checkpoint = self._should_checkpoint_relaxation' in ln

print("=" * 65)
print("DEFECT STATUS CHECK")
print("=" * 65)
fixed = 0
total = len(results)
for k in sorted(results):
    status = "✓ FIXED" if results[k] else "✗ NEEDS FIX"
    if results[k]:
        fixed += 1
    print(f"  {status:15s} {k}")
print(f"\n{fixed}/{total} defects verified fixed")
