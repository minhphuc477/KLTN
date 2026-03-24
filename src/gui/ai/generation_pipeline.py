"""Core AI-generation pipeline helpers extracted from gui_runner."""

import copy
import json
import os
import random
from pathlib import Path

from src.pipeline.block_contracts import BlockContractError, validate_checkpoint_metadata


def resolve_checkpoint_path():
    """Resolve checkpoint path, allowing an explicit environment override."""
    override = str(os.environ.get("KLTN_CHECKPOINT_PATH", "")).strip()
    if override:
        return Path(override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "checkpoints" / "final_model.pth"


def _load_checkpoint_metadata(checkpoint_path: Path):
    """Load optional checkpoint metadata sidecar JSON."""
    metadata_path = Path(f"{checkpoint_path}.meta.json")
    if not metadata_path.exists():
        return None, metadata_path
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f), metadata_path


def _validate_checkpoint_metadata_for_gui(
    *,
    checkpoint_path: Path,
    model_type: str,
    logger,
    strict_checkpoint_mode: bool,
):
    """Validate checkpoint sidecar metadata for GUI loading flow."""
    metadata, metadata_path = _load_checkpoint_metadata(checkpoint_path)
    if metadata is None:
        message = (
            f"Checkpoint metadata sidecar missing for {model_type} at {metadata_path}"
        )
        if strict_checkpoint_mode:
            raise RuntimeError(message)
        logger.warning(message)
        return

    try:
        validate_checkpoint_metadata(metadata=metadata, model_name=model_type)
    except BlockContractError as exc:
        if strict_checkpoint_mode:
            raise RuntimeError(str(exc)) from exc
        logger.warning("Checkpoint metadata validation warning: %s", exc)


def _compute_editor_layout(mission_graph):
    """Compute stable normalized 2D positions for mission-graph editor rendering."""
    try:
        import networkx as nx
    except (AttributeError, RuntimeError, ValueError, TypeError, ImportError):
        nx = None

    node_ids = sorted(list(mission_graph.nodes.keys()))
    if not node_ids:
        return {}

    layout = {}
    if nx is not None:
        try:
            g = nx.DiGraph()
            g.add_nodes_from(node_ids)
            for e in mission_graph.edges:
                g.add_edge(e.source, e.target)
            pos = nx.spring_layout(g, seed=17, k=max(0.2, 1.8 / max(1, len(node_ids))))
            xs = [float(pos[n][0]) for n in node_ids]
            ys = [float(pos[n][1]) for n in node_ids]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            dx = max(1e-6, max_x - min_x)
            dy = max(1e-6, max_y - min_y)
            for nid in node_ids:
                x = (float(pos[nid][0]) - min_x) / dx
                y = (float(pos[nid][1]) - min_y) / dy
                layout[int(nid)] = (0.08 + 0.84 * x, 0.12 + 0.76 * y)
            return layout
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

    # Fallback: topological-order line layout.
    count = len(node_ids)
    for idx, nid in enumerate(node_ids):
        x = 0.08 + 0.84 * (float(idx) / float(max(1, count - 1)))
        y = 0.5
        layout[int(nid)] = (x, y)
    return layout


def _mission_graph_constraints_from_gui(gui):
    """Collect staged mission-graph constraints from GUI state."""
    boss_node = getattr(gui, "ai_mission_graph_boss_node", None)
    locked_edges = list(getattr(gui, "ai_mission_graph_locked_edges", []) or [])
    cleaned_edges = []
    for pair in locked_edges:
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            continue
        try:
            src = int(pair[0])
            dst = int(pair[1])
        except (TypeError, ValueError):
            continue
        if src == dst:
            continue
        cleaned_edges.append((src, dst))
    return {
        "boss_node": int(boss_node) if isinstance(boss_node, (int, float)) else boss_node,
        "locked_edges": cleaned_edges,
    }


def apply_mission_graph_constraints(mission_graph, constraints, logger):
    """Apply staged boss-node and locked-edge constraints directly to mission graph."""
    from src.generation.grammar import EdgeType, NodeType

    if mission_graph is None or not isinstance(constraints, dict):
        return mission_graph, {"boss_applied": False, "locked_edges_applied": 0}

    boss_applied = False
    lock_applied = 0

    boss_node = constraints.get("boss_node")
    if boss_node is not None:
        try:
            boss_node = int(boss_node)
            if boss_node in mission_graph.nodes:
                mission_graph.nodes[boss_node].node_type = NodeType.BOSS
                boss_applied = True
        except (AttributeError, RuntimeError, ValueError, TypeError):
            boss_applied = False

    locked_pairs = constraints.get("locked_edges") or []
    for src, dst in locked_pairs:
        try:
            src_i = int(src)
            dst_i = int(dst)
        except (TypeError, ValueError):
            continue
        if src_i not in mission_graph.nodes or dst_i not in mission_graph.nodes or src_i == dst_i:
            continue

        existing = None
        for edge in mission_graph.edges:
            if int(edge.source) == src_i and int(edge.target) == dst_i:
                existing = edge
                break
        if existing is None:
            mission_graph.add_edge(src_i, dst_i, edge_type=EdgeType.LOCKED)
        else:
            existing.edge_type = EdgeType.LOCKED
        lock_applied += 1

    try:
        mission_graph.sanitize()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

    if boss_applied or lock_applied > 0:
        logger.info(
            "Mission-graph constraints applied: boss=%s, locked_edges=%d",
            boss_applied,
            lock_applied,
        )
    return mission_graph, {"boss_applied": boss_applied, "locked_edges_applied": int(lock_applied)}


def mission_graph_to_gnn_input(mission_graph):
    """Convert mission graph to GNN inputs and return metadata."""
    from src.generation.grammar import graph_to_gnn_input

    gnn_input = graph_to_gnn_input(mission_graph, current_node_idx=0)
    return {
        "mission_graph": mission_graph,
        "edge_index": gnn_input["edge_index"],
        "num_nodes": len(mission_graph.nodes),
        "num_edges": len(mission_graph.edges),
    }


def ensure_mission_graph_editor_draft(gui, random_module, logger=None):
    """Create a draft mission graph for editor interactions when absent."""
    if getattr(gui, "ai_mission_graph_draft", None) is not None:
        return

    configured_seed = getattr(gui, "ai_seed", None)
    try:
        configured_seed = int(configured_seed) if configured_seed is not None else None
    except (TypeError, ValueError):
        configured_seed = None

    mission_data = generate_mission_graph(random_module, seed=configured_seed)
    gui.ai_mission_graph_draft = copy.deepcopy(mission_data["mission_graph"])
    gui.ai_mission_graph_seed = int(mission_data["seed"])
    gui.ai_mission_graph_layout = _compute_editor_layout(gui.ai_mission_graph_draft)
    if logger is not None:
        logger.info(
            "Prepared mission-graph draft for editor: seed=%d, nodes=%d, edges=%d",
            gui.ai_mission_graph_seed,
            int(mission_data["num_nodes"]),
            int(mission_data["num_edges"]),
        )


def generate_mission_graph(random_module, *, seed=None, num_rooms=None):
    """Generate a medium-difficulty mission graph and return metadata."""
    from src.generation.grammar import MissionGrammar, Difficulty as GrammarDifficulty, graph_to_gnn_input

    if seed is None:
        chosen_seed = int(random_module.randint(0, 999999))
        room_count = int(random_module.randint(5, 10)) if num_rooms is None else int(num_rooms)
    else:
        deterministic_rng = random.Random(int(seed))
        chosen_seed = int(seed)
        room_count = int(deterministic_rng.randint(5, 10)) if num_rooms is None else int(num_rooms)

    room_count = max(5, min(10, int(room_count)))
    grammar = MissionGrammar(seed=chosen_seed)
    mission_graph = grammar.generate(
        difficulty=GrammarDifficulty.MEDIUM,
        num_rooms=room_count,
        max_keys=2,
    )
    out = mission_graph_to_gnn_input(mission_graph)
    out["seed"] = chosen_seed
    return out


def load_models_and_weights(
    checkpoint_path,
    device,
    torch_module,
    logger,
    strict_checkpoint_mode=False,
):
    """Construct model components and load checkpoint weights."""
    from src.core.latent_diffusion import create_latent_diffusion
    from src.core.vqvae import create_vqvae
    from src.core.condition_encoder import create_condition_encoder
    from src.core.logic_net import LogicNet as _LogicNet

    vqvae = create_vqvae(num_classes=44, latent_dim=64)
    diffusion = create_latent_diffusion(latent_dim=64, context_dim=256)
    cond_encoder = create_condition_encoder(latent_dim=64, output_dim=256, gnn_type="gcn")
    diffusion.guidance.logic_net = _LogicNet(latent_dim=64, num_classes=44)

    _validate_checkpoint_metadata_for_gui(
        checkpoint_path=Path(checkpoint_path),
        model_type="diffusion",
        logger=logger,
        strict_checkpoint_mode=bool(strict_checkpoint_mode),
    )

    ckpt = torch_module.load(checkpoint_path, map_location=device, weights_only=False)

    if "ema_diffusion_state_dict" in ckpt:
        diffusion.load_state_dict(ckpt["ema_diffusion_state_dict"])
    elif "diffusion_state_dict" in ckpt:
        diffusion.load_state_dict(ckpt["diffusion_state_dict"])

    if "vqvae_state_dict" in ckpt:
        vqvae.load_state_dict(ckpt["vqvae_state_dict"])
        logger.info("  Loaded VQ-VAE from main checkpoint")
    else:
        vqvae_path = Path("checkpoints/vqvae_pretrained.pth")
        if vqvae_path.exists():
            _validate_checkpoint_metadata_for_gui(
                checkpoint_path=vqvae_path,
                model_type="vqvae",
                logger=logger,
                strict_checkpoint_mode=bool(strict_checkpoint_mode),
            )
            vqvae_ckpt = torch_module.load(vqvae_path, map_location=device, weights_only=False)
            vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
            logger.info("  Loaded VQ-VAE from %s", vqvae_path)
        else:
            logger.warning("  No VQ-VAE weights found; decode quality may degrade")

    if "condition_encoder_state_dict" in ckpt:
        cond_encoder.load_state_dict(ckpt["condition_encoder_state_dict"])

    vqvae.eval()
    diffusion.eval()
    cond_encoder.eval()

    return vqvae, diffusion, cond_encoder


def build_conditioning_vector(mission_graph, edge_index, cond_encoder, torch_module, device):
    """Encode mission graph into a pooled conditioning tensor."""
    from src.generation.grammar import NodeType

    num_nodes = len(mission_graph.nodes)
    node_feat_5 = torch_module.zeros(num_nodes, 5, device=device)
    sorted_ids = sorted(mission_graph.nodes.keys())
    for i, nid in enumerate(sorted_ids):
        node_type = mission_graph.nodes[nid].node_type
        if node_type == NodeType.START:
            node_feat_5[i, 0] = 1.0
        elif node_type == NodeType.ENEMY:
            node_feat_5[i, 1] = 1.0
        elif node_type == NodeType.KEY:
            node_feat_5[i, 2] = 1.0
        elif node_type == NodeType.LOCK:
            node_feat_5[i, 3] = 1.0
        elif node_type == NodeType.GOAL:
            node_feat_5[i, 4] = 1.0

    with torch_module.no_grad():
        global_emb = cond_encoder.encode_global_only(node_feat_5, edge_index)
        return global_emb.mean(dim=0, keepdim=True)


def sample_tile_grid(diffusion, vqvae, conditioning, num_nodes, torch_module, np_module, logger):
    """Run DDIM sampling and VQ-VAE decode to obtain a tile grid."""
    scale = max(1, int(num_nodes ** 0.5))
    lat_h = 3 * scale
    lat_w = 4 * scale
    logger.info("  Latent shape: (1, 64, %d, %d) for %d-node graph", lat_h, lat_w, num_nodes)

    with torch_module.no_grad():
        latent = diffusion.ddim_sample(
            context=conditioning,
            shape=(1, 64, lat_h, lat_w),
            num_steps=50,
        )
        target_h = lat_h * 4
        target_w = lat_w * 4
        recon = vqvae.decode(latent, target_size=(target_h, target_w))
        tile_grid = recon.argmax(dim=1).squeeze(0).cpu().numpy().astype(np_module.int32)

    height, width = tile_grid.shape
    logger.info("  Raw generation: %dx%d, unique_tiles=%d", height, width, len(np_module.unique(tile_grid)))
    return tile_grid


def refine_and_fix_terminals(tile_grid, np_module, logger):
    """Apply symbolic refinement and guarantee START/TRIFORCE presence."""
    from src.core.definitions import SEMANTIC_PALETTE as _SP

    height, width = tile_grid.shape
    try:
        from src.core.symbolic_refiner import create_symbolic_refiner

        refiner = create_symbolic_refiner(max_repair_attempts=3)
        start_pos = (2, 2)
        goal_pos = (height - 3, width - 3)

        start_positions = np_module.argwhere(tile_grid == _SP["START"])
        goal_positions = np_module.argwhere(tile_grid == _SP["TRIFORCE"])
        if len(start_positions) > 0:
            start_pos = tuple(start_positions[0])
        if len(goal_positions) > 0:
            goal_pos = tuple(goal_positions[0])

        repaired_grid, success = refiner.repair_room(tile_grid, start_pos, goal_pos)
        if success:
            tile_grid = repaired_grid.astype(np_module.int32)
            logger.info("  Symbolic refinement: SUCCESS")
        else:
            logger.info("  Symbolic refinement: no repair needed or failed")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("  Symbolic refinement skipped: %s", exc)

    if not np_module.any(tile_grid == _SP["START"]):
        floor_positions = np_module.argwhere(tile_grid == _SP["FLOOR"])
        if len(floor_positions) > 0:
            start_point = floor_positions[0]
            tile_grid[start_point[0], start_point[1]] = _SP["START"]
            logger.info("  Placed START at (%d, %d)", start_point[0], start_point[1])

    if not np_module.any(tile_grid == _SP["TRIFORCE"]):
        floor_positions = np_module.argwhere(tile_grid == _SP["FLOOR"])
        if len(floor_positions) > 0:
            goal_point = floor_positions[-1]
            tile_grid[goal_point[0], goal_point[1]] = _SP["TRIFORCE"]
            logger.info("  Placed TRIFORCE at (%d, %d)", goal_point[0], goal_point[1])

    return tile_grid


def apply_generated_dungeon(gui, tile_grid, seed, num_nodes, num_edges, np_module):
    """Apply a generated grid to GUI state exactly like legacy in-method flow."""
    height, width = tile_grid.shape
    dungeon_name = f"AI #{seed} ({num_nodes}rm {height}x{width})"

    gui.maps.append(tile_grid)
    gui.map_names.append(dungeon_name)
    gui.current_map_idx = len(gui.maps) - 1
    gui._load_current_map()
    gui._center_view()

    if gui.effects:
        gui.effects.clear()
    gui.step_count = 0
    gui.auto_path = []
    gui.auto_mode = False

    gui._set_message(f"AI dungeon generated: {num_nodes} rooms, {height}x{width} tiles, seed={seed}")
    return {
        "height": height,
        "width": width,
        "unique_tiles": len(np_module.unique(tile_grid)),
        "name": dungeon_name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


def apply_mixed_initiative_constraints(tile_grid, constraints, np_module, logger):
    """
    Apply user-staged mixed-initiative constraints to a generated tile grid.

    Constraints are normalized anchors in [0, 1] from minimap clicks:
    - boss_norm: prefer placing ENEMY_BOSS near this anchor
    - lock_norm: prefer placing DOOR_LOCKED near this anchor
    - key_norm: prefer placing KEY_SMALL near this anchor
    """
    from src.core.definitions import SEMANTIC_PALETTE as _SP

    grid = np_module.asarray(tile_grid).copy()
    if grid.ndim != 2 or constraints is None:
        return grid, {"boss_applied": False, "lock_applied": False, "key_applied": False}

    h, w = int(grid.shape[0]), int(grid.shape[1])

    floor_id = int(_SP.get("FLOOR", 1))
    wall_id = int(_SP.get("WALL", 2))
    void_id = int(_SP.get("VOID", 0))
    door_open_id = int(_SP.get("DOOR_OPEN", floor_id))
    door_locked_id = int(_SP.get("DOOR_LOCKED", door_open_id))
    enemy_boss_id = int(_SP.get("ENEMY_BOSS", _SP.get("ENEMY", 7)))
    key_small_id = int(_SP.get("KEY_SMALL", _SP.get("KEY", 8)))
    start_id = int(_SP.get("START", 5))
    triforce_id = int(_SP.get("TRIFORCE", 6))

    def _anchor_to_cell(norm):
        if not isinstance(norm, (tuple, list)) or len(norm) < 2:
            return None
        try:
            nr = float(norm[0])
            nc = float(norm[1])
        except (TypeError, ValueError):
            return None
        nr = max(0.0, min(1.0, nr))
        nc = max(0.0, min(1.0, nc))
        rr = int(round(nr * max(0, h - 1)))
        cc = int(round(nc * max(0, w - 1)))
        return rr, cc

    def _find_nearest_walkable(r0, c0):
        best = None
        best_d = 10**9
        for r in range(h):
            for c in range(w):
                tid = int(grid[r, c])
                if tid in {wall_id, void_id, start_id, triforce_id}:
                    continue
                d = abs(r - r0) + abs(c - c0)
                if d < best_d:
                    best = (r, c)
                    best_d = d
        return best

    def _find_nearest_floor_or_walkable(r0, c0):
        best_floor = None
        best_floor_d = 10**9
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) != floor_id:
                    continue
                d = abs(r - r0) + abs(c - c0)
                if d < best_floor_d:
                    best_floor = (r, c)
                    best_floor_d = d
        if best_floor is not None:
            return best_floor
        return _find_nearest_walkable(r0, c0)

    def _find_nearest_door_or_walkable(r0, c0):
        best_door = None
        best_d = 10**9
        for r in range(h):
            for c in range(w):
                tid = int(grid[r, c])
                if tid in {door_open_id, door_locked_id}:
                    d = abs(r - r0) + abs(c - c0)
                    if d < best_d:
                        best_door = (r, c)
                        best_d = d
        if best_door is not None:
            return best_door
        return _find_nearest_walkable(r0, c0)

    applied = {"boss_applied": False, "lock_applied": False, "key_applied": False}

    boss_cell = _anchor_to_cell(constraints.get("boss_norm"))
    if boss_cell is not None:
        target = _find_nearest_walkable(boss_cell[0], boss_cell[1])
        if target is not None:
            grid[target[0], target[1]] = enemy_boss_id
            applied["boss_applied"] = True
            logger.info("Mixed-initiative: placed boss anchor near (%d, %d)", target[0], target[1])

    lock_cell = _anchor_to_cell(constraints.get("lock_norm"))
    if lock_cell is not None:
        target = _find_nearest_door_or_walkable(lock_cell[0], lock_cell[1])
        if target is not None:
            grid[target[0], target[1]] = door_locked_id
            applied["lock_applied"] = True
            logger.info("Mixed-initiative: placed locked door near (%d, %d)", target[0], target[1])

    key_cell = _anchor_to_cell(constraints.get("key_norm"))
    if key_cell is not None:
        target = _find_nearest_floor_or_walkable(key_cell[0], key_cell[1])
        if target is not None:
            grid[target[0], target[1]] = key_small_id
            applied["key_applied"] = True
            logger.info("Mixed-initiative: placed small key near (%d, %d)", target[0], target[1])

    return grid, applied
