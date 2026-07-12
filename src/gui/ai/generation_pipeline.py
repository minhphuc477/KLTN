"""Core AI-generation pipeline helpers extracted from gui_runner."""

import copy
from datetime import datetime
import json
import os
import random
from pathlib import Path

from src.config_system import load_resolved_config_for_artifact
from src.pipeline.block_contracts import BlockContractError, validate_checkpoint_metadata
from src.pipeline.dungeon_pipeline import pipeline_kwargs_from_resolved_config
from src.pipeline.spatial_utils import normalize_node_id, stable_node_sort_key
from src.utils.checkpoint import safe_torch_load


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return max(int(min_value), int(raw))
    except (TypeError, ValueError):
        return int(default)


def discover_best_output_checkpoint(repo_root: Path | None = None):
    """Return the preferred trained GUI-generation checkpoint from outputs/."""
    if _truthy_env("KLTN_DISABLE_OUTPUT_CHECKPOINT_DISCOVERY"):
        return None

    root = Path(repo_root) if repo_root is not None else _repo_root()
    # Ordered by demo suitability: the first entry has a validated solved
    # full-level artifact in outputs and matching checkpoint metadata.
    candidate_relpaths = [
        "outputs/zelda_hmolqd_downstream_stageconditioned_semantics_v3_puzzlefix/checkpoints/diffusion/best_model.pth",
        "outputs/dungeon9_holdout_full_retrain_20260515/checkpoints/diffusion/best_model.pth",
        "outputs/zelda_hmolqd_downstream_global_logicnet_v4_global_logicnet/checkpoints/diffusion/best_model.pth",
        "outputs/zelda_hmolqd_downstream_stageconditioned_semantics_v2/checkpoints/diffusion/best_model.pth",
    ]
    for relpath in candidate_relpaths:
        candidate = (root / relpath).resolve()
        if candidate.exists():
            return candidate
    return None


def resolve_checkpoint_path(explicit_path=None):
    """Resolve checkpoint path, allowing an explicit argument or environment override."""
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    override = str(os.environ.get("KLTN_CHECKPOINT_PATH", "")).strip()
    if override:
        return Path(override).expanduser().resolve()

    discovered = discover_best_output_checkpoint()
    if discovered is not None:
        return discovered

    repo_root = _repo_root()
    return repo_root / "checkpoints" / "final_model.pth"


def _candidate_checkpoint_files(directory: Path, names: tuple[str, ...]):
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_fast_sampler_checkpoint_for_generation(
    checkpoint_path: Path,
    resolved_config: dict | None = None,
):
    """Find a trained fast-sampler adapter that belongs to a diffusion checkpoint."""
    for env_name in ("KLTN_GUI_FAST_SAMPLER_CHECKPOINT", "KLTN_FAST_SAMPLER_CHECKPOINT"):
        override = str(os.environ.get(env_name, "")).strip()
        if override:
            candidate = Path(override).expanduser()
            if not candidate.is_absolute():
                candidate = _repo_root() / candidate
            if candidate.exists():
                return candidate.resolve()

    names = (
        "fast_sampler_best.pth",
        "best_model.pth",
        "latest_resume.pth",
        "fast_sampler_final.pth",
    )
    search_dirs: list[Path] = []
    if isinstance(resolved_config, dict):
        checkpoint_dir = (
            resolved_config.get("fast_sampler", {}).get("checkpoint_dir")
            if isinstance(resolved_config.get("fast_sampler"), dict)
            else None
        )
        if checkpoint_dir:
            configured = Path(str(checkpoint_dir)).expanduser()
            if not configured.is_absolute():
                configured = _repo_root() / configured
            search_dirs.append(configured)

    search_dirs.extend(
        [
            checkpoint_path.parent.parent / "fast_sampler",
            checkpoint_path.parent / "fast_sampler",
            checkpoint_path.parent.parent.parent / "fast_sampler",
        ]
    )

    seen: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        if not directory.exists():
            continue
        candidate = _candidate_checkpoint_files(directory, names)
        if candidate is not None:
            return candidate
    return None


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

    node_ids = sorted(list(mission_graph.nodes.keys()), key=stable_node_sort_key)
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
                layout[nid] = (0.08 + 0.84 * x, 0.12 + 0.76 * y)
            return layout
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

    # Fallback: topological-order line layout.
    count = len(node_ids)
    for idx, nid in enumerate(node_ids):
        x = 0.08 + 0.84 * (float(idx) / float(max(1, count - 1)))
        y = 0.5
        layout[nid] = (x, y)
    return layout


def _normalize_node_ref(value):
    """Best-effort normalization for GUI-staged node references."""
    return normalize_node_id(value)


def _mission_graph_constraints_from_gui(gui):
    """Collect staged mission-graph constraints from GUI state."""
    boss_node = _normalize_node_ref(getattr(gui, "ai_mission_graph_boss_node", None))
    locked_edges = list(getattr(gui, "ai_mission_graph_locked_edges", []) or [])
    cleaned_edges = []
    for pair in locked_edges:
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            continue
        src = _normalize_node_ref(pair[0])
        dst = _normalize_node_ref(pair[1])
        if src is None or dst is None:
            continue
        if src == dst:
            continue
        cleaned_edges.append((src, dst))
    return {
        "boss_node": boss_node,
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
    boss_node = _normalize_node_ref(boss_node)
    if boss_node is not None and boss_node in mission_graph.nodes:
        mission_graph.nodes[boss_node].node_type = NodeType.BOSS
        boss_applied = True

    locked_pairs = constraints.get("locked_edges") or []
    for src, dst in locked_pairs:
        src_ref = _normalize_node_ref(src)
        dst_ref = _normalize_node_ref(dst)
        if src_ref is None or dst_ref is None:
            continue
        if src_ref not in mission_graph.nodes or dst_ref not in mission_graph.nodes or src_ref == dst_ref:
            continue

        existing = None
        for edge in mission_graph.edges:
            if edge.source == src_ref and edge.target == dst_ref:
                existing = edge
                break
        if existing is None:
            mission_graph.add_edge(src_ref, dst_ref, edge_type=EdgeType.LOCKED)
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

    config = getattr(gui, "ai_generation_config", None)
    config = dict(config) if isinstance(config, dict) else {}
    mission_data = generate_mission_graph(
        random_module,
        seed=configured_seed,
        num_rooms=config.get("num_rooms", getattr(gui, "ai_num_rooms", None)),
        difficulty=config.get("difficulty", getattr(gui, "ai_difficulty", "MEDIUM")),
        max_keys=config.get("max_keys", getattr(gui, "ai_max_keys", 2)),
    )
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

def generate_mission_graph(random_module, *, seed=None, num_rooms=None, difficulty="MEDIUM", max_keys=2):
    """Generate a medium-difficulty mission graph and return metadata."""
    from src.generation.grammar import MissionGrammar, Difficulty as GrammarDifficulty

    if seed is None:
        chosen_seed = int(random_module.randint(0, 999999))
        room_count = int(random_module.randint(5, 10)) if num_rooms is None else int(num_rooms)
    else:
        deterministic_rng = random.Random(int(seed))
        chosen_seed = int(seed)
        room_count = int(deterministic_rng.randint(5, 10)) if num_rooms is None else int(num_rooms)

    room_count = max(5, min(24, int(room_count)))
    try:
        max_keys = max(0, min(8, int(max_keys)))
    except (TypeError, ValueError):
        max_keys = 2

    if isinstance(difficulty, GrammarDifficulty):
        grammar_difficulty = difficulty
    else:
        difficulty_text = str(difficulty or "MEDIUM").strip().upper()
        difficulty_lookup = {
            "1": GrammarDifficulty.EASY,
            "EASY": GrammarDifficulty.EASY,
            "2": GrammarDifficulty.MEDIUM,
            "MEDIUM": GrammarDifficulty.MEDIUM,
            "3": GrammarDifficulty.HARD,
            "HARD": GrammarDifficulty.HARD,
            "4": GrammarDifficulty.EXPERT,
            "EXPERT": GrammarDifficulty.EXPERT,
        }
        grammar_difficulty = difficulty_lookup.get(difficulty_text, GrammarDifficulty.MEDIUM)

    def _generate_once(candidate_seed: int, internal_room_count: int):
        grammar = MissionGrammar(seed=int(candidate_seed))
        mission_graph = grammar.generate(
            difficulty=grammar_difficulty,
            num_rooms=int(internal_room_count),
            max_keys=max_keys,
        )
        out = mission_graph_to_gnn_input(mission_graph)
        out["seed"] = int(candidate_seed)
        out["base_seed"] = chosen_seed
        out["requested_num_rooms"] = room_count
        out["internal_num_rooms"] = int(internal_room_count)
        out["difficulty"] = grammar_difficulty.name
        out["max_keys"] = max_keys
        return out

    if num_rooms is None:
        return _generate_once(chosen_seed, room_count)

    internal_counts = sorted(
        range(max(5, room_count - 6), room_count + 1),
        key=lambda value: (abs(int(value) - room_count), -int(value)),
    )
    seed_candidates = [chosen_seed + offset for offset in range(8)]

    best = None
    best_score = None
    for candidate_seed in seed_candidates:
        for internal_room_count in internal_counts:
            candidate = _generate_once(candidate_seed, internal_room_count)
            actual = int(candidate["num_nodes"])
            score = (
                abs(actual - room_count),
                1 if actual > room_count else 0,
                abs(int(internal_room_count) - room_count),
                abs(int(candidate_seed) - chosen_seed),
            )
            if best is None or score < best_score:
                best = candidate
                best_score = score
            if actual == room_count:
                return candidate

    return best


def generate_comprehensive_demo_graph(seed=None):
    """Generate a hardcoded 3x3 comprehensive demo graph that guarantees all mechanics."""
    from src.generation.grammar import MissionGraph, MissionNode, NodeType, EdgeType

    graph = MissionGraph()

    # R7: Start
    graph.add_node(MissionNode(id=7, node_type=NodeType.START, position=(2, 1, 0), difficulty=0.1))

    # R4: Hub with Small Key
    graph.add_node(MissionNode(id=4, node_type=NodeType.KEY, position=(1, 1, 0), difficulty=0.3))
    graph.add_edge(7, 4, EdgeType.PATH)

    # R3: Enemy Gauntlet
    graph.add_node(MissionNode(id=3, node_type=NodeType.ENEMY, position=(1, 0, 0), difficulty=0.8, enemy_count_hint=4))
    graph.add_edge(4, 3, EdgeType.LOCKED) # Small key required

    # R6: Bomb Item
    graph.add_node(MissionNode(id=6, node_type=NodeType.ITEM, position=(2, 0, 0), difficulty=0.4, item_type="BOMB"))
    graph.add_edge(3, 6, EdgeType.PATH)

    # R5: Water Hazard + Boss Key
    graph.add_node(MissionNode(id=5, node_type=NodeType.BIG_KEY, position=(1, 2, 0), difficulty=0.6))
    graph.add_edge(4, 5, EdgeType.ITEM_GATE, item_required="BOMB")

    # R2: Puzzle
    graph.add_node(MissionNode(id=2, node_type=NodeType.PUZZLE, position=(0, 2, 0), difficulty=0.5))
    graph.add_edge(5, 2, EdgeType.ONE_WAY) # Soft drop

    # R1: Pre-boss / Switch Door
    graph.add_node(MissionNode(id=1, node_type=NodeType.BOSS_DOOR, position=(0, 1, 0), difficulty=0.7))
    graph.add_edge(2, 1, EdgeType.ON_OFF_GATE) # Requires puzzle solved

    # R0: Boss and Goal
    graph.add_node(MissionNode(id=0, node_type=NodeType.GOAL, position=(0, 0, 0), difficulty=1.0, enemy_count_hint=2))
    graph.add_edge(1, 0, EdgeType.BOSS_LOCKED) # Requires Boss Key

    graph.sanitize()
    out = mission_graph_to_gnn_input(graph)
    out["seed"] = 42 if seed is None else int(seed)
    return out


def _resolve_vqvae_checkpoint_for_generation(checkpoint_path: Path):
    """Prefer an embedded VQ-VAE, otherwise fall back to sibling pretrain weights."""
    repo_root = _repo_root()
    metadata, _metadata_path = _load_checkpoint_metadata(checkpoint_path)
    metadata_vqvae = None
    if isinstance(metadata, dict):
        extra = metadata.get("extra", {})
        if isinstance(extra, dict):
            metadata_vqvae = extra.get("vqvae_checkpoint")

    metadata_candidates = []
    if metadata_vqvae:
        candidate = Path(str(metadata_vqvae)).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        metadata_candidates.append(candidate.resolve())

    candidate_paths = [
        *metadata_candidates,
        checkpoint_path.parent / "vqvae_pretrained.pth",
        checkpoint_path.parent.parent / "vqvae" / "vqvae_pretrained.pth",
        checkpoint_path.parent.parent / "vqvae" / "latest_resume.pth",
    ]
    try:
        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    except (AttributeError, RuntimeError, ValueError, TypeError, OSError):
        checkpoint = None

    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("vqvae_state_dict"), dict):
            return checkpoint_path
        is_standalone_vqvae = isinstance(checkpoint.get("model_state_dict"), dict) and not any(
            isinstance(checkpoint.get(key), dict)
            for key in ("diffusion_state_dict", "condition_encoder_state_dict", "logic_net_state_dict")
        )
        if is_standalone_vqvae:
            return checkpoint_path

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return checkpoint_path


def load_canonical_generation_pipeline(
    checkpoint_path,
    device,
    logger,
    strict_checkpoint_mode=False,
    gui_fast_mode=False,
):
    """Construct the canonical room-wise neural-symbolic generation pipeline."""
    from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline

    checkpoint_path = Path(checkpoint_path)
    vqvae_checkpoint = _resolve_vqvae_checkpoint_for_generation(checkpoint_path)
    resolved_config = load_resolved_config_for_artifact(checkpoint_path)
    pipeline_kwargs = (
        pipeline_kwargs_from_resolved_config(resolved_config)
        if isinstance(resolved_config, dict)
        else {}
    )
    if bool(gui_fast_mode) and _env_bool("KLTN_GUI_FAST_GENERATION", True):
        fast_sampler_checkpoint = resolve_fast_sampler_checkpoint_for_generation(
            checkpoint_path,
            resolved_config if isinstance(resolved_config, dict) else None,
        )
        if fast_sampler_checkpoint is not None:
            pipeline_kwargs["fast_sampling_checkpoint"] = str(fast_sampler_checkpoint)
            pipeline_kwargs["default_use_fast_sampling"] = _env_bool("KLTN_GUI_USE_FAST_SAMPLING", True)
            logger.info("GUI fast generation: using fast-sampler checkpoint %s", fast_sampler_checkpoint)
        elif _env_bool("KLTN_GUI_USE_FAST_SAMPLING", False):
            pipeline_kwargs["default_use_fast_sampling"] = True

        if "KLTN_GUI_DIFFUSION_STEPS" in os.environ:
            pipeline_kwargs["default_num_diffusion_steps"] = _env_int(
                "KLTN_GUI_DIFFUSION_STEPS",
                int(pipeline_kwargs.get("default_num_diffusion_steps", 50)),
            )
        elif fast_sampler_checkpoint is None:
            pipeline_kwargs["default_num_diffusion_steps"] = min(
                int(pipeline_kwargs.get("default_num_diffusion_steps", 50)),
                16,
            )

    pipeline = NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(vqvae_checkpoint),
        diffusion_checkpoint=str(checkpoint_path),
        logic_net_checkpoint=str(checkpoint_path),
        condition_encoder_checkpoint=str(checkpoint_path),
        device=str(device),
        enable_logging=False,
        strict_checkpoint_mode=bool(strict_checkpoint_mode),
        **pipeline_kwargs,
    )
    logger.info(
        "Loaded canonical generation pipeline from %s (vqvae=%s, resolved_config=%s)",
        checkpoint_path,
        vqvae_checkpoint,
        "yes" if resolved_config is not None else "no",
    )
    return pipeline


def generate_dungeon_with_pipeline(
    pipeline,
    mission_graph,
    *,
    seed,
    logger,
):
    """Generate a stitched dungeon with the canonical per-room pipeline."""
    from src.generation.evolutionary_director import mission_graph_to_networkx

    networkx_graph = mission_graph_to_networkx(mission_graph, directed=True)
    diffusion = getattr(pipeline, "diffusion", None)
    guidance = getattr(diffusion, "guidance", None)
    result = pipeline.generate_dungeon(
        mission_graph=networkx_graph,
        guidance_scale=float(
            getattr(
                pipeline,
                "default_guidance_scale",
                getattr(diffusion, "cfg_scale", 3.0),
            )
        ),
        logic_guidance_scale=float(
            getattr(
                pipeline,
                "default_logic_guidance_scale",
                getattr(guidance, "guidance_scale", 1.0),
            )
        ),
        num_diffusion_steps=int(getattr(pipeline, "default_num_diffusion_steps", 50)),
        use_fast_sampling=bool(getattr(pipeline, "default_use_fast_sampling", False)),
        latent_sampler=str(getattr(pipeline, "default_latent_sampler", "diffusion")),
        categorical_codebook_size=getattr(pipeline, "default_categorical_codebook_size", None),
        use_topological_positional_encoding=bool(
            getattr(pipeline, "default_use_topological_positional_encoding", True)
        ),
        apply_repair=bool(getattr(pipeline, "default_apply_repair", True)),
        enable_map_elites=bool(getattr(pipeline, "default_enable_map_elites", False)),
        seed=(None if seed is None else int(seed)),
    )
    logger.info(
        "Canonical pipeline generated dungeon: rooms=%d shape=%s repair_rate=%.3f",
        int(result.metrics.get("num_rooms", 0)),
        tuple(result.dungeon_grid.shape),
        float(result.metrics.get("repair_rate", 0.0)),
    )
    return result


def load_models_and_weights(
    checkpoint_path,
    device,
    torch_module,
    logger,
    strict_checkpoint_mode=False,
):
    """Compatibility wrapper that returns components from the canonical pipeline."""
    pipeline = load_canonical_generation_pipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
        strict_checkpoint_mode=bool(strict_checkpoint_mode),
    )
    return pipeline.vqvae, pipeline.diffusion, pipeline.condition_encoder


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
    """Run latent sampling and VQ-VAE decode to obtain a tile grid."""
    scale = max(1, int(num_nodes ** 0.5))
    lat_h = 3 * scale
    lat_w = 4 * scale
    latent_dim = int(getattr(diffusion, "latent_dim", 64))
    logger.info(
        "  Latent shape: (1, %d, %d, %d) for %d-node graph",
        latent_dim,
        lat_h,
        lat_w,
        num_nodes,
    )

    latent_shape = (1, latent_dim, lat_h, lat_w)
    training_objective = str(getattr(diffusion, "training_objective", "diffusion")).strip().lower()

    with torch_module.no_grad():
        if training_objective == "flow_matching":
            if not hasattr(diffusion, "flow_ode_sample"):
                raise ValueError("flow_matching diffusion checkpoints require flow_ode_sample() for GUI sampling")
            logger.info("  Sampler: flow_ode (flow_matching)")
            latent = diffusion.flow_ode_sample(
                context=conditioning,
                shape=latent_shape,
                num_steps=50,
            )
        elif training_objective == "diffusion":
            logger.info("  Sampler: ddim")
            latent = diffusion.ddim_sample(
                context=conditioning,
                shape=latent_shape,
                num_steps=50,
            )
        else:
            raise ValueError(f"Unsupported diffusion training_objective={training_objective!r}")
        target_h = lat_h * 4
        target_w = lat_w * 4
        decode_latent = (
            diffusion.unscale_first_stage_latent(latent)
            if hasattr(diffusion, "unscale_first_stage_latent")
            else latent
        )
        recon = vqvae.decode(decode_latent, target_size=(target_h, target_w))
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


def build_generated_dungeon_payload(tile_grid, seed, num_nodes, num_edges, np_module):
    """Build metadata for a generated dungeon without mutating GUI state."""
    height, width = tile_grid.shape
    dungeon_name = f"AI #{seed} ({num_nodes}rm {height}x{width})"
    return {
        "height": height,
        "width": width,
        "unique_tiles": len(np_module.unique(tile_grid)),
        "name": dungeon_name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "message": f"AI dungeon generated: {num_nodes} rooms, {height}x{width} tiles, seed={seed}",
    }


def save_generated_dungeon_txt(
    *,
    tile_grid,
    seed,
    num_nodes,
    num_edges,
    checkpoint_path=None,
    export_dir=None,
    np_module=None,
):
    """Persist a generated dungeon as TXT and PNG files and return the written paths."""
    np_module = np_module or __import__("numpy")
    grid = np_module.asarray(tile_grid, dtype=np_module.int32)
    height, width = grid.shape
    root = Path(export_dir) if export_dir else Path(os.environ.get("KLTN_AI_EXPORT_DIR", "") or "")
    if str(root).strip() in {"", "."} and not export_dir and not os.environ.get("KLTN_AI_EXPORT_DIR"):
        root = _repo_root() / "exports" / "generated_levels"
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_seed = "none" if seed is None else str(int(seed))
    txt_path = root / f"gui_ai_generated_seed{safe_seed}_{height}x{width}_{stamp}.txt"
    latest_path = root / "gui_ai_generated_latest.txt"
    np_module.savetxt(str(txt_path), grid, fmt="%d")
    np_module.savetxt(str(latest_path), grid, fmt="%d")

    png_path = txt_path.with_suffix(".png")
    latest_png_path = latest_path.with_suffix(".png")
    from src.gui.rendering.level_image_export import save_level_grid_png

    save_level_grid_png(grid, png_path, np_module=np_module)
    save_level_grid_png(grid, latest_png_path, np_module=np_module)

    metadata = {
        "seed": None if seed is None else int(seed),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "height": int(height),
        "width": int(width),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "txt_path": str(txt_path),
        "latest_txt_path": str(latest_path),
        "png_path": str(png_path),
        "latest_png_path": str(latest_png_path),
    }
    metadata_path = txt_path.with_suffix(".metadata.json")
    latest_metadata_path = latest_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "txt_path": txt_path,
        "latest_txt_path": latest_path,
        "png_path": png_path,
        "latest_png_path": latest_png_path,
        "metadata_path": metadata_path,
        "latest_metadata_path": latest_metadata_path,
    }


def apply_generated_dungeon(gui, tile_grid, seed, num_nodes, num_edges, np_module):
    """Apply a generated grid to GUI state exactly like legacy in-method flow."""
    payload = build_generated_dungeon_payload(tile_grid, seed, num_nodes, num_edges, np_module)

    gui.maps.append(tile_grid)
    gui.map_names.append(payload["name"])
    gui.current_map_idx = len(gui.maps) - 1
    gui._load_current_map()
    gui._center_view()

    if gui.effects:
        gui.effects.clear()
    gui.step_count = 0
    gui.auto_path = []
    gui.auto_mode = False

    gui._set_message(payload["message"])
    return payload


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
