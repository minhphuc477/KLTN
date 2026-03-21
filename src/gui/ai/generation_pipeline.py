"""Core AI-generation pipeline helpers extracted from gui_runner."""

from pathlib import Path


def resolve_checkpoint_path():
    """Resolve canonical checkpoint path under repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "checkpoints" / "final_model.pth"


def generate_mission_graph(random_module):
    """Generate a medium-difficulty mission graph and return metadata."""
    from src.generation.grammar import MissionGrammar, Difficulty as GrammarDifficulty, graph_to_gnn_input

    seed = random_module.randint(0, 999999)
    grammar = MissionGrammar(seed=seed)
    mission_graph = grammar.generate(
        difficulty=GrammarDifficulty.MEDIUM,
        num_rooms=random_module.randint(5, 10),
        max_keys=2,
    )
    gnn_input = graph_to_gnn_input(mission_graph, current_node_idx=0)
    return {
        "seed": seed,
        "mission_graph": mission_graph,
        "edge_index": gnn_input["edge_index"],
        "num_nodes": len(mission_graph.nodes),
        "num_edges": len(mission_graph.edges),
    }


def load_models_and_weights(checkpoint_path, device, torch_module, logger):
    """Construct model components and load checkpoint weights."""
    from src.core.latent_diffusion import create_latent_diffusion
    from src.core.vqvae import create_vqvae
    from src.core.condition_encoder import create_condition_encoder
    from src.core.logic_net import LogicNet as _LogicNet

    vqvae = create_vqvae(num_classes=44, latent_dim=64)
    diffusion = create_latent_diffusion(latent_dim=64, context_dim=256)
    cond_encoder = create_condition_encoder(latent_dim=64, output_dim=256)
    diffusion.guidance.logic_net = _LogicNet(latent_dim=64, num_classes=44)

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
    except Exception as exc:
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
