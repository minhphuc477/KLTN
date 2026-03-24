"""Worker orchestration for AI dungeon generation."""

import os

from src.gui.ai.generation_pipeline import (
    apply_generated_dungeon,
    build_conditioning_vector,
    generate_mission_graph,
    load_models_and_weights,
    refine_and_fix_terminals,
    resolve_checkpoint_path,
    sample_tile_grid,
)


def run_ai_generation_worker(gui, logger):
    """Execute the full AI generation pipeline from a background worker thread."""
    try:
        import random as _random
        import numpy as np
        import torch

        checkpoint_path = resolve_checkpoint_path()
        if not checkpoint_path.exists():
            gui._set_message("No AI checkpoint found - train first!")
            logger.warning("Checkpoint not found: %s", checkpoint_path)
            return

        strict_checkpoint_mode = str(os.environ.get("KLTN_STRICT_CHECKPOINTS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        configured_seed = None
        env_seed = str(os.environ.get("KLTN_AI_SEED", "")).strip()
        if env_seed:
            try:
                configured_seed = int(env_seed)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid KLTN_AI_SEED=%r", env_seed)
                configured_seed = None

        gui_seed = getattr(gui, "ai_seed", None)
        if gui_seed is not None:
            try:
                configured_seed = int(gui_seed)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid gui.ai_seed=%r", gui_seed)

        gui._set_message("Generating mission graph...")
        logger.info("AI Generation: Loading checkpoint from %s", checkpoint_path)
        device = torch.device("cpu")

        mission_data = generate_mission_graph(_random, seed=configured_seed)
        mission_graph = mission_data["mission_graph"]
        edge_index = mission_data["edge_index"]
        seed = mission_data["seed"]
        num_nodes = mission_data["num_nodes"]
        num_edges = mission_data["num_edges"]
        logger.info(
            "  Mission graph: %d nodes, %d edges, seed=%d%s",
            num_nodes,
            num_edges,
            seed,
            " (deterministic)" if configured_seed is not None else "",
        )

        gui._set_message("Loading AI model...")
        vqvae, diffusion, cond_encoder = load_models_and_weights(
            checkpoint_path=checkpoint_path,
            device=device,
            torch_module=torch,
            logger=logger,
            strict_checkpoint_mode=strict_checkpoint_mode,
        )

        gui._set_message("Running diffusion sampling...")
        conditioning = build_conditioning_vector(
            mission_graph=mission_graph,
            edge_index=edge_index,
            cond_encoder=cond_encoder,
            torch_module=torch,
            device=device,
        )

        tile_grid = sample_tile_grid(
            diffusion=diffusion,
            vqvae=vqvae,
            conditioning=conditioning,
            num_nodes=num_nodes,
            torch_module=torch,
            np_module=np,
            logger=logger,
        )

        gui._set_message("Refining dungeon structure...")
        tile_grid = refine_and_fix_terminals(
            tile_grid=tile_grid,
            np_module=np,
            logger=logger,
        )

        applied = apply_generated_dungeon(
            gui=gui,
            tile_grid=tile_grid,
            seed=seed,
            num_nodes=num_nodes,
            num_edges=num_edges,
            np_module=np,
        )
        logger.info(
            "AI dungeon complete: seed=%d, graph=%dN/%dE, grid=%dx%d, unique_tiles=%d",
            seed,
            applied["num_nodes"],
            applied["num_edges"],
            applied["height"],
            applied["width"],
            applied["unique_tiles"],
        )
    except (AttributeError, RuntimeError, ValueError, TypeError, ImportError, OSError) as exc:
        logger.exception("AI generation failed: %s", exc)
        gui._set_message(f"AI generation failed: {exc}")

