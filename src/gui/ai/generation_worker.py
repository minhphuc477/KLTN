"""Worker orchestration for AI dungeon generation."""

import copy
import os

from src.gui.ai.generation_pipeline import (
    apply_mission_graph_constraints,
    apply_mixed_initiative_constraints,
    build_generated_dungeon_payload,
    ensure_mission_graph_editor_draft,
    generate_dungeon_with_pipeline,
    generate_mission_graph,
    load_canonical_generation_pipeline,
    mission_graph_to_gnn_input,
    resolve_checkpoint_path,
)


def run_ai_generation_worker(gui, logger):
    """Execute the full AI generation pipeline from a background worker thread."""
    def _finish_failure(message):
        gui.ai_gen_result = {"success": False, "error": str(message), "message": str(message)}
        gui.ai_gen_done = True

    try:
        import random as _random
        import numpy as np
        import torch

        gui_checkpoint_path = getattr(gui, "ai_checkpoint_path", None)
        checkpoint_path = (
            resolve_checkpoint_path(gui_checkpoint_path)
            if gui_checkpoint_path
            else resolve_checkpoint_path()
        )
        if not checkpoint_path.exists():
            message = "No AI checkpoint found - train first!"
            gui._set_message(message)
            logger.warning("Checkpoint not found: %s", checkpoint_path)
            _finish_failure(message)
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

        if getattr(gui, "ai_mission_graph_editor_enabled", False):
            ensure_mission_graph_editor_draft(gui, _random, logger=logger)

        draft_graph = getattr(gui, "ai_mission_graph_draft", None)
        if draft_graph is not None:
            mission_graph = copy.deepcopy(draft_graph)
            seed = int(getattr(gui, "ai_mission_graph_seed", configured_seed or 0) or 0)
            mission_data = mission_graph_to_gnn_input(mission_graph)
            mission_data["seed"] = seed
        else:
            mission_data = generate_mission_graph(_random, seed=configured_seed)
            mission_graph = mission_data["mission_graph"]
            seed = mission_data["seed"]

        graph_constraints = {
            "boss_node": getattr(gui, "ai_mission_graph_boss_node", None),
            "locked_edges": list(getattr(gui, "ai_mission_graph_locked_edges", []) or []),
        }
        mission_graph, graph_constraint_info = apply_mission_graph_constraints(
            mission_graph,
            constraints=graph_constraints,
            logger=logger,
        )
        mission_data = mission_graph_to_gnn_input(mission_graph)
        mission_data["seed"] = seed

        edge_index = mission_data["edge_index"]
        num_nodes = mission_data["num_nodes"]
        num_edges = mission_data["num_edges"]
        logger.info(
            "  Mission graph: %d nodes, %d edges, seed=%d%s",
            num_nodes,
            num_edges,
            seed,
            " (deterministic)" if configured_seed is not None else "",
        )
        if graph_constraint_info.get("boss_applied") or int(graph_constraint_info.get("locked_edges_applied", 0)) > 0:
            logger.info(
                "  Mission constraints: boss=%s, locked_edges=%d",
                graph_constraint_info.get("boss_applied"),
                int(graph_constraint_info.get("locked_edges_applied", 0)),
            )

        gui._set_message("Loading AI model...")
        pipeline = load_canonical_generation_pipeline(
            checkpoint_path=checkpoint_path,
            device=device,
            logger=logger,
            strict_checkpoint_mode=strict_checkpoint_mode,
        )

        gui._set_message("Generating rooms with canonical pipeline...")
        dungeon_result = generate_dungeon_with_pipeline(
            pipeline=pipeline,
            mission_graph=mission_graph,
            seed=seed,
            logger=logger,
        )
        tile_grid = dungeon_result.dungeon_grid.astype(np.int32, copy=False)

        staged_constraints = {
            "boss_norm": getattr(gui, "ai_constraint_boss_norm", None),
            "lock_norm": getattr(gui, "ai_constraint_lock_norm", None),
            "key_norm": getattr(gui, "ai_constraint_key_norm", None),
        }
        tile_grid, applied_constraints = apply_mixed_initiative_constraints(
            tile_grid=tile_grid,
            constraints=staged_constraints,
            np_module=np,
            logger=logger,
        )

        applied = build_generated_dungeon_payload(
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
        result_message = applied["message"]
        clear_mixed_constraints = False
        if (
            applied_constraints.get("boss_applied")
            or applied_constraints.get("lock_applied")
            or applied_constraints.get("key_applied")
        ):
            result_message = (
                "AI dungeon generated with mixed-initiative constraints "
                f"(boss={applied_constraints.get('boss_applied')}, "
                f"lock={applied_constraints.get('lock_applied')}, "
                f"key={applied_constraints.get('key_applied')})"
            )
            clear_mixed_constraints = True
        gui.ai_gen_result = {
            "success": True,
            "grid": tile_grid,
            "name": applied["name"],
            "message": result_message,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "height": applied["height"],
            "width": applied["width"],
            "unique_tiles": applied["unique_tiles"],
            "clear_mixed_constraints": clear_mixed_constraints,
            "mission_graph_draft": copy.deepcopy(mission_graph),
        }
        gui.ai_gen_done = True
    except (AttributeError, RuntimeError, ValueError, TypeError, ImportError, OSError) as exc:
        logger.exception("AI generation failed: %s", exc)
        message = f"AI generation failed: {exc}"
        gui._set_message(message)
        _finish_failure(message)

