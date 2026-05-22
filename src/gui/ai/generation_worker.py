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
    save_generated_dungeon_txt,
)


_PIPELINE_CACHE_ENV_KEYS = (
    "KLTN_GUI_FAST_GENERATION",
    "KLTN_GUI_USE_FAST_SAMPLING",
    "KLTN_GUI_DIFFUSION_STEPS",
    "KLTN_GUI_FAST_SAMPLER_CHECKPOINT",
    "KLTN_FAST_SAMPLER_CHECKPOINT",
)


def _resolve_generation_device(torch_module, logger):
    device_name = str(os.environ.get("KLTN_AI_DEVICE", "auto") or "auto").strip().lower()
    if device_name in {"", "auto"}:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    try:
        return torch_module.device(device_name)
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Invalid KLTN_AI_DEVICE=%r; falling back to CPU: %s", device_name, exc)
        return torch_module.device("cpu")


def _pipeline_cache_key(checkpoint_path, device, strict_checkpoint_mode, *, use_fast_sampler=False):
    speed_signature = tuple(os.environ.get(name, "") for name in _PIPELINE_CACHE_ENV_KEYS)
    return (
        str(checkpoint_path),
        str(device),
        bool(strict_checkpoint_mode),
        speed_signature,
        bool(use_fast_sampler),
    )


def _bounded_optional_int(value, *, min_value, max_value, default=None):
    if value is None or str(value).strip() == "":
        return default
    try:
        return max(int(min_value), min(int(max_value), int(value)))
    except (TypeError, ValueError):
        return default


def _effective_generation_config(gui):
    """Return sanitized AI generation settings from GUI state."""
    config = getattr(gui, "ai_generation_config", None)
    config = dict(config) if isinstance(config, dict) else {}

    num_rooms = _bounded_optional_int(
        config.get("num_rooms", getattr(gui, "ai_num_rooms", None)),
        min_value=5,
        max_value=24,
        default=None,
    )
    max_keys = _bounded_optional_int(
        config.get("max_keys", getattr(gui, "ai_max_keys", 3)),
        min_value=0,
        max_value=8,
        default=3,
    )
    diffusion_steps = _bounded_optional_int(
        config.get("diffusion_steps", getattr(gui, "ai_diffusion_steps", 50)),
        min_value=8,
        max_value=100,
        default=50,
    )
    difficulty = str(config.get("difficulty", getattr(gui, "ai_difficulty", "HARD")) or "HARD").upper()
    seed = config.get("seed", getattr(gui, "ai_seed", None))
    use_fast_sampler = bool(config.get("use_fast_sampler", getattr(gui, "ai_use_fast_sampler", False)))
    return {
        "num_rooms": num_rooms,
        "max_keys": max_keys,
        "difficulty": difficulty,
        "seed": seed,
        "diffusion_steps": diffusion_steps,
        "use_fast_sampler": use_fast_sampler,
    }


def _set_pipeline_eval(pipeline) -> None:
    for attr_name in ("vqvae", "diffusion", "condition_encoder", "logic_net"):
        model = getattr(pipeline, attr_name, None)
        if hasattr(model, "eval"):
            model.eval()


def load_or_reuse_gui_generation_pipeline(
    *,
    gui,
    checkpoint_path,
    device,
    logger,
    strict_checkpoint_mode,
):
    """Reuse an already-loaded GUI pipeline when the checkpoint/options match."""
    generation_config = _effective_generation_config(gui)
    use_fast_sampler = bool(generation_config.get("use_fast_sampler", False))
    cache_key = _pipeline_cache_key(
        checkpoint_path,
        device,
        strict_checkpoint_mode,
        use_fast_sampler=use_fast_sampler,
    )
    cache = getattr(gui, "_ai_generation_pipeline_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key and cache.get("pipeline") is not None:
        logger.info("AI Generation: reusing cached model pipeline for %s", checkpoint_path)
        return cache["pipeline"]

    pipeline = load_canonical_generation_pipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
        strict_checkpoint_mode=strict_checkpoint_mode,
        gui_fast_mode=use_fast_sampler,
    )
    _set_pipeline_eval(pipeline)
    try:
        gui._ai_generation_pipeline_cache = {"key": cache_key, "pipeline": pipeline}
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass
    return pipeline


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
        device = _resolve_generation_device(torch, logger)
        logger.info("AI Generation: using device %s", device)

        if getattr(gui, "ai_mission_graph_editor_enabled", False):
            ensure_mission_graph_editor_draft(gui, _random, logger=logger)

        draft_graph = getattr(gui, "ai_mission_graph_draft", None)
        demo_comprehensive = str(os.environ.get("KLTN_DEMO_COMPREHENSIVE", "")).strip() == "1"
        generation_config = _effective_generation_config(gui)
        config_seed = generation_config.get("seed")
        if config_seed is not None:
            try:
                configured_seed = int(config_seed)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid AI config seed=%r", config_seed)

        if demo_comprehensive:
            from src.gui.ai.generation_pipeline import generate_comprehensive_demo_graph
            mission_data = generate_comprehensive_demo_graph(seed=configured_seed)
            mission_graph = mission_data["mission_graph"]
            seed = mission_data["seed"]
            gui._set_message("Generating comprehensive demo graph...")
        elif draft_graph is not None:
            mission_graph = copy.deepcopy(draft_graph)
            seed = int(getattr(gui, "ai_mission_graph_seed", configured_seed or 0) or 0)
            mission_data = mission_graph_to_gnn_input(mission_graph)
            mission_data["seed"] = seed
        else:
            mission_data = generate_mission_graph(
                _random,
                seed=configured_seed,
                num_rooms=generation_config.get("num_rooms"),
                difficulty=generation_config.get("difficulty", "HARD"),
                max_keys=generation_config.get("max_keys", 3),
            )
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
        pipeline = load_or_reuse_gui_generation_pipeline(
            gui=gui,
            checkpoint_path=checkpoint_path,
            device=device,
            logger=logger,
            strict_checkpoint_mode=strict_checkpoint_mode,
        )

        gui._set_message("Generating rooms with canonical pipeline...")
        try:
            pipeline.default_use_fast_sampling = bool(generation_config.get("use_fast_sampler", False))
            pipeline.default_num_diffusion_steps = int(generation_config.get("diffusion_steps") or 50)
            if not bool(generation_config.get("use_fast_sampler", False)):
                pipeline.default_latent_sampler = "diffusion"
                logger.info(
                    "AI Generation: using canonical diffusion sampler (%d steps, fast sampler disabled)",
                    int(pipeline.default_num_diffusion_steps),
                )
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Failed to apply GUI generation sampler config: %s", exc)
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
        generated_txt_info = None
        try:
            generated_txt_info = save_generated_dungeon_txt(
                tile_grid=tile_grid,
                seed=seed,
                num_nodes=num_nodes,
                num_edges=num_edges,
                checkpoint_path=checkpoint_path,
                export_dir=getattr(gui, "ai_generated_level_export_dir", None),
                np_module=np,
            )
            logger.info("AI dungeon TXT exported to %s", generated_txt_info["txt_path"])
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to export generated AI dungeon TXT: %s", exc)

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
            "generated_txt_path": (
                None if generated_txt_info is None else str(generated_txt_info["txt_path"])
            ),
            "generated_latest_txt_path": (
                None if generated_txt_info is None else str(generated_txt_info["latest_txt_path"])
            ),
            "generated_png_path": (
                None if generated_txt_info is None else str(generated_txt_info.get("png_path"))
            ),
            "generated_latest_png_path": (
                None if generated_txt_info is None else str(generated_txt_info.get("latest_png_path"))
            ),
            "generated_metadata_path": (
                None if generated_txt_info is None else str(generated_txt_info["metadata_path"])
            ),
        }
        gui.ai_gen_done = True
    except (AttributeError, RuntimeError, ValueError, TypeError, ImportError, OSError) as exc:
        logger.exception("AI generation failed: %s", exc)
        message = f"AI generation failed: {exc}"
        gui._set_message(message)
        _finish_failure(message)

