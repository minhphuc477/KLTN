"""Headless canonical pipeline helpers shared by non-GUI CLI entrypoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx

from src.config_system import load_resolved_config_for_artifact
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator
from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.utils.checkpoint import safe_torch_load
from src.zelda_data.vglc_utils import filter_virtual_nodes, validate_topology


def generate_mission_graph(
    random_module,
    *,
    seed: Optional[int] = None,
    num_rooms: Optional[int] = None,
    difficulty: str = "MEDIUM",
    max_keys: int = 2,
) -> Dict[str, Any]:
    """Generate a validated directed mission graph without any GUI dependencies."""
    _ = (difficulty, max_keys)
    chosen_seed = int(seed) if seed is not None else int(random_module.randint(0, 2**31 - 1))
    room_count = int(num_rooms) if num_rooms is not None else int(8 + (chosen_seed % 5))
    room_count = max(3, room_count)
    curve = [0.2 + 0.6 * (i / max(1, room_count - 1)) for i in range(room_count)]

    best_graph: Optional[nx.DiGraph] = None
    best_score: Optional[tuple[int, int]] = None
    for offset in range(6):
        candidate_seed = chosen_seed + offset
        generator = EvolutionaryTopologyGenerator(
            target_curve=curve,
            population_size=max(16, room_count * 2),
            generations=max(20, min(40, room_count * 4)),
            max_nodes=room_count,
            seed=candidate_seed,
        )
        candidate = generator.evolve(directed_output=True)
        physical = filter_virtual_nodes(candidate)
        is_valid, _errors = validate_topology(physical)
        actual = int(physical.number_of_nodes())
        score = (0 if is_valid else 1, abs(actual - room_count))
        if best_graph is None or (best_score is not None and score < best_score):
            best_graph = candidate
            best_score = score
        if is_valid and actual == room_count:
            break

    if best_graph is None:
        raise RuntimeError(f"Mission grammar could not generate a graph for seed={chosen_seed}.")
    return {"mission_graph": best_graph, "seed": chosen_seed}


def _resolve_vqvae_checkpoint_for_generation(checkpoint_path: Path) -> Path:
    """Prefer embedded VQ-VAE weights, otherwise resolve sibling pretrained checkpoints."""
    meta_path = Path(f"{checkpoint_path}.meta.json")
    metadata_candidates = []
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict):
            extra = metadata.get("extra", {})
            if isinstance(extra, dict) and extra.get("vqvae_checkpoint"):
                cand = Path(str(extra["vqvae_checkpoint"])).expanduser()
                if not cand.is_absolute():
                    cand = Path(__file__).resolve().parents[2] / cand
                metadata_candidates.append(cand.resolve())

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
    checkpoint_path: str | Path,
    device: Any,
    logger: Optional[logging.Logger] = None,
    strict_checkpoint_mode: bool = False,
) -> NeuralSymbolicDungeonPipeline:
    """Construct the canonical room-wise neural-symbolic generation pipeline."""
    ckpt = Path(checkpoint_path)
    vqvae_checkpoint = _resolve_vqvae_checkpoint_for_generation(ckpt)
    resolved_config = load_resolved_config_for_artifact(ckpt)
    pipeline_kwargs = (
        pipeline_kwargs_from_resolved_config(resolved_config)
        if isinstance(resolved_config, dict)
        else {}
    )
    pipeline = NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(vqvae_checkpoint),
        diffusion_checkpoint=str(ckpt),
        logic_net_checkpoint=str(ckpt),
        condition_encoder_checkpoint=str(ckpt),
        device=str(device),
        enable_logging=False,
        strict_checkpoint_mode=bool(strict_checkpoint_mode),
        **pipeline_kwargs,
    )
    if logger is not None:
        logger.info(
            "Loaded canonical generation pipeline from %s (vqvae=%s, resolved_config=%s)",
            ckpt,
            vqvae_checkpoint,
            "yes" if resolved_config is not None else "no",
        )
    return pipeline


def generate_dungeon_with_pipeline(
    pipeline: NeuralSymbolicDungeonPipeline,
    mission_graph: nx.Graph,
    *,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    """Run canonical headless dungeon generation from a mission graph."""
    if logger is not None:
        logger.debug("Generating dungeon via canonical headless pipeline (seed=%s)", seed)
    return pipeline.generate_dungeon(
        mission_graph=mission_graph,
        seed=seed,
        enable_map_elites=False,
    )
