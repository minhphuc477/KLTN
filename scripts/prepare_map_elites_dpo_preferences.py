#!/usr/bin/env python
"""Convert raw same-condition MAP-Elites room pairs into Diffusion-DPO tensors."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import merge_config
from src.evaluation.preference_buffer import deserialize_condition_graph
from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.utils.checkpoint import atomic_torch_save, checkpoint_sha256, safe_torch_load


def _cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            key: _cpu_value(nested)
            for key, nested in value.items()
            if key != "mission_graph"
        }
    if isinstance(value, list):
        return [_cpu_value(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_cpu_value(nested) for nested in value)
    return value


def _parse_room_id(value: Any) -> Any:
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return value


def _encode_tiles(pipeline: NeuralSymbolicDungeonPipeline, tiles: torch.Tensor) -> torch.Tensor:
    num_classes = int(getattr(pipeline.vqvae, "num_classes", 44))
    ids = tiles.to(device=pipeline.device, dtype=torch.long)
    if ids.dim() == 2:
        ids = ids.unsqueeze(0)
    one_hot = F.one_hot(ids.clamp(0, num_classes - 1), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=torch.float32)
    with torch.no_grad():
        latent, _indices = pipeline.vqvae.encode(one_hot)
        diffusion = getattr(pipeline, "diffusion", None)
        if diffusion is not None and hasattr(diffusion, "scale_first_stage_latent"):
            latent = diffusion.scale_first_stage_latent(latent)
    return latent.detach()


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    raw = safe_torch_load(str(args.raw_pairs), map_location="cpu")
    if not isinstance(raw, dict) or raw.get("format") != "hmolqd_raw_room_preferences_v1":
        raise ValueError("Expected hmolqd_raw_room_preferences_v1 payload.")
    preferred_tiles = raw.get("preferred_tiles")
    rejected_tiles = raw.get("rejected_tiles")
    pair_metadata = raw.get("pairs")
    if not isinstance(preferred_tiles, torch.Tensor) or not isinstance(rejected_tiles, torch.Tensor):
        raise ValueError("Raw preference payload is missing preferred/rejected tile tensors.")
    if not isinstance(pair_metadata, list) or len(pair_metadata) != int(preferred_tiles.shape[0]):
        raise ValueError("Raw preference metadata count does not match tile-pair count.")

    resolved = merge_config(yaml_path=str(args.config), cli_overrides=None)
    pipeline_kwargs = pipeline_kwargs_from_resolved_config(resolved)
    pipeline_kwargs.update(
        {
            "vqvae_checkpoint": str(args.vqvae_checkpoint),
            "diffusion_checkpoint": str(args.diffusion_checkpoint),
            "condition_encoder_checkpoint": str(args.condition_encoder_checkpoint or args.diffusion_checkpoint),
            "logic_net_checkpoint": None,
            "strict_checkpoint_mode": True,
            "require_logic_net": False,
            "default_enable_map_elites": False,
            "device": str(args.device),
        }
    )
    pipeline = NeuralSymbolicDungeonPipeline.from_kwargs(**pipeline_kwargs)
    pipeline.vqvae.eval()
    pipeline.condition_encoder.eval()

    examples = []
    empty_neighbors = {direction: None for direction in ("N", "S", "E", "W")}
    for index, metadata in enumerate(pair_metadata):
        graph = deserialize_condition_graph(metadata["graph_payload"])
        room_id = _parse_room_id(metadata["room_id"])
        if room_id not in graph:
            raise ValueError(f"Preference pair {index} room {room_id!r} is absent from its condition graph.")
        graph_data = pipeline._prepare_graph_context(graph)
        room_graph_data = pipeline._build_room_graph_context(
            graph_data=graph_data,
            mission_graph=graph,
            room_id=room_id,
        )
        position_raw = graph.nodes[room_id].get("position", (0, 0, 0))
        position = torch.tensor(
            [[float(position_raw[0]), float(position_raw[1])]],
            device=pipeline.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            context = pipeline._compute_room_condition(
                neighbor_latents=empty_neighbors,
                reference_room_maps=None,
                graph_context=room_graph_data,
                boundary_constraints=torch.zeros(1, 8, device=pipeline.device),
                position=position,
            )
            preferred = _encode_tiles(pipeline, preferred_tiles[index, 0])
            rejected = _encode_tiles(pipeline, rejected_tiles[index, 0])
        examples.append(
            {
                "preferred": preferred.cpu(),
                "rejected": rejected.cpu(),
                "context": context.detach().cpu(),
                "graph_data": _cpu_value(room_graph_data),
                "metadata": dict(metadata),
            }
        )

    payload = {
        "format": "hmolqd_diffusion_dpo_preferences_v1",
        "examples": examples,
        "provenance": {
            "raw_pairs": str(args.raw_pairs),
            "vqvae_checkpoint": str(args.vqvae_checkpoint),
            "vqvae_sha256": checkpoint_sha256(args.vqvae_checkpoint),
            "diffusion_checkpoint": str(args.diffusion_checkpoint),
            "diffusion_sha256": checkpoint_sha256(args.diffusion_checkpoint),
            "condition_encoder_checkpoint": str(args.condition_encoder_checkpoint or args.diffusion_checkpoint),
            "condition_encoder_sha256": checkpoint_sha256(args.condition_encoder_checkpoint or args.diffusion_checkpoint),
            "config": str(args.config),
        },
    }
    atomic_torch_save(payload, args.output)
    return {"output": str(args.output), "pairs": len(examples)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/zelda_hmolqd.yaml"))
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(prepare(parse_args()))
