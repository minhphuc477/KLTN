#!/usr/bin/env python3
"""Probe graph conditioning and optional room generation on synthetic OOD topologies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import extract_graph_descriptor
from src.pipeline import NeuralSymbolicDungeonPipeline


def _add_room(graph: nx.DiGraph, node: int, label: str, x: int, y: int, difficulty: float) -> None:
    graph.add_node(
        node,
        label=label,
        type=label,
        position=(x, y, 0),
        difficulty=float(difficulty),
        has_enemy=("ENEMY" in label or "BOSS" in label),
        has_key=("KEY" in label),
        is_start=(label == "START"),
        is_goal=(label == "GOAL"),
    )


def build_chain_control_graph() -> nx.DiGraph:
    graph = nx.DiGraph(name="id_chain_control")
    labels = ["START", "ROOM", "ENEMY", "KEY", "ROOM", "LOCK", "BOSS", "GOAL"]
    for idx, label in enumerate(labels):
        _add_room(graph, idx, label, idx, 0, idx / max(1, len(labels) - 1))
    for idx in range(len(labels) - 1):
        graph.add_edge(idx, idx + 1, edge_type="LOCKED" if idx == 4 else "PATH")
    return graph


def build_metroidvania_ood_graph() -> nx.DiGraph:
    """Build a reconvergent, ability-gated macro topology outside a Zelda chain."""
    graph = nx.DiGraph(name="ood_metroidvania_reconvergent")
    rooms = {
        0: ("START", 0, 2, 0.05),
        1: ("HUB", 1, 2, 0.12),
        2: ("ENEMY", 2, 1, 0.22),
        3: ("KEY", 3, 0, 0.28),
        4: ("ROOM", 2, 3, 0.20),
        5: ("ITEM", 3, 4, 0.36),
        6: ("LOCK", 4, 2, 0.43),
        7: ("PUZZLE", 5, 1, 0.52),
        8: ("ENEMY", 5, 3, 0.58),
        9: ("HUB", 6, 2, 0.62),
        10: ("KEY", 7, 3, 0.69),
        11: ("LOCK", 8, 2, 0.76),
        12: ("BOSS", 9, 2, 0.90),
        13: ("GOAL", 10, 2, 1.00),
    }
    for node, (label, x, y, difficulty) in rooms.items():
        _add_room(graph, node, label, x, y, difficulty)
    for source, target, edge_type in (
        (0, 1, "PATH"),
        (1, 2, "PATH"),
        (2, 3, "PATH"),
        (1, 4, "PATH"),
        (4, 5, "PATH"),
        (3, 6, "LOCKED"),
        (5, 6, "ITEM_GATE"),
        (6, 7, "PATH"),
        (6, 8, "PATH"),
        (7, 9, "PATH"),
        (8, 9, "PATH"),
        (9, 10, "PATH"),
        (9, 11, "PATH"),
        (10, 11, "LOCKED"),
        (11, 12, "BOSS_LOCKED"),
        (12, 13, "PATH"),
    ):
        graph.add_edge(source, target, edge_type=edge_type, label=edge_type.lower())
    return graph


def _descriptor_payload(graph: nx.DiGraph) -> Dict[str, Any]:
    descriptor = extract_graph_descriptor(graph, grammar=None)
    return {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "linearity": float(descriptor.linearity),
        "leniency": float(descriptor.leniency),
        "progression_complexity": float(descriptor.progression_complexity),
        "topology_complexity": float(descriptor.topology_complexity),
        "path_length": float(descriptor.path_length),
    }


def _condition_graph(pipeline: NeuralSymbolicDungeonPipeline, graph: nx.DiGraph) -> Tuple[Dict[str, Any], np.ndarray]:
    graph_data = pipeline._prepare_graph_context(graph, use_tpe=True)
    encoder = pipeline.condition_encoder
    if encoder is None:
        raise RuntimeError("Condition encoder is unavailable.")
    with torch.no_grad():
        global_tokens = encoder.encode_global_only(
            graph_data["node_features"],
            graph_data["edge_index"],
            edge_features=graph_data["edge_features"],
            tpe=graph_data["tpe"],
        )
    embedding = global_tokens.detach().cpu().numpy()
    return graph_data, embedding


def _room_generation_payload(
    pipeline: NeuralSymbolicDungeonPipeline,
    graph: nx.DiGraph,
    *,
    seed: int,
    diffusion_steps: int,
) -> Dict[str, Any]:
    prepared = pipeline.prepare_dungeon_generation(graph)
    generated = pipeline.generate_rooms_for_graph(
        prepared,
        seed=int(seed),
        num_diffusion_steps=int(diffusion_steps),
        apply_repair=False,
    )
    unique_tiles = sorted(
        {
            int(tile)
            for room in generated.rooms.values()
            for tile in np.unique(np.asarray(room.room_grid))
        }
    )
    return {
        "rooms_generated": int(len(generated.rooms)),
        "unique_semantic_tiles": unique_tiles,
        "batch_runtime_diagnostics": list(generated.batch_runtime_diagnostics),
    }


def _require_existing(paths: Iterable[Path | None], message: str) -> None:
    missing = [str(path) for path in paths if path is None or not path.exists()]
    if missing:
        raise SystemExit(f"{message}: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "synthetic_metroidvania_ood")
    parser.add_argument("--condition-encoder-checkpoint", type=Path)
    parser.add_argument("--vqvae-checkpoint", type=Path)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--logic-net-checkpoint", type=Path)
    parser.add_argument("--generate-rooms", action="store_true")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.condition_encoder_checkpoint is not None:
        _require_existing([args.condition_encoder_checkpoint], "Missing trained condition-encoder checkpoint")
    if args.generate_rooms:
        _require_existing(
            [args.condition_encoder_checkpoint, args.vqvae_checkpoint, args.diffusion_checkpoint],
            "--generate-rooms requires trained neural checkpoints",
        )

    pipeline = NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=str(args.vqvae_checkpoint) if args.vqvae_checkpoint else None,
        diffusion_checkpoint=str(args.diffusion_checkpoint) if args.diffusion_checkpoint else None,
        logic_net_checkpoint=str(args.logic_net_checkpoint) if args.logic_net_checkpoint else None,
        condition_encoder_checkpoint=(
            str(args.condition_encoder_checkpoint) if args.condition_encoder_checkpoint else None
        ),
        device=str(args.device),
        enable_logging=False,
        default_num_diffusion_steps=int(args.diffusion_steps),
    )
    graphs = {
        "id_chain_control": build_chain_control_graph(),
        "ood_metroidvania_reconvergent": build_metroidvania_ood_graph(),
    }
    embeddings: Dict[str, np.ndarray] = {}
    graph_reports: Dict[str, Any] = {}
    for name, graph in graphs.items():
        graph_data, embedding = _condition_graph(pipeline, graph)
        embeddings[name] = embedding
        graph_reports[name] = {
            "descriptor": _descriptor_payload(graph),
            "conditioning": {
                "node_feature_shape": list(graph_data["node_features"].shape),
                "edge_feature_shape": list(graph_data["edge_features"].shape),
                "global_token_shape": list(embedding.shape),
                "global_token_finite": bool(np.isfinite(embedding).all()),
                "global_token_norm_mean": float(np.linalg.norm(embedding, axis=-1).mean()),
            },
        }
        if args.generate_rooms:
            graph_reports[name]["room_generation"] = _room_generation_payload(
                pipeline,
                graph,
                seed=int(args.seed),
                diffusion_steps=int(args.diffusion_steps),
            )

    id_mean = embeddings["id_chain_control"].mean(axis=0)
    ood_mean = embeddings["ood_metroidvania_reconvergent"].mean(axis=0)
    evidence_scope = "schema_smoke_random_weights"
    if args.condition_encoder_checkpoint:
        evidence_scope = "trained_condition_encoder_probe"
    if args.generate_rooms:
        evidence_scope = "trained_condition_encoder_and_room_generation_probe"
    report = {
        "evidence_scope": evidence_scope,
        "warning": (
            "Random-weight schema smoke only; do not cite as OOD generalization evidence."
            if not args.condition_encoder_checkpoint
            else "Synthetic structural-shift probe; report alongside held-out and external evidence."
        ),
        "checkpoints": {
            "condition_encoder": str(args.condition_encoder_checkpoint) if args.condition_encoder_checkpoint else None,
            "vqvae": str(args.vqvae_checkpoint) if args.vqvae_checkpoint else None,
            "diffusion": str(args.diffusion_checkpoint) if args.diffusion_checkpoint else None,
        },
        "embedding_mean_l2_shift": float(np.linalg.norm(ood_mean - id_mean)),
        "graphs": graph_reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "synthetic_metroidvania_ood_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
