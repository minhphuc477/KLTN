#!/usr/bin/env python
"""Run the real 20-room/3-key H-MOLQD end-to-end integration protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import merge_config
from src.generation.evolutionary_director.converters import mission_graph_to_networkx
from src.generation.grammar import (
    InsertChallengeRule,
    InsertLockKeyRule,
    MissionGraph,
    MissionGrammar,
    NodeType,
    StartRule,
)
from src.pipeline.config_bridge import pipeline_kwargs_from_resolved_config
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.utils.checkpoint import checkpoint_sha256


def build_protocol_graph(seed: int) -> MissionGraph:
    rng = random.Random(int(seed))
    context = {
        "rng": rng,
        "difficulty": 0.5,
        "goal_row": 5,
        "goal_col": 5,
        "spatial_compilable": True,
    }
    graph = StartRule().apply(MissionGraph(), context)
    lock_rule = InsertLockKeyRule()
    for _ in range(3):
        before = (len(graph.nodes), len(graph.edges))
        graph = lock_rule.apply(graph, context)
        if (len(graph.nodes), len(graph.edges)) == before:
            raise RuntimeError("InsertLockKeyRule failed while constructing the protocol graph.")
    while len(graph.nodes) < 20:
        challenge_type = NodeType.PUZZLE if len(graph.nodes) % 2 else NodeType.ENEMY
        before_nodes = len(graph.nodes)
        graph = InsertChallengeRule(challenge_type).apply(graph, context)
        if len(graph.nodes) == before_nodes:
            raise RuntimeError("InsertChallengeRule could not reach the required 20 rooms.")
    if len(graph.nodes) != 20:
        raise RuntimeError(f"Protocol graph has {len(graph.nodes)} nodes, expected exactly 20.")
    keys = sum(node.node_type == NodeType.KEY for node in graph.nodes.values())
    locks = sum(node.node_type == NodeType.LOCK for node in graph.nodes.values())
    if (keys, locks) != (3, 3):
        raise RuntimeError(f"Protocol graph has keys/locks={(keys, locks)}, expected (3, 3).")
    positions = [tuple(node.position) for node in graph.nodes.values()]
    if len(set(positions)) != len(positions):
        raise RuntimeError("Protocol graph contains overlapping mission-node positions.")
    undirected_degree: Dict[Any, set[Any]] = {node_id: set() for node_id in graph.nodes}
    for edge in graph.edges:
        undirected_degree[edge.source].add(edge.target)
        undirected_degree[edge.target].add(edge.source)
    max_degree = max((len(neighbors) for neighbors in undirected_degree.values()), default=0)
    if max_degree > 4:
        raise RuntimeError(f"Protocol graph maximum physical degree is {max_degree}, expected <= 4.")
    grammar = MissionGrammar()
    if not grammar.validate_lock_key_ordering(graph, log_failures=True):
        raise RuntimeError("Protocol graph violates key-before-lock ordering.")
    if not grammar.validate_progression_constraints(graph, log_failures=True):
        raise RuntimeError("Protocol graph violates progression constraints.")
    return graph


def _gradient_probe(pipeline: NeuralSymbolicDungeonPipeline, graph: Any) -> float:
    graph_data = pipeline._prepare_graph_context(graph)
    room_id = graph_data["node_order"][0]
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
    empty_neighbors = {direction: None for direction in ("N", "S", "E", "W")}
    pipeline.condition_encoder.zero_grad(set_to_none=True)
    pipeline.diffusion.zero_grad(set_to_none=True)
    condition = pipeline._compute_room_condition(
        neighbor_latents=empty_neighbors,
        reference_room_maps=None,
        graph_context=room_graph_data,
        boundary_constraints=torch.zeros(1, 8, device=pipeline.device),
        position=position,
    )
    latent = torch.randn(
        1,
        int(getattr(pipeline.diffusion, "latent_dim", 64)),
        4,
        3,
        device=pipeline.device,
    )
    loss = pipeline.diffusion.training_loss(latent, condition, graph_data=room_graph_data)
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.detach().norm().item())
        for parameter in pipeline.condition_encoder.parameters()
        if parameter.grad is not None
    )
    pipeline.condition_encoder.zero_grad(set_to_none=True)
    pipeline.diffusion.zero_grad(set_to_none=True)
    if not gradient_norm > 0.0:
        raise RuntimeError("Condition encoder received zero gradient through the diffusion objective.")
    return gradient_norm


def run(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoints = {
        "vqvae": args.vqvae_checkpoint,
        "diffusion": args.diffusion_checkpoint,
        "condition_encoder": args.condition_encoder_checkpoint or args.diffusion_checkpoint,
        "logic_net": args.logic_net_checkpoint,
    }
    missing = [name for name, path in checkpoints.items() if path is None or not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Master integration requires real checkpoints; missing: {missing}")

    mission_graph = build_protocol_graph(args.seed)
    graph = mission_graph_to_networkx(mission_graph, directed=True)
    resolved = merge_config(yaml_path=str(args.config), cli_overrides=None)
    kwargs = pipeline_kwargs_from_resolved_config(resolved)
    kwargs.update(
        {
            "vqvae_checkpoint": str(checkpoints["vqvae"]),
            "diffusion_checkpoint": str(checkpoints["diffusion"]),
            "condition_encoder_checkpoint": str(checkpoints["condition_encoder"]),
            "logic_net_checkpoint": str(checkpoints["logic_net"]),
            "strict_checkpoint_mode": True,
            "require_logic_net": True,
            "device": str(args.device),
        }
    )
    pipeline = NeuralSymbolicDungeonPipeline.from_kwargs(**kwargs)
    condition_gradient_norm = _gradient_probe(pipeline, graph)
    result = pipeline.generate_dungeon(
        mission_graph=graph,
        generate_topology=False,
        apply_repair=bool(args.apply_repair),
        enable_map_elites=True,
        seed=int(args.seed),
        num_diffusion_steps=int(args.diffusion_steps),
    )
    validation = dict(result.map_elites_score or {})
    interactions = dict(validation.get("path_interactions", {}) or {})
    if not bool(validation.get("is_exact", False)):
        raise RuntimeError(f"Final validator did not return an exact verdict: {validation}")
    if not bool(validation.get("solvable", False)):
        raise RuntimeError(f"Generated dungeon is not solvable: {validation}")
    if int(validation.get("path_length", 0)) <= 0:
        raise RuntimeError("Exact solver returned an empty path.")
    if int(interactions.get("small_keys_collected", -1)) != 3:
        raise RuntimeError(f"Solver path collected {interactions.get('small_keys_collected')} small keys, expected 3.")
    if int(interactions.get("locked_doors_traversed", -1)) != 3:
        raise RuntimeError(f"Solver path traversed {interactions.get('locked_doors_traversed')} locked doors, expected 3.")

    report = {
        "status": "passed",
        "seed": int(args.seed),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "condition_gradient_norm": float(condition_gradient_norm),
        "path_length": int(validation["path_length"]),
        "path_interactions": interactions,
        "final_hard_solvable": result.metrics.get("final_hard_solvable"),
        "repair_count": int(result.metrics.get("repair_count", 0)),
        "checkpoint_sha256": {
            name: checkpoint_sha256(path)
            for name, path in checkpoints.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp.replace(args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/zelda_hmolqd.yaml"))
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--logic-net-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/master_pipeline_integration.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--apply-repair", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
