#!/usr/bin/env python
"""Generate reproducible config patches and commands for LogicNet loss ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "full",
        "description": "All diffusion, topology, global reachability, and tile-classifier losses enabled.",
        "overrides": {},
    },
    {
        "name": "no_logic_grid_reach",
        "description": "Disable local grid-reachability pressure by zeroing alpha_logic.",
        "overrides": {"diffusion": {"alpha_logic": 0.0}},
    },
    {
        "name": "no_tile_classifier",
        "description": "Disable supervised LogicNet tile-classifier cross entropy.",
        "overrides": {"diffusion": {"alpha_logic_tile": 0.0}},
    },
    {
        "name": "no_topology_trace",
        "description": "Disable room-topology traversability trace loss.",
        "overrides": {"diffusion": {"logic_topology_trace_weight": 0.0}},
    },
    {
        "name": "no_topology_anchor",
        "description": "Disable start/goal/door topology anchor walkability loss.",
        "overrides": {"diffusion": {"logic_topology_anchor_weight": 0.0}},
    },
    {
        "name": "no_global_graph_reach",
        "description": "Disable mission-graph reachability loss.",
        "overrides": {"diffusion": {"logic_global_reach_weight": 0.0}},
    },
    {
        "name": "no_global_room_lift",
        "description": "Disable room-passability lifting into mission-graph node costs.",
        "overrides": {"diffusion": {"logic_global_room_weight": 0.0}},
    },
    {
        "name": "no_spatial_alignment",
        "description": "Disable graph-node to grid-position attention alignment.",
        "overrides": {"diffusion": {"graph_spatial_alignment_weight": 0.0}},
    },
]


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    runs = []
    for variant in VARIANTS:
        config_patch = _deep_merge(
            {
                "diffusion": {
                    "checkpoint_dir": str(args.output_dir / variant["name"] / "checkpoints"),
                },
                "runtime": {"seed": int(args.seed)},
            },
            variant["overrides"],
        )
        config_path = args.output_dir / variant["name"] / "config.json"
        command = [
            "python",
            "-m",
            "src.train_diffusion",
            "--config",
            str(config_path),
        ]
        runs.append(
            {
                "name": variant["name"],
                "description": variant["description"],
                "base_config": str(args.base_config),
                "config_path": str(config_path),
                "config": config_patch,
                "manual_command": " ".join(command),
            }
        )
    return {
        "base_config": str(args.base_config),
        "seed": int(args.seed),
        "notes": (
            "Run each variant with identical train/test splits and seeds. Report Dungeon 9 holdout "
            "solvability, key-lock violation rate, tile-pattern JS divergence, and paired deltas vs full."
        ),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/logic_loss_ablation_manifest"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        config_path = Path(run["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(run["config"], indent=2), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
