#!/usr/bin/env python
"""Generate or run LogicNet proof ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "baseline_no_logicnet",
        "description": "Diffusion baseline with LogicNet disabled.",
        "diffusion": {
            "logic_net_enabled": False,
            "alpha_logic": 0.0,
            "alpha_logic_tile": 0.0,
            "guidance_scale": 0.0,
        },
    },
    {
        "name": "tile_only",
        "description": "Train the tile classifier, but disable pathfinder logic loss.",
        "diffusion": {
            "logic_net_enabled": True,
            "alpha_logic": 0.0,
            "alpha_logic_tile": 0.1,
            "logic_grid_pathfinder": "bellman_ford",
        },
    },
    {
        "name": "full_logicnet_bellman_ford",
        "description": "Full LogicNet with tile classifier and explicit Bellman-Ford grid pathfinder.",
        "diffusion": {
            "logic_net_enabled": True,
            "alpha_logic": 0.1,
            "alpha_logic_tile": 0.1,
            "logic_grid_pathfinder": "bellman_ford",
        },
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


def _load_base_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Base config {path} must contain a mapping.")
    return loaded


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    base_config = _load_base_config(args.base_config)
    runs = []
    for variant in VARIANTS:
        config = _deep_merge(
            base_config,
            {
                "runtime": {"seed": int(args.seed)},
                "diffusion": {
                    "epochs": int(args.epochs),
                    "checkpoint_dir": str(args.output_dir / variant["name"] / "checkpoints"),
                    "validation_num_samples": int(args.validation_samples),
                    "validation_num_diffusion_samples": int(args.validation_diffusion_samples),
                },
            },
        )
        config = _deep_merge(config, {"diffusion": variant["diffusion"]})
        config_path = args.output_dir / variant["name"] / "config.json"
        command = [
            sys.executable,
            "-m",
            "src.train_diffusion",
            "--config",
            str(config_path),
        ]
        if args.quick:
            command.append("--quick")
        runs.append(
            {
                "name": variant["name"],
                "description": variant["description"],
                "config_path": str(config_path),
                "config": config,
                "command": command,
            }
        )
    return {
        "base_config": str(args.base_config),
        "seed": int(args.seed),
        "proof_metrics": [
            "val_logic_tile_accuracy",
            "val_grid_reach_loss",
            "val_graph_reach_loss",
            "val_hard_solvability",
            "val_solvability",
        ],
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-config", type=Path, default=Path("configs/zelda_hmolqd.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/logicnet_ablation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--validation-diffusion-samples", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        path = Path(run["config_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run["config"], indent=2), encoding="utf-8")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)

    if args.execute:
        for run in manifest["runs"]:
            subprocess.run(run["command"], check=True)


if __name__ == "__main__":
    main()
