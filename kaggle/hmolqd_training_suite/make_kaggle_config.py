#!/usr/bin/env python
"""Create a validated Kaggle training config from the canonical repo config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_system import merge_config, validate_config


STAGE_SECTIONS = ("diffusion", "fast_sampler", "masked_room")


def _set(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    node = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _apply_stage_branch(config: dict[str, Any], branch: str) -> None:
    branch = str(branch).strip().lower()
    presets: Mapping[str, Mapping[str, Any]] = {
        "base": {
            "puzzle_stage_conditioning_enabled": False,
            "puzzle_stage_topology_enabled": False,
            "puzzle_stage_semantics_loss_weight": 0.0,
        },
        "stage_full": {
            "puzzle_stage_conditioning_enabled": True,
            "puzzle_stage_topology_enabled": True,
            "puzzle_stage_semantics_loss_weight": 0.25,
        },
        "stage_tokens_only": {
            "puzzle_stage_conditioning_enabled": True,
            "puzzle_stage_topology_enabled": False,
            "puzzle_stage_semantics_loss_weight": 0.25,
        },
        "stage_trace_only": {
            "puzzle_stage_conditioning_enabled": False,
            "puzzle_stage_topology_enabled": True,
            "puzzle_stage_semantics_loss_weight": 0.25,
        },
        "stage_loss010": {
            "puzzle_stage_conditioning_enabled": True,
            "puzzle_stage_topology_enabled": True,
            "puzzle_stage_semantics_loss_weight": 0.10,
        },
        "stage_loss050": {
            "puzzle_stage_conditioning_enabled": True,
            "puzzle_stage_topology_enabled": True,
            "puzzle_stage_semantics_loss_weight": 0.50,
        },
    }
    if branch not in presets:
        raise ValueError(f"Unknown branch: {branch}")
    for section in STAGE_SECTIONS:
        for key, value in presets[branch].items():
            _set(config, f"{section}.{key}", value)


def _apply_tokenizer(config: dict[str, Any], tokenizer: str) -> None:
    tokenizer = str(tokenizer).strip().lower()
    if tokenizer == "vqvae":
        _set(config, "vqvae.architecture", "vqvae")
        _set(config, "vqvae.top_codebook_size", None)
        _set(config, "vqvae.top_latent_dim", None)
    elif tokenizer == "vqvae2":
        _set(config, "vqvae.architecture", "vqvae2")
        _set(config, "vqvae.codebook_size", 256)
        _set(config, "vqvae.top_codebook_size", 128)
        _set(config, "vqvae.top_latent_dim", 32)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


def _apply_profile(config: dict[str, Any], profile: str) -> None:
    profile = str(profile).strip().lower()
    if profile == "t4x2":
        _set(config, "dataset.batch_size", 4)
        _set(config, "dataset.num_workers", 2)
        _set(config, "runtime.checkpoint_storage_budget_gb", 6.0)
        _set(config, "distributed.backend", "nccl")
        _set(config, "distributed.nproc_per_node", 2)
        _set(config, "distributed.cuda_visible_devices", "0,1")
    elif profile == "p100":
        _set(config, "dataset.batch_size", 4)
        _set(config, "dataset.num_workers", 2)
        _set(config, "runtime.checkpoint_storage_budget_gb", 6.0)
        _set(config, "distributed.backend", "nccl")
        _set(config, "distributed.nproc_per_node", 1)
        _set(config, "distributed.cuda_visible_devices", "0")
    elif profile == "cpu":
        _set(config, "dataset.batch_size", 2)
        _set(config, "dataset.num_workers", 0)
        _set(config, "runtime.device", "cpu")
        _set(config, "distributed.backend", "gloo")
        _set(config, "distributed.nproc_per_node", 1)
        _set(config, "distributed.cuda_visible_devices", "")
    else:
        raise ValueError(f"Unknown profile: {profile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "configs" / "zelda_hmolqd.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--profile", choices=("t4x2", "p100", "cpu"), default="t4x2")
    parser.add_argument(
        "--branch",
        choices=("base", "stage_full", "stage_tokens_only", "stage_trace_only", "stage_loss010", "stage_loss050"),
        default="stage_full",
    )
    parser.add_argument("--tokenizer", choices=("vqvae", "vqvae2"), default="vqvae2")
    parser.add_argument("--vqvae-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vqvae-epochs", type=int, default=None)
    parser.add_argument("--diffusion-epochs", type=int, default=None)
    parser.add_argument("--fast-sampler-epochs", type=int, default=None)
    parser.add_argument("--masked-room-epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = merge_config(yaml_path=str(args.base_config), cli_overrides=None)

    _set(config, "runtime.experiment_name", args.experiment_name)
    _set(config, "runtime.output_dir", str(args.output_dir))
    _set(config, "runtime.device", "cuda" if args.profile != "cpu" else "cpu")
    _set(config, "runtime.seed", int(args.seed))
    _set(config, "runtime.verbose", True)
    _set(config, "runtime.quick", bool(args.quick))
    _set(config, "dataset.data_dir", str(args.data_dir))

    _apply_profile(config, args.profile)
    _apply_tokenizer(config, args.tokenizer)
    _apply_stage_branch(config, args.branch)

    if args.batch_size is not None:
        _set(config, "dataset.batch_size", int(args.batch_size))
    if args.vqvae_epochs is not None:
        _set(config, "vqvae.epochs", int(args.vqvae_epochs))
    if args.diffusion_epochs is not None:
        _set(config, "diffusion.epochs", int(args.diffusion_epochs))
    if args.fast_sampler_epochs is not None:
        _set(config, "fast_sampler.epochs", int(args.fast_sampler_epochs))
    if args.masked_room_epochs is not None:
        _set(config, "masked_room.epochs", int(args.masked_room_epochs))
    if args.quick:
        _set(config, "vqvae.save_every", 1)
        _set(config, "diffusion.save_every", 1)
        _set(config, "fast_sampler.save_every", 1)
        _set(config, "masked_room.save_every", 1)

    if args.vqvae_checkpoint is not None:
        _set(config, "diffusion.vqvae_checkpoint", str(args.vqvae_checkpoint))

    config = validate_config(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = {
        "config": str(args.output),
        "profile": args.profile,
        "tokenizer": args.tokenizer,
        "branch": args.branch,
        "output_dir": str(config["runtime"]["output_dir"]),
        "data_dir": str(config["dataset"]["data_dir"]),
        "vqvae_checkpoint": config["diffusion"]["vqvae_checkpoint"],
        "batch_size": config["dataset"]["batch_size"],
        "epochs": {
            "vqvae": config["vqvae"]["epochs"],
            "diffusion": config["diffusion"]["epochs"],
            "fast_sampler": config["fast_sampler"]["epochs"],
            "masked_room": config["masked_room"]["epochs"],
        },
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
