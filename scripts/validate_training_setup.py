#!/usr/bin/env python
"""Verify LogicNet is included in diffusion training optimization."""

import sys
from dataclasses import fields
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig


def _flatten_yaml_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested YAML sections into a single-level dict.

    For each nested dict, both the dotted key (``section.key``) *and* the
    bare leaf key are emitted.  Bare leaf keys are only kept when they do
    not collide with an existing top-level key so that explicit top-level
    overrides always win.
    """
    items: dict = {}
    for k, v in d.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_yaml_dict(v, full_key, sep=sep))
        else:
            items[full_key] = v
            # Also store the bare leaf key if it doesn't collide.
            if k not in items:
                items[k] = v
    return items


def check_training_setup():
    """Load the training config and verify LogicNet parameters are optimized."""
    print("[*] Loading Diffusion Training Config")
    config_path = Path("configs/zelda_hmolqd.yaml")

    with config_path.open(encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)

    flat_dict = _flatten_yaml_dict(config_dict)

    config_args = {
        field.name: flat_dict[field.name]
        for field in fields(DiffusionTrainingConfig)
        if field.name in flat_dict
    }

    config = DiffusionTrainingConfig(**config_args)
    print("    [OK] Loaded config")

    print("\n[*] Creating DiffusionTrainer")
    trainer = DiffusionTrainer(config)
    print("    [OK] Created trainer")

    print("\n[*] Checking Training Setup")
    logic_net = trainer.logic_net
    print(f"    LogicNet exists: {logic_net is not None}")

    optimizer_param_ids = {
        id(param)
        for param_group in trainer.optimizer.param_groups
        for param in param_group["params"]
    }

    logic_net_params = list(logic_net.parameters())
    logic_net_in_optimizer = sum(1 for param in logic_net_params if id(param) in optimizer_param_ids)

    print(f"    LogicNet parameter tensors: {len(logic_net_params)}")
    print(f"    LogicNet params in optimizer: {logic_net_in_optimizer}")
    print(f"    Total LogicNet parameters: {sum(param.numel() for param in logic_net_params):,}")
    print(
        "    Total LogicNet params in optimizer: "
        f"{sum(param.numel() for param in logic_net_params if id(param) in optimizer_param_ids):,}"
    )

    print("\n[*] Training Configuration")
    print(f"    alpha_logic: {config.alpha_logic}")
    print(f"    logic_loss_mode: {config.logic_loss_mode}")
    print(f"    include_logic_loss: {getattr(config, 'include_logic_loss', 'not set')}")

    print("\n[*] Module Structure")
    print(f"    diffusion.guidance.logic_net is trainer.logic_net: {trainer.diffusion.guidance.logic_net is logic_net}")

    if logic_net_in_optimizer == len(logic_net_params):
        print("\n[OK] SUCCESS: All LogicNet parameters are in the optimizer.")
        print("    LogicNet WILL be trained during training")
        return 0

    print("\n[FAIL] PROBLEM: Not all LogicNet parameters are in optimizer.")
    print(f"    Only {logic_net_in_optimizer}/{len(logic_net_params)} parameter tensors")
    return 1


if __name__ == "__main__":
    sys.exit(check_training_setup())
