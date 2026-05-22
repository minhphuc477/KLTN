#!/usr/bin/env python
"""Verify LogicNet is actually being trained in the diffusion training loop."""

import sys
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train_diffusion import DiffusionTrainingConfig, DiffusionTrainer

def check_training_setup():
    print("[*] Loading Diffusion Training Config")
    config_path = Path("configs/zelda_hmolqd.yaml")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to proper config object
    from src.train_diffusion import DiffusionTrainingConfig
    from dataclasses import fields
    
    # Extract only the fields that DiffusionTrainingConfig expects
    config_args = {}
    for field in fields(DiffusionTrainingConfig):
        if field.name in config_dict:
            config_args[field.name] = config_dict[field.name]
    
    config = DiffusionTrainingConfig(**config_args)
    print(f"    ✓ Loaded config")
    
    print("\n[*] Creating DiffusionTrainer")
    trainer = DiffusionTrainer(config)
    print(f"    ✓ Created trainer")
    
    print("\n[*] Checking Training Setup")
    
    # Check LogicNet exists
    logic_net = trainer.logic_net
    print(f"    LogicNet exists: {logic_net is not None}")
    
    # Get all trainable parameters from optimizer
    optimizer_param_ids = set()
    for param_group in trainer.optimizer.param_groups:
        for param in param_group['params']:
            optimizer_param_ids.add(id(param))
    
    # Check how many LogicNet params are in optimizer
    logic_net_params = list(logic_net.parameters())
    logic_net_in_optimizer = sum(1 for p in logic_net_params if id(p) in optimizer_param_ids)
    
    print(f"    LogicNet parameter tensors: {len(logic_net_params)}")
    print(f"    LogicNet params in optimizer: {logic_net_in_optimizer}")
    print(f"    Total LogicNet parameters: {sum(p.numel() for p in logic_net_params):,}")
    print(f"    Total LogicNet params in optimizer: {sum(p.numel() for p in logic_net_params if id(p) in optimizer_param_ids):,}")
    
    # Check training config
    print(f"\n[*] Training Configuration")
    print(f"    alpha_logic: {config.alpha_logic}")
    print(f"    logic_loss_mode: {config.logic_loss_mode}")
    print(f"    include_logic_loss: {getattr(config, 'include_logic_loss', 'not set')}")
    
    # Check if LogicNet is in diffusion.guidance
    print(f"\n[*] Module Structure")
    print(f"    diffusion.guidance.logic_net is trainer.logic_net: {trainer.diffusion.guidance.logic_net is logic_net}")
    
    if logic_net_in_optimizer == len(logic_net_params):
        print(f"\n[✓] SUCCESS: All LogicNet parameters are in the optimizer!")
        print(f"    LogicNet WILL be trained during training")
    else:
        print(f"\n[✗] PROBLEM: Not all LogicNet parameters are in optimizer!")
        print(f"    Only {logic_net_in_optimizer}/{len(logic_net_params)} parameter tensors")

if __name__ == "__main__":
    check_training_setup()
