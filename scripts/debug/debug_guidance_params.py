#!/usr/bin/env python
"""Diagnose why GradientGuidance only sees 54 params from LogicNet."""

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logic_net import LogicNet
from src.core.latent_diffusion import GradientGuidance

def main():
    print("[*] Creating LogicNet")
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    logic_net_params = sum(p.numel() for p in logic_net.parameters())
    print(f"    LogicNet total parameters: {logic_net_params:,}")
    
    print("\n[*] Creating GradientGuidance with LogicNet")
    guidance = GradientGuidance(logic_net=logic_net)
    
    print("\n[*] Analyzing GradientGuidance structure")
    print(f"    guidance._modules keys: {list(guidance._modules.keys())}")
    
    if 'logic_net' in guidance._modules:
        gn = guidance._modules['logic_net']
        gn_params = sum(p.numel() for p in gn.parameters()) if gn is not None else 0
        print(f"    guidance._modules['logic_net'] params: {gn_params:,}")
        print(f"    guidance._modules['logic_net'] is logic_net: {gn is logic_net}")
    else:
        print(f"    logic_net NOT in guidance._modules!")
    
    print("\n[*] Calling guidance.parameters()")
    guidance_params_list = list(guidance.parameters())
    guidance_params_count = sum(p.numel() for p in guidance_params_list)
    print(f"    Total parameters: {guidance_params_count:,}")
    print(f"    Number of parameter tensors: {len(guidance_params_list)}")
    
    # Check if any are from LogicNet
    logic_net_param_ids = set(id(p) for p in logic_net.parameters())
    guidance_param_ids = set(id(p) for p in guidance.parameters())
    overlap_ids = logic_net_param_ids & guidance_param_ids
    overlap_count = sum(p.numel() for id_val in overlap_ids for p in logic_net.parameters() if id(p) == id_val)
    
    print(f"\n[*] Parameter overlap analysis")
    print(f"    LogicNet parameter object IDs: {len(logic_net_param_ids)}")
    print(f"    Guidance parameter object IDs: {len(guidance_param_ids)}")
    print(f"    Overlapping IDs: {len(overlap_ids)}")
    print(f"    Overlapping parameter count: {overlap_count:,}")
    
    print("\n[*] Checking requires_grad on guidance.parameters()")
    all_require_grad = all(p.requires_grad for p in guidance.parameters())
    print(f"    All guidance parameters have requires_grad=True: {all_require_grad}")
    
    print("\n[*] Getting param groups for optimizer")
    from torch.optim import AdamW
    optimizer = AdamW(guidance.parameters(), lr=0.001)
    param_count_in_optimizer = sum(p.numel() for param_group in optimizer.param_groups for p in param_group['params'])
    print(f"    Parameters in optimizer: {param_count_in_optimizer:,}")

if __name__ == "__main__":
    main()
