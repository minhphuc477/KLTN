#!/usr/bin/env python
"""Simple check: Verify LogicNet is in diffusion optimizer."""

import torch
from src.core.logic_net import LogicNet
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion

def main():
    print("[*] Test 1: Standalone LogicNet + Optimizer")
    
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    optimizer = torch.optim.AdamW(logic_net.parameters(), lr=0.001)
    
    logic_net_param_ids = set(id(p) for p in logic_net.parameters())
    optimizer_param_ids = set()
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            optimizer_param_ids.add(id(param))
    
    overlap = len(logic_net_param_ids & optimizer_param_ids)
    print(f"    LogicNet params: {len(logic_net_param_ids)}")
    print(f"    Optimizer has LogicNet params: {overlap}/{len(logic_net_param_ids)}")
    
    print("\n[*] Test 2: Diffusion + LogicNet in optimizer")
    
    logic_net2 = LogicNet(latent_dim=64, num_classes=44)
    diffusion = create_latent_diffusion(
        latent_dim=64,
        model_channels=128,
        context_dim=256,
    )
    
    # This is what training code does:
    diffusion.guidance.logic_net = logic_net2
    
    # Optimizer setup from training
    optimizer2 = torch.optim.AdamW(
        list(diffusion.parameters()),
        lr=0.001
    )
    
    diffusion_param_ids = set(id(p) for p in diffusion.parameters())
    logic_net_param_ids2 = set(id(p) for p in logic_net2.parameters())
    
    optimizer_param_ids2 = set()
    for param_group in optimizer2.param_groups:
        for param in param_group['params']:
            optimizer_param_ids2.add(id(param))
    
    # Check overlap
    logic_net_in_diffusion = len(logic_net_param_ids2 & diffusion_param_ids)
    logic_net_in_optimizer = len(logic_net_param_ids2 & optimizer_param_ids2)
    
    print(f"    LogicNet param tensors: {len(list(logic_net2.parameters()))}")
    print(f"    LogicNet in diffusion.parameters(): {logic_net_in_diffusion}/{len(list(logic_net2.parameters()))}")
    print(f"    LogicNet in optimizer: {logic_net_in_optimizer}/{len(list(logic_net2.parameters()))}")
    
    if logic_net_in_optimizer == len(list(logic_net2.parameters())):
        print("\n[✓] All LogicNet parameters ARE in the optimizer!")
    else:
        print(f"\n[✗] Problem: Only {logic_net_in_optimizer} out of {len(list(logic_net2.parameters()))} LogicNet param tensors in optimizer")

if __name__ == "__main__":
    main()
