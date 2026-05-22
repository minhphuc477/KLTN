#!/usr/bin/env python
"""Simple test: verify LogicNet parameter registration in module hierarchy."""

import torch
import torch.nn as nn
from src.core.logic_net import LogicNet
from src.core.latent_diffusion import GradientGuidance

def main():
    print("[*] Test 1: Create GradientGuidance without LogicNet, then assign LogicNet")
    
    # Step 1: Create guidance without logic_net (as training code does)
    guidance1 = GradientGuidance(logic_net=None)
    print(f"  guidance.parameters() count (before): {sum(p.numel() for p in guidance1.parameters())}")
    
    # Step 2: Create LogicNet
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    logic_net_param_count = sum(p.numel() for p in logic_net.parameters())
    print(f"  LogicNet total parameters: {logic_net_param_count:,}")
    
    # Step 3: Assign LogicNet (as training code does at line 792)
    guidance1.logic_net = logic_net
    guidance1_param_count = sum(p.numel() for p in guidance1.parameters())
    print(f"  guidance.parameters() count (after): {guidance1_param_count:,}")
    
    # Check actual overlap
    guidance1_param_ids = set(id(p) for p in guidance1.parameters())
    logicnet_param_ids = set(id(p) for p in logic_net.parameters())
    overlap = len(guidance1_param_ids & logicnet_param_ids)
    print(f"  Parameters overlap: {overlap:,} / {logic_net_param_count:,}")
    
    print("\n[*] Test 2: Create GradientGuidance WITH LogicNet (correct way)")
    
    logic_net2 = LogicNet(latent_dim=64, num_classes=44)
    guidance2 = GradientGuidance(logic_net=logic_net2)
    
    guidance2_param_count = sum(p.numel() for p in guidance2.parameters())
    print(f"  guidance2.parameters() count: {guidance2_param_count:,}")
    
    # Check overlap
    guidance2_param_ids = set(id(p) for p in guidance2.parameters())
    logicnet2_param_ids = set(id(p) for p in logic_net2.parameters())
    overlap2 = len(guidance2_param_ids & logicnet2_param_ids)
    print(f"  Parameters overlap: {overlap2:,} / {logic_net_param_count:,}")
    
    print("\n[*] Test 3: Check requires_grad status")
    
    print(f"  LogicNet requires_grad:")
    for i, param in enumerate(logic_net.parameters()):
        if i < 3:  # Show first 3
            print(f"    Param {i}: {param.requires_grad}")
        elif i == 3:
            print(f"    ... ({len(list(logic_net.parameters()))} total parameters)")
            break
    
    trainable_count = sum(1 for p in logic_net.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_count} / {len(list(logic_net.parameters()))}")
    
    print("\n[!] VERDICT:")
    print(f"  Test 1 (assign after): {overlap:,} / {logic_net_param_count:,} parameters registered")
    print(f"  Test 2 (pass at creation): {overlap2:,} / {logic_net_param_count:,} parameters registered")
    
    if overlap == logic_net_param_count:
        print("  ✓ Assigning LogicNet after creation works fine")
    else:
        print(f"  ✗ PROBLEM: Only {overlap/logic_net_param_count*100:.2f}% of LogicNet registered when assigned after!")

if __name__ == "__main__":
    main()
