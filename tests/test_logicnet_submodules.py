#!/usr/bin/env python
"""Analyze which submodules of LogicNet are being registered."""

import torch
from src.core.logic_net import LogicNet

def main():
    logic_net = LogicNet(latent_dim=64, num_classes=44)
    
    print("[*] LogicNet Submodule Analysis")
    print(f"    Total parameters: {sum(p.numel() for p in logic_net.parameters()):,}")
    print(f"    Trainable parameters: {sum(p.numel() for p in logic_net.parameters() if p.requires_grad):,}")
    
    print("\n[*] Registered submodules (_modules):")
    for name, module in logic_net._modules.items():
        if module is None:
            print(f"    {name}: None")
        else:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"    {name}: {param_count:,} parameters")
    
    print("\n[*] Registered buffers (_buffers):")
    for name, buf in logic_net._buffers.items():
        if buf is None:
            print(f"    {name}: None")
        else:
            print(f"    {name}: {buf.numel():,} elements ({buf.shape})")
    
    print("\n[*] Direct attributes (checking for incorrectly stored modules):")
    for name in dir(logic_net):
        if name.startswith('_') or name in logic_net._modules:
            continue
        attr = getattr(logic_net, name, None)
        if isinstance(attr, torch.nn.Module):
            param_count = sum(p.numel() for p in attr.parameters())
            print(f"    WARNING: {name} is an nn.Module but not in _modules! {param_count:,} params")

if __name__ == "__main__":
    main()
