#!/usr/bin/env python
"""
Quick diagnostic: Test if LogicNet guidance is being applied during diffusion sampling.

This script:
1. Creates a minimal diffusion model + LogicNet setup
2. Runs a few denoising steps with guidance enabled/disabled
3. Logs all guidance computation to verify it's actually happening
"""

import os
import torch
import numpy as np
import logging
from pathlib import Path

# Enable debug logging
os.environ['KLTN_DEBUG_GUIDANCE'] = '1'

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(name)s: %(message)s',
)

from src.core.latent_diffusion import create_latent_diffusion
from src.core.logic_net import LogicNet
from src.core.condition_encoder import create_condition_encoder

def _run_guidance_application() -> bool:
    """Test if LogicNet guidance is applied during sampling."""
    
    print("\n" + "="*70)
    print("TESTING LOGICNET GUIDANCE APPLICATION")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create minimal components
    print("\n[1] Creating LogicNet...")
    logic_net = LogicNet(latent_dim=64, num_classes=44).to(device)
    logic_net.eval()
    print(f"    ✓ LogicNet created with {sum(p.numel() for p in logic_net.parameters()):,} parameters")
    
    print("\n[2] Creating Latent Diffusion Model...")
    diffusion = create_latent_diffusion(
        latent_dim=64,
        model_channels=64,
        context_dim=128,
        num_timesteps=100,  # Use small number for speed
    ).to(device)
    diffusion.eval()
    
    # Wire LogicNet into guidance
    diffusion.guidance.logic_net = logic_net
    print(f"    ✓ Diffusion model created")
    print(f"    ✓ LogicNet wired into diffusion.guidance")
    
    print("\n[3] Creating mock conditioning and graph data...")
    batch_size = 2
    context = torch.randn(batch_size, 128, device=device)
    graph_data = {
        'node_features': torch.randn(8, 16, device=device),
        'edge_index': torch.randint(0, 8, (2, 12), device=device),
    }
    print(f"    ✓ Context shape: {context.shape}")
    print(f"    ✓ Graph nodes: {graph_data['node_features'].shape}")
    
    print("\n[4] Testing guidance computation...")
    
    # Test WITH guidance (guidance_scale=1.0)
    diffusion.guidance.guidance_scale = 1.0
    print(f"    Testing with guidance_scale={diffusion.guidance.guidance_scale}")
    
    x_t = torch.randn(batch_size, 64, 16, 16, device=device)
    t = 50
    
    with torch.no_grad():
        guidance_grad = diffusion.guidance.compute_guidance(
            x_t,
            graph_data,
            t=t,
            num_timesteps=100,
        )
    
    guidance_norm = guidance_grad.view(batch_size, -1).norm(dim=1)
    print(f"    Guidance norms: {guidance_norm.tolist()}")
    print(f"    Any nonzero guidance? {(guidance_norm > 1e-8).any().item()}")
    
    # Test WITHOUT guidance (guidance_scale=0.0)
    diffusion.guidance.guidance_scale = 0.0
    print(f"\n    Testing with guidance_scale={diffusion.guidance.guidance_scale}")
    
    with torch.no_grad():
        guidance_grad_zero = diffusion.guidance.compute_guidance(
            x_t,
            graph_data,
            t=t,
            num_timesteps=100,
        )
    
    guidance_norm_zero = guidance_grad_zero.view(batch_size, -1).norm(dim=1)
    print(f"    Guidance norms: {guidance_norm_zero.tolist()}")
    
    print("\n[5] VERIFICATION")
    print(f"    ✓ guidance_scale=1.0 produces nonzero gradients: {(guidance_norm > 1e-8).any().item()}")
    print(f"    ✓ guidance_scale=0.0 produces zero gradients: {(guidance_norm_zero < 1e-8).all().item()}")
    
    if (guidance_norm > 1e-8).any() and (guidance_norm_zero < 1e-8).all():
        print("\n    [✓] PASS: Guidance is being applied correctly!")
        return True
    else:
        print("\n    [✗] FAIL: Guidance is not being applied as expected!")
        if not (guidance_norm > 1e-8).any():
            print("        - guidance_scale=1.0 produces zero gradients (guidance broken)")
        if not (guidance_norm_zero < 1e-8).all():
            print("        - guidance_scale=0.0 doesn't produce zero gradients (scale not working)")
        return False

def test_guidance_application():
    assert _run_guidance_application() is True


def _run_full_pipeline_guidance() -> bool:
    """Test guidance through a full ddim_sample call."""
    
    print("\n" + "="*70)
    print("TESTING FULL PIPELINE GUIDANCE")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n[1] Creating pipeline components...")
    logic_net = LogicNet(latent_dim=64, num_classes=44).to(device)
    logic_net.eval()
    
    diffusion = create_latent_diffusion(
        latent_dim=64,
        model_channels=64,
        context_dim=128,
        num_timesteps=10,  # Very short for speed
    ).to(device)
    diffusion.eval()
    diffusion.guidance.logic_net = logic_net
    
    condition_encoder = create_condition_encoder(
        latent_dim=64,
        output_dim=128,
    ).to(device)
    condition_encoder.eval()
    
    print("    ✓ Created logic_net, diffusion, condition_encoder")
    
    print("\n[2] Running DDIM sample with guidance...")
    
    x_t = torch.randn(1, 64, 16, 16, device=device)
    context = torch.randn(1, 128, device=device)
    graph_data = {
        'node_features': torch.randn(4, 16, device=device),
        'edge_index': torch.randint(0, 4, (2, 6), device=device),
    }
    
    diffusion.guidance.guidance_scale = 1.0
    num_steps = 3  # Very few steps for speed
    
    try:
        with torch.no_grad():
            x_0 = diffusion.ddim_sample(
                context=context,
                shape=(1, 64, 16, 16),
                num_steps=num_steps,
                eta=1.0,
                graph_data=graph_data,
            )
        print(f"    ✓ DDIM sample completed successfully")
        print(f"    Output shape: {x_0.shape}")
        return True
    except Exception as e:
        print(f"    ✗ DDIM sample failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_guidance():
    assert _run_full_pipeline_guidance() is True


if __name__ == "__main__":
    test1_pass = _run_guidance_application()
    test2_pass = _run_full_pipeline_guidance()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Guidance application test: {'PASS' if test1_pass else 'FAIL'}")
    print(f"Full pipeline test: {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n[✓] Guidance system is working correctly!")
    else:
        print("\n[✗] Guidance system has issues - see logs above")
