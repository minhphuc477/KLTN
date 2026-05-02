#!/usr/bin/env python
"""Regenerate Block I descriptor scatter with improved sizing."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import extract_graph_descriptor, generate_block_i_graphs, load_vglc_reference_graphs

def main():
    # Generate Block I graphs
    generated, _ = generate_block_i_graphs(
        num_samples=9,
        seed=42,
        population_size=24,
        generations=24,
        min_rooms=8,
        max_rooms=16,
        rule_space="full",
        search_strategy="ga",
    )
    
    # Load reference graphs
    references = load_vglc_reference_graphs(limit=18)
    
    # Extract descriptors
    gen_desc = [extract_graph_descriptor(g, grammar=None) for g in generated]
    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in references]
    
    # Create enlarged, improved scatter figure
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=220)
    ax0, ax1 = axes
    
    # Linearity vs Leniency
    ax0.scatter(
        [d.linearity for d in ref_desc],
        [d.leniency for d in ref_desc],
        c="#2563eb",
        label="Reference",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax0.scatter(
        [d.linearity for d in gen_desc],
        [d.leniency for d in gen_desc],
        c="#dc2626",
        label="Generated",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax0.set_title("Linearity vs Leniency", fontsize=12, fontweight="bold", pad=10)
    ax0.set_xlim(-0.05, 1.05)
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlabel("Linearity", fontsize=11)
    ax0.set_ylabel("Leniency", fontsize=11)
    ax0.grid(alpha=0.25, linestyle="--")
    ax0.legend(loc="best", fontsize=10, framealpha=0.95)
    
    # Progression vs Topology Complexity
    ax1.scatter(
        [d.progression_complexity for d in ref_desc],
        [d.topology_complexity for d in ref_desc],
        c="#2563eb",
        label="Reference",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.scatter(
        [d.progression_complexity for d in gen_desc],
        [d.topology_complexity for d in gen_desc],
        c="#dc2626",
        label="Generated",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.set_title("Progression vs Topology Complexity", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Progression Complexity", fontsize=11)
    ax1.set_ylabel("Topology Complexity", fontsize=11)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="best", fontsize=10, framealpha=0.95)
    
    fig.suptitle("Block I Descriptor Distribution: Generated vs Reference", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    
    out_path = ROOT / "REPORT_LATEX" / "figures" / "ch4" / "real_block_i" / "descriptor_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    
    print(f"✓ Regenerated: {out_path}")
    print(f"  Generated graphs: {len(generated)}")
    print(f"  Reference graphs: {len(references)}")

if __name__ == "__main__":
    main()
