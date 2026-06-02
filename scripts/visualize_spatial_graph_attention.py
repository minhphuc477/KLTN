"""Save graph-to-grid attention heatmaps from a captured attention NPY file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.attention_visualization import save_attention_map_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize captured graph-to-grid attention maps.")
    parser.add_argument("--attention-npy", required=True, type=Path, help="NPY with [B,H,W,N] or [B,heads,H,W,N].")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="attention")
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--node-labels", type=str, default="", help="Comma-separated labels for graph nodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention = np.load(args.attention_npy)
    labels = [part.strip() for part in args.node_labels.split(",") if part.strip()] or None
    payload = save_attention_map_images(
        attention,
        args.output_dir,
        prefix=args.prefix,
        node_labels=labels,
        max_nodes=args.max_nodes,
    )
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
