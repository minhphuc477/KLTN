"""Utilities for saving graph-to-grid attention maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch


def set_graph_attention_capture(module: torch.nn.Module, enabled: bool = True) -> int:
    """Enable attention capture on all submodules that expose set_attention_capture."""
    count = 0
    for child in module.modules():
        setter = getattr(child, "set_attention_capture", None)
        if callable(setter):
            setter(bool(enabled))
            count += 1
    return count


def collect_graph_attention_maps(module: torch.nn.Module, *, reduce_heads: str = "mean") -> Dict[str, torch.Tensor]:
    """Collect latest captured attention maps from named submodules."""
    maps: Dict[str, torch.Tensor] = {}
    for name, child in module.named_modules():
        getter = getattr(child, "get_last_attention_map", None)
        if not callable(getter):
            continue
        attention = getter(reduce_heads=reduce_heads)
        if isinstance(attention, torch.Tensor):
            maps[name or "root"] = attention.detach().cpu()
    return maps


def save_attention_map_images(
    attention: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    *,
    prefix: str = "attention",
    node_labels: Optional[Sequence[str]] = None,
    max_nodes: int = 16,
) -> Dict[str, Any]:
    """Save [B,H,W,N] or [B,heads,H,W,N] attention maps as PNG heatmaps plus NPY."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array = attention.detach().cpu().numpy() if isinstance(attention, torch.Tensor) else np.asarray(attention)
    if array.ndim == 5:
        array = array.mean(axis=1)
    if array.ndim != 4:
        raise ValueError(f"attention must have shape [B,H,W,N] or [B,heads,H,W,N], got {array.shape}.")

    npy_path = out_dir / f"{prefix}.npy"
    np.save(npy_path, array)

    saved_pngs = []
    try:
        import matplotlib.pyplot as plt

        batch = 0
        num_nodes = int(min(array.shape[-1], max(1, int(max_nodes))))
        for node_idx in range(num_nodes):
            label = (
                str(node_labels[node_idx])
                if node_labels is not None and node_idx < len(node_labels)
                else f"node_{node_idx}"
            )
            safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)[:80]
            png_path = out_dir / f"{prefix}_{safe_label}.png"
            fig = plt.figure(figsize=(4, 4))
            try:
                plt.imshow(array[batch, :, :, node_idx], cmap="viridis", interpolation="nearest")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(label)
                plt.tight_layout()
                plt.savefig(png_path, dpi=160)
            finally:
                plt.close(fig)
            saved_pngs.append(str(png_path))
    except ImportError:
        saved_pngs = []

    return {
        "npy": str(npy_path),
        "pngs": saved_pngs,
        "shape": tuple(int(v) for v in array.shape),
    }


__all__ = [
    "set_graph_attention_capture",
    "collect_graph_attention_maps",
    "save_attention_map_images",
]
