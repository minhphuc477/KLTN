"""Helper utilities for WFC-guided and boundary-constrained diffusion inpainting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_neighbor_boundary_inpaint_inputs(
    base_latent: torch.Tensor,
    neighbor_latents: Dict[str, Optional[torch.Tensor]],
    *,
    band: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Build latent reference + edit mask to hard-preserve shared boundaries.

    Args:
        base_latent: Current room latent [B, C, H, W]
        neighbor_latents: Directional neighbor latents {'N','S','E','W'}
        band: Number of latent cells to preserve per edge

    Returns:
        reference_latent: [B, C, H, W] with constrained edges copied from neighbors
        edit_mask: [B, 1, H, W], 1=editable, 0=preserve
        has_constraints: Whether any neighbor constraint was applied
    """
    if base_latent.dim() != 4:
        raise ValueError("base_latent must be [B, C, H, W]")

    ref = base_latent.clone()
    bsz, _ch, h, w = ref.shape
    edge_band_h = max(1, min(int(band), h))
    edge_band_w = max(1, min(int(band), w))

    edit_mask = torch.ones((bsz, 1, h, w), device=ref.device, dtype=ref.dtype)
    constrained = False

    def _valid_neighbor(lat: Optional[torch.Tensor]) -> bool:
        return isinstance(lat, torch.Tensor) and lat.dim() == 4 and tuple(lat.shape) == tuple(ref.shape)

    north = neighbor_latents.get("N")
    if _valid_neighbor(north):
        n = north.to(device=ref.device, dtype=ref.dtype)
        ref[:, :, :edge_band_h, :] = n[:, :, -edge_band_h:, :]
        edit_mask[:, :, :edge_band_h, :] = 0.0
        constrained = True

    south = neighbor_latents.get("S")
    if _valid_neighbor(south):
        s = south.to(device=ref.device, dtype=ref.dtype)
        ref[:, :, -edge_band_h:, :] = s[:, :, :edge_band_h, :]
        edit_mask[:, :, -edge_band_h:, :] = 0.0
        constrained = True

    west = neighbor_latents.get("W")
    if _valid_neighbor(west):
        w_lat = west.to(device=ref.device, dtype=ref.dtype)
        ref[:, :, :, :edge_band_w] = w_lat[:, :, :, -edge_band_w:]
        edit_mask[:, :, :, :edge_band_w] = 0.0
        constrained = True

    east = neighbor_latents.get("E")
    if _valid_neighbor(east):
        e = east.to(device=ref.device, dtype=ref.dtype)
        ref[:, :, :, -edge_band_w:] = e[:, :, :, :edge_band_w]
        edit_mask[:, :, :, -edge_band_w:] = 0.0
        constrained = True

    return ref, edit_mask, constrained


def build_latent_edit_mask(
    room_mask: np.ndarray,
    *,
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Downsample a room-space boolean mask to latent-space with max pooling."""
    mask = np.asarray(room_mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError("room_mask must be 2D")

    mask_t = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0)
    k_h = max(1, int(np.ceil(mask.shape[0] / max(1, latent_h))))
    k_w = max(1, int(np.ceil(mask.shape[1] / max(1, latent_w))))
    pooled = F.max_pool2d(mask_t, kernel_size=(k_h, k_w), stride=(k_h, k_w), ceil_mode=True)
    resized = F.interpolate(pooled, size=(latent_h, latent_w), mode="nearest")
    return torch.clamp(resized, 0.0, 1.0)


@torch.no_grad()
def wfc_guided_inpaint_room(
    current_grid: np.ndarray,
    dead_end_mask: np.ndarray,
    condition: torch.Tensor,
    graph_data: Optional[Dict[str, Any]],
    num_diffusion_steps: int,
    *,
    seed: Optional[int],
    device: torch.device,
    vqvae: Any,
    diffusion: Any,
    num_classes: int = 44,
) -> np.ndarray:
    """Regenerate only dead-end regions in latent space and merge back."""
    grid_int = np.asarray(current_grid, dtype=np.int64)
    grid_int = np.clip(grid_int, 0, int(num_classes) - 1)
    one_hot = np.eye(int(num_classes), dtype=np.float32)[grid_int]
    x_0 = torch.from_numpy(one_hot).to(device).permute(2, 0, 1).unsqueeze(0).contiguous()

    z_0, _ = vqvae.encode(x_0)
    latent_h, latent_w = int(z_0.shape[2]), int(z_0.shape[3])
    latent_mask = build_latent_edit_mask(
        dead_end_mask,
        latent_h=latent_h,
        latent_w=latent_w,
        device=device,
    )

    if seed is not None:
        torch.manual_seed(int(seed))

    z_inpaint = diffusion.inpaint(
        x_0=z_0,
        mask=latent_mask,
        context=condition,
        graph_data=graph_data,
        num_steps=max(8, int(num_diffusion_steps)),
    )
    logits = vqvae.decode(z_inpaint)
    inpainted_grid = logits.argmax(dim=1).detach().cpu().numpy()[0]

    keep = ~np.asarray(dead_end_mask, dtype=bool)
    merged = inpainted_grid.copy()
    merged[keep] = grid_int[keep]
    return merged.astype(np.int32, copy=False)
