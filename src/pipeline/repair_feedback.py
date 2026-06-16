"""Helper utilities for LogicNet-guided and boundary-constrained diffusion inpainting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _maybe_transpose_spatial_to_target(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Transpose [H,W] axes when the source is the exact reversed target shape."""
    if tensor.dim() < 2:
        return tensor
    source_h, source_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    if source_h == int(target_w) and source_w == int(target_h) and source_h != source_w:
        return tensor.transpose(-2, -1).contiguous()
    return tensor


def _mask_aspect_error(mask_h: int, mask_w: int, latent_h: int, latent_w: int) -> float:
    if min(mask_h, mask_w, latent_h, latent_w) <= 0:
        return float("inf")
    return abs((float(mask_h) / float(mask_w)) - (float(latent_h) / float(latent_w)))


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

    def _align_neighbor(lat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Normalize neighbor latent to [B, C, H, W] for boundary copying.

        Accepts:
        - Exact shape match
        - Batch-1 neighbor broadcast to base batch size
        - Spatial mismatch resolved via nearest resize
        """
        if not isinstance(lat, torch.Tensor) or lat.dim() != 4:
            return None

        n = lat.to(device=ref.device, dtype=ref.dtype)
        if int(n.shape[1]) != int(_ch):
            return None

        if int(n.shape[0]) == int(bsz):
            pass
        elif int(n.shape[0]) == 1 and int(bsz) > 1:
            n = n.expand(bsz, -1, -1, -1)
        else:
            return None

        n = _maybe_transpose_spatial_to_target(n, h, w)
        if int(n.shape[2]) != int(h) or int(n.shape[3]) != int(w):
            n = F.interpolate(n, size=(h, w), mode="nearest")

        return n

    north = neighbor_latents.get("N")
    n = _align_neighbor(north)
    if n is not None:
        ref[:, :, :edge_band_h, :] = n[:, :, -edge_band_h:, :]
        edit_mask[:, :, :edge_band_h, :] = 0.0
        constrained = True

    south = neighbor_latents.get("S")
    s = _align_neighbor(south)
    if s is not None:
        ref[:, :, -edge_band_h:, :] = s[:, :, :edge_band_h, :]
        edit_mask[:, :, -edge_band_h:, :] = 0.0
        constrained = True

    west = neighbor_latents.get("W")
    w_lat = _align_neighbor(west)
    if w_lat is not None:
        ref[:, :, :, :edge_band_w] = w_lat[:, :, :, -edge_band_w:]
        edit_mask[:, :, :, :edge_band_w] = 0.0
        constrained = True

    east = neighbor_latents.get("E")
    e = _align_neighbor(east)
    if e is not None:
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

    direct_error = _mask_aspect_error(mask.shape[0], mask.shape[1], latent_h, latent_w)
    transposed_error = _mask_aspect_error(mask.shape[1], mask.shape[0], latent_h, latent_w)
    if transposed_error + 1e-6 < direct_error:
        mask = np.ascontiguousarray(mask.T)

    mask_t = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0)
    k_h = max(1, int(np.ceil(mask.shape[0] / max(1, latent_h))))
    k_w = max(1, int(np.ceil(mask.shape[1] / max(1, latent_w))))
    pooled = F.max_pool2d(mask_t, kernel_size=(k_h, k_w), stride=(k_h, k_w), ceil_mode=True)
    resized = F.interpolate(pooled, size=(latent_h, latent_w), mode="nearest")
    return torch.clamp(resized, 0.0, 1.0)


@torch.no_grad()
def logicnet_guided_inpaint_room(
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
    noise_strength: float = 0.5,
    guidance_scale_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Regenerate contradiction regions with latent diffusion inpainting.

    The symbolic repair stage supplies the dead-end mask. The neural fallback
    encodes the partially repaired room, runs diffusion.inpaint() on the masked
    latent cells, and lets the diffusion model's configured LogicNet guidance
    shape the regenerated region.
    """
    grid_int = np.asarray(current_grid, dtype=np.int64)
    if grid_int.ndim != 2:
        raise ValueError(f"current_grid must be 2D, got shape={tuple(grid_int.shape)}")

    mask_bool = np.asarray(dead_end_mask, dtype=bool)
    if mask_bool.shape != grid_int.shape:
        raise ValueError(
            f"dead_end_mask shape {tuple(mask_bool.shape)} must match room grid shape {tuple(grid_int.shape)}"
        )
    if not bool(np.any(mask_bool)):
        # No repair region requested.
        return grid_int.astype(np.int32, copy=False)

    grid_int = np.clip(grid_int, 0, int(num_classes) - 1)
    one_hot = np.eye(int(num_classes), dtype=np.float32)[grid_int]
    x_0 = torch.from_numpy(one_hot).to(device).permute(2, 0, 1).unsqueeze(0).contiguous()

    with torch.no_grad():
        z_0, _ = vqvae.encode(x_0)
    latent_h, latent_w = int(z_0.shape[2]), int(z_0.shape[3])
    latent_mask = build_latent_edit_mask(
        mask_bool,
        latent_h=latent_h,
        latent_w=latent_w,
        device=device,
    )

    guidance_module = getattr(diffusion, "guidance", None)
    old_guidance_scale = getattr(guidance_module, "guidance_scale", None)
    if old_guidance_scale is not None:
        guidance_module.guidance_scale = float(old_guidance_scale) * float(max(0.0, guidance_scale_multiplier))
    try:
        rng_devices = []
        if device.type == "cuda":
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            if seed is not None:
                torch.manual_seed(int(seed))
            z_inpaint = diffusion.inpaint(
                x_0=z_0,
                mask=latent_mask,
                context=condition,
                graph_data=graph_data,
                num_steps=max(8, int(num_diffusion_steps)),
                noise_strength=float(max(0.0, min(1.0, noise_strength))),
            )
    finally:
        if old_guidance_scale is not None:
            guidance_module.guidance_scale = old_guidance_scale
    with torch.no_grad():
        logits = vqvae.decode(z_inpaint)
    inpainted_grid = logits.argmax(dim=1).detach().cpu().numpy()[0]

    keep = ~mask_bool
    merged = inpainted_grid.copy()
    merged[keep] = grid_int[keep]
    return merged.astype(np.int32, copy=False)


def wfc_guided_inpaint_room(*args: Any, **kwargs: Any) -> np.ndarray:
    """Backward-compatible alias for logicnet_guided_inpaint_room."""
    return logicnet_guided_inpaint_room(*args, **kwargs)
