"""LogicNet-guided symbolic repair coupling.

The symbolic refiner remains deterministic and ablation-friendly, but it no
longer has to repair independently from the neural model. LogicNet supplies a
soft walkability field for repair costs and optional topology masks for hard
floor constraints.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.core.symbolic_refiner import SymbolicRefiner


@dataclass
class NeuralRepairGuidance:
    """Structured LogicNet outputs consumed by symbolic repair."""

    walkability: np.ndarray
    cost_map: Optional[np.ndarray]
    required_floor_mask: Optional[np.ndarray]
    logic_loss: float
    logic_info: Dict[str, Any]


class NeuralGuidedRepair:
    """
    Coupling layer between LogicNet and SymbolicRefiner.

    M1: LogicNet walkability -> symbolic repair cost map.
    M2: LogicNet topology/anchor targets -> hard floor mask.
    M3: optional neural feedback callback for WFC contradictions. When enabled,
    contradiction masks are sent to diffusion inpainting, then WFC continues
    from the neural patch.
    """

    def __init__(
        self,
        logic_net: Any,
        refiner: SymbolicRefiner,
        *,
        use_logicnet_cost: bool = True,
        use_logicnet_floor_mask: bool = True,
        use_neural_feedback: bool = False,
        obstacle_weight: float = 5.0,
        trace_threshold: float = 0.5,
        anchor_threshold: float = 0.3,
        transpose_logicnet_masks: bool = False,
        repair_inpaint_noise_strength: float = 0.5,
        repair_inpaint_guidance_scale_multiplier: float = 1.0,
    ):
        self.logic_net = logic_net
        self.refiner = refiner
        self.use_logicnet_cost = bool(use_logicnet_cost)
        self.use_logicnet_floor_mask = bool(use_logicnet_floor_mask)
        self.use_neural_feedback = bool(use_neural_feedback)
        self.obstacle_weight = float(max(0.0, obstacle_weight))
        self.trace_threshold = float(trace_threshold)
        self.anchor_threshold = float(anchor_threshold)
        self.transpose_logicnet_masks = bool(transpose_logicnet_masks)
        self.repair_inpaint_noise_strength = float(max(0.0, min(1.0, repair_inpaint_noise_strength)))
        self.repair_inpaint_guidance_scale_multiplier = float(max(0.0, repair_inpaint_guidance_scale_multiplier))

    @torch.no_grad()
    def get_logicnet_guidance(
        self,
        tile_logits: torch.Tensor,
        *,
        grid_shape: Tuple[int, int],
        graph_data: Optional[dict] = None,
    ) -> NeuralRepairGuidance:
        """Run LogicNet and convert outputs to row/col numpy repair tensors."""
        if not isinstance(tile_logits, torch.Tensor):
            raise TypeError(f"tile_logits must be a torch.Tensor, got {type(tile_logits).__name__}.")
        if tile_logits.dim() != 4:
            raise ValueError(f"tile_logits must have shape [B,C,H,W], got {tuple(tile_logits.shape)}.")
        if int(tile_logits.shape[0]) != 1:
            raise ValueError("NeuralGuidedRepair currently repairs one room at a time; expected batch size 1.")

        was_training = bool(getattr(self.logic_net, "training", False))
        try:
            self.logic_net.eval()
            loss, info = self.logic_net(tile_logits, graph_data=graph_data)
            walkability_t = info.get("walkability")
            if not isinstance(walkability_t, torch.Tensor):
                raise RuntimeError("LogicNet did not return info['walkability']; cannot guide repair.")
            walkability_t = self._resize_mask(
                walkability_t,
                grid_shape,
                transpose_mask=self.transpose_logicnet_masks,
            ).clamp(0.0, 1.0)
            walkability = walkability_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

            cost_map = None
            if self.use_logicnet_cost:
                cost_map = (1.0 + (1.0 - walkability) * self.obstacle_weight).astype(np.float32, copy=False)

            floor_mask = None
            if self.use_logicnet_floor_mask:
                floor_mask = self._build_floor_mask(tile_logits, graph_data, grid_shape)

            return NeuralRepairGuidance(
                walkability=walkability,
                cost_map=cost_map,
                required_floor_mask=floor_mask,
                logic_loss=float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss),
                logic_info=info,
            )
        finally:
            if was_training:
                self.logic_net.train()

    def repair_room_with_neural_guidance(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        tile_logits: torch.Tensor,
        *,
        graph_data: Optional[dict] = None,
        required_floor_mask: Optional[np.ndarray] = None,
        inpaint_callback: Optional[Any] = None,
        inpaint_context: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 16,
        seed: Optional[int] = None,
        feedback_callback: Optional[Any] = None,
        max_feedback_rounds: int = 0,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Repair a room using LogicNet-derived cost and topology constraints."""
        grid_array = np.asarray(grid)
        guidance = self.get_logicnet_guidance(
            tile_logits,
            grid_shape=tuple(int(v) for v in grid_array.shape[:2]),
            graph_data=graph_data,
        )
        floor_mask = guidance.required_floor_mask
        if isinstance(required_floor_mask, np.ndarray):
            external_mask = np.asarray(required_floor_mask, dtype=bool)
            if tuple(external_mask.shape[:2]) == tuple(grid_array.shape[:2]):
                floor_mask = external_mask if floor_mask is None else (floor_mask | external_mask)
        neural_feedback_callback = self._build_neural_callback(
            inpaint_callback=inpaint_callback,
            condition=inpaint_context,
            graph_data=graph_data,
            num_diffusion_steps=num_diffusion_steps,
            seed=seed,
        )
        if neural_feedback_callback is None:
            neural_feedback_callback = feedback_callback
        effective_feedback_rounds = int(max_feedback_rounds)
        if neural_feedback_callback is not None and effective_feedback_rounds <= 0:
            effective_feedback_rounds = 2
        repaired, success, diagnostics = self.refiner.repair_room_with_feedback(
            grid=grid_array,
            start=start,
            goal=goal,
            feedback_callback=neural_feedback_callback,
            max_feedback_rounds=max(0, effective_feedback_rounds),
            required_floor_mask=floor_mask,
            cost_map=guidance.cost_map,
            seed=seed,
        )
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "neural_guidance_used": True,
                "neural_feedback_enabled": bool(self.use_neural_feedback),
                "neural_feedback_callback_used": bool(neural_feedback_callback is not None),
                "logicnet_loss_before_repair": float(guidance.logic_loss),
                "logicnet_cost_used": bool(guidance.cost_map is not None),
                "logicnet_floor_mask_pixels": (
                    int(np.sum(floor_mask))
                    if isinstance(floor_mask, np.ndarray)
                    else 0
                ),
            }
        )
        return repaired, bool(success), diagnostics

    def _build_neural_callback(
        self,
        *,
        inpaint_callback: Optional[Any],
        condition: Optional[torch.Tensor],
        graph_data: Optional[dict],
        num_diffusion_steps: int,
        seed: Optional[int],
    ) -> Optional[Any]:
        if not self.use_neural_feedback or not callable(inpaint_callback) or not isinstance(condition, torch.Tensor):
            return None

        def _callback(
            current_grid: np.ndarray,
            dead_end_mask: np.ndarray,
            _start: Tuple[int, int],
            _goal: Tuple[int, int],
            attempt_idx: int,
        ) -> np.ndarray:
            return inpaint_callback(
                current_grid=current_grid,
                dead_end_mask=dead_end_mask,
                condition=condition,
                graph_data=graph_data,
                num_diffusion_steps=max(8, int(num_diffusion_steps)),
                seed=(None if seed is None else int(seed) + 1000 + int(attempt_idx)),
                noise_strength=self.repair_inpaint_noise_strength,
                guidance_scale_multiplier=self.repair_inpaint_guidance_scale_multiplier,
            )

        return _callback

    def _build_floor_mask(
        self,
        tile_logits: torch.Tensor,
        graph_data: Optional[dict],
        grid_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if graph_data is None:
            return None
        resolver = getattr(self.logic_net, "_resolve_room_logic_targets", None)
        if not callable(resolver):
            return None
        resolver_kwargs = {
            "batch_size": int(tile_logits.shape[0]),
            "device": tile_logits.device,
            "dtype": tile_logits.dtype,
        }
        try:
            if "spatial_hw" in inspect.signature(resolver).parameters:
                resolver_kwargs["spatial_hw"] = (
                    int(grid_shape[0]),
                    int(grid_shape[1]),
                )
        except (TypeError, ValueError):
            pass
        targets = resolver(graph_data, **resolver_kwargs)
        masks = []
        trace = targets.get("trace_target") if isinstance(targets, dict) else None
        if isinstance(trace, torch.Tensor):
            masks.append(
                self._resize_mask(
                    trace,
                    grid_shape,
                    transpose_mask=self.transpose_logicnet_masks,
                ) > self.trace_threshold
            )
        anchor = targets.get("anchor_target") if isinstance(targets, dict) else None
        if isinstance(anchor, torch.Tensor):
            masks.append(
                self._resize_mask(
                    anchor,
                    grid_shape,
                    transpose_mask=self.transpose_logicnet_masks,
                ) > self.anchor_threshold
            )
        if not masks:
            return None
        stacked = torch.stack([mask[0, 0] for mask in masks], dim=0).any(dim=0)
        return stacked.detach().cpu().numpy().astype(bool, copy=False)

    @staticmethod
    def _resize_mask(
        mask: torch.Tensor,
        grid_shape: Tuple[int, int],
        *,
        transpose_mask: bool = False,
    ) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4 or int(mask.shape[1]) != 1:
            raise ValueError(f"Expected mask [B,1,H,W], got {tuple(mask.shape)}.")
        target_h, target_w = int(grid_shape[0]), int(grid_shape[1])
        if transpose_mask:
            mask = mask.transpose(-1, -2).contiguous()
        if tuple(mask.shape[-2:]) == (target_h, target_w):
            return mask
        if target_h != target_w and tuple(mask.shape[-2:]) == (target_w, target_h):
            return mask.transpose(-1, -2).contiguous()
        return F.interpolate(mask.float(), size=(target_h, target_w), mode="nearest")


__all__ = ["NeuralGuidedRepair", "NeuralRepairGuidance"]
