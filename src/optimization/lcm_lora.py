"""
Fast sampling via consistency-distilled LoRA adapters.

Important terminology note:

This module does not implement paper-faithful LCM-LoRA inference. In the LCM
and LCM-LoRA papers, the accelerated model is distilled with the latent
consistency objective and used with LCM-specific sampling semantics. This repo's
current fast path is narrower:

- it distills LoRA adapters against the repo's existing latent diffusion model
- it preserves the graph-aware conditioning path already used by the project
- it reuses the project's DDIM sampler at inference time

So this is a graph-aware consistency-LoRA fast sampler for this codebase, not a
drop-in implementation of the published Stable Diffusion LCM-LoRA stack.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from src.pipeline.room_topology_conditioning import build_topology_anchor_policy_metadata
from src.utils.checkpoint import atomic_torch_save, log_checkpoint_artifact, write_checkpoint_metadata

logger = logging.getLogger(__name__)


DEFAULT_LORA_TARGETS = (
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.proj",
    "graph_cross_attn.q_proj",
    "graph_cross_attn.k_proj",
    "graph_cross_attn.v_proj",
    "graph_cross_attn.out_proj",
)


@dataclass
class FastSamplerCheckpointInfo:
    checkpoint_path: str
    distillation_type: str
    base_diffusion_checkpoint: Optional[str]
    num_inference_steps: int
    lora_rank: int
    lora_alpha: float
    target_modules: Tuple[str, ...]


@dataclass
class PerformanceMetrics:
    sampling_strategy: str
    num_steps: int
    generation_time: float
    throughput: float
    quality_score: Optional[float] = None


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.rank = int(max(1, rank))
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(self.rank)
        self.lora_A = nn.Parameter(
            torch.randn(in_features, self.rank, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.rank, out_features, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.dropout(x) @ self.lora_A) @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora(x)


def _iter_target_linear_modules(
    model: nn.Module,
    target_modules: Iterable[str],
):
    targets = tuple(str(t) for t in target_modules)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in targets):
            yield name, module


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    target_modules: Optional[Iterable[str]] = None,
) -> int:
    targets = tuple(target_modules or DEFAULT_LORA_TARGETS)
    params_added = 0
    seen = set()
    for name, module in _iter_target_linear_modules(model, targets):
        if name in seen:
            continue
        seen.add(name)
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        params_added += rank * (module.in_features + module.out_features)
    logger.info("Injected LoRA adapters into %d modules (%d parameters).", len(seen), params_added)
    return params_added


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = ".lora." in name


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora." in key
    }


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> Tuple[list[str], list[str]]:
    incompatible = model.load_state_dict(lora_state_dict, strict=False)
    missing = [k for k in getattr(incompatible, "missing_keys", []) if ".lora." in k]
    unexpected = [k for k in getattr(incompatible, "unexpected_keys", []) if ".lora." in k]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return missing, unexpected


def save_fast_sampler_checkpoint(
    path: str,
    *,
    lora_state_dict: Dict[str, torch.Tensor],
    base_diffusion_checkpoint: Optional[str],
    num_inference_steps: int,
    lora_rank: int,
    lora_alpha: float,
    target_modules: Iterable[str] = DEFAULT_LORA_TARGETS,
    metrics: Optional[Dict[str, Any]] = None,
    distillation_type: str = "consistency_lora",
    ema_decay: Optional[float] = None,
    target_update: Optional[str] = None,
    distillation_objective: Optional[str] = None,
    adapter_export: Optional[str] = None,
    topology_alignment_weight: float = 0.0,
    topology_marker_weight: float = 2.0,
    topology_trace_weight: float = 0.75,
    topology_focus_dilation: int = 1,
    topology_anchor_policy: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "lora_state_dict": dict(lora_state_dict),
        "metadata": {
            "distillation_type": str(distillation_type),
            "base_diffusion_checkpoint": None if base_diffusion_checkpoint is None else str(base_diffusion_checkpoint),
            "num_inference_steps": int(max(1, num_inference_steps)),
            "lora_rank": int(max(1, lora_rank)),
            "lora_alpha": float(lora_alpha),
            "target_modules": [str(t) for t in target_modules],
            "ema_decay": None if ema_decay is None else float(ema_decay),
            "target_update": None if target_update is None else str(target_update),
            "distillation_objective": (
                None if distillation_objective is None else str(distillation_objective)
            ),
            "adapter_export": None if adapter_export is None else str(adapter_export),
            "topology_alignment_weight": float(max(0.0, topology_alignment_weight)),
            "topology_marker_weight": float(max(0.0, topology_marker_weight)),
            "topology_trace_weight": float(max(0.0, topology_trace_weight)),
            "topology_focus_dilation": int(max(0, topology_focus_dilation)),
            "topology_anchor_policy": dict(
                topology_anchor_policy
                or build_topology_anchor_policy_metadata()
            ),
            "metrics": dict(metrics or {}),
        },
    }
    atomic_torch_save(payload, path)
    write_checkpoint_metadata(
        path,
        model_type="fast_sampler_adapter",
        architecture={
            "distillation_type": str(distillation_type),
            "num_inference_steps": int(max(1, num_inference_steps)),
            "lora_rank": int(max(1, lora_rank)),
            "ema_decay": None if ema_decay is None else float(ema_decay),
            "target_update": None if target_update is None else str(target_update),
            "distillation_objective": (
                None if distillation_objective is None else str(distillation_objective)
            ),
            "adapter_export": None if adapter_export is None else str(adapter_export),
            "topology_alignment_weight": float(max(0.0, topology_alignment_weight)),
            "topology_marker_weight": float(max(0.0, topology_marker_weight)),
            "topology_trace_weight": float(max(0.0, topology_trace_weight)),
            "topology_focus_dilation": int(max(0, topology_focus_dilation)),
        },
        extra={
            "base_diffusion_checkpoint": None if base_diffusion_checkpoint is None else str(base_diffusion_checkpoint),
            "target_modules": [str(t) for t in target_modules],
            "topology_anchor_policy": dict(
                topology_anchor_policy
                or build_topology_anchor_policy_metadata()
            ),
        },
    )
    log_checkpoint_artifact(
        logger,
        path,
        checkpoint_dir=Path(path).parent,
        label="Saved fast-sampler adapter checkpoint",
    )


def load_fast_sampler_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], FastSamplerCheckpointInfo]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "lora_state_dict" not in payload or "metadata" not in payload:
        raise ValueError(f"Invalid fast-sampler checkpoint format at {path!r}.")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid fast-sampler metadata at {path!r}.")
    distillation_type = str(metadata.get("distillation_type", "")).strip().lower()
    if distillation_type == "lcm_lora":
        raise ValueError(
            "This runtime does not support paper-faithful LCM-LoRA checkpoints yet. "
            "Expected distillation_type='consistency_lora' for the repo's graph-aware fast sampler."
        )
    if distillation_type != "consistency_lora":
        raise ValueError(
            f"Unsupported fast-sampler distillation_type={distillation_type!r} in {path!r}."
        )
    info = FastSamplerCheckpointInfo(
        checkpoint_path=str(path),
        distillation_type=distillation_type,
        base_diffusion_checkpoint=metadata.get("base_diffusion_checkpoint"),
        num_inference_steps=int(max(1, int(metadata.get("num_inference_steps", 4)))),
        lora_rank=int(max(1, int(metadata.get("lora_rank", 8)))),
        lora_alpha=float(metadata.get("lora_alpha", 8.0)),
        target_modules=tuple(str(t) for t in metadata.get("target_modules", DEFAULT_LORA_TARGETS)),
    )
    return dict(payload["lora_state_dict"]), info


class GraphConditionedFastSampler:
    """
    Graph-aware fast sampler for a LatentDiffusionModel.

    The underlying model must already have a compatible consistency-LoRA adapter.
    Runtime intentionally reuses the full graph-aware DDIM path so room topology
    conditioning, room anchors, and cross-attention semantics stay aligned with
    the main architecture.
    """

    def __init__(
        self,
        diffusion_model: Any,
        *,
        adapter_checkpoint: str,
        num_inference_steps: Optional[int] = None,
        use_fp16: bool = False,
        compile_model: bool = False,
        strict: bool = True,
    ):
        self.diffusion_model = diffusion_model
        self.adapter_checkpoint = str(adapter_checkpoint)
        lora_state_dict, info = load_fast_sampler_checkpoint(self.adapter_checkpoint)
        self.info = info
        base_num_timesteps = int(max(1, int(getattr(self.diffusion_model, "num_timesteps", 1))))
        requested_steps = int(max(1, num_inference_steps or info.num_inference_steps))
        if requested_steps > base_num_timesteps:
            raise ValueError(
                "Fast sampler adapter requests "
                f"{requested_steps} inference steps, but the base diffusion model only defines "
                f"{base_num_timesteps} timesteps."
            )
        self.num_inference_steps = int(max(1, num_inference_steps or info.num_inference_steps))
        if num_inference_steps is not None and int(num_inference_steps) != int(info.num_inference_steps):
            logger.info(
                "Fast sampler overriding adapter num_inference_steps from %d to %d.",
                int(info.num_inference_steps),
                int(self.num_inference_steps),
            )
        if int(self.num_inference_steps) >= base_num_timesteps:
            logger.warning(
                "Fast sampler is configured with %d steps against a %d-step base schedule; "
                "this provides little or no acceleration.",
                int(self.num_inference_steps),
                base_num_timesteps,
            )
        self.use_fp16 = bool(use_fp16 and torch.cuda.is_available())
        self.compile_model = bool(compile_model)

        inject_lora_into_model(
            self.diffusion_model.denoiser,
            rank=info.lora_rank,
            alpha=info.lora_alpha,
            target_modules=info.target_modules,
        )
        missing, unexpected = load_lora_state_dict(
            self.diffusion_model,
            lora_state_dict,
            strict=strict,
        )
        if missing or unexpected:
            logger.warning(
                "Fast sampler adapter loaded with LoRA mismatch: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )

        if self.use_fp16:
            self.diffusion_model = self.diffusion_model.half()
        if self.compile_model:
            try:
                self.diffusion_model = torch.compile(self.diffusion_model, mode="reduce-overhead")
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.warning("torch.compile() unavailable for fast sampler: %s", exc)

    @torch.inference_mode()
    def sample_fast(
        self,
        *,
        context: torch.Tensor,
        latent_shape: Tuple[int, ...],
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(int(seed))
        original_cfg = float(self.diffusion_model.cfg_scale)
        try:
            if guidance_scale is not None:
                self.diffusion_model.cfg_scale = float(guidance_scale)
            return self.diffusion_model.ddim_sample(
                context=context,
                shape=latent_shape,
                num_steps=int(max(1, self.num_inference_steps)),
                graph_data=graph_data,
            )
        finally:
            self.diffusion_model.cfg_scale = original_cfg

    def benchmark(
        self,
        *,
        context: torch.Tensor,
        latent_shape: Tuple[int, ...],
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        num_trials: int = 10,
    ) -> PerformanceMetrics:
        times = []
        for _ in range(2):
            self.sample_fast(context=context, latent_shape=latent_shape, graph_data=graph_data)
        for _ in range(max(1, int(num_trials))):
            start = time.time()
            self.sample_fast(context=context, latent_shape=latent_shape, graph_data=graph_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
        avg_time = sum(times) / max(1, len(times))
        return PerformanceMetrics(
            sampling_strategy=self.info.distillation_type,
            num_steps=self.num_inference_steps,
            generation_time=avg_time,
            throughput=(1.0 / avg_time) if avg_time > 0 else 0.0,
        )


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "FastSamplerCheckpointInfo",
    "GraphConditionedFastSampler",
    "LoRALayer",
    "LoRALinear",
    "PerformanceMetrics",
    "extract_lora_state_dict",
    "freeze_non_lora_parameters",
    "inject_lora_into_model",
    "load_fast_sampler_checkpoint",
    "load_lora_state_dict",
    "save_fast_sampler_checkpoint",
]
