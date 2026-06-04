#!/usr/bin/env python
"""Run a latent diffusion DPO preference fine-tuning step/loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.train_diffusion import DiffusionTrainingConfig
from src.utils.checkpoint import safe_torch_load

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - exercised only in minimal envs
    Accelerator = None  # type: ignore[assignment]


def _load_model_from_checkpoint(path: Path, config: DiffusionTrainingConfig) -> LatentDiffusionModel:
    model = create_latent_diffusion(
        latent_dim=config.latent_dim,
        model_channels=config.model_channels,
        context_dim=config.context_dim,
        denoiser_backbone=config.denoiser_backbone,
        unet_channel_mult=config.unet_channel_mult,
        unet_num_res_blocks=config.unet_num_res_blocks,
        unet_attention_resolutions=config.unet_attention_resolutions,
        unet_num_heads=config.unet_num_heads,
        unet_dropout=config.unet_dropout,
        dit_depth=config.dit_depth,
        dit_patch_size=config.dit_patch_size,
        dit_mlp_ratio=config.dit_mlp_ratio,
        num_timesteps=config.num_timesteps,
        schedule_type=config.schedule_type,
        prediction_type=config.prediction_type,
        cfg_dropout_prob=config.cfg_dropout_prob,
        cfg_scale=config.cfg_scale,
        pag_scale=config.pag_scale,
        min_snr_gamma=config.min_snr_gamma,
        topology_refinement_mode=config.topology_refinement_mode,
        attention_mode=config.attention_mode,
        topology_conditioning_mode=config.topology_conditioning_mode,
        hedgehog_feature_dim=config.hedgehog_feature_dim,
        graph_auto_linear_attention_nodes=config.graph_auto_linear_attention_nodes,
        spatial_graph_gate_init=config.spatial_graph_gate_init,
        spatial_topology_gate_init=config.spatial_topology_gate_init,
        room_topology_channels=config.room_topology_channels,
    )
    checkpoint = safe_torch_load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_diffusion_state_dict") or checkpoint.get("diffusion_state_dict") or checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a diffusion state dict.")
    model.load_state_dict(state, strict=False)
    return model


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(nested, device) for key, nested in value.items()}
    if isinstance(value, list):
        return [_to_device(nested, device) for nested in value]
    if isinstance(value, tuple):
        return tuple(_to_device(nested, device) for nested in value)
    return value


def _load_preference_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    payload = safe_torch_load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Preference payload must be a dict with preferred, rejected, and context tensors.")
    required = {"preferred", "rejected", "context"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Preference payload is missing required keys: {missing}")
    return {key: _to_device(value, device) for key, value in payload.items()}


def run_dpo(args: argparse.Namespace) -> Dict[str, float]:
    accelerator = None
    if Accelerator is not None:
        accelerator = Accelerator(
            mixed_precision=None if str(args.mixed_precision).lower() == "no" else str(args.mixed_precision),
            gradient_accumulation_steps=max(1, int(args.gradient_accumulation_steps)),
        )
    device = (
        accelerator.device
        if accelerator is not None and str(args.device).strip().lower() == "auto"
        else torch.device("cuda" if str(args.device).strip().lower() == "auto" and torch.cuda.is_available() else args.device)
    )
    config = DiffusionTrainingConfig(
        denoiser_backbone=args.denoiser_backbone,
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        context_dim=args.context_dim,
        unet_num_heads=args.num_heads,
        dit_depth=args.dit_depth,
        dit_patch_size=args.dit_patch_size,
        num_timesteps=args.num_timesteps,
        prediction_type=args.prediction_type,
    )
    model = _load_model_from_checkpoint(Path(args.checkpoint), config).to(device)
    model.train()

    reference_model: Optional[LatentDiffusionModel] = None
    if args.reference_checkpoint:
        reference_model = _load_model_from_checkpoint(Path(args.reference_checkpoint), config).to(device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    payload = _load_preference_payload(Path(args.preference_pairs), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    if accelerator is not None:
        model, optimizer = accelerator.prepare(model, optimizer)
    metrics: Dict[str, float] = {}
    for step in range(int(args.steps)):
        accumulate_ctx = accelerator.accumulate(model) if accelerator is not None else torch.enable_grad()
        with accumulate_ctx:
            optimizer.zero_grad(set_to_none=True)
            loss, aux = model(
                payload["preferred"],
                payload["rejected"],
                payload["context"],
                reference_model=reference_model,
                beta=float(args.beta),
                graph_data=payload.get("graph_data") if isinstance(payload.get("graph_data"), dict) else None,
                forward_mode="dpo_preference_loss",
            )
            if accelerator is not None:
                accelerator.backward(loss)
                if float(args.grad_clip_norm) > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            else:
                loss.backward()
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()
        metrics = {"loss": float(loss.detach().item()), **{k: float(v.detach().item()) for k, v in aux.items()}}
        is_main = bool(accelerator is None or accelerator.is_main_process)
        if is_main and step % max(1, int(args.log_every)) == 0:
            print({"step": step, **metrics})

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if args.output_checkpoint and bool(accelerator is None or accelerator.is_main_process):
        state_dict = accelerator.get_state_dict(model) if accelerator is not None else model.state_dict()
        torch.save(
            {
                "diffusion_state_dict": state_dict,
                "dpo_metrics": metrics,
                "config": config.to_dict(),
            },
            str(args.output_checkpoint),
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Base diffusion checkpoint.")
    parser.add_argument("--reference-checkpoint", default=None, help="Frozen reference checkpoint. Defaults to no-reference DPO.")
    parser.add_argument("--preference-pairs", required=True, help=".pt dict with preferred, rejected, context tensors.")
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--denoiser-backbone", choices=["unet", "dit"], default="unet")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--model-channels", type=int, default=96)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dit-depth", type=int, default=4)
    parser.add_argument("--dit-patch-size", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--prediction-type", choices=["epsilon", "v"], default="epsilon")
    return parser.parse_args()


def main() -> None:
    run_dpo(parse_args())


if __name__ == "__main__":
    main()
