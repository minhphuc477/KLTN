#!/usr/bin/env python
"""Run a latent diffusion DPO preference fine-tuning step/loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.train_diffusion import DiffusionTrainingConfig
from src.utils.checkpoint import atomic_torch_save, checkpoint_sha256, safe_torch_load

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - exercised only in minimal envs
    Accelerator = None  # type: ignore[assignment]


def _tensor_is_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    try:
        return bool(torch.isfinite(torch.as_tensor(float(value))).item())
    except (TypeError, ValueError):
        return False


def _gradients_are_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for param in parameters:
        if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
            return False
    return True


def _dpo_metrics_payload(
    loss: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    *,
    skipped_nonfinite_batch: bool,
) -> Dict[str, float]:
    payload: Dict[str, float] = {
        "loss": float(loss.detach().item()) if _tensor_is_finite(loss) else 0.0,
        "skipped_nonfinite_batch": 1.0 if bool(skipped_nonfinite_batch) else 0.0,
    }
    for key, value in aux.items():
        payload[str(key)] = float(value.detach().item()) if _tensor_is_finite(value) else 0.0
    return payload


def _load_model_from_checkpoint(path: Path, config: DiffusionTrainingConfig) -> LatentDiffusionModel:
    checkpoint = safe_torch_load(str(path), map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}

    def _saved(name: str, fallback: Any) -> Any:
        return checkpoint_config.get(name, fallback)
    model = create_latent_diffusion(
        latent_dim=int(_saved("latent_dim", config.latent_dim)),
        latent_scale_factor=float(_saved("latent_scale_factor", config.latent_scale_factor)),
        model_channels=int(_saved("model_channels", config.model_channels)),
        context_dim=int(_saved("context_dim", config.context_dim)),
        denoiser_backbone=str(_saved("denoiser_backbone", config.denoiser_backbone)),
        unet_channel_mult=tuple(_saved("unet_channel_mult", config.unet_channel_mult)),
        unet_num_res_blocks=int(_saved("unet_num_res_blocks", config.unet_num_res_blocks)),
        unet_attention_resolutions=tuple(_saved("unet_attention_resolutions", config.unet_attention_resolutions)),
        unet_num_heads=int(_saved("unet_num_heads", config.unet_num_heads)),
        unet_dropout=float(_saved("unet_dropout", config.unet_dropout)),
        dit_depth=int(_saved("dit_depth", config.dit_depth)),
        dit_patch_size=int(_saved("dit_patch_size", config.dit_patch_size)),
        dit_mlp_ratio=float(_saved("dit_mlp_ratio", config.dit_mlp_ratio)),
        num_timesteps=int(_saved("num_timesteps", config.num_timesteps)),
        schedule_type=str(_saved("schedule_type", config.schedule_type)),
        prediction_type=str(_saved("prediction_type", config.prediction_type)),
        cfg_dropout_prob=float(_saved("cfg_dropout_prob", config.cfg_dropout_prob)),
        cfg_scale=float(_saved("cfg_scale", config.cfg_scale)),
        pag_scale=float(_saved("pag_scale", config.pag_scale)),
        min_snr_gamma=float(_saved("min_snr_gamma", config.min_snr_gamma)),
        topology_refinement_mode=str(_saved("topology_refinement_mode", config.topology_refinement_mode)),
        attention_mode=str(_saved("attention_mode", config.attention_mode)),
        topology_conditioning_mode=str(_saved("topology_conditioning_mode", config.topology_conditioning_mode)),
        hedgehog_feature_dim=int(_saved("hedgehog_feature_dim", config.hedgehog_feature_dim)),
        graph_auto_linear_attention_nodes=int(_saved("graph_auto_linear_attention_nodes", config.graph_auto_linear_attention_nodes)),
        spatial_graph_gate_init=float(_saved("spatial_graph_gate_init", config.spatial_graph_gate_init)),
        spatial_topology_gate_init=float(_saved("spatial_topology_gate_init", config.spatial_topology_gate_init)),
        room_topology_channels=int(_saved("room_topology_channels", config.room_topology_channels)),
        training_objective=str(
            _saved("diffusion_training_objective", _saved("training_objective", "diffusion"))
        ),
    )
    state = checkpoint.get("ema_diffusion_state_dict") or checkpoint.get("diffusion_state_dict") or checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a diffusion state dict.")
    model.load_state_dict(state, strict=True)
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
    examples = payload.get("examples")
    if isinstance(examples, list):
        if not examples:
            raise ValueError("Preference payload examples list is empty.")
        for index, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Preference example {index} must be a dict.")
            missing = sorted(required - set(example))
            if missing:
                raise ValueError(f"Preference example {index} is missing required keys: {missing}")
    else:
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
    checkpoint_path = Path(args.checkpoint)
    base_checkpoint = (
        safe_torch_load(str(checkpoint_path), map_location="cpu")
        if checkpoint_path.exists()
        else None
    )
    saved_config = base_checkpoint.get("config") if isinstance(base_checkpoint, dict) else None
    if isinstance(saved_config, dict):
        config = DiffusionTrainingConfig.from_dict(saved_config)
    else:
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
    provenance = payload.get("provenance")
    if isinstance(provenance, dict) and provenance.get("diffusion_sha256"):
        expected_sha = str(provenance["diffusion_sha256"])
        actual_sha = checkpoint_sha256(Path(args.checkpoint))
        if actual_sha != expected_sha:
            raise ValueError(
                "Preference latents were prepared for a different diffusion checkpoint: "
                f"expected sha256={expected_sha}, got {actual_sha}."
            )
    config.latent_scale_factor = float(
        getattr(model, "latent_scale_factor", config.latent_scale_factor)
    )
    examples = payload.get("examples")
    if not isinstance(examples, list):
        examples = [payload]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    if accelerator is not None:
        model, optimizer = accelerator.prepare(model, optimizer)
    metrics: Dict[str, float] = {}
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    pending_micro_batches = 0
    if accelerator is None:
        optimizer.zero_grad(set_to_none=True)
    for step in range(int(args.steps)):
        example = examples[step % len(examples)]
        accumulate_ctx = accelerator.accumulate(model) if accelerator is not None else torch.enable_grad()
        with accumulate_ctx:
            if accelerator is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, aux = model(
                example["preferred"],
                example["rejected"],
                example["context"],
                reference_model=reference_model,
                beta=float(args.beta),
                graph_data=example.get("graph_data") if isinstance(example.get("graph_data"), dict) else None,
                forward_mode="dpo_preference_loss",
            )
            if not _tensor_is_finite(loss):
                optimizer.zero_grad(set_to_none=True)
                pending_micro_batches = 0
                metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=True)
                if accelerator is None or accelerator.is_main_process:
                    print({"step": step, "skipped": "nonfinite_loss", **metrics})
                continue
            if accelerator is not None:
                accelerator.backward(loss)
                params = list(model.parameters())
                if accelerator.sync_gradients and not _gradients_are_finite(params):
                    optimizer.zero_grad(set_to_none=True)
                    metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=True)
                    if accelerator.is_main_process:
                        print({"step": step, "skipped": "nonfinite_gradient", **metrics})
                    continue
                if float(args.grad_clip_norm) > 0 and accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params, float(args.grad_clip_norm))
                    if not _tensor_is_finite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=True)
                        if accelerator.is_main_process:
                            print({"step": step, "skipped": "nonfinite_clipped_grad_norm", **metrics})
                        continue
            else:
                (loss / float(accumulation_steps)).backward()
                pending_micro_batches += 1
                should_step = (
                    pending_micro_batches >= accumulation_steps
                    or step + 1 >= int(args.steps)
                )
                if not should_step:
                    metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=False)
                    continue
                params = list(model.parameters())
                if not _gradients_are_finite(params):
                    optimizer.zero_grad(set_to_none=True)
                    pending_micro_batches = 0
                    metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=True)
                    print({"step": step, "skipped": "nonfinite_gradient", **metrics})
                    continue
                if float(args.grad_clip_norm) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_norm))
                    if not _tensor_is_finite(grad_norm):
                        optimizer.zero_grad(set_to_none=True)
                        pending_micro_batches = 0
                        metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=True)
                        print({"step": step, "skipped": "nonfinite_clipped_grad_norm", **metrics})
                        continue
            optimizer.step()
            if accelerator is None:
                optimizer.zero_grad(set_to_none=True)
                pending_micro_batches = 0
        metrics = _dpo_metrics_payload(loss, aux, skipped_nonfinite_batch=False)
        is_main = bool(accelerator is None or accelerator.is_main_process)
        if is_main and step % max(1, int(args.log_every)) == 0:
            print({"step": step, **metrics})

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if args.output_checkpoint and bool(accelerator is None or accelerator.is_main_process):
        state_dict = accelerator.get_state_dict(model) if accelerator is not None else model.state_dict()
        atomic_torch_save(
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
