"""
Train a graph-aware consistency-LoRA fast sampler from a base diffusion checkpoint.

This trainer is intentionally repo-specific. It is a practical consistency-style
LoRA distillation path for the project's graph-aware latent diffusion model, not
an exact reproduction of the published LCM-LoRA training procedure.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config_system import merge_config, seed_everything
from src.optimization.lcm_lora import (
    DEFAULT_LORA_TARGETS,
    extract_lora_state_dict,
    freeze_non_lora_parameters,
    inject_lora_into_model,
    load_lora_state_dict,
    save_fast_sampler_checkpoint,
)
from src.pipeline.room_topology_conditioning import (
    DEFAULT_SEMANTIC_PUZZLE_OFFSET,
    DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
    apply_puzzle_structure_control_to_conditioning,
    apply_puzzle_structure_dropout_batch,
    build_topology_anchor_policy_metadata,
    build_topology_loss_focus_map,
)
from src.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig
from src.train_vqvae import split_dataset_for_vqvae_validation
from src.utils.checkpoint import (
    LATEST_RESUME_FILENAME,
    MetricsLogger,
    atomic_torch_save,
    enforce_checkpoint_storage_budget,
    log_checkpoint_artifact,
    prune_checkpoints,
    resolve_resume_checkpoint,
    write_checkpoint_metadata,
)
from src.zelda_data.zelda_loader import create_dataloader, graph_collate_fn

logger = logging.getLogger(__name__)


class FastSamplerTrainingConfig:
    def __init__(
        self,
        *,
        base_diffusion_checkpoint: str,
        data_dir: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        use_vglc: bool = True,
        normalize: bool = True,
        room_level: bool = True,
        topology_supervision_mode: str = "runtime_aligned",
        semantic_role_prior_strength: float = DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH,
        semantic_puzzle_offset: int = DEFAULT_SEMANTIC_PUZZLE_OFFSET,
        puzzle_structure_dropout_prob: float = 0.35,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        optimizer_weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        num_inference_steps: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        prediction_loss_weight: float = 0.25,
        decode_alignment_weight: float = 0.25,
        topology_alignment_weight: float = 0.25,
        topology_marker_weight: float = 2.0,
        topology_trace_weight: float = 0.75,
        topology_focus_dilation: int = 1,
        validation_fraction: float = 0.1,
        validation_max_batches: int = 16,
        best_checkpoint_metric: str = "val_decode_ce_loss",
        save_every: int = 5,
        keep_last: int = 2,
        auto_resume: bool = True,
        resume_checkpoint: Optional[str] = None,
        checkpoint_storage_budget_gb: Optional[float] = None,
        checkpoint_storage_warning_fraction: float = 0.8,
        checkpoint_storage_cleanup_enabled: bool = True,
        checkpoint_storage_cleanup_target_fraction: float = 0.6,
        checkpoint_dir: str = "./checkpoints/fast_sampler",
        device: str = "auto",
        seed: int = 42,
        quick: bool = False,
    ):
        self.base_diffusion_checkpoint = str(base_diffusion_checkpoint)
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(max(0, num_workers))
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.shuffle_train = bool(shuffle_train)
        self.shuffle_val = bool(shuffle_val)
        self.use_vglc = bool(use_vglc)
        self.normalize = bool(normalize)
        self.room_level = bool(room_level)
        self.topology_supervision_mode = str(topology_supervision_mode).strip().lower()
        self.semantic_role_prior_strength = float(max(0.0, min(1.0, semantic_role_prior_strength)))
        self.semantic_puzzle_offset = int(max(0, semantic_puzzle_offset))
        self.puzzle_structure_dropout_prob = float(max(0.0, min(1.0, puzzle_structure_dropout_prob)))
        self.epochs = 1 if quick else int(epochs)
        self.learning_rate = float(learning_rate)
        self.optimizer_weight_decay = float(max(0.0, optimizer_weight_decay))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.num_inference_steps = int(max(1, num_inference_steps))
        self.lora_rank = int(max(1, lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.prediction_loss_weight = float(max(0.0, prediction_loss_weight))
        self.decode_alignment_weight = float(max(0.0, decode_alignment_weight))
        self.topology_alignment_weight = float(max(0.0, topology_alignment_weight))
        self.topology_marker_weight = float(max(0.0, topology_marker_weight))
        self.topology_trace_weight = float(max(0.0, topology_trace_weight))
        self.topology_focus_dilation = int(max(0, topology_focus_dilation))
        self.validation_fraction = float(max(0.0, min(0.5, validation_fraction)))
        self.validation_max_batches = int(max(1, validation_max_batches))
        self.best_checkpoint_metric = str(best_checkpoint_metric).strip().lower()
        if self.best_checkpoint_metric not in {
            "val_loss",
            "val_decode_ce_loss",
            "val_topology_decode_ce_loss",
            "train_loss",
        }:
            raise ValueError(
                "best_checkpoint_metric must be 'val_loss', 'val_decode_ce_loss', "
                "'val_topology_decode_ce_loss', or 'train_loss'."
            )
        self.save_every = int(max(1, save_every))
        self.keep_last = int(max(0, keep_last))
        self.auto_resume = bool(auto_resume)
        self.resume_checkpoint = None if resume_checkpoint is None else str(resume_checkpoint)
        self.checkpoint_storage_budget_gb = (
            None if checkpoint_storage_budget_gb is None else float(max(0.0, checkpoint_storage_budget_gb))
        )
        self.checkpoint_storage_warning_fraction = float(max(0.0, min(1.0, checkpoint_storage_warning_fraction)))
        self.checkpoint_storage_cleanup_enabled = bool(checkpoint_storage_cleanup_enabled)
        self.checkpoint_storage_cleanup_target_fraction = float(
            max(0.0, min(1.0, checkpoint_storage_cleanup_target_fraction))
        )
        self.checkpoint_dir = str(checkpoint_dir)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
        self.seed = int(seed)
        self.quick = bool(quick)
        if self.topology_supervision_mode not in {"runtime_aligned", "oracle_room_grid"}:
            raise ValueError("topology_supervision_mode must be 'runtime_aligned' or 'oracle_room_grid'.")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def fast_sampler_training_kwargs_from_resolved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build FastSamplerTrainingConfig kwargs from the validated global config payload."""
    stage = config["fast_sampler"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    return {
        "base_diffusion_checkpoint": stage["base_diffusion_checkpoint"],
        "data_dir": dataset["data_dir"],
        "batch_size": dataset["batch_size"],
        "num_workers": dataset["num_workers"],
        "pin_memory": dataset["pin_memory"],
        "drop_last": dataset["drop_last"],
        "shuffle_train": dataset["shuffle_train"],
        "shuffle_val": dataset["shuffle_val"],
        "use_vglc": dataset["use_vglc"],
        "normalize": dataset["normalize"],
        "room_level": dataset["room_level"],
        "topology_supervision_mode": dataset["topology_supervision_mode"],
        "semantic_role_prior_strength": config["generation"]["semantic_role_prior_strength"],
        "semantic_puzzle_offset": config["generation"]["semantic_puzzle_offset"],
        "puzzle_structure_dropout_prob": stage.get("puzzle_structure_dropout_prob", 0.35),
        "epochs": stage["epochs"],
        "learning_rate": stage["learning_rate"],
        "optimizer_weight_decay": stage["optimizer_weight_decay"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "num_inference_steps": stage["num_inference_steps"],
        "lora_rank": stage["lora_rank"],
        "lora_alpha": stage["lora_alpha"],
        "prediction_loss_weight": stage["prediction_loss_weight"],
        "decode_alignment_weight": stage.get("decode_alignment_weight", 0.25),
        "topology_alignment_weight": stage.get("topology_alignment_weight", 0.25),
        "topology_marker_weight": stage.get("topology_marker_weight", 2.0),
        "topology_trace_weight": stage.get("topology_trace_weight", 0.75),
        "topology_focus_dilation": stage.get("topology_focus_dilation", 1),
        "validation_fraction": stage.get("validation_fraction", 0.1),
        "validation_max_batches": stage.get("validation_max_batches", 16),
        "best_checkpoint_metric": stage.get("best_checkpoint_metric", "val_decode_ce_loss"),
        "save_every": stage["save_every"],
        "keep_last": stage["keep_last"],
        "auto_resume": runtime["auto_resume"],
        "resume_checkpoint": runtime["resume"],
        "checkpoint_storage_budget_gb": runtime["checkpoint_storage_budget_gb"],
        "checkpoint_storage_warning_fraction": runtime["checkpoint_storage_warning_fraction"],
        "checkpoint_storage_cleanup_enabled": runtime["checkpoint_storage_cleanup_enabled"],
        "checkpoint_storage_cleanup_target_fraction": runtime["checkpoint_storage_cleanup_target_fraction"],
        "checkpoint_dir": stage["checkpoint_dir"],
        "device": runtime["device"],
        "seed": runtime["seed"],
        "quick": runtime["quick"],
    }


def _legacy_fast_sampler_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    def _set(name: str, value: Any) -> None:
        if value is None:
            return
        overrides[name] = value

    _set("base_diffusion_checkpoint", getattr(args, "base_diffusion_checkpoint", None))
    _set("data_dir", getattr(args, "data_dir", None))
    _set("batch_size", getattr(args, "batch_size", None))
    _set("use_vglc", getattr(args, "use_vglc", None))
    _set("puzzle_structure_dropout_prob", getattr(args, "puzzle_structure_dropout_prob", None))
    _set("epochs", getattr(args, "epochs", None))
    _set("learning_rate", getattr(args, "lr", None))
    _set("num_inference_steps", getattr(args, "num_inference_steps", None))
    _set("lora_rank", getattr(args, "lora_rank", None))
    _set("lora_alpha", getattr(args, "lora_alpha", None))
    _set("decode_alignment_weight", getattr(args, "decode_alignment_weight", None))
    _set("topology_alignment_weight", getattr(args, "topology_alignment_weight", None))
    _set("topology_marker_weight", getattr(args, "topology_marker_weight", None))
    _set("topology_trace_weight", getattr(args, "topology_trace_weight", None))
    _set("topology_focus_dilation", getattr(args, "topology_focus_dilation", None))
    _set("validation_fraction", getattr(args, "validation_fraction", None))
    _set("validation_max_batches", getattr(args, "validation_max_batches", None))
    _set("best_checkpoint_metric", getattr(args, "best_checkpoint_metric", None))
    _set("save_every", getattr(args, "save_every", None))
    _set("keep_last", getattr(args, "keep_last", None))
    _set("auto_resume", getattr(args, "auto_resume", None))
    _set("resume_checkpoint", getattr(args, "resume", None))
    _set("checkpoint_storage_budget_gb", getattr(args, "checkpoint_storage_budget_gb", None))
    _set("checkpoint_storage_warning_fraction", getattr(args, "checkpoint_storage_warning_fraction", None))
    _set("checkpoint_storage_cleanup_enabled", getattr(args, "checkpoint_storage_cleanup_enabled", None))
    _set("checkpoint_storage_cleanup_target_fraction", getattr(args, "checkpoint_storage_cleanup_target_fraction", None))
    _set("checkpoint_dir", getattr(args, "checkpoint_dir", None))
    _set("device", getattr(args, "device", None))
    _set("seed", getattr(args, "seed", None))
    _set("quick", getattr(args, "quick", None))
    return overrides


def build_fast_sampler_training_config_from_args(args: argparse.Namespace) -> FastSamplerTrainingConfig:
    base_kwargs: Dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        resolved = merge_config(yaml_path=str(config_path), cli_overrides=None)
        base_kwargs = fast_sampler_training_kwargs_from_resolved_config(resolved)
        if not base_kwargs.get("base_diffusion_checkpoint"):
            candidate = Path(resolved["diffusion"]["checkpoint_dir"]) / "best_model.pth"
            if candidate.exists():
                base_kwargs["base_diffusion_checkpoint"] = str(candidate)
        if getattr(args, "verbose", None) is None:
            setattr(args, "verbose", bool(resolved["runtime"]["verbose"]))
    legacy_overrides = _legacy_fast_sampler_overrides_from_args(args)
    merged = {**base_kwargs, **legacy_overrides}
    if not merged.get("base_diffusion_checkpoint"):
        raise ValueError(
            "Fast sampler training requires --base-diffusion-checkpoint or --config with "
            "fast_sampler.base_diffusion_checkpoint / an existing diffusion best_model.pth."
        )
    return FastSamplerTrainingConfig(**merged)


class ConsistencyLoRATrainer:
    def __init__(self, config: FastSamplerTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.base_bundle = self._load_base_bundle(config.base_diffusion_checkpoint)
        self.teacher = copy.deepcopy(self.base_bundle.ema_diffusion).to(self.device).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = copy.deepcopy(self.teacher).to(self.device)
        inject_lora_into_model(
            self.student.denoiser,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            target_modules=DEFAULT_LORA_TARGETS,
        )
        freeze_non_lora_parameters(self.student)
        self.student.train()

        self.optimizer = optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.optimizer_weight_decay,
        )
        self.global_step = 0
        self.epoch = 0

        self.target_timesteps = self._build_target_timestep_schedule()

    def _load_base_bundle(self, checkpoint_path: str) -> DiffusionTrainer:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        base_cfg_raw = checkpoint.get("config", {})
        if not isinstance(base_cfg_raw, dict):
            raise ValueError(f"Base diffusion checkpoint {checkpoint_path!r} is missing config metadata.")

        cfg_kwargs = dict(base_cfg_raw)
        cfg_kwargs["data_dir"] = self.config.data_dir or cfg_kwargs.get("data_dir", "Data/The Legend of Zelda")
        cfg_kwargs["batch_size"] = int(self.config.batch_size)
        cfg_kwargs["room_level"] = bool(self.config.room_level)
        cfg_kwargs["device"] = self.config.device
        cfg_kwargs["quick"] = bool(self.config.quick)
        base_config = DiffusionTrainingConfig(**cfg_kwargs)
        del checkpoint

        bundle = DiffusionTrainer(base_config)
        bundle.load_checkpoint(checkpoint_path)
        bundle.vqvae.eval()
        bundle.condition_encoder.eval()
        bundle.logic_net.eval()
        bundle.diffusion.eval()
        bundle.ema_diffusion.eval()
        for module in (bundle.vqvae, bundle.condition_encoder, bundle.logic_net, bundle.diffusion, bundle.ema_diffusion):
            for param in module.parameters():
                param.requires_grad = False
        return bundle

    def _build_target_timestep_schedule(self) -> torch.Tensor:
        num_train = int(self.teacher.num_timesteps)
        steps = int(self.config.num_inference_steps)
        return torch.linspace(num_train - 1, 0, steps).long().to(self.device)

    def _room_tile_targets(self, real_maps: torch.Tensor) -> torch.Tensor:
        num_classes = int(self.base_bundle.vqvae.num_classes)
        if real_maps.shape[1] == 1:
            tile_ids = (real_maps.squeeze(1) * (num_classes - 1)).round().long()
            return tile_ids.clamp_(0, num_classes - 1)
        if real_maps.shape[1] == num_classes:
            return real_maps.argmax(dim=1)
        raise ValueError(
            f"Unexpected real_maps channel count {int(real_maps.shape[1])}; "
            f"expected 1 or {num_classes}."
        )

    def _sample_batch_timesteps(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, len(self.target_timesteps), (batch_size,), device=self.device)
        return self.target_timesteps[idx]

    def _sample_batch_timesteps_deterministic(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        idx = torch.randint(
            0,
            len(self.target_timesteps),
            (batch_size,),
            generator=generator,
        )
        return self.target_timesteps[idx.to(device=self.device)]

    def _build_conditioning(
        self,
        graph_list: Optional[List[dict]],
        batch_size: int,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        if not graph_list:
            return self.base_bundle.get_dummy_conditioning(batch_size), None

        cond_vectors = []
        for graph_dict in graph_list:
            cond_vectors.append(self.base_bundle._encode_graph_conditioning(graph_dict))
        conditioning = self.base_bundle._stack_conditioning_vectors(cond_vectors)
        diffusion_graph_data = self.base_bundle._stack_diffusion_graph_batch(graph_list)
        return conditioning, diffusion_graph_data

    def _topology_focus_map(
        self,
        graph_list: Optional[List[dict]],
        batch_size: int,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if (
            not graph_list
            or float(getattr(self.config, "topology_alignment_weight", 0.0)) <= 0.0
        ):
            return None
        topo_maps: List[torch.Tensor] = []
        for graph_dict in graph_list:
            topo = graph_dict.get("room_topology_map")
            if topo is None:
                return None
            if not isinstance(topo, torch.Tensor):
                topo = torch.as_tensor(topo)
            if topo.dim() == 4:
                if int(topo.shape[0]) != 1:
                    return None
                topo = topo.squeeze(0)
            if topo.dim() != 3:
                return None
            topo_maps.append(topo.to(device=device, dtype=torch.float32))
        if len(topo_maps) != int(batch_size):
            return None
        stacked = torch.stack(topo_maps, dim=0)
        return build_topology_loss_focus_map(
            stacked,
            marker_weight=float(getattr(self.config, "topology_marker_weight", 2.0)),
            trace_weight=float(getattr(self.config, "topology_trace_weight", 0.75)),
            dilation=int(getattr(self.config, "topology_focus_dilation", 1)),
        )

    def distill_step(
        self,
        real_maps: torch.Tensor,
        graph_list: Optional[List[dict]] = None,
    ) -> Dict[str, float]:
        real_maps = real_maps.to(self.device)
        batch_size = int(real_maps.shape[0])
        conditioning, diffusion_graph_data = self._build_conditioning(graph_list, batch_size)
        z_0 = self.base_bundle.encode_to_latent(real_maps)

        t = self._sample_batch_timesteps(batch_size)
        noise = torch.randn_like(z_0)
        x_t = self.teacher.q_sample(z_0, t, noise)

        with torch.no_grad():
            teacher_pred = self.teacher._predict_noise_cfg(x_t, t, conditioning, graph_data=diffusion_graph_data)
            teacher_x0, _teacher_noise = self.teacher._convert_prediction(teacher_pred, x_t, t)
            teacher_x0 = torch.clamp(teacher_x0, -1.0, 1.0)

        student_pred = self.student._predict_noise_cfg(x_t, t, conditioning, graph_data=diffusion_graph_data)
        student_x0, _student_noise = self.student._convert_prediction(student_pred, x_t, t)
        student_x0 = torch.clamp(student_x0, -1.0, 1.0)

        x0_loss = F.mse_loss(student_x0, teacher_x0)
        pred_loss = F.mse_loss(student_pred, teacher_pred)
        decode_ce_loss = torch.zeros((), device=self.device, dtype=student_x0.dtype)
        topology_decode_ce_loss = torch.zeros((), device=self.device, dtype=student_x0.dtype)
        if self.config.decode_alignment_weight > 0.0:
            target_tiles = self._room_tile_targets(real_maps)
            student_logits = self.base_bundle.vqvae.decode(student_x0)
            decode_ce_loss = F.cross_entropy(student_logits, target_tiles)
            focus_map = self._topology_focus_map(graph_list, batch_size, device=target_tiles.device)
            if focus_map is not None and bool((focus_map > 0).any()):
                ce_map = F.cross_entropy(student_logits, target_tiles, reduction="none")
                denom = focus_map.sum().clamp(min=1.0)
                topology_decode_ce_loss = (ce_map * focus_map).sum() / denom
        elif self.config.topology_alignment_weight > 0.0:
            target_tiles = self._room_tile_targets(real_maps)
            student_logits = self.base_bundle.vqvae.decode(student_x0)
            focus_map = self._topology_focus_map(graph_list, batch_size, device=target_tiles.device)
            if focus_map is not None and bool((focus_map > 0).any()):
                ce_map = F.cross_entropy(student_logits, target_tiles, reduction="none")
                denom = focus_map.sum().clamp(min=1.0)
                topology_decode_ce_loss = (ce_map * focus_map).sum() / denom
        loss = (
            x0_loss
            + (self.config.prediction_loss_weight * pred_loss)
            + (self.config.decode_alignment_weight * decode_ce_loss)
            + (self.config.topology_alignment_weight * topology_decode_ce_loss)
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.student.parameters() if p.requires_grad],
                max_norm=self.config.grad_clip_norm,
            )
        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": float(loss.item()),
            "x0_loss": float(x0_loss.item()),
            "prediction_loss": float(pred_loss.item()),
            "decode_ce_loss": float(decode_ce_loss.item()),
            "topology_decode_ce_loss": float(topology_decode_ce_loss.item()),
        }

    @torch.no_grad()
    def validate(
        self,
        dataloader,
        *,
        max_batches: Optional[int] = None,
        eval_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        self.student.eval()
        metrics = {
            "val_loss": 0.0,
            "val_x0_loss": 0.0,
            "val_prediction_loss": 0.0,
            "val_decode_ce_loss": 0.0,
            "val_topology_decode_ce_loss": 0.0,
        }
        count = 0
        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                real_maps, graph_list = batch_data
            else:
                real_maps, graph_list = batch_data, None
            step_metrics = self.distill_step_eval(
                real_maps,
                graph_list,
                batch_index=batch_idx,
                eval_seed=eval_seed,
            )
            for key, value in step_metrics.items():
                metrics[key] += float(value)
            count += 1
            if max_batches is not None and count >= int(max_batches):
                break
        self.student.train()
        return {k: (v / max(1, count)) for k, v in metrics.items()}

    @torch.no_grad()
    def distill_step_eval(
        self,
        real_maps: torch.Tensor,
        graph_list: Optional[List[dict]] = None,
        *,
        batch_index: int = 0,
        eval_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        real_maps = real_maps.to(self.device)
        batch_size = int(real_maps.shape[0])
        conditioning, diffusion_graph_data = self._build_conditioning(graph_list, batch_size)
        z_0 = self.base_bundle.encode_to_latent(real_maps)
        if eval_seed is None:
            t = self._sample_batch_timesteps(batch_size)
            noise = torch.randn_like(z_0)
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(eval_seed) + int(batch_index))
            t = self._sample_batch_timesteps_deterministic(batch_size, generator=generator)
            noise = torch.randn(
                tuple(z_0.shape),
                generator=generator,
                dtype=torch.float32,
            ).to(device=self.device, dtype=z_0.dtype)
        x_t = self.teacher.q_sample(z_0, t, noise)

        teacher_pred = self.teacher._predict_noise_cfg(x_t, t, conditioning, graph_data=diffusion_graph_data)
        teacher_x0, _ = self.teacher._convert_prediction(teacher_pred, x_t, t)
        teacher_x0 = torch.clamp(teacher_x0, -1.0, 1.0)

        student_pred = self.student._predict_noise_cfg(x_t, t, conditioning, graph_data=diffusion_graph_data)
        student_x0, _ = self.student._convert_prediction(student_pred, x_t, t)
        student_x0 = torch.clamp(student_x0, -1.0, 1.0)

        x0_loss = F.mse_loss(student_x0, teacher_x0)
        pred_loss = F.mse_loss(student_pred, teacher_pred)
        decode_ce_loss = torch.zeros((), device=self.device, dtype=student_x0.dtype)
        topology_decode_ce_loss = torch.zeros((), device=self.device, dtype=student_x0.dtype)
        if self.config.decode_alignment_weight > 0.0:
            target_tiles = self._room_tile_targets(real_maps)
            student_logits = self.base_bundle.vqvae.decode(student_x0)
            decode_ce_loss = F.cross_entropy(student_logits, target_tiles)
            focus_map = self._topology_focus_map(graph_list, batch_size, device=target_tiles.device)
            if focus_map is not None and bool((focus_map > 0).any()):
                ce_map = F.cross_entropy(student_logits, target_tiles, reduction="none")
                denom = focus_map.sum().clamp(min=1.0)
                topology_decode_ce_loss = (ce_map * focus_map).sum() / denom
        elif self.config.topology_alignment_weight > 0.0:
            target_tiles = self._room_tile_targets(real_maps)
            student_logits = self.base_bundle.vqvae.decode(student_x0)
            focus_map = self._topology_focus_map(graph_list, batch_size, device=target_tiles.device)
            if focus_map is not None and bool((focus_map > 0).any()):
                ce_map = F.cross_entropy(student_logits, target_tiles, reduction="none")
                denom = focus_map.sum().clamp(min=1.0)
                topology_decode_ce_loss = (ce_map * focus_map).sum() / denom
        loss = (
            x0_loss
            + (self.config.prediction_loss_weight * pred_loss)
            + (self.config.decode_alignment_weight * decode_ce_loss)
            + (self.config.topology_alignment_weight * topology_decode_ce_loss)
        )
        return {
            "val_loss": float(loss.item()),
            "val_x0_loss": float(x0_loss.item()),
            "val_prediction_loss": float(pred_loss.item()),
            "val_decode_ce_loss": float(decode_ce_loss.item()),
            "val_topology_decode_ce_loss": float(topology_decode_ce_loss.item()),
        }

    def save_checkpoint(self, path: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        save_fast_sampler_checkpoint(
            path,
            lora_state_dict=extract_lora_state_dict(self.student),
            base_diffusion_checkpoint=self.config.base_diffusion_checkpoint,
            num_inference_steps=self.config.num_inference_steps,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=DEFAULT_LORA_TARGETS,
            metrics=metrics,
            distillation_type="consistency_lora",
            topology_alignment_weight=float(getattr(self.config, "topology_alignment_weight", 0.0)),
            topology_marker_weight=float(getattr(self.config, "topology_marker_weight", 2.0)),
            topology_trace_weight=float(getattr(self.config, "topology_trace_weight", 0.75)),
            topology_focus_dilation=int(getattr(self.config, "topology_focus_dilation", 1)),
            topology_anchor_policy=build_topology_anchor_policy_metadata(
                semantic_role_prior_strength=float(
                    getattr(self.config, "semantic_role_prior_strength", DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH)
                ),
                semantic_puzzle_offset=int(
                    getattr(self.config, "semantic_puzzle_offset", DEFAULT_SEMANTIC_PUZZLE_OFFSET)
                ),
                topology_supervision_mode=str(
                    getattr(self.config, "topology_supervision_mode", "runtime_aligned")
                ),
            ),
        )

    def save_resume_checkpoint(self, path: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        topology_anchor_policy = build_topology_anchor_policy_metadata(
            semantic_role_prior_strength=float(
                getattr(self.config, "semantic_role_prior_strength", DEFAULT_SEMANTIC_ROLE_PRIOR_STRENGTH)
            ),
            semantic_puzzle_offset=int(
                getattr(self.config, "semantic_puzzle_offset", DEFAULT_SEMANTIC_PUZZLE_OFFSET)
            ),
            topology_supervision_mode=str(
                getattr(self.config, "topology_supervision_mode", "runtime_aligned")
            ),
        )
        payload = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "lora_state_dict": extract_lora_state_dict(self.student),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": dict(metrics or {}),
            "metadata": {
                "distillation_type": "consistency_lora",
                "base_diffusion_checkpoint": str(self.config.base_diffusion_checkpoint),
                "num_inference_steps": int(self.config.num_inference_steps),
                "lora_rank": int(self.config.lora_rank),
                "lora_alpha": float(self.config.lora_alpha),
                "topology_alignment_weight": float(getattr(self.config, "topology_alignment_weight", 0.0)),
                "topology_marker_weight": float(getattr(self.config, "topology_marker_weight", 2.0)),
                "topology_trace_weight": float(getattr(self.config, "topology_trace_weight", 0.75)),
                "topology_focus_dilation": int(getattr(self.config, "topology_focus_dilation", 1)),
                "topology_anchor_policy": dict(topology_anchor_policy),
                "target_modules": [str(t) for t in DEFAULT_LORA_TARGETS],
            },
        }
        atomic_torch_save(payload, path)
        write_checkpoint_metadata(
            path,
            model_type="fast_sampler_resume",
            architecture={
                "distillation_type": "consistency_lora",
                "num_inference_steps": int(self.config.num_inference_steps),
                "lora_rank": int(self.config.lora_rank),
                "topology_alignment_weight": float(getattr(self.config, "topology_alignment_weight", 0.0)),
                "topology_marker_weight": float(getattr(self.config, "topology_marker_weight", 2.0)),
                "topology_trace_weight": float(getattr(self.config, "topology_trace_weight", 0.75)),
                "topology_focus_dilation": int(getattr(self.config, "topology_focus_dilation", 1)),
            },
            extra={
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "base_diffusion_checkpoint": str(self.config.base_diffusion_checkpoint),
                "checkpoint_kind": "resume",
                "contains": ["lora", "optimizer"],
                "topology_anchor_policy": dict(topology_anchor_policy),
            },
        )
        log_checkpoint_artifact(
            logger,
            path,
            checkpoint_dir=Path(path).parent,
            label="Saved fast-sampler resume checkpoint",
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        lora_state = payload.get("lora_state_dict")
        if not isinstance(lora_state, dict):
            raise ValueError(f"Invalid fast-sampler resume checkpoint at {path!r}: missing lora_state_dict.")
        load_lora_state_dict(self.student, lora_state, strict=True)
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("global_step", 0))
        logger.info("Loaded fast-sampler checkpoint from %s (epoch %d)", path, self.epoch)
        return payload


def _create_fast_sampler_dataloaders(
    config: FastSamplerTrainingConfig,
    data_dir: str,
) -> tuple[DataLoader, DataLoader, str, int, int]:
    base_loader = create_dataloader(
        data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        use_vglc=config.use_vglc,
        normalize=config.normalize,
        room_level=config.room_level,
        load_graphs=True,
        topology_supervision_mode=config.topology_supervision_mode,
        semantic_role_prior_strength=config.semantic_role_prior_strength,
        semantic_puzzle_offset=config.semantic_puzzle_offset,
    )
    dataset = base_loader.dataset
    train_dataset, val_dataset = split_dataset_for_vqvae_validation(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=graph_collate_fn,
    )
    eval_source = val_dataset if val_dataset is not None else train_dataset
    eval_split_name = "val" if val_dataset is not None else "train"
    val_loader = DataLoader(
        eval_source,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=graph_collate_fn,
    )
    return train_loader, val_loader, eval_split_name, len(train_dataset), len(eval_source)


def _resolve_fast_sampler_best_metric_name(config: FastSamplerTrainingConfig) -> str:
    if config.best_checkpoint_metric == "train_loss":
        return "train_loss"
    if config.best_checkpoint_metric == "val_topology_decode_ce_loss":
        return "val_topology_decode_ce_loss"
    if config.best_checkpoint_metric == "val_decode_ce_loss":
        return "val_decode_ce_loss"
    return "val_loss"


def reevaluate_fast_sampler_checkpoint_candidates(
    config: FastSamplerTrainingConfig,
    *,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    target_dir = Path(checkpoint_dir or config.checkpoint_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Fast-sampler checkpoint directory not found: {target_dir}")

    trainer = ConsistencyLoRATrainer(config)
    data_dir = config.data_dir or trainer.base_bundle.config.data_dir
    _train_loader, val_loader, eval_split_name, train_size, eval_size = _create_fast_sampler_dataloaders(
        config,
        data_dir,
    )

    candidate_paths: List[Path] = []
    for filename in ("fast_sampler_best.pth", "fast_sampler_final.pth"):
        candidate = target_dir / filename
        if candidate.exists():
            candidate_paths.append(candidate)
    candidate_paths.extend(sorted(target_dir.glob("fast_sampler_resume_epoch_*.pth")))
    if not candidate_paths:
        raise FileNotFoundError(f"No fast-sampler checkpoints found under {target_dir}")

    eval_seed = int(config.seed) + 10_000
    rankings: List[Dict[str, Any]] = []
    metric_name = _resolve_fast_sampler_best_metric_name(config)
    if metric_name == "train_loss":
        metric_name = "val_loss"
    best_path: Optional[Path] = None
    best_metric_value = float("inf")

    for candidate in candidate_paths:
        trainer.load_checkpoint(str(candidate))
        metrics = trainer.validate(
            val_loader,
            max_batches=config.validation_max_batches,
            eval_seed=eval_seed,
        )
        if metric_name == "val_topology_decode_ce_loss":
            metric_value = float(metrics["val_topology_decode_ce_loss"])
        elif metric_name == "val_decode_ce_loss":
            metric_value = float(metrics["val_decode_ce_loss"])
        else:
            metric_value = float(metrics["val_loss"])
        rankings.append(
            {
                "checkpoint": candidate.name,
                "metric_name": metric_name,
                "metric_value": metric_value,
                **metrics,
            }
        )
        if metric_value < best_metric_value:
            best_metric_value = metric_value
            best_path = candidate

    assert best_path is not None
    trainer.load_checkpoint(str(best_path))
    reselected_path = target_dir / "fast_sampler_best_reselected.pth"
    trainer.save_checkpoint(
        str(reselected_path),
        {
            "reselected_metric_name": metric_name,
            "reselected_metric_value": best_metric_value,
            "reselected_from": best_path.name,
            "eval_split": eval_split_name,
        },
    )
    ranking_payload = {
        "metric_name": metric_name,
        "eval_split": eval_split_name,
        "train_size": int(train_size),
        "eval_size": int(eval_size),
        "eval_seed": int(eval_seed),
        "validation_max_batches": int(config.validation_max_batches),
        "selected_checkpoint": best_path.name,
        "selected_metric_value": float(best_metric_value),
        "reselected_checkpoint": reselected_path.name,
        "rankings": rankings,
    }
    (target_dir / "fast_sampler_checkpoint_ranking.json").write_text(
        json.dumps(ranking_payload, indent=2),
        encoding="utf-8",
    )
    return ranking_payload


def train_fast_sampler(config: FastSamplerTrainingConfig) -> ConsistencyLoRATrainer:
    resolved_seed = seed_everything(int(getattr(config, "seed", 42)))
    logger.info("Fast-sampler trainer seeds initialized: seed=%d", resolved_seed)
    trainer = ConsistencyLoRATrainer(config)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_dir = config.data_dir or trainer.base_bundle.config.data_dir
    train_loader, val_loader, eval_split_name, train_size, eval_size = _create_fast_sampler_dataloaders(
        config,
        data_dir,
    )

    metrics_logger = MetricsLogger(
        log_dir=str(checkpoint_dir / "logs"),
        experiment_name="fast_sampler_training",
    )
    if float(getattr(config, "validation_fraction", 0.0)) > 0.0:
        best_metric_name = _resolve_fast_sampler_best_metric_name(config)
    else:
        best_metric_name = "train_loss"
    if config.best_checkpoint_metric == "train_loss":
        best_metric_name = "train_loss"
    best_metric_value = float("inf")
    metrics: Dict[str, Any] = {}
    logger.info(
        "Fast sampler split: train=%d rooms | %s=%d rooms | best_metric=%s",
        int(train_size),
        eval_split_name,
        int(eval_size),
        best_metric_name,
    )
    resume_path = resolve_resume_checkpoint(
        explicit_path=getattr(config, "resume_checkpoint", None),
        checkpoint_dir=str(checkpoint_dir),
        auto_resume=bool(getattr(config, "auto_resume", True)),
        latest_filename=LATEST_RESUME_FILENAME,
    )
    if resume_path is not None:
        resume_payload = trainer.load_checkpoint(str(resume_path))
        latest_metrics = resume_payload.get("metrics", {})
        if isinstance(latest_metrics, dict):
            best_metric_name = str(latest_metrics.get("best_metric_name", best_metric_name))
            best_metric_value = float(
                latest_metrics.get("best_metric_value", latest_metrics.get("best_val_loss", best_metric_value))
            )
        logger.info("Auto-resumed fast-sampler training from %s", resume_path)

    for epoch in range(int(getattr(trainer, "epoch", -1)) + 1, config.epochs):
        trainer.epoch = epoch
        running = {
            "loss": 0.0,
            "x0_loss": 0.0,
            "prediction_loss": 0.0,
            "decode_ce_loss": 0.0,
            "topology_decode_ce_loss": 0.0,
        }
        count = 0
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                real_maps, graph_list = batch_data
            else:
                real_maps, graph_list = batch_data, None
            if graph_list is not None and float(getattr(config, "puzzle_structure_dropout_prob", 0.0)) > 0.0:
                real_maps, graph_list = apply_puzzle_structure_dropout_batch(
                    real_maps,
                    graph_list,
                    num_classes=int(getattr(trainer.base_bundle.config, "num_classes", 44)),
                    dropout_prob=float(config.puzzle_structure_dropout_prob),
                )
            step_metrics = trainer.distill_step(real_maps, graph_list)
            for key, value in step_metrics.items():
                running[key] += float(value)
            count += 1
            if batch_idx % 10 == 0:
                logger.debug(
                    "Fast sampler batch %d: loss=%.4f x0=%.4f pred=%.4f topo_ce=%.4f",
                    batch_idx,
                    step_metrics["loss"],
                    step_metrics["x0_loss"],
                    step_metrics["prediction_loss"],
                    step_metrics["topology_decode_ce_loss"],
                )

        train_metrics = {k: (v / max(1, count)) for k, v in running.items()}
        val_metrics = trainer.validate(
            val_loader,
            max_batches=config.validation_max_batches,
            eval_seed=int(config.seed) + 10_000,
        )
        metrics = {"epoch": epoch, "eval_split": eval_split_name, **train_metrics, **val_metrics}
        metrics_logger.log(metrics)
        logger.info(
            "Fast sampler epoch %d/%d: loss=%.4f val_loss=%.4f val_decode_ce=%.4f val_topo_ce=%.4f",
            epoch + 1,
            config.epochs,
            train_metrics["loss"],
            val_metrics["val_loss"],
            val_metrics["val_decode_ce_loss"],
            val_metrics["val_topology_decode_ce_loss"],
        )

        if (epoch + 1) % config.save_every == 0:
            trainer.save_resume_checkpoint(str(checkpoint_dir / f"fast_sampler_resume_epoch_{epoch+1:04d}.pth"), metrics)
            prune_checkpoints(
                checkpoint_dir=str(checkpoint_dir),
                pattern="fast_sampler_resume_epoch_*.pth",
                keep_last=int(getattr(config, "keep_last", 2)),
            )
        if best_metric_name == "val_topology_decode_ce_loss":
            current_metric_value = float(val_metrics["val_topology_decode_ce_loss"])
        elif best_metric_name == "val_decode_ce_loss":
            current_metric_value = float(val_metrics["val_decode_ce_loss"])
        elif best_metric_name == "val_loss":
            current_metric_value = float(val_metrics["val_loss"])
        else:
            current_metric_value = float(train_metrics["loss"])
        if current_metric_value < best_metric_value:
            best_metric_value = current_metric_value
            trainer.save_checkpoint(str(checkpoint_dir / "fast_sampler_best.pth"), metrics)

        latest_metrics = dict(metrics)
        latest_metrics["best_metric_name"] = str(best_metric_name)
        latest_metrics["best_metric_value"] = float(best_metric_value)
        latest_metrics["best_val_loss"] = float(best_metric_value if best_metric_name == "val_loss" else val_metrics["val_loss"])
        trainer.save_resume_checkpoint(str(checkpoint_dir / LATEST_RESUME_FILENAME), latest_metrics)
        enforce_checkpoint_storage_budget(
            logger,
            checkpoint_dir=checkpoint_dir,
            budget_gb=getattr(config, "checkpoint_storage_budget_gb", None),
            warning_fraction=float(getattr(config, "checkpoint_storage_warning_fraction", 0.8)),
            cleanup_enabled=bool(getattr(config, "checkpoint_storage_cleanup_enabled", True)),
            cleanup_target_fraction=float(getattr(config, "checkpoint_storage_cleanup_target_fraction", 0.6)),
            removable_patterns=("fast_sampler_resume_epoch_*.pth",),
        )

    trainer.save_checkpoint(str(checkpoint_dir / "fast_sampler_final.pth"), metrics)
    metrics_logger.save()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the repo's graph-aware consistency-LoRA fast sampler."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path using the shared validated config system. "
             "When provided, omitted legacy flags inherit values from that config.",
    )
    parser.add_argument("--base-diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-vglc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--puzzle-structure-dropout-prob", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--decode-alignment-weight", type=float, default=None)
    parser.add_argument("--topology-alignment-weight", type=float, default=None)
    parser.add_argument("--topology-marker-weight", type=float, default=None)
    parser.add_argument("--topology-trace-weight", type=float, default=None)
    parser.add_argument("--topology-focus-dilation", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--validation-max-batches", type=int, default=None)
    parser.add_argument(
        "--best-checkpoint-metric",
        type=str,
        choices=("val_loss", "val_decode_ce_loss", "val_topology_decode_ce_loss", "train_loss"),
        default=None,
    )
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--keep-last", type=int, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-storage-budget-gb", type=float, default=None)
    parser.add_argument("--checkpoint-storage-warning-fraction", type=float, default=None)
    parser.add_argument("--checkpoint-storage-cleanup-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-storage-cleanup-target-fraction", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", "-v", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    try:
        config = build_fast_sampler_training_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=logging.DEBUG if bool(args.verbose) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train_fast_sampler(config)


if __name__ == "__main__":
    main()
