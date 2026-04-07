"""
Train a graph-aware consistency-LoRA fast sampler from a base diffusion checkpoint.

This trainer is intentionally repo-specific. It is a practical consistency-style
LoRA distillation path for the project's graph-aware latent diffusion model, not
an exact reproduction of the published LCM-LoRA training procedure.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

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
    build_topology_anchor_policy_metadata,
)
from src.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig
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
from src.zelda_data.zelda_loader import create_dataloader

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
        epochs: int = 10,
        learning_rate: float = 1e-4,
        optimizer_weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        num_inference_steps: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        prediction_loss_weight: float = 0.25,
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
        self.epochs = 1 if quick else int(epochs)
        self.learning_rate = float(learning_rate)
        self.optimizer_weight_decay = float(max(0.0, optimizer_weight_decay))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.num_inference_steps = int(max(1, num_inference_steps))
        self.lora_rank = int(max(1, lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.prediction_loss_weight = float(max(0.0, prediction_loss_weight))
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
        "epochs": stage["epochs"],
        "learning_rate": stage["learning_rate"],
        "optimizer_weight_decay": stage["optimizer_weight_decay"],
        "grad_clip_norm": stage["grad_clip_norm"],
        "num_inference_steps": stage["num_inference_steps"],
        "lora_rank": stage["lora_rank"],
        "lora_alpha": stage["lora_alpha"],
        "prediction_loss_weight": stage["prediction_loss_weight"],
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
    _set("epochs", getattr(args, "epochs", None))
    _set("learning_rate", getattr(args, "lr", None))
    _set("num_inference_steps", getattr(args, "num_inference_steps", None))
    _set("lora_rank", getattr(args, "lora_rank", None))
    _set("lora_alpha", getattr(args, "lora_alpha", None))
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

    def _sample_batch_timesteps(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, len(self.target_timesteps), (batch_size,), device=self.device)
        return self.target_timesteps[idx]

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

    def distill_step(
        self,
        real_maps: torch.Tensor,
        graph_list: Optional[List[dict]] = None,
    ) -> Dict[str, float]:
        batch_size = int(real_maps.shape[0])
        conditioning, diffusion_graph_data = self._build_conditioning(graph_list, batch_size)
        z_0 = self.base_bundle.encode_to_latent(real_maps.to(self.device))

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
        loss = x0_loss + (self.config.prediction_loss_weight * pred_loss)

        self.optimizer.zero_grad()
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
        }

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        self.student.eval()
        metrics = {"val_loss": 0.0, "val_x0_loss": 0.0, "val_prediction_loss": 0.0}
        count = 0
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                real_maps, graph_list = batch_data
            else:
                real_maps, graph_list = batch_data, None
            step_metrics = self.distill_step_eval(real_maps.to(self.device), graph_list)
            for key, value in step_metrics.items():
                metrics[key] += float(value)
            count += 1
        self.student.train()
        return {k: (v / max(1, count)) for k, v in metrics.items()}

    @torch.no_grad()
    def distill_step_eval(self, real_maps: torch.Tensor, graph_list: Optional[List[dict]] = None) -> Dict[str, float]:
        batch_size = int(real_maps.shape[0])
        conditioning, diffusion_graph_data = self._build_conditioning(graph_list, batch_size)
        z_0 = self.base_bundle.encode_to_latent(real_maps.to(self.device))
        t = self._sample_batch_timesteps(batch_size)
        noise = torch.randn_like(z_0)
        x_t = self.teacher.q_sample(z_0, t, noise)

        teacher_pred = self.teacher._predict_noise_cfg(x_t, t, conditioning, graph_data=diffusion_graph_data)
        teacher_x0, _ = self.teacher._convert_prediction(teacher_pred, x_t, t)
        teacher_x0 = torch.clamp(teacher_x0, -1.0, 1.0)

        student_pred = self.student._predict_noise_cfg(x_t, t, conditioning, graph_data=diffusion_graph_data)
        student_x0, _ = self.student._convert_prediction(student_pred, x_t, t)
        student_x0 = torch.clamp(student_x0, -1.0, 1.0)

        x0_loss = F.mse_loss(student_x0, teacher_x0)
        pred_loss = F.mse_loss(student_pred, teacher_pred)
        loss = x0_loss + (self.config.prediction_loss_weight * pred_loss)
        return {
            "val_loss": float(loss.item()),
            "val_x0_loss": float(x0_loss.item()),
            "val_prediction_loss": float(pred_loss.item()),
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
            topology_anchor_policy=build_topology_anchor_policy_metadata(
                semantic_role_prior_strength=self.config.semantic_role_prior_strength,
                semantic_puzzle_offset=self.config.semantic_puzzle_offset,
                topology_supervision_mode=self.config.topology_supervision_mode,
            ),
        )

    def save_resume_checkpoint(self, path: str, metrics: Optional[Dict[str, Any]] = None) -> None:
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
            },
            extra={
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "base_diffusion_checkpoint": str(self.config.base_diffusion_checkpoint),
                "checkpoint_kind": "resume",
                "contains": ["lora", "optimizer"],
                "topology_anchor_policy": build_topology_anchor_policy_metadata(
                    semantic_role_prior_strength=self.config.semantic_role_prior_strength,
                    semantic_puzzle_offset=self.config.semantic_puzzle_offset,
                    topology_supervision_mode=self.config.topology_supervision_mode,
                ),
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


def train_fast_sampler(config: FastSamplerTrainingConfig) -> ConsistencyLoRATrainer:
    resolved_seed = seed_everything(int(getattr(config, "seed", 42)))
    logger.info("Fast-sampler trainer seeds initialized: seed=%d", resolved_seed)
    trainer = ConsistencyLoRATrainer(config)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_dir = config.data_dir or trainer.base_bundle.config.data_dir
    train_loader = create_dataloader(
        data_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
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
    val_loader = create_dataloader(
        data_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle_val,
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

    metrics_logger = MetricsLogger(
        log_dir=str(checkpoint_dir / "logs"),
        experiment_name="fast_sampler_training",
    )
    best_val = float("inf")
    metrics: Dict[str, Any] = {}
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
            best_val = float(latest_metrics.get("best_val_loss", latest_metrics.get("val_loss", best_val)))
        logger.info("Auto-resumed fast-sampler training from %s", resume_path)

    for epoch in range(int(getattr(trainer, "epoch", -1)) + 1, config.epochs):
        trainer.epoch = epoch
        running = {"loss": 0.0, "x0_loss": 0.0, "prediction_loss": 0.0}
        count = 0
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                real_maps, graph_list = batch_data
            else:
                real_maps, graph_list = batch_data, None
            step_metrics = trainer.distill_step(real_maps.to(trainer.device), graph_list)
            for key, value in step_metrics.items():
                running[key] += float(value)
            count += 1
            if batch_idx % 10 == 0:
                logger.debug(
                    "Fast sampler batch %d: loss=%.4f x0=%.4f pred=%.4f",
                    batch_idx,
                    step_metrics["loss"],
                    step_metrics["x0_loss"],
                    step_metrics["prediction_loss"],
                )

        train_metrics = {k: (v / max(1, count)) for k, v in running.items()}
        val_metrics = trainer.validate(val_loader)
        metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        metrics_logger.log(metrics)
        logger.info(
            "Fast sampler epoch %d/%d: loss=%.4f val_loss=%.4f",
            epoch + 1,
            config.epochs,
            train_metrics["loss"],
            val_metrics["val_loss"],
        )

        if (epoch + 1) % config.save_every == 0:
            trainer.save_resume_checkpoint(str(checkpoint_dir / f"fast_sampler_resume_epoch_{epoch+1:04d}.pth"), metrics)
            prune_checkpoints(
                checkpoint_dir=str(checkpoint_dir),
                pattern="fast_sampler_resume_epoch_*.pth",
                keep_last=int(getattr(config, "keep_last", 2)),
            )
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            trainer.save_checkpoint(str(checkpoint_dir / "fast_sampler_best.pth"), metrics)

        latest_metrics = dict(metrics)
        latest_metrics["best_val_loss"] = float(best_val)
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
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
