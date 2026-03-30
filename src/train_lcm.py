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

from src.optimization.lcm_lora import (
    DEFAULT_LORA_TARGETS,
    extract_lora_state_dict,
    freeze_non_lora_parameters,
    inject_lora_into_model,
    save_fast_sampler_checkpoint,
)
from src.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig
from src.utils.checkpoint import MetricsLogger
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
        normalize: bool = True,
        room_level: bool = True,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        optimizer_weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        num_inference_steps: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        prediction_loss_weight: float = 0.25,
        save_every: int = 5,
        checkpoint_dir: str = "./checkpoints/fast_sampler",
        device: str = "auto",
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
        self.normalize = bool(normalize)
        self.room_level = bool(room_level)
        self.epochs = 1 if quick else int(epochs)
        self.learning_rate = float(learning_rate)
        self.optimizer_weight_decay = float(max(0.0, optimizer_weight_decay))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))
        self.num_inference_steps = int(max(1, num_inference_steps))
        self.lora_rank = int(max(1, lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.prediction_loss_weight = float(max(0.0, prediction_loss_weight))
        self.save_every = int(max(1, save_every))
        self.checkpoint_dir = str(checkpoint_dir)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
        self.quick = bool(quick)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
        )


def train_fast_sampler(config: FastSamplerTrainingConfig) -> ConsistencyLoRATrainer:
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
        use_vglc=True,
        normalize=config.normalize,
        room_level=config.room_level,
        load_graphs=True,
    )
    val_loader = create_dataloader(
        data_dir,
        batch_size=config.batch_size,
        shuffle=config.shuffle_val,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        use_vglc=True,
        normalize=config.normalize,
        room_level=config.room_level,
        load_graphs=True,
    )

    metrics_logger = MetricsLogger(
        log_dir=str(checkpoint_dir / "logs"),
        experiment_name="fast_sampler_training",
    )
    best_val = float("inf")

    for epoch in range(config.epochs):
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
            trainer.save_checkpoint(str(checkpoint_dir / f"fast_sampler_epoch_{epoch+1:04d}.pth"), metrics)
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            trainer.save_checkpoint(str(checkpoint_dir / "fast_sampler_best.pth"), metrics)

    trainer.save_checkpoint(str(checkpoint_dir / "fast_sampler_final.pth"), metrics)
    metrics_logger.save()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the repo's graph-aware consistency-LoRA fast sampler."
    )
    parser.add_argument("--base-diffusion-checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/fast_sampler")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = FastSamplerTrainingConfig(
        base_diffusion_checkpoint=args.base_diffusion_checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_inference_steps=args.num_inference_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        quick=args.quick,
    )
    train_fast_sampler(config)


if __name__ == "__main__":
    main()
