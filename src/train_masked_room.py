"""
Training pipeline for the graph-conditioned discrete masked room model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
from src.core.discrete_masked_model import (
    DiscreteMaskedRoomModel,
    create_discrete_masked_model,
)
from src.pipeline.room_topology_conditioning import ROOM_TOPOLOGY_CHANNEL_COUNT
from src.utils.checkpoint import MetricsLogger, write_checkpoint_metadata
from src.zelda_data.zelda_loader import create_dataloader

logger = logging.getLogger(__name__)


class MaskedRoomTrainingConfig:
    def __init__(
        self,
        data_dir: str = "Data/The Legend of Zelda",
        batch_size: int = 8,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        context_dim: int = 256,
        condition_gnn_type: str = "gcn",
        graph_conditioning_mode: str = "node_sequence",
        model_channels: int = 128,
        masked_steps: int = 8,
        checkpoint_dir: str = "./checkpoints/masked_room",
        save_every: int = 10,
        device: str = "auto",
        quick: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.epochs = 2 if quick else int(epochs)
        self.learning_rate = float(learning_rate)
        self.context_dim = int(context_dim)
        self.condition_gnn_type = str(condition_gnn_type).strip().lower()
        self.graph_conditioning_mode = str(graph_conditioning_mode).strip().lower()
        self.model_channels = int(model_channels)
        self.masked_steps = int(max(1, masked_steps))
        self.checkpoint_dir = str(checkpoint_dir)
        self.save_every = int(save_every)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else str(device)
        self.quick = bool(quick)

        if self.condition_gnn_type not in {"gcn", "gat", "sage"}:
            raise ValueError("condition_gnn_type must be 'gcn', 'gat', or 'sage'.")
        if self.graph_conditioning_mode not in {"node_sequence", "pooled"}:
            raise ValueError("graph_conditioning_mode must be 'node_sequence' or 'pooled'.")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class MaskedRoomTrainer:
    def __init__(
        self,
        config: MaskedRoomTrainingConfig,
        *,
        model: Optional[DiscreteMaskedRoomModel] = None,
        condition_encoder: Optional[DualStreamConditionEncoder] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = (model or create_discrete_masked_model(
            num_classes=44,
            hidden_dim=64,
            model_channels=config.model_channels,
            context_dim=config.context_dim,
            num_steps=config.masked_steps,
            room_topology_channels=ROOM_TOPOLOGY_CHANNEL_COUNT,
        )).to(self.device)
        self.condition_encoder = (condition_encoder or create_condition_encoder(
            latent_dim=64,
            output_dim=config.context_dim,
            gnn_type=config.condition_gnn_type,
        )).to(self.device)
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.condition_encoder.parameters()),
            lr=config.learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.epochs),
            eta_min=1e-6,
        )

    @staticmethod
    def _to_token_ids(real_maps: torch.Tensor, num_classes: int = 44) -> torch.Tensor:
        if real_maps.dim() != 4 or int(real_maps.shape[1]) != 1:
            raise ValueError(f"Expected room tensors [B,1,H,W], got {tuple(real_maps.shape)}")
        tile_ids = (real_maps.squeeze(1) * float(num_classes - 1)).round().long()
        return tile_ids.clamp_(0, num_classes - 1)

    @staticmethod
    def _encode_edge_features(graph_dict: dict, device: torch.device) -> Optional[torch.Tensor]:
        edge_attr = graph_dict.get("edge_attr")
        if edge_attr is None:
            return None
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        edge_attr = edge_attr.to(device)
        if edge_attr.numel() == 0:
            return None
        num_edge_types = 8
        return torch.nn.functional.one_hot(edge_attr.clamp(0, num_edge_types - 1), num_classes=num_edge_types).float()

    def _stack_conditioning_vectors(self, cond_vectors: List[torch.Tensor]) -> torch.Tensor:
        if not cond_vectors:
            raise ValueError("cond_vectors must be non-empty")
        if self.config.graph_conditioning_mode == "node_sequence":
            max_nodes = max(int(c.shape[0]) for c in cond_vectors)
            padded = []
            for c in cond_vectors:
                if int(c.shape[0]) < max_nodes:
                    pad = torch.zeros(max_nodes - int(c.shape[0]), int(c.shape[1]), device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=0)
                padded.append(c.unsqueeze(0))
            return torch.cat(padded, dim=0)
        return torch.cat(cond_vectors, dim=0)

    def _encode_graph_conditioning(self, graph_dict: dict) -> torch.Tensor:
        node_features = graph_dict["node_features"].to(self.device)
        edge_index = graph_dict["edge_index"].to(self.device)
        edge_features = self._encode_edge_features(graph_dict, self.device)
        tpe = graph_dict.get("tpe")
        tpe = tpe.to(self.device) if isinstance(tpe, torch.Tensor) else None

        c_global = self.condition_encoder.encode_global_only(
            node_features,
            edge_index,
            edge_features=edge_features,
            tpe=tpe,
        )

        boundary_constraints = graph_dict.get("boundary_constraints")
        room_position = graph_dict.get("room_position")
        current_node_idx = graph_dict.get("current_node_idx")
        has_room_anchor = bool(graph_dict.get("has_room_anchor", False)) or (
            isinstance(boundary_constraints, torch.Tensor)
            and isinstance(room_position, torch.Tensor)
        )
        if has_room_anchor:
            boundary_constraints = boundary_constraints.to(self.device, dtype=torch.float32)
            room_position = room_position.to(self.device, dtype=torch.float32)
            if boundary_constraints.dim() == 1:
                boundary_constraints = boundary_constraints.unsqueeze(0)
            if room_position.dim() == 1:
                room_position = room_position.unsqueeze(0)
            room_anchor = self.condition_encoder(
                neighbor_latents={"N": None, "S": None, "E": None, "W": None},
                boundary_constraints=boundary_constraints,
                position=room_position,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                tpe=tpe,
                current_node_idx=int(current_node_idx) if current_node_idx is not None else None,
            )
            if self.config.graph_conditioning_mode == "node_sequence":
                return torch.cat([room_anchor, c_global], dim=0)
            return room_anchor

        if self.config.graph_conditioning_mode == "node_sequence":
            return c_global
        return c_global.mean(dim=0, keepdim=True)

    def _normalize_graph_sample(self, graph_dict: dict) -> Dict[str, torch.Tensor]:
        node_features = graph_dict["node_features"]
        edge_index = graph_dict["edge_index"]
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_features = node_features.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)

        num_nodes = int(node_features.shape[0])
        tpe = graph_dict.get("tpe")
        if not isinstance(tpe, torch.Tensor):
            tpe = torch.zeros(num_nodes, 8, dtype=torch.float32)
        tpe = tpe.to(self.device, dtype=torch.float32)

        node_positions = graph_dict.get("node_positions")
        if not isinstance(node_positions, torch.Tensor):
            node_positions = torch.zeros(num_nodes, 2, dtype=torch.float32)
        node_positions = node_positions.to(self.device, dtype=torch.float32)

        node_mask = graph_dict.get("node_mask")
        if not isinstance(node_mask, torch.Tensor):
            node_mask = torch.ones(num_nodes, dtype=torch.float32)
        node_mask = node_mask.to(self.device, dtype=torch.float32)
        if node_mask.dim() == 2:
            node_mask = node_mask.squeeze(0)

        room_topology_map = graph_dict.get("room_topology_map")
        if isinstance(room_topology_map, torch.Tensor):
            room_topology_map = room_topology_map.to(self.device, dtype=torch.float32)
            if room_topology_map.dim() == 4:
                room_topology_map = room_topology_map.squeeze(0)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "tpe": tpe,
            "node_positions": node_positions,
            "node_mask": node_mask,
            "has_room_anchor": bool(graph_dict.get("has_room_anchor", False)) or (
                isinstance(graph_dict.get("boundary_constraints"), torch.Tensor)
                and isinstance(graph_dict.get("room_position"), torch.Tensor)
            ),
            **({"room_topology_map": room_topology_map} if isinstance(room_topology_map, torch.Tensor) else {}),
        }

    def _stack_graph_batch(self, graph_list: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
        if not graph_list:
            return None
        samples = [self._normalize_graph_sample(graph_dict) for graph_dict in graph_list]
        max_nodes = max(int(sample["node_features"].shape[0]) for sample in samples)
        feat_dim = max(int(sample["node_features"].shape[1]) for sample in samples)
        tpe_dim = max(int(sample["tpe"].shape[1]) for sample in samples)
        max_edges = max(int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0 for sample in samples)

        node_features_batch = torch.zeros(len(samples), max_nodes, feat_dim, device=self.device, dtype=torch.float32)
        tpe_batch = torch.zeros(len(samples), max_nodes, tpe_dim, device=self.device, dtype=torch.float32)
        node_positions_batch = torch.zeros(len(samples), max_nodes, 2, device=self.device, dtype=torch.float32)
        node_mask_batch = torch.zeros(len(samples), max_nodes, device=self.device, dtype=torch.float32)
        edge_index_batch = torch.full((len(samples), 2, max_edges), -1, device=self.device, dtype=torch.long)
        topo_maps = []
        for i, sample in enumerate(samples):
            n = int(sample["node_features"].shape[0])
            if n > 0:
                node_features_batch[i, :n] = sample["node_features"]
                tpe_batch[i, :n] = sample["tpe"]
                node_positions_batch[i, :n] = sample["node_positions"]
                node_mask_batch[i, :n] = sample["node_mask"]
            e = int(sample["edge_index"].shape[1]) if sample["edge_index"].dim() == 2 else 0
            if e > 0:
                edge_index_batch[i, :, :e] = sample["edge_index"]
            topo = sample.get("room_topology_map")
            if isinstance(topo, torch.Tensor):
                topo_maps.append(topo.unsqueeze(0) if topo.dim() == 3 else topo)

        batch_graph = {
            "node_features": node_features_batch,
            "edge_index": edge_index_batch,
            "tpe": tpe_batch,
            "node_positions": node_positions_batch,
            "node_mask": node_mask_batch,
            "has_room_anchor": bool(samples[0].get("has_room_anchor", False)),
        }
        if len(topo_maps) == len(samples):
            batch_graph["room_topology_map"] = torch.cat(topo_maps, dim=0)
        return batch_graph

    def _step(
        self,
        real_maps: torch.Tensor,
        graph_list: Optional[List[dict]],
        *,
        train: bool,
    ) -> Dict[str, float]:
        token_ids = self._to_token_ids(real_maps.to(self.device))
        if graph_list:
            cond_vectors = [self._encode_graph_conditioning(graph_dict) for graph_dict in graph_list]
            conditioning = self._stack_conditioning_vectors(cond_vectors)
            graph_batch = self._stack_graph_batch(graph_list)
            topo = graph_batch.get("room_topology_map") if isinstance(graph_batch, dict) else None
        else:
            conditioning = torch.zeros(token_ids.shape[0], 1, self.config.context_dim, device=self.device)
            graph_batch = None
            topo = None

        fixed_tokens, fixed_mask = DiscreteMaskedRoomModel.build_fixed_mask_from_topology_map(
            token_ids,
            topo,
            num_classes=44,
        )
        loss, metrics = self.model.training_loss(
            token_ids,
            conditioning,
            graph_data=graph_batch,
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
        )
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.condition_encoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
        return metrics

    def save_checkpoint(self, path: str, metrics: Dict[str, Any]) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "condition_encoder_state_dict": self.condition_encoder.state_dict(),
            "config": self.config.to_dict(),
            "metrics": dict(metrics),
        }
        torch.save(payload, path)
        write_checkpoint_metadata(
            path,
            model_type="masked_room_model",
            architecture={
                "num_classes": 44,
                "hidden_dim": int(self.model.hidden_dim),
                "context_dim": int(self.model.context_dim),
                "masked_steps": int(self.config.masked_steps),
            },
            extra={"graph_conditioning_mode": self.config.graph_conditioning_mode},
        )


def train_masked_room(config: MaskedRoomTrainingConfig) -> MaskedRoomTrainer:
    trainer = MaskedRoomTrainer(config)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=True,
        use_vglc=True,
        normalize=True,
        room_level=True,
        load_graphs=True,
    )
    val_loader = create_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        shuffle=False,
        use_vglc=True,
        normalize=True,
        room_level=True,
        load_graphs=True,
    )

    metrics_logger = MetricsLogger(
        log_dir=str(checkpoint_dir / "logs"),
        experiment_name="masked_room_training",
    )
    best_val = float("inf")

    for epoch in range(config.epochs):
        trainer.model.train()
        trainer.condition_encoder.train()
        train_sum = {"loss": 0.0, "mask_ratio": 0.0, "masked_fraction": 0.0}
        train_batches = 0
        for batch in train_loader:
            real_maps, graph_list = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
            metrics = trainer._step(real_maps, graph_list, train=True)
            for key, value in metrics.items():
                train_sum[key] += float(value)
            train_batches += 1

        trainer.model.eval()
        trainer.condition_encoder.eval()
        val_sum = {"val_loss": 0.0, "val_mask_ratio": 0.0, "val_masked_fraction": 0.0}
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                real_maps, graph_list = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
                metrics = trainer._step(real_maps, graph_list, train=False)
                val_sum["val_loss"] += float(metrics["loss"])
                val_sum["val_mask_ratio"] += float(metrics["mask_ratio"])
                val_sum["val_masked_fraction"] += float(metrics["masked_fraction"])
                val_batches += 1

        trainer.scheduler.step()
        epoch_metrics = {
            "epoch": epoch,
            **{k: v / max(1, train_batches) for k, v in train_sum.items()},
            **{k: v / max(1, val_batches) for k, v in val_sum.items()},
        }
        metrics_logger.log(epoch_metrics)
        logger.info(
            "Masked room epoch %d/%d: loss=%.4f val_loss=%.4f",
            epoch + 1,
            config.epochs,
            epoch_metrics["loss"],
            epoch_metrics["val_loss"],
        )
        if (epoch + 1) % config.save_every == 0:
            trainer.save_checkpoint(str(checkpoint_dir / f"masked_room_epoch_{epoch + 1:04d}.pth"), epoch_metrics)
        if epoch_metrics["val_loss"] < best_val:
            best_val = epoch_metrics["val_loss"]
            trainer.save_checkpoint(str(checkpoint_dir / "masked_room_best.pth"), epoch_metrics)

    trainer.save_checkpoint(str(checkpoint_dir / "masked_room_final.pth"), epoch_metrics)
    metrics_logger.save()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the graph-conditioned discrete masked room model.")
    parser.add_argument("--data-dir", type=str, default="Data/The Legend of Zelda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--condition-gnn-type", type=str, default="gcn")
    parser.add_argument("--graph-conditioning-mode", type=str, default="node_sequence")
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--masked-steps", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/masked_room")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = MaskedRoomTrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        context_dim=args.context_dim,
        condition_gnn_type=args.condition_gnn_type,
        graph_conditioning_mode=args.graph_conditioning_mode,
        model_channels=args.model_channels,
        masked_steps=args.masked_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        device=args.device,
        quick=args.quick,
    )
    train_masked_room(config)


if __name__ == "__main__":
    main()
