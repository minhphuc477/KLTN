"""Graph Collation and Batch Sampling for Zelda Dungeons.

Provides:
- `graph_collate_fn`: PyTorch collate function for variable-size mission graphs.
- `DungeonBatchSampler`: Custom sampler yielding coherent room sets per dungeon.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

__all__ = ["graph_collate_fn", "DungeonBatchSampler"]


def graph_collate_fn(batch):
    """Custom collation function for batches containing (image, graph_dict) pairs.

    Handles variable-size graphs by:
    1. Stacking image tensors normally (already padded to target shape)
    2. Storing graph dicts as a list (since node counts differ per dungeon)
    """
    if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
        images = torch.stack([item[0] for item in batch])
        graphs = []
        for _, raw_graph in batch:
            graph = dict(raw_graph)
            node_features = graph.get("node_features")
            num_nodes = 0
            if isinstance(node_features, torch.Tensor) and node_features.dim() >= 1:
                num_nodes = int(node_features.shape[0])

            existing_batch_idx = graph.get("batch_idx")
            needs_reset = (
                "batch_idx" not in graph
                or not isinstance(existing_batch_idx, torch.Tensor)
                or (existing_batch_idx.numel() > 0 and int(existing_batch_idx.max()) > 0)
            )
            if needs_reset:
                graph["batch_idx"] = torch.zeros(num_nodes, dtype=torch.long)

            graphs.append(graph)

        return images, graphs
    else:
        return torch.stack(batch)


class DungeonBatchSampler(Sampler[List[int]]):
    """Batch sampler that yields all room samples from one dungeon variant.

    Room-level diffusion needs this when training dungeon-scope graph losses:
    the batch dimension must correspond to the graph's node set, not to random
    rooms sampled from unrelated dungeons.
    """

    def __init__(
        self,
        groups: List[List[int]],
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self.groups = [list(group) for group in groups if group]
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> "DungeonBatchSampler":
        base_dataset = getattr(dataset, "dataset", dataset)
        subset_indices = list(getattr(dataset, "indices", range(len(base_dataset))))
        metadata = getattr(base_dataset, "sample_metadata", None)
        if metadata is None:
            raise ValueError("DungeonBatchSampler requires dataset.sample_metadata.")

        grouped: Dict[Any, List[Tuple[int, int]]] = {}
        for local_idx, base_idx in enumerate(subset_indices):
            if int(base_idx) >= len(metadata):
                continue
            meta = dict(metadata[int(base_idx)])
            key = meta.get("dungeon_id")
            if key is None:
                key = (meta.get("dungeon_num"), meta.get("variant"))
            order = (
                int(meta.get("current_node_idx", local_idx))
                if str(meta.get("current_node_idx", "")).strip()
                else local_idx
            )
            grouped.setdefault(key, []).append((order, local_idx))

        groups = [
            [local_idx for _order, local_idx in sorted(items, key=lambda item: item[0])]
            for _key, items in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
        return cls(groups, shuffle=shuffle, drop_last=drop_last, seed=seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(order), generator=generator).tolist()
            order = [order[i] for i in perm]
        for group_idx in order:
            group = self.groups[group_idx]
            if self.drop_last and not group:
                continue
            yield list(group)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(1 for group in self.groups if group)
        return len(self.groups)
