"""Bounded cache for frozen tokenizer latents."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch


class FrozenLatentCache:
    """Small LRU cache for deterministic frozen-encoder outputs."""

    def __init__(self, *, enabled: bool = True, max_items: int = 4096):
        self.enabled = bool(enabled)
        self.max_items = int(max(0, max_items))
        self._items: "OrderedDict[Tuple[Any, ...], torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def clear(self, *, reset_stats: bool = False) -> None:
        """Discard cached latents, optionally resetting observability counters."""
        self._items.clear()
        if reset_stats:
            self.hits = 0
            self.misses = 0

    @staticmethod
    def module_version_token(module: Any) -> Tuple[Any, ...]:
        """Return a cheap token that changes after ordinary in-place state updates."""
        named_parameters = getattr(module, "named_parameters", None)
        named_buffers = getattr(module, "named_buffers", None)
        if not callable(named_parameters) or not callable(named_buffers):
            return (type(module).__module__, type(module).__qualname__, id(module))
        entries = []
        for kind, iterator in (
            ("parameter", named_parameters()),
            ("buffer", named_buffers()),
        ):
            for name, tensor in iterator:
                entries.append(
                    (
                        kind,
                        str(name),
                        id(tensor),
                        int(getattr(tensor, "_version", 0)),
                        tuple(int(dim) for dim in tensor.shape),
                        str(tensor.dtype),
                    )
                )
        return tuple(entries)

    @staticmethod
    def module_state_fingerprint(module: Any) -> str:
        """Hash the actual module state once for cache namespace isolation."""
        digest = hashlib.blake2b(digest_size=20)
        digest.update(f"{type(module).__module__}.{type(module).__qualname__}".encode("utf-8"))
        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            digest.update(str(id(module)).encode("ascii"))
            return digest.hexdigest()

        for name, value in state_dict().items():
            digest.update(str(name).encode("utf-8"))
            if not isinstance(value, torch.Tensor):
                digest.update(repr(value).encode("utf-8"))
                continue
            sample = value.detach().to(device="cpu").contiguous()
            digest.update(str(tuple(int(dim) for dim in sample.shape)).encode("ascii"))
            digest.update(str(sample.dtype).encode("ascii"))
            if sample.numel() > 0:
                digest.update(sample.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def key_for_tensor(
        self,
        tensor: torch.Tensor,
        *,
        namespace: Tuple[Any, ...] = (),
    ) -> Optional[Tuple[Any, ...]]:
        """Build a stable CPU-content key for one tensor."""
        if not self.enabled or self.max_items <= 0:
            return None
        if not isinstance(tensor, torch.Tensor) or tensor.requires_grad:
            return None
        sample = tensor.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(sample).all()):
            return None
        digest = hashlib.blake2b(sample.numpy().tobytes(), digest_size=16).hexdigest()
        return (*namespace, tuple(int(dim) for dim in sample.shape), str(sample.dtype), digest)

    def get(self, key: Optional[Tuple[Any, ...]], *, device: torch.device) -> Optional[torch.Tensor]:
        if key is None:
            return None
        cached = self._items.get(key)
        if cached is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return cached.to(device=device, non_blocking=True)

    def put(self, key: Optional[Tuple[Any, ...]], value: torch.Tensor) -> None:
        if key is None or not self.enabled or self.max_items <= 0:
            return
        self._items[key] = value.detach().to(device="cpu")
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    @property
    def total_lookups(self) -> int:
        return int(self.hits + self.misses)

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        return float(self.hits) / float(total) if total > 0 else 0.0
