from __future__ import annotations

import torch
import pytest

from src.utils.frozen_latent_cache import FrozenLatentCache


def test_module_state_fingerprint_is_stable_and_tracks_state_changes() -> None:
    module = torch.nn.Linear(16, 8)

    first = FrozenLatentCache.module_state_fingerprint(module)
    second = FrozenLatentCache.module_state_fingerprint(module)

    assert first == second
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    assert FrozenLatentCache.module_state_fingerprint(module) != first


def test_module_state_fingerprint_handles_large_tensor_state() -> None:
    module = torch.nn.Linear(2048, 2048, bias=False)
    fingerprint = FrozenLatentCache.module_state_fingerprint(module)

    assert len(fingerprint) == 40


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bfloat16])
def test_scalar_state_can_be_fingerprinted(dtype) -> None:
    module = torch.nn.Module()
    module.register_buffer("counter", torch.tensor(1, dtype=dtype))
    first = FrozenLatentCache.module_state_fingerprint(module)
    module.counter.add_(1)
    assert FrozenLatentCache.module_state_fingerprint(module) != first


def test_bfloat16_cache_keys_use_exact_tensor_bytes() -> None:
    cache = FrozenLatentCache()
    first = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert cache.key_for_tensor(first) == cache.key_for_tensor(first.clone())
    assert cache.key_for_tensor(first) != cache.key_for_tensor(first + 1)


def test_cache_does_not_alias_inserted_or_returned_cpu_tensors() -> None:
    cache = FrozenLatentCache()
    value = torch.tensor([1.0, 2.0])
    key = cache.key_for_tensor(value)
    cache.put(key, value)
    value.zero_()
    retrieved = cache.get(key, device=torch.device("cpu"))
    torch.testing.assert_close(retrieved, torch.tensor([1.0, 2.0]))
    retrieved.add_(10)
    torch.testing.assert_close(cache.get(key, device=torch.device("cpu")), torch.tensor([1.0, 2.0]))
