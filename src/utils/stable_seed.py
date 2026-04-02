"""
Stable helpers for reproducible seed derivation.

Python's built-in ``hash(...)`` is salted per process, so it must not be used
to derive experiment seeds or persistent IDs that are expected to be stable
across runs. These helpers provide deterministic integer hashing instead.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Optional


def stable_int_hash(value: Any, *, digest_size: int = 8) -> int:
    """Return a deterministic unsigned integer hash for arbitrary JSON-ish values."""
    if isinstance(value, Integral):
        return int(value) & ((1 << (digest_size * 8)) - 1)

    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    digest = hashlib.blake2b(payload.encode("utf-8", errors="ignore"), digest_size=digest_size).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def stable_seed_offset(value: Any, *, modulo: Optional[int] = None, digest_size: int = 4) -> int:
    """Return a deterministic seed offset, optionally reduced to a bounded range."""
    offset = stable_int_hash(value, digest_size=digest_size)
    if modulo is not None:
        modulo = int(modulo)
        if modulo <= 0:
            raise ValueError(f"modulo must be positive when provided, got {modulo}.")
        offset %= modulo
    return int(offset)


__all__ = ["stable_int_hash", "stable_seed_offset"]
