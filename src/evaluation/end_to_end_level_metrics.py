"""
End-to-end structural evaluation helpers for generated dungeon exports.

This module complements the topology benchmark suite with report-facing metrics
for stitched room-generation artifacts. The goal is not to replace the mission-
graph descriptors, but to expose literature-aligned structural diversity and
novelty signals at the final room-grid level.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.core.definitions import semantic_to_vglc_char
from src.evaluation.benchmark_suite import load_vglc_reference_rooms


DEFAULT_REFERENCE_ROOM_LIMIT = 256


def _sanitize_grid_text(text: str) -> str:
    return str(text or "").replace("\r", "").rstrip("\n")


@lru_cache(maxsize=4096)
def _compressed_size(text: str) -> int:
    payload = _sanitize_grid_text(text).encode("utf-8", errors="ignore")
    if not payload:
        payload = b"-"
    return int(len(zlib.compress(payload, level=9)))


def normalized_compression_distance(text_a: str, text_b: str) -> float:
    """
    Compute the normalized compression distance (NCD) between two strings.

    NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    """

    a = _sanitize_grid_text(text_a)
    b = _sanitize_grid_text(text_b)
    if not a and not b:
        return 0.0
    if a == b:
        return 0.0
    ca = _compressed_size(a)
    cb = _compressed_size(b)
    cab = int(len(zlib.compress((a + "\n\0\n" + b).encode("utf-8", errors="ignore"), level=9)))
    denom = max(1, ca, cb)
    return float((cab - min(ca, cb)) / float(denom))


def _safe_stats(values: Sequence[float]) -> Dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def pairwise_ncd_stats(texts: Sequence[str]) -> Dict[str, Any]:
    cleaned = [_sanitize_grid_text(text) for text in texts if _sanitize_grid_text(text)]
    values: list[float] = []
    for idx in range(len(cleaned)):
        for jdx in range(idx + 1, len(cleaned)):
            values.append(normalized_compression_distance(cleaned[idx], cleaned[jdx]))
    return _safe_stats(values)


def nearest_reference_ncd_stats(texts: Sequence[str], reference_texts: Sequence[str]) -> Dict[str, Any]:
    cleaned = [_sanitize_grid_text(text) for text in texts if _sanitize_grid_text(text)]
    references = [_sanitize_grid_text(text) for text in reference_texts if _sanitize_grid_text(text)]
    values: list[float] = []
    if not cleaned or not references:
        return _safe_stats(values)
    for text in cleaned:
        values.append(min(normalized_compression_distance(text, ref) for ref in references))
    return _safe_stats(values)


def symbol_entropy(text: str, *, ignore_chars: str = "\n\r") -> float:
    cleaned = [ch for ch in _sanitize_grid_text(text) if ch not in set(ignore_chars)]
    if not cleaned:
        return 0.0
    counts = Counter(cleaned)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        prob = float(count) / total
        if prob > 0.0:
            entropy -= prob * math.log2(prob)
    return float(entropy)


def _grid_to_vglc_text(grid: np.ndarray) -> str:
    lines = []
    for row in np.asarray(grid, dtype=np.int32):
        lines.append("".join(str(semantic_to_vglc_char(int(value))) for value in row))
    return "\n".join(lines)


@lru_cache(maxsize=8)
def load_reference_room_texts(data_root: str, max_rooms: int = DEFAULT_REFERENCE_ROOM_LIMIT) -> tuple[str, ...]:
    rooms = load_vglc_reference_rooms(Path(data_root), max_rooms=max(1, int(max_rooms)))
    return tuple(_grid_to_vglc_text(np.asarray(room, dtype=np.int32)) for room in rooms)


def compute_end_to_end_structural_metrics(
    *,
    room_texts: Mapping[int, str],
    dungeon_text: str,
    reference_room_texts: Sequence[str],
) -> Dict[str, Any]:
    ordered_room_texts = [
        _sanitize_grid_text(text)
        for _, text in sorted(room_texts.items(), key=lambda kv: int(kv[0]))
        if _sanitize_grid_text(text)
    ]
    unique_room_texts = set(ordered_room_texts)
    room_entropies = [symbol_entropy(text) for text in ordered_room_texts]

    return {
        "room_count": int(len(ordered_room_texts)),
        "unique_room_count": int(len(unique_room_texts)),
        "room_unique_ratio": (
            float(len(unique_room_texts)) / float(len(ordered_room_texts))
            if ordered_room_texts
            else None
        ),
        "room_symbol_entropy_mean": float(np.mean(room_entropies)) if room_entropies else None,
        "room_symbol_entropy_std": float(np.std(room_entropies)) if room_entropies else None,
        "dungeon_symbol_entropy_non_void": float(symbol_entropy(dungeon_text, ignore_chars="\n\r-")),
        "room_pairwise_ncd": pairwise_ncd_stats(ordered_room_texts),
        "room_nearest_reference_ncd": nearest_reference_ncd_stats(
            ordered_room_texts,
            reference_room_texts,
        ),
        "reference_room_count": int(len(reference_room_texts)),
    }
