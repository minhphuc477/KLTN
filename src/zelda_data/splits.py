"""Canonical dataset splits for the local Zelda corpus.

The split unit is the dungeon id, not an individual room.  This keeps every
room from Dungeon 9 unseen during room-model training while preserving both
Quest variants for the held-out evaluation protocol.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple


ALL_DUNGEONS: Tuple[int, ...] = tuple(range(1, 10))
DEFAULT_TRAIN_DUNGEONS: Tuple[int, ...] = tuple(range(1, 9))
DEFAULT_TEST_DUNGEONS: Tuple[int, ...] = (9,)
DEFAULT_VARIANTS: Tuple[int, ...] = (1, 2)


def normalize_dungeon_ids(values: Optional[Iterable[int]]) -> Optional[Tuple[int, ...]]:
    """Return sorted unique dungeon ids, or ``None`` to mean all dungeons."""
    if values is None:
        return None
    normalized = tuple(sorted({int(v) for v in values}))
    if not normalized:
        return None
    invalid = [v for v in normalized if v not in ALL_DUNGEONS]
    if invalid:
        raise ValueError(f"Unsupported Zelda dungeon ids {invalid}; expected ids in {ALL_DUNGEONS}.")
    return normalized


def normalize_variants(values: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """Return sorted unique Quest variants."""
    if values is None:
        return DEFAULT_VARIANTS
    normalized = tuple(sorted({int(v) for v in values}))
    if not normalized:
        return DEFAULT_VARIANTS
    invalid = [v for v in normalized if v not in DEFAULT_VARIANTS]
    if invalid:
        raise ValueError(f"Unsupported Zelda variants {invalid}; expected variants in {DEFAULT_VARIANTS}.")
    return normalized


def validate_disjoint_dungeon_splits(
    train_dungeons: Iterable[int],
    test_dungeons: Iterable[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Normalize train/test dungeon ids and reject holdout leakage."""
    normalized_train = normalize_dungeon_ids(train_dungeons)
    normalized_test = normalize_dungeon_ids(test_dungeons)
    if normalized_train is None or normalized_test is None:
        raise ValueError("Training and test dungeon splits must both be non-empty.")
    overlap = sorted(set(normalized_train).intersection(normalized_test))
    if overlap:
        raise ValueError(
            "Training/test dungeon split overlap would leak holdout data: "
            f"{overlap}. Use disjoint dungeon ids."
        )
    return normalized_train, normalized_test


def split_summary(
    *,
    train_dungeons: Sequence[int] = DEFAULT_TRAIN_DUNGEONS,
    test_dungeons: Sequence[int] = DEFAULT_TEST_DUNGEONS,
    variants: Sequence[int] = DEFAULT_VARIANTS,
) -> dict[str, list[int]]:
    """Small JSON-serializable split description for logs and reports."""
    return {
        "train_dungeons": [int(v) for v in train_dungeons],
        "test_dungeons": [int(v) for v in test_dungeons],
        "variants": [int(v) for v in variants],
    }
