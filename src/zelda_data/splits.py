"""Canonical dataset splits for the local Zelda corpus.

The split unit is the dungeon id, not an individual room.  This keeps every
room from Dungeon 9 unseen during room-model training while preserving both
Quest variants for the held-out evaluation protocol.
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Optional, Sequence, Tuple


ALL_DUNGEONS: Tuple[int, ...] = tuple(range(1, 10))
DEFAULT_TRAIN_DUNGEONS: Tuple[int, ...] = tuple(range(1, 9))
DEFAULT_TEST_DUNGEONS: Tuple[int, ...] = (9,)
DEFAULT_VARIANTS: Tuple[int, ...] = (1, 2)


def normalize_dungeon_ids(values: Optional[Iterable[int]]) -> Optional[Tuple[int, ...]]:
    """Return sorted unique dungeon ids, or ``None`` to mean all dungeons."""
    if values is None:
        return None
    normalized = _strict_ids(values)
    if not normalized:
        raise ValueError("Dungeon ids must be non-empty; use None for the default selection.")
    invalid = [v for v in normalized if v not in ALL_DUNGEONS]
    if invalid:
        raise ValueError(f"Unsupported Zelda dungeon ids {invalid}; expected ids in {ALL_DUNGEONS}.")
    return normalized


def normalize_variants(values: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """Return sorted unique Quest variants."""
    if values is None:
        return DEFAULT_VARIANTS
    normalized = _strict_ids(values)
    if not normalized:
        raise ValueError("Quest variants must be non-empty; use None for the default selection.")
    invalid = [v for v in normalized if v not in DEFAULT_VARIANTS]
    if invalid:
        raise ValueError(f"Unsupported Zelda variants {invalid}; expected variants in {DEFAULT_VARIANTS}.")
    return normalized


def _strict_ids(values: Iterable[int]) -> Tuple[int, ...]:
    """Reject lossy casts that could silently change a held-out identity."""
    ids = list(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in ids):
        raise ValueError("Dataset identities must be integers, without coercion.")
    return tuple(sorted({int(value) for value in ids}))


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


def build_dungeon_kfold_splits(
    train_dungeons: Optional[Iterable[int]],
    *,
    num_folds: int,
    fold_index: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Deterministically split training dungeons into disjoint train and validation folds."""
    dungeons = normalize_dungeon_ids(train_dungeons)
    if dungeons is None:
        dungeons = DEFAULT_TRAIN_DUNGEONS
    folds = int(num_folds)
    fold = int(fold_index)
    if folds < 2 or folds > len(dungeons):
        raise ValueError(
            f"num_folds must be in [2, {len(dungeons)}] for dungeons {dungeons}, got {num_folds}."
        )
    if fold < 0 or fold >= folds:
        raise ValueError(f"fold_index must be in [0, {folds - 1}], got {fold_index}.")
    val_dungeons = tuple(d for idx, d in enumerate(dungeons) if idx % folds == fold)
    fold_train_dungeons = tuple(d for d in dungeons if d not in val_dungeons)
    if not val_dungeons or not fold_train_dungeons:
        raise ValueError("K-fold split produced an empty train or validation dungeon split.")
    return fold_train_dungeons, val_dungeons


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
