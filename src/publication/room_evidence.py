"""Measurements of unchanged room outputs, before boundary carving or compilation.

Exact training matches and floor-mask connectivity are descriptive diagnostics;
neither is a test of memorization or full game solvability on its own.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE

from .protocol import payload_sha256


def categorical_room(grid):
    array = np.asarray(grid)
    if array.ndim != 2 or not array.size or array.dtype.kind not in "iuf":
        raise ValueError("Room evidence requires a nonempty two-dimensional categorical grid.")
    # ZeldaRoomDataset stores categorical IDs as float32 for training tensors.
    # Accept exact integers in that storage format, never round probabilities.
    if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
        raise ValueError("Room evidence requires finite integral tile IDs.")
    if np.any(array < 0) or np.any(array > np.iinfo(np.int32).max):
        raise ValueError("Room tile IDs must fit nonnegative int32.")
    return np.array(array, dtype=np.int32, copy=True)


def shape_preserving_dihedral_transforms(grid):
    """Return all shape-preserving dihedral (V4 for H!=W, D4 for H==W) views."""
    array = categorical_room(grid)
    transforms = [
        ("identity", array),
        ("flip_h", np.flip(array, axis=1).copy()),
        ("flip_v", np.flip(array, axis=0).copy()),
        ("rot180", np.flip(array, axis=(0, 1)).copy()),
    ]
    if array.shape[0] == array.shape[1]:
        transforms.extend([
            ("rot90", np.rot90(array, k=1).copy()),
            ("rot270", np.rot90(array, k=3).copy()),
            ("transpose", array.T.copy()),
            ("antitranspose", np.rot90(np.flip(array, axis=1), k=1).copy()),
        ])
    return transforms


def room_fingerprint(grid):
    array = categorical_room(grid)
    return hashlib.sha256(
        np.asarray(array.shape, dtype="<i8").tobytes() + array.astype("<i4").tobytes()
    ).hexdigest()


def floor_structure(grid):
    """Count four-connected FLOOR components; gates/entities are not floors."""
    array = categorical_room(grid)
    floor = array == int(SEMANTIC_PALETTE["FLOOR"])
    unseen = set(zip(*np.where(floor)))
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    counts, frequencies = np.unique(array, return_counts=True)
    return {
        "predicate": "four_connected_floor_tiles_only",
        "floor_tiles": int(floor.sum()), "floor_components": len(sizes),
        "largest_floor_component_fraction": max(sizes, default=0) / max(1, int(floor.sum())),
        "unknown_tile_fraction": float(np.mean(~np.isin(array, list(SEMANTIC_PALETTE.values())))),
        "tile_counts": {str(int(tile)): int(count) for tile, count in zip(counts, frequencies)},
    }


class TrainingRoomReference:
    """Immutable, fold-specific source rooms; no held-out rooms enter fitting."""

    def __init__(self, rooms):
        arrays = [categorical_room(room) for room in rooms]
        if not arrays or len({room.shape for room in arrays}) != 1:
            raise ValueError("Training references require nonempty rooms of one shape.")
        self._rooms = np.stack(arrays)
        self._rooms.flags.writeable = False
        self.fingerprints = tuple(room_fingerprint(room) for room in arrays)
        self.sha256 = payload_sha256({"ordered_room_sha256": self.fingerprints})

    @classmethod
    def from_config(cls, config):
        from src.zelda_data.zelda_loader import ZeldaRoomDataset
        data = config["dataset"]
        dataset = ZeldaRoomDataset(data_dir=data["data_dir"], dungeon_ids=data["train_dungeons"],
                                   variants=data["variants"], load_graphs=False)
        return cls(dataset.rooms)

    def rooms(self):
        return self._rooms.copy()

    def compare(self, grid):
        array = categorical_room(grid)
        if array.shape != self._rooms.shape[1:]:
            raise ValueError("Generated room shape differs from the training reference.")
        distances = np.count_nonzero(self._rooms != array, axis=(1, 2))
        closest = int(np.argmin(distances))
        floor_id = int(SEMANTIC_PALETTE["FLOOR"])
        floor_matches = np.all((self._rooms == floor_id) == (array == floor_id), axis=(1, 2))

        sym_best_dist = int(distances[closest])
        sym_best_name = "identity"
        sym_best_idx = closest
        sym_exact_mask = (distances == 0).copy()
        for name, view in shape_preserving_dihedral_transforms(array)[1:]:
            view_distances = np.count_nonzero(self._rooms != view, axis=(1, 2))
            sym_exact_mask |= (view_distances == 0)
            idx = int(np.argmin(view_distances))
            dist = int(view_distances[idx])
            if dist < sym_best_dist:
                sym_best_dist = dist
                sym_best_name = name
                sym_best_idx = idx
        sym_group = "D4" if array.shape[0] == array.shape[1] else "V4_klein_four"

        return {"reference_sha256": self.sha256, "reference_room_count": len(self._rooms),
                "exact_training_match": bool(distances[closest] == 0),
                "matching_training_room_count": int(np.count_nonzero(distances == 0)),
                "nearest_training_room_sha256": self.fingerprints[closest],
                "nearest_training_changed_tiles": int(distances[closest]),
                "nearest_training_hamming_fraction": float(distances[closest] / array.size),
                "exact_training_floor_mask_match": bool(np.any(floor_matches)),
                "symmetry_group": sym_group,
                "symmetry_exact_training_match": bool(sym_best_dist == 0),
                "symmetry_matching_training_room_count": int(np.count_nonzero(sym_exact_mask)),
                "symmetry_nearest_training_room_sha256": self.fingerprints[sym_best_idx],
                "symmetry_nearest_transform": sym_best_name,
                "symmetry_nearest_training_changed_tiles": int(sym_best_dist),
                "symmetry_nearest_training_hamming_fraction": float(sym_best_dist / array.size)}


def heldout_overlap(reference, heldout_rooms):
    """Retain per-room strata so repeated layouts are not called unseen layouts."""
    records = [{"room_sha256": room_fingerprint(room), **reference.compare(room)} for room in heldout_rooms]
    if not records:
        raise ValueError("Held-out overlap audit requires nonempty source rooms.")
    repeated = sum(item["exact_training_match"] for item in records)
    sym_repeated = sum(item["symmetry_exact_training_match"] for item in records)
    return {"comparison": "exact_tiles_same_orientation", "rooms": records,
            "heldout_room_count": len(records), "repeated_layout_count": repeated,
            "unseen_layout_count": len(records) - repeated,
            "symmetry_repeated_layout_count": sym_repeated,
            "symmetry_unseen_layout_count": len(records) - sym_repeated}
