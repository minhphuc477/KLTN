#!/usr/bin/env python3
"""Run the Dungeon 9 holdout audit requested for the thesis.

The protocol is intentionally lightweight enough to run on a local machine:

* fixed split audit: train dungeons 1-8, test dungeon 9, variants 1 and 2;
* nearest-train Hamming novelty for real holdout rooms and generated rooms;
* same room-level MAP-Elites archive bins for every baseline/configuration;
* symbolic baselines (WFC and random walk PCG);
* a small GAN-style ML baseline trained only on the train split;
* diagnostic component ablations that isolate graph conditioning, rule/route
  guidance, and symbolic repair effects without requiring a full diffusion
  retrain in the current run.

The diagnostic ablations are not a replacement for retraining the diffusion
checkpoint under the fixed split. They produce thesis-facing audit tables and a
reproducible runner for the final full-budget pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_system import merge_config
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, TileID
from src.core.latent_diffusion import CrossAttention, SelfAttention, create_latent_diffusion
from src.generation.weighted_bayesian_wfc import (
    WeightedBayesianWFC,
    WeightedBayesianWFCConfig,
    extract_tile_priors_from_vqvae,
)
from src.zelda_data.splits import DEFAULT_TEST_DUNGEONS, DEFAULT_TRAIN_DUNGEONS, DEFAULT_VARIANTS
from src.zelda_data.zelda_core import ZeldaDungeonAdapter
from src.zelda_data.zelda_loader import ZeldaRoomDataset


PASSABLE_TILES = {
    int(TileID.FLOOR),
    int(TileID.DOOR_OPEN),
    int(TileID.DOOR_LOCKED),
    int(TileID.DOOR_BOMB),
    int(TileID.DOOR_PUZZLE),
    int(TileID.DOOR_BOSS),
    int(TileID.DOOR_SOFT),
    int(TileID.START),
    int(TileID.TRIFORCE),
    int(TileID.KEY_SMALL),
    int(TileID.KEY_BOSS),
    int(TileID.KEY_ITEM),
    int(TileID.ITEM_MINOR),
    int(TileID.ELEMENT_FLOOR),
    int(TileID.STAIR),
}

DOOR_TILES = {
    int(TileID.DOOR_OPEN),
    int(TileID.DOOR_LOCKED),
    int(TileID.DOOR_BOMB),
    int(TileID.DOOR_PUZZLE),
    int(TileID.DOOR_BOSS),
    int(TileID.DOOR_SOFT),
}

IMPASSABLE_TILES = {
    int(TileID.VOID),
    int(TileID.WALL),
    int(TileID.BLOCK),
    int(TileID.ELEMENT),
    int(TileID.PUZZLE),
}

DIRECTIONS: Mapping[str, Tuple[int, int]] = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}


@dataclass(frozen=True)
class RoomSpec:
    dungeon_num: int
    variant: int
    coord: Tuple[int, int]
    required_doors: Tuple[str, ...]
    is_start: bool = False
    is_goal: bool = False


class TinyRoomGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_tiles: int) -> None:
        super().__init__()
        self.num_tiles = int(num_tiles)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.num_tiles * ROOM_HEIGHT * ROOM_WIDTH),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        return logits.view(z.shape[0], self.num_tiles, ROOM_HEIGHT, ROOM_WIDTH)


class TinyRoomDiscriminator(nn.Module):
    def __init__(self, num_tiles: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_tiles * ROOM_HEIGHT * ROOM_WIDTH, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k)) for k in keys})


def _load_rooms(
    data_root: Path,
    dungeon_ids: Sequence[int],
    variants: Sequence[int],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    dataset = ZeldaRoomDataset(
        str(data_root),
        normalize=False,
        load_graphs=False,
        dungeon_ids=dungeon_ids,
        variants=variants,
    )
    rooms = [np.asarray(room, dtype=np.int32) for room in dataset.rooms]
    metadata = [dict(item) for item in dataset.sample_metadata]
    return rooms, metadata


def _build_dungeon9_specs(data_root: Path, variants: Sequence[int]) -> List[RoomSpec]:
    adapter = ZeldaDungeonAdapter(str(data_root))
    specs: List[RoomSpec] = []
    for variant in variants:
        dungeon = adapter.load_dungeon(9, int(variant))
        room_coords = {tuple(coord) for coord in dungeon.rooms.keys()}
        sorted_coords = sorted(room_coords)
        for coord in sorted_coords:
            required: List[str] = []
            for direction, (dr, dc) in DIRECTIONS.items():
                if (coord[0] + dr, coord[1] + dc) in room_coords:
                    required.append(direction)
            specs.append(
                RoomSpec(
                    dungeon_num=9,
                    variant=int(variant),
                    coord=tuple(coord),
                    required_doors=tuple(required),
                )
            )

    if specs:
        first_by_variant: Dict[int, int] = {}
        last_by_variant: Dict[int, int] = {}
        for idx, spec in enumerate(specs):
            first_by_variant.setdefault(spec.variant, idx)
            last_by_variant[spec.variant] = idx
        mutable = list(specs)
        for idx in first_by_variant.values():
            spec = mutable[idx]
            mutable[idx] = RoomSpec(
                spec.dungeon_num,
                spec.variant,
                spec.coord,
                spec.required_doors,
                is_start=True,
                is_goal=spec.is_goal,
            )
        for idx in last_by_variant.values():
            spec = mutable[idx]
            mutable[idx] = RoomSpec(
                spec.dungeon_num,
                spec.variant,
                spec.coord,
                spec.required_doors,
                is_start=spec.is_start,
                is_goal=True,
            )
        specs = mutable
    return specs


def _select_specs(specs: Sequence[RoomSpec], budget: int, variants: Sequence[int]) -> List[RoomSpec]:
    selected: List[RoomSpec] = []
    per_variant = max(1, int(math.ceil(float(budget) / max(1, len(variants)))))
    for variant in variants:
        by_variant = [spec for spec in specs if spec.variant == int(variant)]
        selected.extend(by_variant[:per_variant])
    return selected[:budget]


def _door_position(direction: str) -> Tuple[int, int]:
    mid_r = ROOM_HEIGHT // 2
    mid_c = ROOM_WIDTH // 2
    if direction == "N":
        return (0, mid_c)
    if direction == "S":
        return (ROOM_HEIGHT - 1, mid_c)
    if direction == "W":
        return (mid_r, 0)
    if direction == "E":
        return (mid_r, ROOM_WIDTH - 1)
    raise ValueError(f"Unknown direction: {direction}")


def _inside_from_door(direction: str) -> Tuple[int, int]:
    r, c = _door_position(direction)
    if direction == "N":
        return (min(ROOM_HEIGHT - 1, r + 1), c)
    if direction == "S":
        return (max(0, r - 1), c)
    if direction == "W":
        return (r, min(ROOM_WIDTH - 1, c + 1))
    if direction == "E":
        return (r, max(0, c - 1))
    raise ValueError(f"Unknown direction: {direction}")


def _carve_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> None:
    r, c = start
    gr, gc = goal
    while r != gr:
        r += 1 if gr > r else -1
        if 0 < r < ROOM_HEIGHT - 1 and 0 < c < ROOM_WIDTH - 1:
            grid[r, c] = int(TileID.FLOOR)
    while c != gc:
        c += 1 if gc > c else -1
        if 0 < r < ROOM_HEIGHT - 1 and 0 < c < ROOM_WIDTH - 1:
            grid[r, c] = int(TileID.FLOOR)


def _apply_graph_scaffold(
    grid: np.ndarray,
    spec: RoomSpec,
    *,
    repair_boundary: bool,
    ensure_route: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int32).copy()
    out = np.clip(out, 0, int(TileID.PUZZLE)).astype(np.int32)

    if repair_boundary:
        out[0, :] = int(TileID.WALL)
        out[-1, :] = int(TileID.WALL)
        out[:, 0] = int(TileID.WALL)
        out[:, -1] = int(TileID.WALL)
        out[1:-1, 1:-1][out[1:-1, 1:-1] == int(TileID.VOID)] = int(TileID.FLOOR)

    center = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)
    if out[center] in IMPASSABLE_TILES:
        out[center] = int(TileID.FLOOR)

    for direction in spec.required_doors:
        door = _door_position(direction)
        inner = _inside_from_door(direction)
        out[door] = int(TileID.DOOR_OPEN)
        out[inner] = int(TileID.FLOOR)
        if ensure_route:
            _carve_path(out, center, inner)

    if spec.is_start:
        out[center] = int(TileID.START)
    elif spec.is_goal:
        out[center] = int(TileID.TRIFORCE)

    return out


def _random_walk_room(spec: RoomSpec, rng: np.random.Generator) -> np.ndarray:
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), int(TileID.WALL), dtype=np.int32)
    grid[1:-1, 1:-1] = int(TileID.FLOOR)
    center = (ROOM_HEIGHT // 2, ROOM_WIDTH // 2)
    for _ in range(int(rng.integers(5, 12))):
        r = int(rng.integers(2, ROOM_HEIGHT - 2))
        c = int(rng.integers(2, ROOM_WIDTH - 2))
        if (r, c) == center:
            continue
        roll = float(rng.random())
        if roll < 0.65:
            grid[r, c] = int(TileID.BLOCK)
        elif roll < 0.85:
            grid[r, c] = int(TileID.ENEMY)
        else:
            grid[r, c] = int(TileID.ELEMENT)
    return _apply_graph_scaffold(grid, spec, repair_boundary=True, ensure_route=True)


def _wfc_room(
    spec: RoomSpec,
    tile_priors: Mapping[int, Any],
    rng: np.random.Generator,
    *,
    seed: int,
) -> np.ndarray:
    wfc = WeightedBayesianWFC(
        width=ROOM_WIDTH,
        height=ROOM_HEIGHT,
        tile_priors=dict(tile_priors),
        config=WeightedBayesianWFCConfig(
            max_iterations=ROOM_HEIGHT * ROOM_WIDTH * 8,
            max_backtracks=64,
            max_restarts=1,
            prior_weight=1.0,
            adjacency_weight=1.0,
        ),
        seed=seed,
    )
    grid = np.asarray(wfc.generate(seed=seed), dtype=np.int32)
    # Keep WFC as the content source, then add graph-mandated apertures so the
    # symbolic baseline is evaluated on the same Dungeon 9 room specs.
    return _apply_graph_scaffold(grid, spec, repair_boundary=True, ensure_route=True)


def _train_tiny_gan(
    train_rooms: Sequence[np.ndarray],
    *,
    steps: int,
    seed: int,
    batch_size: int,
) -> Tuple[TinyRoomGenerator, np.ndarray]:
    rng = np.random.default_rng(seed)
    tile_vocab = np.array(sorted({int(v) for grid in train_rooms for v in np.unique(grid)}), dtype=np.int64)
    tile_to_idx = {int(tile): int(i) for i, tile in enumerate(tile_vocab.tolist())}
    encoded = np.stack(
        [
            np.vectorize(lambda x: tile_to_idx.get(int(x), 0), otypes=[np.int64])(grid)
            for grid in train_rooms
        ],
        axis=0,
    )

    device = torch.device("cpu")
    latent_dim = 64
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = TinyRoomGenerator(latent_dim=latent_dim, num_tiles=len(tile_vocab)).to(device)
        discriminator = TinyRoomDiscriminator(num_tiles=len(tile_vocab)).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    torch_gen = torch.Generator(device=device).manual_seed(int(seed))

    real_targets = torch.ones((batch_size, 1), device=device)
    fake_targets = torch.zeros((batch_size, 1), device=device)
    encoded_t = torch.from_numpy(encoded).long()

    for _ in range(max(0, int(steps))):
        idx = torch.from_numpy(rng.integers(0, encoded_t.shape[0], size=batch_size)).long()
        real_idx = encoded_t.index_select(0, idx).to(device)
        real = F.one_hot(real_idx, num_classes=len(tile_vocab)).permute(0, 3, 1, 2).float()

        z = torch.randn(batch_size, latent_dim, device=device, generator=torch_gen)
        fake_logits = generator(z)
        fake_soft = torch.softmax(fake_logits, dim=1)

        opt_d.zero_grad(set_to_none=True)
        d_loss = F.binary_cross_entropy_with_logits(discriminator(real), real_targets)
        d_loss = d_loss + F.binary_cross_entropy_with_logits(discriminator(fake_soft.detach()), fake_targets)
        d_loss.backward()
        opt_d.step()

        z = torch.randn(batch_size, latent_dim, device=device, generator=torch_gen)
        fake_soft = torch.softmax(generator(z), dim=1)
        opt_g.zero_grad(set_to_none=True)
        g_loss = F.binary_cross_entropy_with_logits(discriminator(fake_soft), real_targets)
        g_loss.backward()
        opt_g.step()

    generator.eval()
    return generator, tile_vocab


def _generate_gan_rooms(
    generator: TinyRoomGenerator,
    tile_vocab: np.ndarray,
    count: int,
    *,
    seed: int,
    temperature: float = 0.85,
) -> List[np.ndarray]:
    latent_dim = int(generator.net[0].in_features)
    device = next(generator.parameters()).device
    torch_gen = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        logits = generator(torch.randn(count, latent_dim, device=device, generator=torch_gen)) / max(0.05, float(temperature))
        probs = torch.softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, len(tile_vocab))
        sampled = torch.multinomial(probs, num_samples=1, generator=torch_gen).view(count, ROOM_HEIGHT, ROOM_WIDTH).cpu().numpy()
    return [tile_vocab[grid].astype(np.int32) for grid in sampled]


def _is_passable(value: int) -> bool:
    return int(value) in PASSABLE_TILES


def _room_playable(grid: np.ndarray, spec: RoomSpec) -> bool:
    points: List[Tuple[int, int]] = []
    for direction in spec.required_doors:
        points.append(_door_position(direction))
        points.append(_inside_from_door(direction))
    points.append((ROOM_HEIGHT // 2, ROOM_WIDTH // 2))

    starts = [p for p in points if 0 <= p[0] < ROOM_HEIGHT and 0 <= p[1] < ROOM_WIDTH and _is_passable(int(grid[p]))]
    if not starts:
        passable = np.argwhere(np.vectorize(_is_passable)(grid))
        if passable.size == 0:
            return False
        starts = [tuple(int(v) for v in passable[0])]

    start = starts[0]
    queue: deque[Tuple[int, int]] = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= ROOM_HEIGHT or nc >= ROOM_WIDTH:
                continue
            if (nr, nc) in seen or not _is_passable(int(grid[nr, nc])):
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))

    required_points = [_door_position(direction) for direction in spec.required_doors]
    if not required_points:
        return bool(seen)
    return all(point in seen for point in required_points)


def _missing_required_doors(grid: np.ndarray, spec: RoomSpec) -> int:
    missing = 0
    for direction in spec.required_doors:
        if int(grid[_door_position(direction)]) not in DOOR_TILES:
            missing += 1
    return missing


def _boundary_artifacts(grid: np.ndarray) -> int:
    allowed = {int(TileID.WALL), int(TileID.VOID), *DOOR_TILES}
    count = 0
    for r in range(ROOM_HEIGHT):
        for c in range(ROOM_WIDTH):
            if r not in {0, ROOM_HEIGHT - 1} and c not in {0, ROOM_WIDTH - 1}:
                continue
            if int(grid[r, c]) not in allowed:
                count += 1
    return count


def _logic_violations(grid: np.ndarray, spec: RoomSpec) -> int:
    violations = 0
    if not _room_playable(grid, spec):
        violations += 1
    if spec.is_start and int(TileID.START) not in set(int(v) for v in grid.ravel()):
        violations += 1
    if spec.is_goal and int(TileID.TRIFORCE) not in set(int(v) for v in grid.ravel()):
        violations += 1
    return violations


def _nearest_hamming_stats(generated: Sequence[np.ndarray], train_rooms: Sequence[np.ndarray]) -> Dict[str, float]:
    if not generated or not train_rooms:
        return {
            "nearest_train_hamming_mean": float("nan"),
            "nearest_train_hamming_min": float("nan"),
            "nearest_train_hamming_p05": float("nan"),
            "exact_copy_rate": float("nan"),
            "novelty_gt_zero_rate": float("nan"),
        }
    train_flat = np.stack([np.asarray(room, dtype=np.int16).reshape(-1) for room in train_rooms], axis=0)
    distances: List[float] = []
    for room in generated:
        flat = np.asarray(room, dtype=np.int16).reshape(1, -1)
        d = np.mean(train_flat != flat, axis=1)
        distances.append(float(np.min(d)))
    arr = np.asarray(distances, dtype=np.float64)
    return {
        "nearest_train_hamming_mean": float(np.mean(arr)),
        "nearest_train_hamming_min": float(np.min(arr)),
        "nearest_train_hamming_p05": float(np.quantile(arr, 0.05)),
        "exact_copy_rate": float(np.mean(arr <= 0.0)),
        "novelty_gt_zero_rate": float(np.mean(arr > 0.0)),
    }


def _pairwise_hamming(rooms: Sequence[np.ndarray]) -> float:
    if len(rooms) < 2:
        return 0.0
    flats = [np.asarray(room, dtype=np.int16).reshape(-1) for room in rooms]
    total = 0.0
    count = 0
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            total += float(np.mean(flats[i] != flats[j]))
            count += 1
    return float(total / max(1, count))


def _room_feature_cell(grid: np.ndarray) -> Tuple[int, int]:
    arr = np.asarray(grid, dtype=np.int32)
    floor_like = np.isin(arr, list(PASSABLE_TILES))
    obstacle_like = np.isin(
        arr,
        [
            int(TileID.WALL),
            int(TileID.BLOCK),
            int(TileID.ELEMENT),
            int(TileID.PUZZLE),
            int(TileID.ENEMY),
            int(TileID.BOSS),
        ],
    )
    floor_ratio = float(np.mean(floor_like))
    obstacle_ratio = float(np.mean(obstacle_like))
    x = min(3, max(0, int(floor_ratio * 4.0)))
    y = min(3, max(0, int(obstacle_ratio * 4.0)))
    return x, y


def _map_elites_metrics(rooms: Sequence[np.ndarray], specs: Sequence[RoomSpec]) -> Dict[str, float]:
    occupied: Set[Tuple[int, int]] = set()
    playable_cells: Counter[Tuple[int, int]] = Counter()
    for room, spec in zip(rooms, specs):
        cell = _room_feature_cell(room)
        occupied.add(cell)
        if _room_playable(room, spec):
            playable_cells[cell] += 1
    total_cells = 16
    return {
        "map_elites_coverage": float(len(occupied) / total_cells),
        "map_elites_qd_score": float(sum(playable_cells.values()) / max(1, len(rooms))),
        "map_elites_occupied_cells": float(len(occupied)),
    }


def _observed_boundary_doors(grid: np.ndarray) -> Tuple[str, ...]:
    observed: List[str] = []
    for direction in DIRECTIONS:
        if int(grid[_door_position(direction)]) in DOOR_TILES:
            observed.append(direction)
    return tuple(observed)


def _linearity_score_for_doors(doors: Sequence[str]) -> float:
    unique = tuple(sorted({str(v) for v in doors}))
    degree = len(unique)
    if degree <= 1:
        return 1.0
    if degree == 2:
        door_set = set(unique)
        return 1.0 if door_set in ({"N", "S"}, {"E", "W"}) else 0.75
    if degree == 3:
        return 0.35
    return 0.0


def _linearity_scores(rooms: Sequence[np.ndarray], specs: Sequence[RoomSpec]) -> List[float]:
    scores: List[float] = []
    for room, spec in zip(rooms, specs):
        doors = spec.required_doors or _observed_boundary_doors(room)
        scores.append(_linearity_score_for_doors(doors))
    return scores


def _leniency_score(grid: np.ndarray) -> float:
    arr = np.asarray(grid, dtype=np.int32)
    danger = np.zeros(arr.shape, dtype=np.float32)
    danger += (arr == int(TileID.ENEMY)).astype(np.float32)
    danger += 2.0 * (arr == int(TileID.BOSS)).astype(np.float32)
    danger += 0.75 * np.isin(arr, [int(TileID.ELEMENT), int(TileID.PUZZLE)]).astype(np.float32)
    danger += 0.50 * np.isin(
        arr,
        [
            int(TileID.DOOR_LOCKED),
            int(TileID.DOOR_BOMB),
            int(TileID.DOOR_PUZZLE),
            int(TileID.DOOR_BOSS),
            int(TileID.DOOR_SOFT),
        ],
    ).astype(np.float32)
    return float(np.clip(1.0 - float(np.mean(danger)), 0.0, 1.0))


def _summarize_method(
    name: str,
    rooms: Sequence[np.ndarray],
    specs: Sequence[RoomSpec],
    train_rooms: Sequence[np.ndarray],
    *,
    generation_time_sec: float,
) -> Dict[str, Any]:
    playable_values = [_room_playable(room, spec) for room, spec in zip(rooms, specs)]
    missing_doors = [_missing_required_doors(room, spec) for room, spec in zip(rooms, specs)]
    required_door_count = [len(spec.required_doors) for spec in specs]
    logic_violations = [_logic_violations(room, spec) for room, spec in zip(rooms, specs)]
    boundary_artifacts = [_boundary_artifacts(room) for room in rooms]
    linearity = _linearity_scores(rooms, specs)
    leniency = [_leniency_score(room) for room in rooms]
    total_required = max(1, int(sum(required_door_count)))
    row: Dict[str, Any] = {
        "method": name,
        "n": int(len(rooms)),
        "playable_rate": float(statistics.fmean(float(v) for v in playable_values)) if playable_values else 0.0,
        "linearity": float(statistics.fmean(linearity)) if linearity else 0.0,
        "leniency": float(statistics.fmean(leniency)) if leniency else 0.0,
        "graph_door_violation_rate": float(sum(missing_doors) / total_required),
        "logic_violation_rate": float(sum(1 for v in logic_violations if v > 0) / max(1, len(rooms))),
        "boundary_artifacts_mean": float(statistics.fmean(boundary_artifacts)) if boundary_artifacts else 0.0,
        "boundary_artifacts_total": int(sum(boundary_artifacts)),
        "pairwise_diversity_hamming": _pairwise_hamming(rooms),
        "generation_time_sec": float(generation_time_sec),
    }
    row.update(_nearest_hamming_stats(rooms, train_rooms))
    row.update(_map_elites_metrics(rooms, specs))
    return row


def _profile_unet_cost(config_path: Path, *, room_budget: int) -> Dict[str, Any]:
    cfg = merge_config(yaml_path=str(config_path))
    diffusion = dict(cfg["diffusion"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = create_latent_diffusion(
            latent_dim=int(diffusion["latent_dim"]),
            model_channels=int(diffusion["model_channels"]),
            context_dim=int(diffusion["context_dim"]),
            num_timesteps=int(diffusion["num_timesteps"]),
            schedule_type=str(diffusion["schedule_type"]),
            prediction_type=str(diffusion["prediction_type"]),
            cfg_dropout_prob=float(diffusion["cfg_dropout_prob"]),
            cfg_scale=float(diffusion["cfg_scale"]),
            cfg_schedule_mode=str(diffusion["cfg_schedule_mode"]),
            cfg_schedule_min_scale=float(diffusion["cfg_schedule_min_scale"]),
            cfg_schedule_power=float(diffusion["cfg_schedule_power"]),
            min_snr_gamma=float(diffusion["min_snr_gamma"]),
            topology_refinement_mode=str(diffusion["topology_refinement_mode"]),
            attention_mode=str(diffusion["attention_mode"]),
            topology_conditioning_mode=str(diffusion["topology_conditioning_mode"]),
            hedgehog_feature_dim=int(diffusion["hedgehog_feature_dim"]),
            graph_auto_linear_attention_nodes=int(diffusion["graph_auto_linear_attention_nodes"]),
            spatial_graph_gate_init=float(diffusion["spatial_graph_gate_init"]),
            spatial_topology_gate_init=float(diffusion["spatial_topology_gate_init"]),
            room_topology_channels=int(diffusion["room_topology_channels"]),
            unet_channel_mult=tuple(int(v) for v in diffusion["unet_channel_mult"]),
            unet_num_res_blocks=int(diffusion["unet_num_res_blocks"]),
            unet_attention_resolutions=tuple(int(v) for v in diffusion["unet_attention_resolutions"]),
            unet_num_heads=int(diffusion["unet_num_heads"]),
            unet_dropout=float(diffusion["unet_dropout"]),
        )
    denoiser = model.denoiser.eval()
    params = int(sum(p.numel() for p in denoiser.parameters()))
    trainable_params = int(sum(p.numel() for p in denoiser.parameters() if p.requires_grad))

    macs = {"value": 0.0}
    handles = []

    def conv_hook(module: nn.Conv2d, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        out = output
        if not isinstance(out, torch.Tensor):
            return
        batch = int(out.shape[0])
        out_ch = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        kernel_ops = int(module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups)
        macs["value"] += float(batch * out_ch * out_h * out_w * kernel_ops)

    def linear_hook(module: nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        out = output
        if not isinstance(out, torch.Tensor):
            return
        instances = int(np.prod(list(out.shape[:-1]))) if out.dim() > 1 else 1
        macs["value"] += float(instances * module.in_features * module.out_features)

    def self_attn_hook(module: SelfAttention, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return
        batch, tokens, _channels = [int(v) for v in x.shape]
        macs["value"] += float(2 * batch * int(module.num_heads) * tokens * tokens * int(module.head_dim))

    def cross_attn_hook(module: CrossAttention, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0]
        context = inputs[1] if len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor) or not isinstance(context, torch.Tensor):
            return
        if x.dim() != 3:
            return
        batch, query_tokens, _channels = [int(v) for v in x.shape]
        context_tokens = 1 if context.dim() == 2 else int(context.shape[1])
        macs["value"] += float(
            2 * batch * int(module.num_heads) * query_tokens * context_tokens * int(module.head_dim)
        )

    for module in denoiser.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, SelfAttention):
            handles.append(module.register_forward_hook(self_attn_hook))
        elif isinstance(module, CrossAttention):
            handles.append(module.register_forward_hook(cross_attn_hook))

    profile_error: Optional[str] = None
    try:
        with torch.no_grad():
            torch_gen = torch.Generator(device="cpu").manual_seed(0)
            x = torch.randn(1, int(diffusion["latent_dim"]), ROOM_HEIGHT, ROOM_WIDTH, generator=torch_gen)
            t = torch.tensor([int(diffusion["num_timesteps"]) // 2], dtype=torch.long)
            context = torch.randn(1, int(room_budget), int(diffusion["context_dim"]), generator=torch_gen)
            node_mask = torch.ones(1, int(room_budget), dtype=torch.bool)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            _ = denoiser(
                x,
                t,
                context,
                context_edge_index=edge_index,
                context_node_mask=node_mask,
            )
    except Exception as exc:  # pragma: no cover - defensive reporting path
        profile_error = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()

    forward_macs = float(macs["value"])
    forward_flops = float(2.0 * forward_macs) if forward_macs > 0.0 else None
    sampling_steps = int(min(50, int(diffusion["num_timesteps"])))
    return {
        "unet_params": params,
        "unet_trainable_params": trainable_params,
        "profile_input_shape": [1, int(diffusion["latent_dim"]), ROOM_HEIGHT, ROOM_WIDTH],
        "profile_context_tokens": int(room_budget),
        "estimated_unet_forward_macs": forward_macs if forward_macs > 0.0 else None,
        "estimated_unet_forward_flops": forward_flops,
        "estimated_50_step_room_flops": (forward_flops * sampling_steps) if forward_flops else None,
        "estimated_50_step_30_room_flops": (forward_flops * sampling_steps * room_budget) if forward_flops else None,
        "profile_error": profile_error,
    }


def _method_markdown(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals: List[str] = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(_json_safe(value)))
        out.append("|" + "|".join(vals) + "|")
    return "\n".join(out)


def run_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dungeons = tuple(int(v) for v in DEFAULT_TRAIN_DUNGEONS)
    test_dungeons = tuple(int(v) for v in DEFAULT_TEST_DUNGEONS)
    variants = tuple(int(v) for v in DEFAULT_VARIANTS)

    train_rooms, train_meta = _load_rooms(args.data_root, train_dungeons, variants)
    test_rooms, test_meta = _load_rooms(args.data_root, test_dungeons, variants)
    all_d9_specs = _build_dungeon9_specs(args.data_root, variants)
    selected_specs = _select_specs(all_d9_specs, int(args.room_budget), variants)

    split_rows = [
        {
            "split": "train",
            "dungeons": ",".join(str(v) for v in train_dungeons),
            "variants": ",".join(str(v) for v in variants),
            "rooms": len(train_rooms),
            "dungeon_files": len(sorted({m["dungeon_id"] for m in train_meta})),
        },
        {
            "split": "test",
            "dungeons": ",".join(str(v) for v in test_dungeons),
            "variants": ",".join(str(v) for v in variants),
            "rooms": len(test_rooms),
            "dungeon_files": len(sorted({m["dungeon_id"] for m in test_meta})),
        },
    ]
    _write_csv(out_dir / "split_summary.csv", split_rows)

    d9_graph_rows: List[Dict[str, Any]] = []
    adapter = ZeldaDungeonAdapter(str(args.data_root))
    for variant in variants:
        dungeon = adapter.load_dungeon(9, variant)
        physical_nodes = [
            n for n, attrs in dungeon.graph.nodes(data=True)
            if not bool(attrs.get("is_start_pointer", False))
        ]
        edge_types = Counter(str(attrs.get("edge_type", "open")) for _, _, attrs in dungeon.graph.edges(data=True))
        d9_graph_rows.append(
            {
                "dungeon": 9,
                "variant": variant,
                "rooms": len(dungeon.rooms),
                "graph_nodes": dungeon.graph.number_of_nodes(),
                "physical_graph_nodes": len(physical_nodes),
                "graph_edges": dungeon.graph.number_of_edges(),
                "key_locked_edges": edge_types.get("key_locked", 0),
                "bombable_edges": edge_types.get("bombable", 0),
                "soft_locked_edges": edge_types.get("soft_locked", 0),
            }
        )
    _write_csv(out_dir / "dungeon9_graph_summary.csv", d9_graph_rows)

    tile_priors = extract_tile_priors_from_vqvae(np.zeros((1, 1), dtype=np.float32), list(train_rooms))

    gan_start = time.perf_counter()
    generator, tile_vocab = _train_tiny_gan(
        train_rooms,
        steps=int(args.gan_steps),
        seed=int(args.seed),
        batch_size=int(args.gan_batch_size),
    )
    gan_training_time = float(time.perf_counter() - gan_start)
    raw_gan_rooms = _generate_gan_rooms(
        generator,
        tile_vocab,
        len(selected_specs),
        seed=int(args.seed) + 1000,
        temperature=float(args.gan_temperature),
    )

    method_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []

    # Real held-out rooms are reported as a novelty reference. Their raw VGLC
    # door semantics are generic and graph matching is separate, so do not score
    # them with the generated-room door-scaffold contract.
    real_specs = [
        RoomSpec(
            dungeon_num=9,
            variant=int(meta.get("variant", 0)),
            coord=tuple(meta.get("room_coord", (0, 0))),
            required_doors=tuple(),
        )
        for meta in test_meta
    ]
    method_rows.append(
        _summarize_method(
            "DUNGEON9_REAL_TEST",
            test_rooms[: len(real_specs)],
            real_specs,
            train_rooms,
            generation_time_sec=0.0,
        )
    )

    t0 = time.perf_counter()
    random_rooms = [_random_walk_room(spec, rng) for spec in selected_specs]
    method_rows.append(
        _summarize_method(
            "RANDOM_WALK_SYMBOLIC",
            random_rooms,
            selected_specs,
            train_rooms,
            generation_time_sec=float(time.perf_counter() - t0),
        )
    )

    t0 = time.perf_counter()
    wfc_rooms = [
        _wfc_room(spec, tile_priors, rng, seed=int(args.seed) + 2000 + i)
        for i, spec in enumerate(selected_specs)
    ]
    method_rows.append(
        _summarize_method(
            "WFC_SYMBOLIC",
            wfc_rooms,
            selected_specs,
            train_rooms,
            generation_time_sec=float(time.perf_counter() - t0),
        )
    )

    method_rows.append(
        _summarize_method(
            "GAN_LITE_ML",
            raw_gan_rooms,
            selected_specs,
            train_rooms,
            generation_time_sec=gan_training_time,
        )
    )

    # Diagnostic ablations over the same raw GAN rooms.
    t0 = time.perf_counter()
    full_diag = [
        _apply_graph_scaffold(room, spec, repair_boundary=True, ensure_route=True)
        for room, spec in zip(raw_gan_rooms, selected_specs)
    ]
    ablation_rows.append(
        _summarize_method(
            "FULL_DIAGNOSTIC",
            full_diag,
            selected_specs,
            train_rooms,
            generation_time_sec=float(time.perf_counter() - t0),
        )
    )

    t0 = time.perf_counter()
    no_graph = [
        _apply_graph_scaffold(
            room,
            RoomSpec(spec.dungeon_num, spec.variant, spec.coord, tuple(), spec.is_start, spec.is_goal),
            repair_boundary=True,
            ensure_route=True,
        )
        for room, spec in zip(raw_gan_rooms, selected_specs)
    ]
    ablation_rows.append(
        _summarize_method(
            "NO_GRAPH",
            no_graph,
            selected_specs,
            train_rooms,
            generation_time_sec=float(time.perf_counter() - t0),
        )
    )

    t0 = time.perf_counter()
    no_logic = [
        _apply_graph_scaffold(room, spec, repair_boundary=True, ensure_route=False)
        for room, spec in zip(raw_gan_rooms, selected_specs)
    ]
    ablation_rows.append(
        _summarize_method(
            "NO_LOGICNET",
            no_logic,
            selected_specs,
            train_rooms,
            generation_time_sec=float(time.perf_counter() - t0),
        )
    )

    t0 = time.perf_counter()
    no_wfc = [
        _apply_graph_scaffold(room, spec, repair_boundary=False, ensure_route=True)
        for room, spec in zip(raw_gan_rooms, selected_specs)
    ]
    ablation_rows.append(
        _summarize_method(
            "NO_WFC",
            no_wfc,
            selected_specs,
            train_rooms,
            generation_time_sec=float(time.perf_counter() - t0),
        )
    )

    # Include the diagnostic full row in the method table as the local hybrid
    # reference for baseline comparison.
    method_rows.append(dict(ablation_rows[0], method="HYBRID_DIAGNOSTIC_FULL"))

    cost = _profile_unet_cost(Path(args.config), room_budget=int(args.room_budget))
    mean_full_diag_room_time = float(ablation_rows[0]["generation_time_sec"]) / max(1, len(selected_specs))
    cost.update(
        {
            "room_budget": int(args.room_budget),
            "diagnostic_full_30_room_postprocess_time_sec": float(ablation_rows[0]["generation_time_sec"]),
            "diagnostic_full_mean_room_postprocess_time_sec": mean_full_diag_room_time,
            "gan_lite_training_time_sec": gan_training_time,
            "random_walk_30_room_time_sec": float(method_rows[1]["generation_time_sec"]),
            "wfc_30_room_time_sec": float(method_rows[2]["generation_time_sec"]),
            "realtime_assessment": (
                "offline_recommended"
                if (
                    cost.get("estimated_50_step_30_room_flops") is None
                    or float(method_rows[2]["generation_time_sec"]) > 1.0
                )
                else "realtime_possible_for_symbolic_only"
            ),
        }
    )

    _write_csv(out_dir / "method_summary.csv", method_rows)
    _write_csv(out_dir / "ablation_summary.csv", ablation_rows)
    (out_dir / "unet_cost.json").write_text(json.dumps(_json_safe(cost), indent=2), encoding="utf-8")

    payload = {
        "protocol": {
            "name": "dungeon9_holdout_protocol",
            "seed": int(args.seed),
            "room_budget": int(args.room_budget),
            "data_root": str(args.data_root),
            "train_dungeons": list(train_dungeons),
            "test_dungeons": list(test_dungeons),
            "variants": list(variants),
            "notes": [
                "Diffusion training code now enforces the fixed split, but this lightweight audit does not retrain the full diffusion checkpoint.",
                "GAN_LITE_ML is a local GAN-style baseline inspired by ZeldaGAN/MarioGAN literature, not an imported external checkpoint.",
                "Ablation rows are diagnostic component toggles over the same generated room candidates.",
            ],
        },
        "split_summary": split_rows,
        "dungeon9_graph_summary": d9_graph_rows,
        "method_summary": method_rows,
        "ablation_summary": ablation_rows,
        "unet_cost": cost,
        "runtime_sec": float(time.perf_counter() - started),
    }
    (out_dir / "holdout_protocol_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Dungeon 9 Holdout Protocol",
        "",
        "## Split",
        "",
        _method_markdown(split_rows, ["split", "dungeons", "variants", "rooms", "dungeon_files"]),
        "",
        "## Dungeon 9 Graph",
        "",
        _method_markdown(
            d9_graph_rows,
            [
                "variant",
                "rooms",
                "graph_nodes",
                "physical_graph_nodes",
                "graph_edges",
                "key_locked_edges",
                "bombable_edges",
                "soft_locked_edges",
            ],
        ),
        "",
        "## Methods",
        "",
        _method_markdown(
            method_rows,
            [
                "method",
                "n",
                "playable_rate",
                "linearity",
                "leniency",
                "nearest_train_hamming_mean",
                "exact_copy_rate",
                "pairwise_diversity_hamming",
                "map_elites_coverage",
                "generation_time_sec",
            ],
        ),
        "",
        "## Ablations",
        "",
        _method_markdown(
            ablation_rows,
            [
                "method",
                "n",
                "playable_rate",
                "graph_door_violation_rate",
                "logic_violation_rate",
                "boundary_artifacts_mean",
                "nearest_train_hamming_mean",
            ],
        ),
        "",
        "## Cost",
        "",
        f"- U-Net params: {cost['unet_params']}",
        f"- Estimated one-step U-Net FLOPs: {_json_safe(cost.get('estimated_unet_forward_flops'))}",
        f"- Estimated 50-step, 30-room FLOPs: {_json_safe(cost.get('estimated_50_step_30_room_flops'))}",
        f"- WFC 30-room wall time: {cost['wfc_30_room_time_sec']:.4f} sec",
        f"- Assessment: {cost['realtime_assessment']}",
        "",
        "The full diffusion checkpoint must be retrained with Dungeon 9 held out before using neural numbers as final generalization evidence.",
    ]
    (out_dir / "holdout_protocol_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed Dungeon 9 holdout audit protocol.")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--config", type=Path, default=Path("configs") / "zelda_hmolqd.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "dungeon9_holdout_protocol")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--room-budget", type=int, default=30)
    parser.add_argument("--gan-steps", type=int, default=80)
    parser.add_argument("--gan-batch-size", type=int, default=32)
    parser.add_argument("--gan-temperature", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_protocol(args)
    print(json.dumps(_json_safe({
        "output_dir": str(args.output_dir),
        "train_rooms": payload["split_summary"][0]["rooms"],
        "test_rooms": payload["split_summary"][1]["rooms"],
        "runtime_sec": payload["runtime_sec"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
