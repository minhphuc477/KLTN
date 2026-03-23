"""Standalone validation helper classes extracted from validator monolith."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE

WALKABLE_IDS = {
    SEMANTIC_PALETTE["FLOOR"],
    SEMANTIC_PALETTE["DOOR_OPEN"],
    SEMANTIC_PALETTE["DOOR_SOFT"],
    SEMANTIC_PALETTE["START"],
    SEMANTIC_PALETTE["TRIFORCE"],
    SEMANTIC_PALETTE["KEY_SMALL"],
    SEMANTIC_PALETTE["KEY_BOSS"],
    SEMANTIC_PALETTE["KEY_ITEM"],
    SEMANTIC_PALETTE["ITEM_MINOR"],
    SEMANTIC_PALETTE["ELEMENT_FLOOR"],
    SEMANTIC_PALETTE["STAIR"],
    SEMANTIC_PALETTE["ENEMY"],
    SEMANTIC_PALETTE["BOSS"],
    SEMANTIC_PALETTE["PUZZLE"],
}


class SanityChecker:
    """Pre-validation checks for map structural validity."""

    def __init__(self, semantic_grid: np.ndarray):
        self.grid = semantic_grid
        self.height, self.width = self.grid.shape

    def check_all(self) -> Tuple[bool, List[str]]:
        """Run all sanity checks."""
        errors = []

        starts = np.where(self.grid == SEMANTIC_PALETTE["START"])
        if len(starts[0]) == 0:
            errors.append("No start position (S) found")
        elif len(starts[0]) > 1:
            errors.append(f"Multiple start positions found: {len(starts[0])}")

        goals = np.where(self.grid == SEMANTIC_PALETTE["TRIFORCE"])
        if len(goals[0]) == 0:
            errors.append("No goal (Triforce) found")

        walkable_count = np.sum(np.isin(self.grid, list(WALKABLE_IDS)))
        void_count = np.sum(self.grid == SEMANTIC_PALETTE["VOID"])
        total_cells = self.height * self.width
        non_void_cells = total_cells - void_count

        if non_void_cells > 0 and walkable_count < 0.05 * non_void_cells:
            errors.append(
                f"Map is mostly blocked ({walkable_count}/{non_void_cells} walkable, excluding void)"
            )

        locked_doors = np.sum(self.grid == SEMANTIC_PALETTE["DOOR_LOCKED"])
        keys = np.sum(self.grid == SEMANTIC_PALETTE["KEY_SMALL"])
        if locked_doors > 0 and keys == 0:
            errors.append(f"Locked doors ({locked_doors}) but no keys")

        boss_doors = np.sum(self.grid == SEMANTIC_PALETTE["DOOR_BOSS"])
        boss_keys = np.sum(self.grid == SEMANTIC_PALETTE["KEY_BOSS"])
        if boss_doors > 0 and boss_keys == 0:
            errors.append("Boss door present but no boss key")

        return len(errors) == 0, errors

    def count_elements(self) -> Dict[str, int]:
        """Count occurrences of each semantic element."""
        counts = {}
        for name, id_val in SEMANTIC_PALETTE.items():
            count = int(np.sum(self.grid == id_val))
            if count > 0:
                counts[name] = count
        return counts


class MetricsEngine:
    """Calculate validation metrics for a solved map."""

    @staticmethod
    def calculate_reachability(env: Any, path: List[Tuple[int, int]]) -> float:
        visited = set(path)
        walkable = 0
        for r in range(env.height):
            for c in range(env.width):
                if env.original_grid[r, c] in WALKABLE_IDS:
                    walkable += 1

        if walkable == 0:
            return 0.0

        return len(visited) / walkable

    @staticmethod
    def calculate_backtracking(path: List[Tuple[int, int]]) -> float:
        if len(path) <= 1:
            return 0.0

        unique_positions = len(set(path))
        total_steps = len(path)
        return (total_steps - unique_positions) / total_steps

    @staticmethod
    def calculate_linearity(
        path: List[Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> float:
        if len(path) <= 1:
            return 1.0

        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if manhattan == 0:
            return 1.0

        return min(1.0, manhattan / len(path))

    @staticmethod
    def find_logical_errors(env: Any, path: List[Tuple[int, int]]) -> List[str]:
        errors = []
        visited = set(path)

        find_all_positions = getattr(env, "_find_all_positions", None)
        if not callable(find_all_positions):
            return errors

        key_positions = find_all_positions(SEMANTIC_PALETTE["KEY_SMALL"])
        if not isinstance(key_positions, Iterable):
            key_positions = []
        for kp in key_positions:
            if kp not in visited and kp not in env.state.collected_items:
                errors.append(f"Unreachable key at {kp}")

        boss_key_positions = find_all_positions(SEMANTIC_PALETTE["KEY_BOSS"])
        if not isinstance(boss_key_positions, Iterable):
            boss_key_positions = []
        for bp in boss_key_positions:
            if bp not in visited and not env.state.has_boss_key:
                errors.append(f"Unreachable boss key at {bp}")

        return errors


class DiversityEvaluator:
    """Evaluate diversity across a batch of generated maps."""

    @staticmethod
    def hamming_distance(grid1: np.ndarray, grid2: np.ndarray) -> float:
        if grid1.shape != grid2.shape:
            return 1.0

        total_cells = grid1.size
        different_cells = np.sum(grid1 != grid2)
        return different_cells / total_cells

    @staticmethod
    def batch_diversity(grids: List[np.ndarray]) -> float:
        n = len(grids)
        if n < 2:
            return 0.0

        total_dist = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += DiversityEvaluator.hamming_distance(grids[i], grids[j])
                pairs += 1

        return total_dist / pairs if pairs > 0 else 0.0

    @staticmethod
    def structural_diversity(paths: List[List[Tuple[int, int]]]) -> float:
        if len(paths) < 2:
            return 0.0

        path_sets = [set(p) for p in paths if p]
        if len(path_sets) < 2:
            return 0.0

        total_dist = 0.0
        pairs = 0
        for i in range(len(path_sets)):
            for j in range(i + 1, len(path_sets)):
                intersection = len(path_sets[i] & path_sets[j])
                union = len(path_sets[i] | path_sets[j])
                if union > 0:
                    jaccard = intersection / union
                    total_dist += 1 - jaccard
                    pairs += 1

        return total_dist / pairs if pairs > 0 else 0.0
