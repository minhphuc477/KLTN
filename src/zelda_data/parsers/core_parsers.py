"""Core parser implementations extracted from zelda_core."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from src.core.definitions import (
    CHAR_TO_SEMANTIC,
    DOOR_POSITIONS,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    SEMANTIC_PALETTE,
    normalize_node_label,
    parse_edge_type_tokens,
    parse_node_label_tokens,
    select_primary_edge_type,
)


class GridBasedRoomExtractor:
    """Extract rooms from fixed-size VGLC slots."""

    SLOT_WIDTH = 11
    SLOT_HEIGHT = 16
    GAP_MARKER = "-"
    WALL_MARKER = "W"

    def _load_grid(self, filepath: str) -> np.ndarray:
        with open(filepath, "r") as file_obj:
            lines = [line.rstrip("\n") for line in file_obj]

        if not lines:
            return np.zeros((0, 0), dtype="<U1")

        width = max(len(line) for line in lines) if lines else 0
        padded = [list(line.ljust(width, self.GAP_MARKER)) for line in lines]
        return np.array(padded)

    def _is_room_slot(self, slot_grid: np.ndarray) -> bool:
        if slot_grid.size == 0:
            return False

        dash_count = np.sum(slot_grid == self.GAP_MARKER)
        total = slot_grid.size
        if dash_count > total * 0.7:
            return False

        wall_count = np.sum(slot_grid == self.WALL_MARKER)
        door_count = int(np.sum(slot_grid == "D"))

        if door_count > 0 and wall_count >= 20:
            return True

        interior_count = total - wall_count - dash_count
        return bool(wall_count >= 20 and interior_count >= 5)

    def extract(self, filepath: str) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        grid = self._load_grid(filepath)

        if grid.size == 0:
            return []

        height, width = grid.shape
        num_row_slots = height // self.SLOT_HEIGHT
        num_col_slots = width // self.SLOT_WIDTH

        rooms = []

        for row_slot in range(num_row_slots):
            row_start = row_slot * self.SLOT_HEIGHT
            row_end = row_start + self.SLOT_HEIGHT

            for col_slot in range(num_col_slots):
                col_start = col_slot * self.SLOT_WIDTH
                col_end = col_start + self.SLOT_WIDTH

                slot_grid = grid[row_start:row_end, col_start:col_end]

                if slot_grid.shape[0] < self.SLOT_HEIGHT:
                    pad = np.full(
                        (self.SLOT_HEIGHT - slot_grid.shape[0], slot_grid.shape[1]), self.GAP_MARKER
                    )
                    slot_grid = np.vstack([slot_grid, pad])
                if slot_grid.shape[1] < self.SLOT_WIDTH:
                    pad = np.full(
                        (slot_grid.shape[0], self.SLOT_WIDTH - slot_grid.shape[1]), self.GAP_MARKER
                    )
                    slot_grid = np.hstack([slot_grid, pad])

                if self._is_room_slot(slot_grid):
                    rooms.append(((row_slot, col_slot), slot_grid.copy()))

        return rooms

    def extract_with_ids(self, filepath: str) -> List[Tuple[int, np.ndarray]]:
        raw_rooms = self.extract(filepath)
        return [(r_idx * 100 + c_idx, grid) for ((r_idx, c_idx), grid) in raw_rooms]


class VGLCParser:
    """Parse VGLC text files into room objects."""

    def __init__(self, room_cls: Any):
        self.room_cls = room_cls

    def parse(self, filepath: str) -> Dict[Tuple[int, int], Any]:
        with open(filepath, "r") as file_obj:
            lines = [line.rstrip("\n") for line in file_obj]

        if not lines:
            return {}

        max_width = max(len(line) for line in lines)
        lines = [line.ljust(max_width, "-") for line in lines]

        num_rows = len(lines) // ROOM_HEIGHT
        num_cols = max_width // ROOM_WIDTH

        rooms: Dict[Tuple[int, int], Any] = {}

        for row in range(num_rows):
            for col in range(num_cols):
                room_chars = []
                is_void = True
                has_stair = False

                for inner_row in range(ROOM_HEIGHT):
                    y = row * ROOM_HEIGHT + inner_row
                    x_start = col * ROOM_WIDTH
                    x_end = x_start + ROOM_WIDTH

                    if y < len(lines):
                        line_slice = lines[y][x_start:x_end]
                        if len(line_slice) < ROOM_WIDTH:
                            line_slice = line_slice.ljust(ROOM_WIDTH, "-")
                    else:
                        line_slice = "-" * ROOM_WIDTH

                    room_chars.append(list(line_slice))

                    if any(char not in "-" for char in line_slice):
                        is_void = False
                    if "S" in line_slice:
                        has_stair = True

                if is_void:
                    continue

                char_grid = np.array(room_chars, dtype="<U1")
                doors = self._detect_doors(char_grid)
                semantic_grid = self._to_semantic(char_grid, doors)

                rooms[(row, col)] = self.room_cls(
                    position=(row, col),
                    char_grid=char_grid,
                    semantic_grid=semantic_grid,
                    doors=doors,
                    has_stair=has_stair,
                )

        return rooms

    def _detect_doors(self, char_grid: np.ndarray) -> Dict[str, bool]:
        doors = {}

        north = DOOR_POSITIONS["N"]
        n_row = int(north["row"])
        n_c0, n_c1 = int(north["col_start"]), int(north["col_end"])
        north_cells = char_grid[n_row, n_c0:n_c1] if char_grid.shape[0] > n_row else []
        doors["N"] = "D" in north_cells

        south = DOOR_POSITIONS["S"]
        s_row = int(south["row"])
        s_c0, s_c1 = int(south["col_start"]), int(south["col_end"])
        south_cells = char_grid[s_row, s_c0:s_c1] if char_grid.shape[0] > s_row else []
        doors["S"] = "D" in south_cells

        west = DOOR_POSITIONS["W"]
        w_col = int(west["col"])
        w_r0, w_r1 = int(west["row_start"]), int(west["row_end"])
        west_cells = char_grid[w_r0:w_r1, w_col] if char_grid.shape[1] > w_col else []
        doors["W"] = "D" in west_cells

        east = DOOR_POSITIONS["E"]
        e_col = int(east["col"])
        e_r0, e_r1 = int(east["row_start"]), int(east["row_end"])
        east_cells = char_grid[e_r0:e_r1, e_col] if char_grid.shape[1] > e_col else []
        doors["E"] = "D" in east_cells

        return doors

    def _to_semantic(self, char_grid: np.ndarray, doors: Dict[str, bool]) -> np.ndarray:
        semantic = np.zeros(char_grid.shape, dtype=np.int32)

        for row in range(char_grid.shape[0]):
            for col in range(char_grid.shape[1]):
                char = char_grid[row, col]
                semantic[row, col] = CHAR_TO_SEMANTIC.get(char, SEMANTIC_PALETTE["VOID"])

        has_any_door = any(doors.values())
        interior = semantic[2:14, 2:9]
        void_count = np.sum(interior == SEMANTIC_PALETTE["VOID"])
        total_interior = interior.size

        if has_any_door and void_count > total_interior * 0.5:
            for row in range(2, 14):
                for col in range(2, 9):
                    if semantic[row, col] == SEMANTIC_PALETTE["VOID"]:
                        semantic[row, col] = SEMANTIC_PALETTE["FLOOR"]

        return semantic


class DOTParser:
    """Parse DOT graph files into NetworkX digraphs."""

    def parse(self, filepath: str) -> nx.DiGraph:
        with open(filepath, "r") as file_obj:
            content = file_obj.read()

        graph = nx.DiGraph()

        node_pattern = r"^\s*(\d+)\s*\[([^\]]*)\]"
        for match in re.finditer(node_pattern, content, re.MULTILINE):
            node_id = int(match.group(1))
            attrs = match.group(2)
            label = ""
            quoted = re.search(r"label\s*=\s*\"", attrs)
            if quoted:
                start = quoted.end()
                end = attrs.find('"', start)
                label = attrs[start:end] if end != -1 else attrs[start:]
            else:
                unquoted = re.search(r"label\s*=\s*", attrs)
                if unquoted:
                    start = unquoted.end()
                    next_attr = re.search(r",\s*\w+\s*=", attrs[start:])
                    end = start + next_attr.start() if next_attr else len(attrs)
                    label = attrs[start:end].strip()

            parts = parse_node_label_tokens(label)
            label_norm = normalize_node_label(label)

            is_start = ("s" in parts) or ("S" in parts)
            is_start_pointer = is_start and all(part in ("s", "") for part in parts)

            graph.add_node(
                node_id,
                label=label_norm,
                is_start=is_start,
                is_start_pointer=is_start_pointer,
                is_triforce="t" in parts,
                is_boss="b" in parts,
                has_key="k" in parts,
                has_item="I" in parts or "i" in parts or "K" in parts,
                has_enemy="e" in parts,
                has_puzzle="p" in parts,
            )

        edge_pattern = r"(\d+)\s*->\s*(\d+)(?:\s*\[([^\]]*)\])?"
        for match in re.finditer(edge_pattern, content, re.MULTILINE):
            src = int(match.group(1))
            dst = int(match.group(2))
            attrs = match.group(3) or ""

            label = ""
            quoted = re.search(r"label\s*=\s*\"", attrs)
            if quoted:
                start = quoted.end()
                end = attrs.find('"', start)
                label = attrs[start:end] if end != -1 else attrs[start:]
            else:
                unquoted = re.search(r"label\s*=\s*", attrs)
                if unquoted:
                    start = unquoted.end()
                    next_attr = re.search(r",\s*\w+\s*=", attrs[start:])
                    end = start + next_attr.start() if next_attr else len(attrs)
                    label = attrs[start:end].strip()

            label_clean = str(label or "").replace("\n", ",").strip()
            if label_clean:
                label_clean = ",".join([part.strip() for part in label_clean.split(",") if part.strip()])
            edge_constraints = parse_edge_type_tokens(label=label_clean)
            edge_type = select_primary_edge_type(edge_constraints)

            graph.add_edge(
                src,
                dst,
                label=label_clean,
                edge_type=edge_type,
                edge_constraints=edge_constraints,
            )

        return graph
