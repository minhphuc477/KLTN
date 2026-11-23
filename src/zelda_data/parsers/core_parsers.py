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


def _read_vglc_lines(filepath: str) -> List[str]:
    """Reject malformed maps instead of silently padding or dropping tiles."""
    with open(filepath, "r", encoding="utf-8") as file_obj:
        lines = file_obj.read().splitlines()
    if not lines:
        return []
    width = len(lines[0])
    if width == 0 or any(len(line) != width for line in lines):
        raise ValueError(f"VGLC map must be a nonempty rectangle: {filepath}")
    if len(lines) % ROOM_HEIGHT or width % ROOM_WIDTH:
        raise ValueError(f"VGLC map dimensions must be multiples of {ROOM_HEIGHT}x{ROOM_WIDTH}: {filepath}")
    unknown = set("".join(lines)).difference(CHAR_TO_SEMANTIC)
    if unknown:
        raise ValueError(f"Unknown VGLC glyphs {sorted(unknown)!r}: {filepath}")
    return lines


def is_vglc_room_slot(slot_grid: np.ndarray) -> bool:
    """Return whether a canonical VGLC slot contains a playable room.

    Some Zelda maps contain framed placeholders made entirely from WALL and
    VOID glyphs.  They are layout blanks, not training rooms: VGLC defines
    ``-`` as VOID, and these slots contain no floor, door, stair, entity, or
    other playable content.  Keep one detector shared by both parser paths so
    training and reference evaluation cannot silently use different corpora.
    """
    if slot_grid.size == 0:
        return False

    normalized = np.char.upper(np.asarray(slot_grid, dtype="<U1"))
    gap_count = int(np.sum(normalized == "-"))
    total = int(normalized.size)
    if gap_count > int(0.7 * total):
        return False

    wall_count = int(np.sum(normalized == "W"))
    door_count = int(np.sum(normalized == "D"))
    if door_count > 0 and wall_count >= 20:
        return True

    content_count = total - wall_count - gap_count
    return bool(wall_count >= 20 and content_count >= 5)


class GridBasedRoomExtractor:
    """Extract rooms from fixed-size VGLC slots."""

    SLOT_WIDTH = 11
    SLOT_HEIGHT = 16
    GAP_MARKER = "-"
    WALL_MARKER = "W"

    def _load_grid(self, filepath: str) -> np.ndarray:
        lines = _read_vglc_lines(filepath)

        if not lines:
            return np.zeros((0, 0), dtype="<U1")

        width = max(len(line) for line in lines) if lines else 0
        padded = [list(line.ljust(width, self.GAP_MARKER)) for line in lines]
        return np.array(padded)

    def _is_room_slot(self, slot_grid: np.ndarray) -> bool:
        return is_vglc_room_slot(slot_grid)

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
        lines = _read_vglc_lines(filepath)

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

                    if "S" in line_slice:
                        has_stair = True

                char_grid = np.array(room_chars, dtype="<U1")
                if not is_vglc_room_slot(char_grid):
                    continue

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
        open_door_glyphs = {"D", "d", "F", "f", "."}

        def _has_open_door(cells: Any) -> bool:
            return any(str(cell) in open_door_glyphs for cell in np.asarray(cells).ravel())

        # VGLC Zelda rooms use a two-tile wall shell in some files.  Door
        # glyphs can therefore appear on either the outer canonical boundary
        # (row 0/15, col 0/10) or the inner wall layer (row 1/14, col 1/9).
        north = DOOR_POSITIONS["N"]
        n_row = int(north["row"])
        n_c0, n_c1 = int(north["col_start"]), int(north["col_end"])
        north_cells = []
        for row_idx in (n_row, min(n_row + 1, char_grid.shape[0] - 1)):
            if 0 <= row_idx < char_grid.shape[0]:
                north_cells.append(char_grid[row_idx, n_c0:n_c1])
        doors["N"] = _has_open_door(north_cells)

        south = DOOR_POSITIONS["S"]
        s_row = int(south["row"])
        s_c0, s_c1 = int(south["col_start"]), int(south["col_end"])
        south_cells = []
        for row_idx in (s_row, max(s_row - 1, 0)):
            if 0 <= row_idx < char_grid.shape[0]:
                south_cells.append(char_grid[row_idx, s_c0:s_c1])
        doors["S"] = _has_open_door(south_cells)

        west = DOOR_POSITIONS["W"]
        w_col = int(west["col"])
        w_r0, w_r1 = int(west["row_start"]), int(west["row_end"])
        west_cells = []
        for col_idx in (w_col, min(w_col + 1, char_grid.shape[1] - 1)):
            if 0 <= col_idx < char_grid.shape[1]:
                west_cells.append(char_grid[w_r0:w_r1, col_idx])
        doors["W"] = _has_open_door(west_cells)

        east = DOOR_POSITIONS["E"]
        e_col = int(east["col"])
        e_r0, e_r1 = int(east["row_start"]), int(east["row_end"])
        east_cells = []
        for col_idx in (e_col, max(e_col - 1, 0)):
            if 0 <= col_idx < char_grid.shape[1]:
                east_cells.append(char_grid[e_r0:e_r1, col_idx])
        doors["E"] = _has_open_door(east_cells)

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
    """Parse VGLC DOT into a traversable mission graph.

    VGLC edge ``s`` means visible but impassable (see the corpus zelda.json).
    Keep those source relations in graph metadata, outside passage adjacency.
    Explicit ``stair``/``warp`` edges remain supported for authored missions.
    """

    def parse(self, filepath: str) -> nx.DiGraph:
        with open(filepath, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
        return self.parse_string(content)

    def parse_string(self, content: str) -> nx.DiGraph:
        """Parse the same source schema from an in-memory DOT document."""

        graph = nx.DiGraph()
        graph.graph["source_schema"] = "vglc_zelda"
        graph.graph["non_traversable_edges"] = []

        node_pattern = r"(?:^|[;{}])\s*(\d+)\s*\[([^\]]*)\]"
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
            # Uppercase S is a switch in the source, whereas legacy internal
            # graphs use it as a start alias. Normalize at the ingestion boundary
            # and preserve the source label for an auditable, lossless mapping.
            switch_tokens = [part for part in parts if re.fullmatch(r"S\d*", part)]
            label_norm = normalize_node_label(
                ",".join("switch" if part in switch_tokens else part for part in parts)
            )

            is_start = "s" in parts
            is_start_pointer = is_start and all(part in ("s", "") for part in parts)

            graph.add_node(
                node_id,
                label=label_norm,
                source_label=label,
                is_start=is_start,
                is_start_pointer=is_start_pointer,
                is_triforce="t" in parts,
                is_boss="b" in parts,
                has_key="k" in parts,
                has_item="I" in parts,
                has_minor_item="i" in parts,
                has_boss_key="K" in parts,
                has_switch=bool(switch_tokens),
                switch_tokens=switch_tokens,
                has_enemy="e" in parts,
                has_puzzle="p" in parts,
            )

        switch_providers = {}
        for node_id, data in graph.nodes(data=True):
            for token in data.get("switch_tokens", []):
                switch_providers.setdefault(token, []).append(node_id)

        edge_pattern = r"(\d+)\s*(->|--)\s*(\d+)(?:\s*\[([^\]]*)\])?"
        for match in re.finditer(edge_pattern, content, re.MULTILINE):
            src = int(match.group(1))
            operator = match.group(2)
            dst = int(match.group(3))
            attrs = match.group(4) or ""

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

            if "impassable" in edge_constraints:
                graph.graph["non_traversable_edges"].append(
                    {"source": src, "target": dst, "label": label_clean,
                     "edge_type": edge_type, "edge_constraints": edge_constraints}
                )
                # DOT permits nodes to be declared implicitly by an edge.
                graph.add_nodes_from((src, dst))
                if operator == "--":
                    graph.graph["non_traversable_edges"].append(
                        {**graph.graph["non_traversable_edges"][-1], "source": dst, "target": src}
                    )
                continue

            graph.add_edge(
                src,
                dst,
                label=label_clean,
                edge_type=edge_type,
                edge_constraints=edge_constraints,
            )
            # Numbered source switches name a particular provider, not any
            # puzzle previously visited. DOT node IDs here are nonnegative;
            # a missing/ambiguous provider gets an impossible negative ID.
            numbered_switches = [token for token in label_clean.split(",") if re.fullmatch(r"S\d+", token)]
            if numbered_switches:
                required = []
                unresolved = []
                for token in numbered_switches:
                    providers = switch_providers.get(token, [])
                    if len(providers) == 1:
                        required.append(providers[0])
                    else:
                        required.append(-1 - int(token[1:]))
                        unresolved.append(token)
                graph.edges[src, dst]["switches_required"] = list(dict.fromkeys(required))
                if len(set(required)) == 1:
                    graph.edges[src, dst]["switch_id"] = required[0]
                if unresolved:
                    graph.edges[src, dst]["unresolved_switch_tokens"] = unresolved
            if operator == "--":
                reverse = dict(graph.edges[src, dst])
                reverse['metadata'] = {**dict(reverse.get('metadata', {})), 'implied_reverse': True}
                graph.add_edge(dst, src, **reverse)

        return graph
