"""Reusable GUI overlay render helpers extracted from gui_runner monolith."""

from __future__ import annotations

import logging
from typing import Any

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE


logger = logging.getLogger(__name__)


def default_topology_semantics() -> dict:
    """Default topology semantics mapping for legend/tooltips."""
    return {
        "nodes": {
            "e": ["room", "enemy"],
            "S": ["room", "switch"],
            "b": ["room", "boss"],
            "k": ["room", "key"],
            "K": ["room", "boss key"],
            "I": ["room", "key item"],
            "p": ["room", "puzzle"],
            "s": ["room", "start"],
            "t": ["room", "triforce"],
        },
        "edges": {
            "S": ["door", "switch locked"],
            "b": ["door", "bombable"],
            "k": ["door", "key locked"],
            "K": ["door", "boss key locked"],
            "I": ["door", "key item locked"],
            "l": ["door", "soft locked"],
            "s": ["visible", "impassable"],
        },
    }


def _as_grid(current: Any):
    grid = getattr(current, "global_grid", current)
    if grid is None:
        return None
    shape = getattr(grid, "shape", None)
    if shape is not None and len(shape) >= 2:
        return grid
    return None


def _grid_tile(grid: Any, row: int, col: int) -> int:
    try:
        return int(grid[row, col])
    except TypeError:
        return int(grid[row][col])


def _infer_topology_from_grid(current: Any) -> tuple[dict, list]:
    """Infer room nodes/edges for imported stitched grids without graph metadata."""
    grid = _as_grid(current)
    if grid is None:
        return {}, []

    height, width = int(grid.shape[0]), int(grid.shape[1])
    void_id = int(SEMANTIC_PALETTE["VOID"])
    wall_id = int(SEMANTIC_PALETTE["WALL"])
    walkable_ids = {
        int(SEMANTIC_PALETTE["FLOOR"]),
        int(SEMANTIC_PALETTE["DOOR_OPEN"]),
        int(SEMANTIC_PALETTE["DOOR_SOFT"]),
        int(SEMANTIC_PALETTE["DOOR_LOCKED"]),
        int(SEMANTIC_PALETTE["DOOR_BOMB"]),
        int(SEMANTIC_PALETTE["DOOR_BOSS"]),
        int(SEMANTIC_PALETTE["DOOR_PUZZLE"]),
        int(SEMANTIC_PALETTE["START"]),
        int(SEMANTIC_PALETTE["TRIFORCE"]),
        int(SEMANTIC_PALETTE["KEY_SMALL"]),
        int(SEMANTIC_PALETTE["KEY_BOSS"]),
        int(SEMANTIC_PALETTE["KEY_ITEM"]),
        int(SEMANTIC_PALETTE["ITEM_MINOR"]),
        int(SEMANTIC_PALETTE["ELEMENT_FLOOR"]),
        int(SEMANTIC_PALETTE["STAIR"]),
        int(SEMANTIC_PALETTE["ENEMY"]),
        int(SEMANTIC_PALETTE["BOSS"]),
        int(SEMANTIC_PALETTE["PUZZLE"]),
    }
    door_type_by_id = {
        int(SEMANTIC_PALETTE["DOOR_LOCKED"]): "key_locked",
        int(SEMANTIC_PALETTE["DOOR_BOMB"]): "bombable",
        int(SEMANTIC_PALETTE["DOOR_BOSS"]): "boss_locked",
        int(SEMANTIC_PALETTE["DOOR_PUZZLE"]): "puzzle_locked",
        int(SEMANTIC_PALETTE["DOOR_SOFT"]): "soft_locked",
        int(SEMANTIC_PALETTE["STAIR"]): "stair",
    }

    nodes = {}
    room_rows = (height + ROOM_HEIGHT - 1) // ROOM_HEIGHT
    room_cols = (width + ROOM_WIDTH - 1) // ROOM_WIDTH
    for room_row in range(room_rows):
        for room_col in range(room_cols):
            row0 = room_row * ROOM_HEIGHT
            col0 = room_col * ROOM_WIDTH
            row1 = min(height, row0 + ROOM_HEIGHT)
            col1 = min(width, col0 + ROOM_WIDTH)
            non_empty = 0
            passable = 0
            for row in range(row0, row1):
                for col in range(col0, col1):
                    tile = _grid_tile(grid, row, col)
                    if tile not in {void_id, wall_id}:
                        non_empty += 1
                    if tile in walkable_ids:
                        passable += 1
            if non_empty > 0 or passable > 0:
                nodes[(room_row, room_col)] = {
                    "center": ((row0 + row1 - 1) / 2.0, (col0 + col1 - 1) / 2.0),
                    "bounds": (row0, col0, row1, col1),
                }

    edges = []
    seen = set()
    for room_pos, meta in nodes.items():
        room_row, room_col = room_pos
        for neighbor in ((room_row, room_col + 1), (room_row + 1, room_col)):
            if neighbor not in nodes:
                continue
            edge_key = tuple(sorted((room_pos, neighbor)))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            r0, c0, r1, c1 = meta["bounds"]
            nr0, nc0, nr1, nc1 = nodes[neighbor]["bounds"]
            edge_type = "open"
            connected = False
            if neighbor[0] == room_row:
                left_col = c1 - 1
                right_col = nc0
                for row in range(max(r0, nr0), min(r1, nr1)):
                    lt = _grid_tile(grid, row, left_col)
                    rt = _grid_tile(grid, row, right_col)
                    if lt in walkable_ids or rt in walkable_ids:
                        connected = True
                        edge_type = door_type_by_id.get(lt, door_type_by_id.get(rt, "open"))
                        break
            else:
                top_row = r1 - 1
                bottom_row = nr0
                for col in range(max(c0, nc0), min(c1, nc1)):
                    tt = _grid_tile(grid, top_row, col)
                    bt = _grid_tile(grid, bottom_row, col)
                    if tt in walkable_ids or bt in walkable_ids:
                        connected = True
                        edge_type = door_type_by_id.get(tt, door_type_by_id.get(bt, "open"))
                        break
            if connected:
                edges.append((room_pos, neighbor, {"edge_type": edge_type, "inferred": True}))

    return nodes, edges


def render_topology_overlay(
    *,
    surface: Any,
    current: Any,
    tile_size: int,
    view_offset_x: int,
    view_offset_y: int,
    pygame: Any,
) -> None:
    """Draw topology nodes/edges for the current stitched dungeon map."""
    graph = getattr(current, "graph", None)

    room_positions = getattr(current, "room_positions", {})
    room_to_node = getattr(current, "room_to_node", {})
    node_to_room = {v: k for k, v in room_to_node.items()} if room_to_node else {}

    try:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        overlay = None

    node_pos = {}
    unmatched_nodes = 0
    inferred_edges = []
    if graph:
        for node in graph.nodes():
            room_pos = node_to_room.get(node)
            if room_pos is None:
                unmatched_nodes += 1
                continue
            rp = room_positions.get(room_pos)
            if not rp:
                unmatched_nodes += 1
                continue
            ry, rx = rp
            cx = (rx + ROOM_WIDTH / 2.0) * tile_size - view_offset_x
            cy = (ry + ROOM_HEIGHT / 2.0) * tile_size - view_offset_y
            node_pos[node] = (cx, cy)
    else:
        inferred_nodes, inferred_edges = _infer_topology_from_grid(current)
        for node, meta in inferred_nodes.items():
            center_row, center_col = meta["center"]
            cx = (center_col + 0.5) * tile_size - view_offset_x
            cy = (center_row + 0.5) * tile_size - view_offset_y
            node_pos[node] = (cx, cy)

    edge_colors = {
        "open": (100, 255, 100, 180),
        "key_locked": (255, 220, 100, 200),
        "bombable": (255, 150, 50, 200),
        "boss_locked": (255, 90, 90, 210),
        "puzzle_locked": (190, 100, 255, 200),
        "soft_locked": (180, 100, 255, 180),
        "stair": (100, 200, 255, 200),
    }
    default_edge_color = (150, 150, 200, 150)

    target_surface = overlay if overlay else surface
    edge_iter = graph.edges(data=True) if graph else inferred_edges
    for u, v, data in edge_iter:
        if u not in node_pos or v not in node_pos:
            continue
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        edge_type = data.get("edge_type", data.get("type", "open")) if data else "open"
        color = edge_colors.get(edge_type, default_edge_color)
        try:
            pygame.draw.line(target_surface, color[:3], (int(x1), int(y1)), (int(x2), int(y2)), 3)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pygame.draw.line(surface, color[:3], (int(x1), int(y1)), (int(x2), int(y2)), 3)

    node_radius = max(8, tile_size // 3)
    font = pygame.font.SysFont("Arial", 12, bold=True)
    for node, (cx, cy) in node_pos.items():
        try:
            pygame.draw.circle(target_surface, (255, 255, 255, 100), (int(cx), int(cy)), node_radius + 3)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Topology overlay halo draw failed for node %s: %s", node, exc)
        pygame.draw.circle(target_surface, (80, 120, 200), (int(cx), int(cy)), node_radius)
        pygame.draw.circle(target_surface, (150, 200, 255), (int(cx), int(cy)), node_radius, 2)
        try:
            label_text = f"{node[0]},{node[1]}" if isinstance(node, tuple) else str(node)
            label = font.render(label_text, True, (255, 255, 255))
            lx = int(cx - label.get_width() / 2)
            ly = int(cy - label.get_height() / 2)
            target_surface.blit(label, (lx, ly))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Topology overlay label draw failed for node %s: %s", node, exc)

    if overlay:
        surface.blit(overlay, (0, 0))

    if unmatched_nodes > 0:
        try:
            warn_font = pygame.font.SysFont("Arial", 14, bold=True)
            warn_text = warn_font.render(f"{unmatched_nodes} unmatched nodes", True, (255, 150, 100))
            surface.blit(warn_text, (10, 10))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Topology overlay warning draw failed: %s", exc)


def render_solver_comparison_overlay(
    *,
    surface: Any,
    results: list,
    screen_w: int,
    sidebar_width: int,
    pygame: Any,
) -> None:
    """Render solver comparison table in the sidebar."""
    if not results:
        return

    sidebar_x = screen_w - sidebar_width
    box_w = sidebar_width - 20
    has_cbs = any("CBS" in r["name"] for r in results)
    row_height = 22 if has_cbs else 18
    box_h = min(300, 24 + row_height * len(results) + 20)
    box_y = 220

    box_rect = pygame.Rect(sidebar_x + 10, box_y, box_w, box_h)
    pygame.draw.rect(surface, (38, 38, 55), box_rect)
    pygame.draw.rect(surface, (100, 150, 255), box_rect, 1)

    font = pygame.font.SysFont("Arial", 11, bold=True)
    if has_cbs:
        header = font.render("Solver Comparison", True, (200, 200, 255))
    else:
        header = font.render("Solver   Success   Len   Nodes   ms", True, (200, 200, 255))
    surface.blit(header, (box_rect.x + 6, box_rect.y + 6))

    y = box_rect.y + 28
    small = pygame.font.SysFont("Arial", 10)
    for row in results:
        if "CBS" in row["name"] and "confusion" in row:
            line1 = f"{row['name'][:15]:15} {str(row.get('success', False))[:5]:5} Len:{row.get('path_len', 0):<4}"
            raw_len = int(row.get("trajectory_len", 0) or 0)
            if raw_len and raw_len != int(row.get("path_len", 0) or 0):
                line1 += f" Raw:{raw_len:<4}"
            line2 = (
                f"  Confusion:{row.get('confusion', 0):.2f} "
                f"Rooms:{int(row.get('unique_rooms', 0) or 0)} "
                f"Load:{row.get('cog_load', 0):.2f} {int(row.get('time_ms', 0))}ms"
            )
            color = (200, 255, 200) if row.get("success") else (255, 150, 150)
            surface.blit(small.render(line1, True, color), (box_rect.x + 6, y))
            surface.blit(small.render(line2, True, (180, 180, 255)), (box_rect.x + 6, y + 11))
            y += row_height
        else:
            fallback_mark = "FB" if row.get("fallback_used") else "  "
            text = (
                f"{row['name'][:7]:7}   {str(row.get('success', False))[:5]:5}   "
                f"{row.get('path_len', 0):3}   {row.get('nodes', 0):6}   {int(row.get('time_ms', 0)):4} {fallback_mark}"
            )
            surface.blit(small.render(text, True, (200, 200, 200)), (box_rect.x + 6, y))
            y += row_height

    legend = small.render("FB = fallback used", True, (150, 150, 150))
    surface.blit(legend, (box_rect.x + 6, box_rect.y + box_rect.h - 30))

    hint = small.render("Press Esc to close", True, (150, 150, 150))
    surface.blit(hint, (box_rect.x + 6, box_rect.y + box_rect.h - 18))

