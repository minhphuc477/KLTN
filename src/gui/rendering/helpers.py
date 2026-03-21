"""Reusable GUI overlay render helpers extracted from gui_runner monolith."""

from __future__ import annotations

import logging
from typing import Any


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
    if not graph:
        return

    room_positions = getattr(current, "room_positions", {})
    room_to_node = getattr(current, "room_to_node", {})
    node_to_room = {v: k for k, v in room_to_node.items()} if room_to_node else {}

    try:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    except Exception:
        overlay = None

    node_pos = {}
    unmatched_nodes = 0
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
        cx = rx * tile_size + tile_size * 0.5 - view_offset_x
        cy = ry * tile_size + tile_size * 0.5 - view_offset_y
        node_pos[node] = (cx, cy)

    edge_colors = {
        "open": (100, 255, 100, 180),
        "key_locked": (255, 220, 100, 200),
        "bombable": (255, 150, 50, 200),
        "soft_locked": (180, 100, 255, 180),
        "stair": (100, 200, 255, 200),
    }
    default_edge_color = (150, 150, 200, 150)

    target_surface = overlay if overlay else surface
    for u, v, data in graph.edges(data=True):
        if u not in node_pos or v not in node_pos:
            continue
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        edge_type = data.get("type", "open") if data else "open"
        color = edge_colors.get(edge_type, default_edge_color)
        try:
            pygame.draw.line(target_surface, color[:3], (int(x1), int(y1)), (int(x2), int(y2)), 3)
        except Exception:
            pygame.draw.line(surface, color[:3], (int(x1), int(y1)), (int(x2), int(y2)), 3)

    node_radius = max(8, tile_size // 3)
    font = pygame.font.SysFont("Arial", 12, bold=True)
    for node, (cx, cy) in node_pos.items():
        try:
            pygame.draw.circle(target_surface, (255, 255, 255, 100), (int(cx), int(cy)), node_radius + 3)
        except Exception as exc:
            logger.debug("Topology overlay halo draw failed for node %s: %s", node, exc)
        pygame.draw.circle(target_surface, (80, 120, 200), (int(cx), int(cy)), node_radius)
        pygame.draw.circle(target_surface, (150, 200, 255), (int(cx), int(cy)), node_radius, 2)
        try:
            label = font.render(str(node), True, (255, 255, 255))
            lx = int(cx - label.get_width() / 2)
            ly = int(cy - label.get_height() / 2)
            target_surface.blit(label, (lx, ly))
        except Exception as exc:
            logger.debug("Topology overlay label draw failed for node %s: %s", node, exc)

    if overlay:
        surface.blit(overlay, (0, 0))

    if unmatched_nodes > 0:
        try:
            warn_font = pygame.font.SysFont("Arial", 14, bold=True)
            warn_text = warn_font.render(f"{unmatched_nodes} unmatched nodes", True, (255, 150, 100))
            surface.blit(warn_text, (10, 10))
        except Exception as exc:
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
            line2 = f"  Confusion:{row.get('confusion', 0):.2f} Load:{row.get('cog_load', 0):.2f} {int(row.get('time_ms', 0))}ms"
            color = (200, 255, 200) if row.get("success") else (255, 150, 150)
            surface.blit(small.render(line1, True, color), (box_rect.x + 6, y))
            surface.blit(small.render(line2, True, (180, 180, 255)), (box_rect.x + 6, y + 11))
            y += row_height
        else:
            text = (
                f"{row['name'][:7]:7}   {str(row.get('success', False))[:5]:5}   "
                f"{row.get('path_len', 0):3}   {row.get('nodes', 0):6}   {int(row.get('time_ms', 0)):4}"
            )
            surface.blit(small.render(text, True, (200, 200, 200)), (box_rect.x + 6, y))
            y += row_height

    hint = small.render("Press Esc to close", True, (150, 150, 150))
    surface.blit(hint, (box_rect.x + 6, box_rect.y + box_rect.h - 18))
