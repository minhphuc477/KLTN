"""Helpers for minimap rendering and click interactions."""

import math
import time
from typing import Any, Tuple

from src.core.definitions import SEMANTIC_PALETTE
from src.pipeline.spatial_utils import normalize_node_id, stable_node_sort_key


def render_minimap(gui: Any, pygame: Any) -> None:
    """Render small dungeon overview map in bottom-right corner."""
    if not getattr(gui, "env", None):
        return

    minimap_margin = 20
    minimap_x = gui.screen_w - gui.SIDEBAR_WIDTH - gui.minimap_size - minimap_margin
    minimap_y = gui.screen_h - gui.HUD_HEIGHT - gui.minimap_size - minimap_margin

    minimap = pygame.Surface((gui.minimap_size, gui.minimap_size), pygame.SRCALPHA)
    pygame.draw.rect(minimap, (40, 40, 60, 220), minimap.get_rect(), border_radius=8)

    editor_mode = bool(getattr(gui, "ai_mission_graph_editor_enabled", False)) and getattr(
        gui, "ai_mission_graph_draft", None
    ) is not None

    title_font = pygame.font.SysFont("Arial", 10, bold=True)
    title_label = "Mission Graph" if editor_mode else "Dungeon Map"
    title_surf = title_font.render(title_label, True, (180, 180, 200))
    minimap.blit(title_surf, (5, 3))

    map_h, map_w = gui.env.height, gui.env.width
    content_area = gui.minimap_size - 30
    scale_x = content_area / map_w
    scale_y = content_area / map_h
    scale = min(scale_x, scale_y)

    scaled_w = int(map_w * scale)
    scaled_h = int(map_h * scale)
    offset_x = (gui.minimap_size - scaled_w) // 2
    offset_y = 18 + (gui.minimap_size - 18 - scaled_h) // 2

    if editor_mode:
        graph = gui.ai_mission_graph_draft
        layout = dict(getattr(gui, "ai_mission_graph_layout", {}) or {})
        node_ids = sorted(list(getattr(graph, "nodes", {}).keys()), key=stable_node_sort_key)
        if node_ids and len(layout) != len(node_ids):
            layout = {}
            total = len(node_ids)
            for idx, node_id in enumerate(node_ids):
                x = 0.08 + 0.84 * (float(idx) / float(max(1, total - 1)))
                y = 0.5
                layout[node_id] = (x, y)
            gui.ai_mission_graph_layout = layout

        staged_locked = set(tuple(edge) for edge in list(getattr(gui, "ai_mission_graph_locked_edges", []) or []))
        boss_node = getattr(gui, "ai_mission_graph_boss_node", None)
        pending_source = getattr(gui, "ai_mission_graph_pending_lock_source", None)

        def _pos(node_id):
            nxn, nyn = layout.get(node_id, (0.5, 0.5))
            px = offset_x + int(float(nxn) * max(1, scaled_w - 1))
            py = offset_y + int(float(nyn) * max(1, scaled_h - 1))
            return px, py

        for edge in getattr(graph, "edges", []):
            src = normalize_node_id(getattr(edge, "source", None))
            dst = normalize_node_id(getattr(edge, "target", None))
            if src is None or dst is None:
                continue
            x1, y1 = _pos(src)
            x2, y2 = _pos(dst)
            is_locked = str(getattr(getattr(edge, "edge_type", None), "name", "")).upper() == "LOCKED"
            if (src, dst) in staged_locked:
                is_locked = True
            color = (255, 170, 70) if is_locked else (130, 165, 230)
            pygame.draw.line(minimap, color, (x1, y1), (x2, y2), 2)

        for node_id in node_ids:
            x, y = _pos(node_id)
            fill = (80, 120, 200)
            if boss_node is not None and normalize_node_id(node_id) == normalize_node_id(boss_node):
                fill = (215, 65, 65)
            if pending_source is not None and normalize_node_id(node_id) == normalize_node_id(pending_source):
                fill = (230, 175, 45)
            pygame.draw.circle(minimap, (230, 230, 245), (x, y), 7)
            pygame.draw.circle(minimap, fill, (x, y), 5)
    else:
        for r in range(map_h):
            for c in range(map_w):
                tile_id = gui.env.grid[r, c]
                if tile_id == SEMANTIC_PALETTE["VOID"]:
                    continue
                if tile_id == SEMANTIC_PALETTE["WALL"] or tile_id == SEMANTIC_PALETTE["BLOCK"]:
                    color = (60, 60, 80)
                elif tile_id == SEMANTIC_PALETTE["START"]:
                    color = (80, 180, 80)
                elif tile_id == SEMANTIC_PALETTE["TRIFORCE"]:
                    color = (255, 215, 0)
                elif tile_id in [SEMANTIC_PALETTE["KEY_SMALL"], SEMANTIC_PALETTE["KEY_BOSS"]]:
                    color = (255, 200, 50)
                elif tile_id in [
                    SEMANTIC_PALETTE["DOOR_LOCKED"],
                    SEMANTIC_PALETTE["DOOR_BOMB"],
                    SEMANTIC_PALETTE["DOOR_BOSS"],
                ]:
                    color = (180, 100, 50)
                elif tile_id == SEMANTIC_PALETTE["STAIR"]:
                    color = (100, 150, 255)
                elif tile_id == SEMANTIC_PALETTE["ENEMY"]:
                    color = (200, 50, 50)
                else:
                    color = (100, 120, 140)

                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                mini_w = max(1, int(scale))
                mini_h = max(1, int(scale))
                pygame.draw.rect(minimap, color, (mini_x, mini_y, mini_w, mini_h))

        pr, pc = gui.env.state.position
        player_x = offset_x + int(pc * scale)
        player_y = offset_y + int(pr * scale)
        player_size = max(2, int(scale * 1.5))
        pygame.draw.circle(minimap, (255, 100, 100), (player_x, player_y), player_size)
        pygame.draw.circle(minimap, (255, 255, 255), (player_x, player_y), player_size + 1, 1)

        current_time = time.time()
        pulse = (math.sin(current_time * 3) + 1) / 2

        for pos in gui.env._find_all_positions(SEMANTIC_PALETTE["KEY_SMALL"]):
            if pos not in gui.env.state.collected_items:
                r, c = pos
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                size = int(2 + pulse * 2)
                pygame.draw.circle(minimap, (255, 255, 0), (mini_x, mini_y), size)

        for pos in gui.env._find_all_positions(SEMANTIC_PALETTE["KEY_BOSS"]):
            if pos not in gui.env.state.collected_items:
                r, c = pos
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                size = int(2 + pulse * 2)
                pygame.draw.circle(minimap, (255, 150, 0), (mini_x, mini_y), size)

        # Draw staged mixed-initiative anchors for the next AI generation run.
        def _draw_norm_anchor(norm, color, radius=4):
            if not isinstance(norm, (tuple, list)) or len(norm) < 2:
                return
            try:
                nr = max(0.0, min(1.0, float(norm[0])))
                nc = max(0.0, min(1.0, float(norm[1])))
            except (TypeError, ValueError):
                return
            rr = int(round(nr * max(0, map_h - 1)))
            cc = int(round(nc * max(0, map_w - 1)))
            px = offset_x + int(cc * scale)
            py = offset_y + int(rr * scale)
            pygame.draw.circle(minimap, color, (px, py), radius)
            pygame.draw.circle(minimap, (255, 255, 255), (px, py), radius + 1, 1)

        _draw_norm_anchor(getattr(gui, "ai_constraint_boss_norm", None), (220, 80, 80), radius=5)
        _draw_norm_anchor(getattr(gui, "ai_constraint_lock_norm", None), (230, 170, 70), radius=4)
        _draw_norm_anchor(getattr(gui, "ai_constraint_key_norm", None), (255, 230, 80), radius=4)

    pygame.draw.rect(minimap, (70, 70, 100), minimap.get_rect(), 2, border_radius=8)
    gui.screen.blit(minimap, (minimap_x, minimap_y))


def handle_minimap_click(gui: Any, mouse_pos: Tuple[int, int], pygame_module: Any = None, button: int = 1) -> bool:
    """Handle click on minimap and recenter view to selected tile."""
    if not getattr(gui, "show_minimap", False) or not getattr(gui, "env", None):
        return False

    minimap_margin = 20
    minimap_x = gui.screen_w - gui.SIDEBAR_WIDTH - gui.minimap_size - minimap_margin
    minimap_y = gui.screen_h - gui.HUD_HEIGHT - gui.minimap_size - minimap_margin

    mx, my = mouse_pos
    if not (
        minimap_x <= mx <= minimap_x + gui.minimap_size
        and minimap_y <= my <= minimap_y + gui.minimap_size
    ):
        return False

    map_h, map_w = gui.env.height, gui.env.width
    content_area = gui.minimap_size - 30
    scale_x = content_area / map_w
    scale_y = content_area / map_h
    scale = min(scale_x, scale_y)

    scaled_w = int(map_w * scale)
    scaled_h = int(map_h * scale)
    offset_x = (gui.minimap_size - scaled_w) // 2
    offset_y = 18 + (gui.minimap_size - 18 - scaled_h) // 2

    local_x = mx - minimap_x - offset_x
    local_y = my - minimap_y - offset_y

    if local_x < 0 or local_y < 0:
        return True

    editor_mode = bool(getattr(gui, "ai_mission_graph_editor_enabled", False)) and getattr(
        gui, "ai_mission_graph_draft", None
    ) is not None
    if editor_mode:
        graph = gui.ai_mission_graph_draft
        layout = dict(getattr(gui, "ai_mission_graph_layout", {}) or {})
        node_ids = sorted(list(getattr(graph, "nodes", {}).keys()), key=stable_node_sort_key)
        if not node_ids:
            return True

        if len(layout) != len(node_ids):
            total = len(node_ids)
            layout = {
                node_id: (0.08 + 0.84 * (float(idx) / float(max(1, total - 1))), 0.5)
                for idx, node_id in enumerate(node_ids)
            }
            gui.ai_mission_graph_layout = layout

        def _node_pixel(node_id):
            nxn, nyn = layout.get(node_id, (0.5, 0.5))
            px = offset_x + int(float(nxn) * max(1, scaled_w - 1))
            py = offset_y + int(float(nyn) * max(1, scaled_h - 1))
            return px, py

        nearest = None
        best_d = float("inf")
        for node_id in node_ids:
            px, py = _node_pixel(node_id)
            d = (float(local_x) - float(px)) ** 2 + (float(local_y) - float(py)) ** 2
            if d < best_d:
                best_d = d
                nearest = node_id

        if nearest is None or best_d > float(max(9, int(scale) + 7) ** 2):
            return True

        if int(button) == 1:
            gui.ai_mission_graph_boss_node = nearest
            gui.message = f"Mission editor: boss room node = {nearest}"
            return True

        if int(button) == 3:
            pending = getattr(gui, "ai_mission_graph_pending_lock_source", None)
            if pending is None:
                gui.ai_mission_graph_pending_lock_source = nearest
                gui.message = f"Mission editor: lock source node = {nearest}"
                return True

            src = normalize_node_id(pending)
            dst = normalize_node_id(nearest)
            gui.ai_mission_graph_pending_lock_source = None
            if src is None or dst is None:
                gui.message = "Mission editor: ignored invalid node reference"
                return True
            if src == dst:
                gui.message = "Mission editor: ignored self-lock edge"
                return True

            staged = list(getattr(gui, "ai_mission_graph_locked_edges", []) or [])
            pair = (src, dst)
            if pair in staged:
                staged = [p for p in staged if tuple(p) != pair]
                gui.message = f"Mission editor: removed locked edge {src}->{dst}"
            else:
                staged.append(pair)
                gui.message = f"Mission editor: staged locked edge {src}->{dst}"
            gui.ai_mission_graph_locked_edges = staged
            return True

        return True

    tile_c = int(local_x / scale)
    tile_r = int(local_y / scale)

    if 0 <= tile_r < map_h and 0 <= tile_c < map_w:
        mods = 0
        try:
            if pygame_module is not None:
                mods = int(pygame_module.key.get_mods())
        except (AttributeError, RuntimeError, ValueError, TypeError):
            mods = 0

        shift_held = bool(mods & getattr(pygame_module, "KMOD_SHIFT", 0)) if pygame_module is not None else False
        ctrl_held = bool(mods & getattr(pygame_module, "KMOD_CTRL", 0)) if pygame_module is not None else False
        alt_held = bool(mods & getattr(pygame_module, "KMOD_ALT", 0)) if pygame_module is not None else False

        # Mixed-initiative authoring shortcuts on minimap:
        # Shift+click: stage Boss Room placement anchor for next AI generation.
        # Ctrl+click: stage Locked Door anchor for next AI generation.
        # Alt+click: stage Small Key anchor for next AI generation.
        if shift_held or ctrl_held or alt_held:
            denom_r = max(1, map_h - 1)
            denom_c = max(1, map_w - 1)
            norm = (float(tile_r) / float(denom_r), float(tile_c) / float(denom_c))
            if shift_held:
                gui.ai_constraint_boss_norm = norm
                gui.message = f"Staged Boss Room anchor at ({tile_r}, {tile_c}) for next AI generation"
            if ctrl_held:
                gui.ai_constraint_lock_norm = norm
                gui.message = f"Staged Locked Door anchor at ({tile_r}, {tile_c}) for next AI generation"
            if alt_held:
                gui.ai_constraint_key_norm = norm
                gui.message = f"Staged Small Key anchor at ({tile_r}, {tile_c}) for next AI generation"
            return True

        gui.view_offset_x = int(tile_c * gui.TILE_SIZE - (gui.screen_w - gui.SIDEBAR_WIDTH) / 2)
        gui.view_offset_y = int(tile_r * gui.TILE_SIZE - (gui.screen_h - gui.HUD_HEIGHT) / 2)
        gui._clamp_view_offset()
        gui.message = f"Jumped to ({tile_r}, {tile_c})"

    return True
