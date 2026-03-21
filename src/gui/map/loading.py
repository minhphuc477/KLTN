"""Map and visual-loading helpers extracted from gui_runner."""

from __future__ import annotations

from typing import Any


def load_visual_assets(
    gui: Any,
    *,
    templates_dir: str | None,
    link_sprite_path: str | None,
    pygame: Any,
    os_module: Any,
    logger: Any,
    semantic_palette: dict,
) -> bool:
    """Optionally override GUI assets with extracted visual tiles/sprites."""
    try:
        from src.data_processing.visual_extractor import extract_grid  # noqa: F401
        from PIL import Image
    except Exception:
        return False

    if templates_dir and os_module.path.isdir(templates_dir):
        for fn in sorted(os_module.listdir(templates_dir)):
            if not fn.lower().endswith(".png"):
                continue
            name = os_module.path.splitext(fn)[0]
            try:
                im = Image.open(os_module.path.join(templates_dir, fn)).convert("RGBA")
                surf = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
                ln = name.lower()
                if "floor" in ln or ln == "f":
                    gui.images[semantic_palette["FLOOR"]] = pygame.transform.scale(surf, (gui.TILE_SIZE, gui.TILE_SIZE))
                if "wall" in ln:
                    gui.images[semantic_palette["WALL"]] = pygame.transform.scale(surf, (gui.TILE_SIZE, gui.TILE_SIZE))
                if "door" in ln:
                    gui.images[semantic_palette["DOOR_OPEN"]] = pygame.transform.scale(surf, (gui.TILE_SIZE, gui.TILE_SIZE))
                if "key" in ln:
                    gui.images[semantic_palette["KEY"]] = pygame.transform.scale(surf, (gui.TILE_SIZE, gui.TILE_SIZE))
            except Exception:
                continue

    if link_sprite_path and os_module.path.exists(link_sprite_path):
        try:
            from PIL import Image

            im = Image.open(link_sprite_path).convert("RGBA")
            im = im.resize((gui.TILE_SIZE - 4, gui.TILE_SIZE - 4), Image.NEAREST)
            gui.link_img = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
        except Exception as e:
            logger.warning("Failed to load link sprite from %s: %s", link_sprite_path, e)

    logger.info("Loaded %d visual assets", len([k for k in gui.images if k in (1, 2, 10, 12)]))
    return True


def load_visual_map(
    gui: Any,
    *,
    image_path: str,
    templates_dir: str | None,
) -> bool:
    """Create a GUI map from a screenshot and switch to it."""
    try:
        from src.data_processing.visual_integration import (
            visual_extract_to_room,
            make_stitched_for_single_room,
            infer_inventory_from_room,  # noqa: F401
        )
    except Exception:
        return False

    try:
        ids, _conf = visual_extract_to_room(image_path, templates_dir or "")
        stitched = make_stitched_for_single_room(ids)
        gui.maps = [stitched.global_grid]
        gui.current_map_idx = 0
        gui._load_current_map()
        gui.message = f"Loaded visual map: {image_path}"
        return True
    except Exception as e:
        gui.message = f"Visual load failed: {e}"
        return False


def place_items_from_graph(
    gui: Any,
    *,
    grid: Any,
    graph: Any,
    room_positions: dict,
    room_to_node: dict,
    logger: Any,
    semantic_palette: dict,
) -> None:
    """Materialize graph-declared items into semantic grid tiles."""
    if not graph or not room_positions or not room_to_node:
        return

    node_to_room = {v: k for k, v in room_to_node.items()}

    room_height = 16
    room_width = 11
    items_placed = 0

    for node_id in graph.nodes():
        attrs = graph.nodes[node_id]
        room_pos = node_to_room.get(node_id)
        if room_pos is None or room_pos not in room_positions:
            continue

        r_off, c_off = room_positions[room_pos]

        def find_floor_position(offset_r: int = 0, offset_c: int = 0):
            center_r = r_off + room_height // 2 + offset_r
            center_c = c_off + room_width // 2 + offset_c
            for dr in range(-5, 6):
                for dc in range(-3, 4):
                    r = center_r + dr
                    c = center_c + dc
                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                        if grid[r, c] == semantic_palette["FLOOR"]:
                            return (r, c)
            return None

        if attrs.get("has_key", False):
            pos = find_floor_position(offset_r=-2, offset_c=0)
            if pos:
                grid[pos[0], pos[1]] = semantic_palette["KEY_SMALL"]
                items_placed += 1
                logger.debug("Placed KEY at %s (room %s, node %d)", pos, room_pos, node_id)

        if attrs.get("has_item", False):
            pos = find_floor_position(offset_r=0, offset_c=-2)
            if pos:
                grid[pos[0], pos[1]] = semantic_palette["KEY_ITEM"]
                items_placed += 1
                logger.debug("Placed KEY_ITEM at %s (room %s, node %d)", pos, room_pos, node_id)

    if items_placed > 0:
        logger.info("_place_items_from_graph: placed %d items into grid", items_placed)


def load_current_map(
    gui: Any,
    *,
    os_module: Any,
    logger: Any,
    zelda_logic_env_cls: Any,
    sanity_checker_cls: Any,
    semantic_palette: dict,
) -> None:
    """Load and initialize current map state in GUI."""
    current_dungeon = gui.maps[gui.current_map_idx]

    if hasattr(current_dungeon, "global_grid"):
        grid = current_dungeon.global_grid.copy()
        graph = current_dungeon.graph
        room_to_node = current_dungeon.room_to_node
        room_positions = current_dungeon.room_positions
        node_to_room = getattr(current_dungeon, "node_to_room", None)

        if graph and room_positions and room_to_node:
            gui._place_items_from_graph(grid, graph, room_positions, room_to_node)
    else:
        grid = current_dungeon
        graph = None
        room_to_node = None
        room_positions = None
        node_to_room = None

    gui.env = zelda_logic_env_cls(
        grid,
        render_mode=False,
        graph=graph,
        room_to_node=room_to_node,
        room_positions=room_positions,
        node_to_room=node_to_room,
    )
    gui.solver = None
    gui.auto_path = []
    gui.auto_step_idx = 0
    gui.auto_mode = False
    gui.block_push_animations = []

    if os_module.environ.get("KLTN_DEBUG_TEST_PATH") == "1":
        gui._test_path = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (8, 8), (8, 9), (8, 10)]
        print(f"[LOAD_MAP] _test_path ENABLED: {len(gui._test_path)} points at {gui._test_path[0]} to {gui._test_path[-1]}")
    else:
        gui._test_path = None

    gui.solver_result = None
    gui.current_keys_held = 0
    gui.current_keys_used = 0
    gui.current_edge_types = []

    gui.total_keys = len(gui.env._find_all_positions(semantic_palette["KEY_SMALL"]))
    bomb_items = gui.env._find_all_positions(semantic_palette["ITEM_MINOR"])
    key_items = gui.env._find_all_positions(semantic_palette["KEY_ITEM"])
    gui.total_bombs = len(bomb_items) + len(key_items)
    gui.total_boss_keys = len(gui.env._find_all_positions(semantic_palette["KEY_BOSS"]))
    gui.keys_collected = 0
    gui.bombs_collected = 0
    gui.boss_keys_collected = 0
    gui.keys_used = 0
    gui.bombs_used = 0
    gui.boss_keys_used = 0
    gui.collected_items = []
    gui.collected_positions = set()
    gui.used_items = []

    gui.path_items_summary = {}
    gui.path_item_positions = {}
    gui.search_heatmap = {}

    if gui.renderer and gui.env.start_pos:
        gui.renderer.set_agent_position(gui.env.start_pos[0], gui.env.start_pos[1], immediate=True)

    checker = sanity_checker_cls(grid)
    is_valid, errors = checker.check_all()

    if not is_valid:
        gui.message = f"Map Error: {errors[0]}"
    else:
        gui.message = f"Map {gui.current_map_idx + 1}/{len(gui.maps)} - Press SPACE to solve"

    gui._auto_fit_zoom()
    try:
        gui._center_view()
    except Exception:
        pass

    try:
        logger.info(
            "Map loaded: %dx%d, TILE_SIZE=%d, view_offset=(%d,%d), images=%d",
            gui.env.width if gui.env else 0,
            gui.env.height if gui.env else 0,
            gui.TILE_SIZE,
            getattr(gui, "view_offset_x", 0),
            getattr(gui, "view_offset_y", 0),
            len(getattr(gui, "images", {})),
        )
    except Exception:
        pass

    try:
        gui._start_preview_for_current_map()
    except Exception:
        logger.exception("Failed to start preview for current map")
