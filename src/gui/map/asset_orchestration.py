"""Map asset and visual-load orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def load_visual_assets(
    *,
    gui,
    templates_dir,
    link_sprite_path,
    pygame,
    os_module,
    logger,
    semantic_palette,
    load_visual_assets_helper,
):
    return load_visual_assets_helper(
        gui,
        templates_dir=templates_dir,
        link_sprite_path=link_sprite_path,
        pygame=pygame,
        os_module=os_module,
        logger=logger,
        semantic_palette=semantic_palette,
    )


def load_visual_map(*, gui, image_path, templates_dir, load_visual_map_helper):
    return load_visual_map_helper(
        gui,
        image_path=image_path,
        templates_dir=templates_dir,
    )


def place_items_from_graph(
    *,
    gui,
    grid,
    graph,
    room_positions,
    room_to_node,
    logger,
    semantic_palette,
    place_items_from_graph_helper,
):
    place_items_from_graph_helper(
        gui,
        grid=grid,
        graph=graph,
        room_positions=room_positions,
        room_to_node=room_to_node,
        logger=logger,
        semantic_palette=semantic_palette,
    )
