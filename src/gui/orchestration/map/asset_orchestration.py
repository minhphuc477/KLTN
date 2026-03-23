"""Map asset and visual-load orchestration bridges for ZeldaGUI."""

from __future__ import annotations

import os

from src.gui.map.loading import (
    load_visual_assets as _load_visual_assets_helper,
    load_visual_map as _load_visual_map_helper,
    place_items_from_graph as _place_items_from_graph_helper,
)


def load_visual_assets(
    *,
    gui,
    templates_dir,
    link_sprite_path,
    pygame,
    logger,
    semantic_palette,
):
    return _load_visual_assets_helper(
        gui,
        templates_dir=templates_dir,
        link_sprite_path=link_sprite_path,
        pygame=pygame,
        os_module=os,
        logger=logger,
        semantic_palette=semantic_palette,
    )


def load_visual_map(*, gui, image_path, templates_dir):
    return _load_visual_map_helper(
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
):
    _place_items_from_graph_helper(
        gui,
        grid=grid,
        graph=graph,
        room_positions=room_positions,
        room_to_node=room_to_node,
        logger=logger,
        semantic_palette=semantic_palette,
    )
