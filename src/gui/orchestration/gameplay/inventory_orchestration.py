"""Inventory and path-item orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.gameplay.inventory_manager import (
    remove_from_path_items as _remove_from_path_items,
    sync_inventory_counters as _sync_inventory_counters,
    track_item_collection as _track_item_collection,
    track_item_usage as _track_item_usage,
    update_inventory_and_hud as _update_inventory_and_hud,
)
from src.gui.gameplay.item_markers import apply_pickup_at as _apply_pickup_at, scan_and_mark_items as _scan_and_mark_items
from src.gui.gameplay.path_analysis import scan_items_along_path as _scan_items_along_path
from src.gui.rendering.inventory_display import (
    get_path_items_display_text as _get_path_items_display_text,
    render_item_legend as _render_item_legend,
)


def update_inventory_and_hud(*, gui, logger):
    _update_inventory_and_hud(gui, logger)


def remove_from_path_items(*, gui, pos, item_type, logger):
    _remove_from_path_items(gui, pos, item_type, logger)


def track_item_collection(
    *,
    gui,
    old_state,
    new_state,
    time_module,
    logger,
    pop_effect_cls,
    item_collection_effect_cls,
):
    _track_item_collection(
        gui,
        old_state,
        new_state,
        time_module,
        logger,
        pop_effect_cls,
        item_collection_effect_cls,
    )


def track_item_usage(
    *,
    gui,
    old_state,
    new_state,
    time_module,
    logger,
    item_usage_effect_cls,
):
    _track_item_usage(gui, old_state, new_state, time_module, logger, item_usage_effect_cls)


def scan_and_mark_items(*, gui, semantic_palette, logger, item_marker_effect_cls):
    _scan_and_mark_items(gui, semantic_palette, logger, item_marker_effect_cls)


def apply_pickup_at(
    *,
    gui,
    pos,
    semantic_palette,
    logger,
    time_module,
    item_collection_effect_cls,
):
    return _apply_pickup_at(
        gui,
        pos,
        semantic_palette,
        logger,
        time_module,
        item_collection_effect_cls,
    )


def render_item_legend(*, gui, surface, pygame):
    _render_item_legend(gui, surface, pygame)


def sync_inventory_counters(*, gui):
    _sync_inventory_counters(gui)


def scan_items_along_path(*, gui, semantic_palette, logger, path):
    return _scan_items_along_path(gui, semantic_palette, logger, path=path)


def get_path_items_display_text(*, gui):
    return _get_path_items_display_text(gui)
