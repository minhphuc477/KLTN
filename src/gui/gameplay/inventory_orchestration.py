"""Inventory and path-item orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def update_inventory_and_hud(*, gui, logger, update_inventory_and_hud_helper):
    update_inventory_and_hud_helper(gui, logger)


def remove_from_path_items(*, gui, pos, item_type, logger, remove_from_path_items_helper):
    remove_from_path_items_helper(gui, pos, item_type, logger)


def track_item_collection(
    *,
    gui,
    old_state,
    new_state,
    time_module,
    logger,
    pop_effect_cls,
    item_collection_effect_cls,
    track_item_collection_helper,
):
    track_item_collection_helper(
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
    track_item_usage_helper,
):
    track_item_usage_helper(gui, old_state, new_state, time_module, logger, item_usage_effect_cls)


def scan_and_mark_items(*, gui, semantic_palette, logger, item_marker_effect_cls, scan_and_mark_items_helper):
    scan_and_mark_items_helper(gui, semantic_palette, logger, item_marker_effect_cls)


def apply_pickup_at(
    *,
    gui,
    pos,
    semantic_palette,
    logger,
    time_module,
    item_collection_effect_cls,
    apply_pickup_at_helper,
):
    return apply_pickup_at_helper(
        gui,
        pos,
        semantic_palette,
        logger,
        time_module,
        item_collection_effect_cls,
    )


def render_item_legend(*, gui, surface, pygame, render_item_legend_helper):
    render_item_legend_helper(gui, surface, pygame)


def sync_inventory_counters(*, gui, sync_inventory_counters_helper):
    sync_inventory_counters_helper(gui)


def scan_items_along_path(*, gui, semantic_palette, logger, path, scan_items_along_path_helper):
    return scan_items_along_path_helper(gui, semantic_palette, logger, path=path)


def get_path_items_display_text(*, gui, get_path_items_display_text_helper):
    return get_path_items_display_text_helper(gui)
