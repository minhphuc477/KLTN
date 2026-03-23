"""Topology orchestration bridges for ZeldaGUI methods."""

from __future__ import annotations


def render_topology_overlay(gui, surface, *, pygame, render_topology_overlay_helper):
    """Render topology overlay for the current map."""
    current = gui.maps[gui.current_map_idx]
    render_topology_overlay_helper(
        surface=surface,
        current=current,
        tile_size=gui.TILE_SIZE,
        view_offset_x=gui.view_offset_x,
        view_offset_y=gui.view_offset_y,
        pygame=pygame,
    )


def capture_precheck_snapshot(gui, current, reason, *, capture_precheck_snapshot_helper):
    """Capture topology state to support undo prune."""
    gui._precheck_snapshot = capture_precheck_snapshot_helper(current, reason=reason)


def update_env_topology_view(gui, current, *, update_env_topology_view_helper):
    """Synchronize topology attributes into current env."""
    update_env_topology_view_helper(getattr(gui, "env", None), current)


def prune_dead_end_topology(
    gui,
    current,
    preserve_rooms,
    *,
    logger,
    prune_dead_end_topology_flow_helper,
    build_room_adjacency_fn,
    node_has_critical_content_fn,
):
    """Delegate dead-end topology pruning with injected dependencies."""
    return prune_dead_end_topology_flow_helper(
        gui=gui,
        current=current,
        preserve_rooms=preserve_rooms,
        logger=logger,
        build_room_adjacency_fn=build_room_adjacency_fn,
        node_has_critical_content_fn=node_has_critical_content_fn,
    )


def run_prechecks_and_optional_prune(
    gui,
    *,
    logger,
    np_module,
    semantic_palette,
    action_deltas,
    topology_has_path_fn,
    min_locked_between_fn,
    walkable_grid_reachable_fn,
    node_has_small_key_fn,
    room_for_global_position_fn,
    zelda_dungeon_adapter,
    capture_snapshot_fn,
    update_env_topology_view_fn,
    prune_dead_end_topology_fn,
    run_prechecks_and_optional_prune_flow_helper,
):
    """Run map prechecks and optional topology pruning before solve."""
    current = gui.maps[gui.current_map_idx]
    return run_prechecks_and_optional_prune_flow_helper(
        gui=gui,
        current=current,
        logger=logger,
        np_module=np_module,
        semantic_palette=semantic_palette,
        action_deltas=action_deltas,
        topology_has_path_fn=topology_has_path_fn,
        min_locked_between_fn=min_locked_between_fn,
        walkable_grid_reachable_fn=walkable_grid_reachable_fn,
        node_has_small_key_fn=node_has_small_key_fn,
        room_for_global_position_fn=room_for_global_position_fn,
        zelda_dungeon_adapter=zelda_dungeon_adapter,
        capture_snapshot_fn=capture_snapshot_fn,
        update_env_topology_view_fn=update_env_topology_view_fn,
        prune_dead_end_topology_fn=prune_dead_end_topology_fn,
    )


def undo_prune(gui, *, logger, update_env_topology_view_fn, undo_prune_flow_helper):
    """Undo the previous prune operation if snapshot data exists."""
    current = gui.maps[gui.current_map_idx]
    return undo_prune_flow_helper(
        gui=gui,
        current=current,
        logger=logger,
        update_env_topology_view_fn=update_env_topology_view_fn,
    )
