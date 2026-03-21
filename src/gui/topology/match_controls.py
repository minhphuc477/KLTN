"""Helpers for topology room-node matching controls in the GUI."""

from typing import Any


def match_missing_nodes(gui: Any, matcher_cls: Any, logger: Any) -> None:
    """Infer missing room-node mappings and apply confident matches."""
    current = gui.maps[gui.current_map_idx]
    graph = getattr(current, "graph", None)
    rooms = getattr(current, "rooms", None)
    room_positions = getattr(current, "room_positions", None)
    room_to_node = dict(getattr(current, "room_to_node", {}) or {})

    if graph is None or rooms is None:
        gui._set_message("No topology available for this map", 3.0)
        return

    matcher = matcher_cls()
    proposed_r2n, proposed_n2r, confidences = matcher.infer_missing_mappings(
        rooms,
        graph,
        room_positions=room_positions,
        room_to_node=room_to_node,
    )

    if not proposed_r2n:
        gui._set_message("No proposals for missing nodes", 3.0)
        return

    current.match_proposals = (proposed_r2n, proposed_n2r, confidences)

    threshold_widget = next(
        (w for w in gui.widget_manager.widgets if getattr(w, "control_name", None) == "match_threshold"),
        None,
    )
    try:
        threshold_val = (
            float(threshold_widget.options[threshold_widget.selected])
            if threshold_widget
            else getattr(gui, "match_apply_threshold", 0.85)
        )
    except Exception:
        threshold_val = getattr(gui, "match_apply_threshold", 0.85)

    confident = {
        room_pos: node_id
        for room_pos, node_id in proposed_r2n.items()
        if confidences.get(node_id, 0) >= threshold_val
    }
    tentative = {
        room_pos: node_id
        for room_pos, node_id in proposed_r2n.items()
        if confidences.get(node_id, 0) < threshold_val
    }

    applied = 0
    if confident:
        snapshot = dict(getattr(current, "room_to_node", {}) or {})
        applied_nodes = list(confident.items())
        gui.match_undo_stack.append(snapshot)

        for room_pos, node_id in confident.items():
            current.room_to_node[room_pos] = node_id
            current.rooms[room_pos].graph_node_id = node_id
            applied += 1

        gui._set_message(f"Applied {applied} confident matches. {len(tentative)} tentative remain", 4.0)
        logger.info("Applied matches: %s", applied_nodes)
    else:
        gui._set_message(
            f"No matches above threshold ({threshold_val}). {len(proposed_r2n)} proposals available",
            5.0,
        )
        logger.info("Match proposals (none auto-applied):")
        for node, room in proposed_n2r.items():
            logger.info("  Node %s -> Room %s (conf=%.2f)", node, room, confidences.get(node, 0.0))


def undo_last_match(gui: Any, logger: Any) -> None:
    """Undo the last applied room-node mapping snapshot."""
    current = gui.maps[gui.current_map_idx]
    if not gui.match_undo_stack:
        gui._set_message("Nothing to undo", 2.0)
        return

    snapshot = gui.match_undo_stack.pop()
    current.room_to_node = dict(snapshot)
    for rpos, room in current.rooms.items():
        room.graph_node_id = current.room_to_node.get(rpos)

    if hasattr(current, "match_proposals"):
        del current.match_proposals

    gui._set_message("Undo: restored previous mapping", 3.0)
    logger.info("Undo applied: restored previous mapping")


def apply_tentative_matches(gui: Any, logger: Any) -> None:
    """Apply staged tentative matches above threshold and keep the rest staged."""
    current = gui.maps[gui.current_map_idx]
    if not hasattr(current, "match_proposals"):
        gui._set_message("No staged proposals to apply", 2.0)
        return

    proposed_r2n, proposed_n2r, confidences = current.match_proposals

    threshold_widget = next(
        (w for w in gui.widget_manager.widgets if getattr(w, "control_name", None) == "match_threshold"),
        None,
    )
    try:
        threshold_val = (
            float(threshold_widget.options[threshold_widget.selected])
            if threshold_widget
            else getattr(gui, "match_apply_threshold", 0.85)
        )
    except Exception:
        threshold_val = getattr(gui, "match_apply_threshold", 0.85)

    to_apply = {
        room_pos: node_id
        for room_pos, node_id in proposed_r2n.items()
        if confidences.get(node_id, 0) >= threshold_val
    }

    if not to_apply:
        gui._set_message("No proposals meet threshold", 2.0)
        return

    snapshot = dict(getattr(current, "room_to_node", {}) or {})
    gui.match_undo_stack.append(snapshot)
    applied = 0
    for room_pos, node_id in to_apply.items():
        current.room_to_node[room_pos] = node_id
        current.rooms[room_pos].graph_node_id = node_id
        applied += 1

    for node_id in list(to_apply.values()):
        proposed_n2r.pop(node_id, None)
    for room_pos in list(to_apply.keys()):
        proposed_r2n.pop(room_pos, None)

    if not proposed_r2n:
        del current.match_proposals

    gui._set_message(f"Applied {applied} tentative matches", 3.0)
    logger.info("Applied tentative matches: %d", applied)