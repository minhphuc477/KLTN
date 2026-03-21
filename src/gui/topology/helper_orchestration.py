"""Topology helper orchestration bridges for ZeldaGUI wrappers."""

from __future__ import annotations


def export_topology(*, gui, export_topology_helper):
    return export_topology_helper(gui)


def room_for_global_position(*, pos, room_positions, room_for_global_position_helper):
    return room_for_global_position_helper(pos, room_positions)


def node_has_small_key(*, attrs, node_has_small_key_helper):
    return node_has_small_key_helper(attrs)


def node_has_critical_content(*, graph, node_id, node_has_critical_content_helper):
    return node_has_critical_content_helper(graph, node_id)


def build_room_adjacency_from_graph(*, graph, room_to_node, node_to_room, build_room_adjacency_from_graph_helper):
    return build_room_adjacency_from_graph_helper(graph, room_to_node, node_to_room)
